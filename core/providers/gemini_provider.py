"""Gemini provider — Google Gemini API with function calling."""

import json
import logging
from typing import Any

from google import genai
from google.genai import types

from core.providers.base import BaseProvider, TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)


class GeminiProvider(BaseProvider):
    """Google Gemini provider with native function calling support."""

    def __init__(self, provider_config: dict, system_prompt: str):
        """Initialize the Google Gemini provider.

        Args:
            provider_config: Provider configuration containing model and api_key.
            system_prompt: The system prompt describing assistant behavior.
        """
        super().__init__(provider_config, system_prompt)
        self._client = genai.Client(api_key=self.api_key)
        self._tools = self._build_gemini_tools()
        self._history: list[types.Content] = []
        logger.info("GeminiProvider initialized with model '%s'", self.model)

    def _json_type_to_gemini(self, json_type: str) -> types.Type:
        """Map JSON schema types to Gemini types.Type enum.

        Args:
            json_type: JSON schema type string.

        Returns:
            The mapped types.Type enum value (defaults to STRING).
        """
        mapping = {
            "string": types.Type.STRING,
            "integer": types.Type.INTEGER,
            "number": types.Type.NUMBER,
            "boolean": types.Type.BOOLEAN,
            "object": types.Type.OBJECT,
            "array": types.Type.ARRAY,
        }
        return mapping.get(
            json_type.lower() if isinstance(json_type, str) else "",
            types.Type.STRING,
        )

    def _build_gemini_tools(self) -> list[types.Tool]:
        """Convert TOOL_DEFINITIONS to Gemini's native Tool format.

        Returns:
            A list containing one types.Tool with all function declarations.
        """
        declarations: list[types.FunctionDeclaration] = []

        for tool_def in TOOL_DEFINITIONS:
            name = tool_def.get("name", "")
            description = tool_def.get("description", "")
            params = tool_def.get("parameters", {})

            properties: dict[str, types.Schema] = {}
            for prop_name, prop_def in params.get("properties", {}).items():
                prop_type = prop_def.get("type", "string")
                prop_desc = prop_def.get("description", "")
                properties[prop_name] = types.Schema(
                    type=self._json_type_to_gemini(prop_type),
                    description=prop_desc,
                )

            parameters_schema = types.Schema(
                type=self._json_type_to_gemini(params.get("type", "object")),
                properties=properties,
                required=params.get("required"),
            )

            declaration = types.FunctionDeclaration(
                name=name,
                description=description,
                parameters=parameters_schema,
            )
            declarations.append(declaration)

        return [types.Tool(function_declarations=declarations)]

    def think(self, user_text: str) -> str:
        """Process user input and return a text response.

        Sends the user query to Gemini, executes any requested function calls,
        feeds the function responses back, and continues until a final text
        response is generated (max 5 tool call iterations).

        Exceptions are intentionally not caught here so the Brain orchestrator
        can detect failures and fall back to the next available provider.

        Args:
            user_text: Transcribed user speech.

        Returns:
            The final text response from the model.
        """
        history_snapshot = list(self._history)
        # Add user message to history
        self._history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_text)],
            )
        )


        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=self._tools,
            temperature=0.7,
        )

        try:
            # Initial generate content call (let exceptions propagate)
            response = self._client.models.generate_content(
                model=self.model,
                contents=self._history,
                config=config,
            )

            # Handle tool calls in a loop (max 5 iterations)
            for _ in range(5):
                function_calls = self._extract_function_calls(response)
                if not function_calls:
                    break

                # Add model response with function call parts to history
                if response.candidates and response.candidates[0].content:
                    self._history.append(response.candidates[0].content)

                # Execute each function call and collect responses
                tool_response_parts: list[types.Part] = []
                for call in function_calls:
                    name = getattr(call, "name", None)
                    args = getattr(call, "args", None)
                    if name is None and hasattr(call, "function_call"):
                        name = getattr(call.function_call, "name", None)
                        args = getattr(call.function_call, "args", None)

                    if not name:
                        continue

                    call_args = dict(args) if args else {}
                    result = execute_tool(name, call_args)

                    tool_response_parts.append(
                        types.Part.from_function_response(
                            name=name,
                            response={"result": result},
                        )
                    )

                # Add function responses to history
                self._history.append(
                    types.Content(
                        role="user",
                        parts=tool_response_parts,
                    )
                )

                # Call generate_content again (let exceptions propagate)
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=self._history,
                    config=config,
                )

            # Once no more function calls, extract final text
            final_text = self._extract_text(response)

            # Add model's final response to history
            if response and response.candidates and response.candidates[0].content:
                self._history.append(response.candidates[0].content)
            elif final_text:
                self._history.append(
                    types.Content(
                        role="model",
                        parts=[types.Part.from_text(text=final_text)],
                    )
                )

            # Trim history if exceeding max limit
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

            return final_text
        except Exception:
            # Revert history so unfulfilled user commands don't linger
            self._history = history_snapshot
            raise


    def _extract_function_calls(self, response: Any) -> list:
        """Try to get function_call parts from response.candidates[0].content.parts.

        Args:
            response: Model response object.

        Returns:
            List of function calls, or empty list on error.
        """
        try:
            if not response or not response.candidates:
                return []
            content = response.candidates[0].content
            if not content or not content.parts:
                return []
            return [
                part.function_call
                for part in content.parts
                if getattr(part, "function_call", None) is not None
            ]
        except Exception as e:
            logger.debug("Failed to extract function calls: %s", e)
            return []

    def _extract_text(self, response: Any) -> str:
        """Get text parts from response.candidates[0].content.parts, join with space.

        Args:
            response: Model response object.

        Returns:
            Combined text or fallback message if empty.
        """
        try:
            if not response or not response.candidates:
                return "I apologize, but I was unable to generate a response."
            content = response.candidates[0].content
            if not content or not content.parts:
                return "I apologize, but I was unable to generate a response."
            text_parts = [
                part.text
                for part in content.parts
                if getattr(part, "text", None)
            ]
            joined = " ".join(text_parts).strip()
            return joined if joined else "I have completed the requested task."
        except Exception as e:
            logger.debug("Failed to extract text: %s", e)
            return "I apologize, but I was unable to generate a response."

    def clear_history(self) -> None:
        """Reset conversation history."""
        self._history.clear()
        logger.info("GeminiProvider history cleared.")

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        """Record an external/local conversation turn into history."""
        clean_text = assistant_text.replace("[FOLLOW_UP]", "").strip()
        self._history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_text)],
            )
        )
        self._history.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=clean_text)],
            )
        )
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

