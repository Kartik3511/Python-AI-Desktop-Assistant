"""OpenAI provider — GPT models with function calling."""

import json
import logging
from typing import Any

from openai import OpenAI

from core.providers.base import BaseProvider, TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """OpenAI provider for JARVIS using the official OpenAI Python SDK.

    Handles chat completions, function/tool calling loop, and conversation
    history management with automatic trimming.
    """

    def __init__(self, provider_config: dict, system_prompt: str) -> None:
        """Initialize the OpenAI provider.

        Args:
            provider_config: Configuration dictionary with 'name', 'model', and 'api_key'.
            system_prompt: System prompt defining JARVIS's personality and instructions.
        """
        super().__init__(provider_config, system_prompt)
        self._client = OpenAI(
            api_key=self.api_key,
            max_retries=0,
            timeout=10.0,
        )
        self._tools = self._build_openai_tools()

        self._history: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        logger.info("Initialized OpenAI provider with model '%s'", self.model)

    def _build_openai_tools(self) -> list[dict]:
        """Convert TOOL_DEFINITIONS to OpenAI tool format.

        Returns:
            List of tool schemas in OpenAI function calling format.
        """
        tools: list[dict] = []
        for tool_def in TOOL_DEFINITIONS:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_def["name"],
                    "description": tool_def["description"],
                    "parameters": tool_def["parameters"],
                },
            })
        return tools

    def think(self, user_text: str) -> str:
        """Process user input and return a text response.

        Handles tool execution in a loop for up to 5 rounds. Exceptions are not
        caught so the Brain orchestrator can catch them and fall back to another
        provider.

        Args:
            user_text: Transcribed user speech.

        Returns:
            The final text response from the model.
        """
        history_snapshot = list(self._history)
        self._history.append({"role": "user", "content": user_text})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=self._history,
                tools=self._tools,
                tool_choice="auto",
                temperature=0.7,
            )

            for _ in range(5):
                message = response.choices[0].message
                if not message.tool_calls:
                    break

                self._history.append(message)

                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    raw_args = tool_call.function.arguments
                    if isinstance(raw_args, str):
                        try:
                            func_args = json.loads(raw_args) if raw_args else {}
                        except (json.JSONDecodeError, TypeError):
                            func_args = {}
                    elif isinstance(raw_args, dict):
                        func_args = raw_args
                    else:
                        func_args = {}

                    result = execute_tool(func_name, func_args)
                    self._history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    })

                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=self._history,
                    tools=self._tools,
                    tool_choice="auto",
                    temperature=0.7,
                )

            final_text = response.choices[0].message.content or ""
            self._history.append({"role": "assistant", "content": final_text})

            # Keep system prompt at index 0 + last _max_history entries
            if len(self._history) > self._max_history + 1:
                self._history = [self._history[0]] + self._history[-self._max_history:]

            return final_text
        except Exception:
            self._history = history_snapshot
            raise


    def clear_history(self) -> None:
        """Reset conversation history back to initial system prompt."""
        self._history = [{"role": "system", "content": self.system_prompt}]
        logger.info("OpenAI conversation history cleared.")

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        """Record an external/local conversation turn into history."""
        clean_text = assistant_text.replace("[FOLLOW_UP]", "").strip()
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": clean_text})
        if len(self._history) > self._max_history + 1:
            self._history = [self._history[0]] + self._history[-self._max_history:]

