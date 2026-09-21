"""Groq provider — uses OpenAI-compatible API for Llama/Mixtral models."""

import json
import logging
from typing import Any

from openai import OpenAI

from core.providers.base import BaseProvider, TOOL_DEFINITIONS, execute_tool, split_sentences

logger = logging.getLogger(__name__)


class GroqProvider(BaseProvider):
    """LLM provider using Groq's OpenAI-compatible inference API."""

    def __init__(self, provider_config: dict[str, Any], system_prompt: str) -> None:
        """Initialize the Groq provider.

        Args:
            provider_config: Configuration dict with 'name', 'model', and 'api_key'.
            system_prompt: The Jarvis system prompt.
        """
        super().__init__(provider_config, system_prompt)
        self._client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1",
            max_retries=0,
            timeout=10.0,
        )

        self._tools = self._build_openai_tools()
        self._history: list[Any] = [{"role": "system", "content": self.system_prompt}]
        logger.info("GroqProvider initialized with model '%s'", self.model)

    def _build_openai_tools(self) -> list[dict[str, Any]]:
        """Convert TOOL_DEFINITIONS to OpenAI tool format.

        Returns:
            A list of OpenAI-formatted tool definitions.
        """
        tools: list[dict[str, Any]] = []
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

    def think_stream(self, user_text: str):
        """Process user input and yield text sentence-by-sentence as tokens stream in."""
        history_snapshot = list(self._history)
        self._history.append({"role": "user", "content": user_text})

        try:
            for round_idx in range(5):
                stream = self._client.chat.completions.create(
                    model=self.model,
                    messages=self._history,
                    tools=self._tools,
                    tool_choice="auto",
                    temperature=0.7,
                    stream=True,
                )

                # Peek at the stream to detect tool call vs text content
                first_delta = None
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta:
                        d = chunk.choices[0].delta
                        if d.tool_calls or d.content:
                            first_delta = d
                            break

                if not first_delta:
                    break

                # ── PATH A: Tool Call ─────────────────────────────
                if first_delta.tool_calls:
                    tool_calls_acc = {}

                    def _accumulate(d):
                        if d.tool_calls:
                            for tc in d.tool_calls:
                                idx = tc.index
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {
                                        "id": tc.id or "",
                                        "name": tc.function.name if tc.function and tc.function.name else "",
                                        "arguments": tc.function.arguments if tc.function and tc.function.arguments else "",
                                    }
                                else:
                                    if tc.id:
                                        tool_calls_acc[idx]["id"] += tc.id
                                    if tc.function:
                                        if tc.function.name:
                                            tool_calls_acc[idx]["name"] += tc.function.name
                                        if tc.function.arguments:
                                            tool_calls_acc[idx]["arguments"] += tc.function.arguments

                    _accumulate(first_delta)
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta:
                            _accumulate(chunk.choices[0].delta)

                    tool_calls_list = []
                    for idx in sorted(tool_calls_acc.keys()):
                        item = tool_calls_acc[idx]
                        tool_calls_list.append({
                            "id": item["id"],
                            "type": "function",
                            "function": {
                                "name": item["name"],
                                "arguments": item["arguments"],
                            }
                        })

                    self._history.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls_list,
                    })

                    logger.info(
                        "GroqProvider executing %d tool call(s) (round %d/5)",
                        len(tool_calls_list),
                        round_idx + 1,
                    )

                    for tc in tool_calls_list:
                        func_name = tc["function"]["name"]
                        raw_args = tc["function"]["arguments"]
                        try:
                            func_args = json.loads(raw_args) if raw_args else {}
                        except Exception:
                            func_args = {}

                        result = execute_tool(func_name, func_args)
                        self._history.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": str(result),
                        })
                    continue

                # ── PATH B: Natural Language Streaming ────────────
                else:
                    def _token_gen():
                        tokens = [first_delta.content]
                        yield first_delta.content
                        for chunk in stream:
                            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                                t = chunk.choices[0].delta.content
                                tokens.append(t)
                                yield t
                        full_text = "".join(tokens)
                        self._history.append({"role": "assistant", "content": full_text})
                        self._trim_history()

                    yield from split_sentences(_token_gen())
                    return
        except Exception:
            # Rollback history so failed requests don't pollute subsequent turns
            self._history = history_snapshot
            raise


    def think(self, user_text: str) -> str:
        """Process user input and return a text response."""
        sentences = list(self.think_stream(user_text))
        return " ".join(sentences)

    def clear_history(self) -> None:
        """Reset conversation history to initial system prompt."""
        self._history = [{"role": "system", "content": self.system_prompt}]
        logger.info("GroqProvider conversation history cleared.")

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        """Record an external/local conversation turn into history."""
        clean_text = assistant_text.replace("[FOLLOW_UP]", "").strip()
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": clean_text})
        self._trim_history()


    def _trim_history(self) -> None:
        """Trim history to keep the system message and last _max_history entries."""
        if len(self._history) > self._max_history + 1:
            system_msg = self._history[0]
            recent = self._history[-self._max_history:]
            # Ensure we do not start trimmed history with an orphaned tool response
            while recent and (
                (isinstance(recent[0], dict) and recent[0].get("role") == "tool")
                or getattr(recent[0], "role", None) == "tool"
            ):
                recent.pop(0)
            self._history = [system_msg] + recent
