"""
brain.py - Orchestrator that manages multiple LLM providers with fallback.

The Brain tries providers in the order defined in config.yaml. If the
primary provider fails (rate limit, high demand, timeout), it automatically
falls back to the next one. Once a provider succeeds, it becomes the
"sticky" provider for subsequent calls until it fails again.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Iterable, Optional

from core.memory_manager import get_memory_manager
from core.providers.base import BaseProvider
from core.intent_router import IntentRouter


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Provider Factory                                                   #
# ------------------------------------------------------------------ #

def _create_provider(provider_config: dict, system_prompt: str) -> Optional[BaseProvider]:
    """Instantiate a provider by name.

    Args:
        provider_config: Dict with 'name', 'model', 'api_key'.
        system_prompt: The Jarvis system prompt.

    Returns:
        A BaseProvider instance, or None if the provider can't be created.
    """
    name = provider_config.get("name", "").lower()
    api_key = provider_config.get("api_key", "")

    if not api_key or api_key.startswith("YOUR_"):
        logger.warning("Skipping provider '%s' — no API key configured.", name)
        return None

    try:
        if name == "gemini":
            from core.providers.gemini_provider import GeminiProvider
            return GeminiProvider(provider_config, system_prompt)

        elif name == "groq":
            from core.providers.groq_provider import GroqProvider
            return GroqProvider(provider_config, system_prompt)

        elif name == "openai":
            from core.providers.openai_provider import OpenAIProvider
            return OpenAIProvider(provider_config, system_prompt)

        else:
            logger.warning("Unknown provider: '%s'", name)
            return None

    except ImportError as e:
        logger.warning(
            "Could not import provider '%s' — missing dependency: %s", name, e
        )
        return None
    except Exception as e:
        logger.error("Failed to initialize provider '%s': %s", name, e)
        return None



# ------------------------------------------------------------------ #
#  Brain Stream Result                                                #
# ------------------------------------------------------------------ #

class BrainStreamResult:
    """Encapsulates a streamed response from Brain.think_stream.

    Yields clean sentences (with [FOLLOW_UP] stripped) and exposes
    metadata: needs_follow_up, full_text, and provider_name.
    """

    def __init__(self, sentence_generator: Iterable[str], provider_name: str = ""):
        self._generator = iter(sentence_generator)
        self.provider_name = provider_name
        self.needs_follow_up = False
        self.full_text = ""
        self._sentences: list[str] = []

    def __iter__(self):
        return self

    def __next__(self) -> str:
        while True:
            try:
                raw_sentence = next(self._generator)
            except StopIteration:
                # Fallback heuristic: check if response concludes with a question mark
                if not self.needs_follow_up and self.full_text:
                    last_sentence = self._sentences[-1] if self._sentences else self.full_text.strip()
                    if "?" in last_sentence:
                        self.needs_follow_up = True
                raise

            clean_sentence = raw_sentence
            if "[FOLLOW_UP]" in clean_sentence:
                self.needs_follow_up = True
                clean_sentence = clean_sentence.replace("[FOLLOW_UP]", "").strip()

            if "?" in clean_sentence:
                self.needs_follow_up = True

            if clean_sentence:
                self._sentences.append(clean_sentence)
                if self.full_text:
                    self.full_text += " " + clean_sentence
                else:
                    self.full_text = clean_sentence
                return clean_sentence


# ------------------------------------------------------------------ #
#  Brain Orchestrator                                                 #
# ------------------------------------------------------------------ #

class Brain:

    """Multi-provider LLM orchestrator with automatic fallback.

    Providers are tried in config order. Once one succeeds, it becomes
    the "sticky" active provider. If it fails later, we move to the
    next available provider.
    """

    def __init__(self, config: dict):
        """Initialize all configured providers.

        Args:
            config: Full application config dict (loaded from config.yaml).
        """
        assistant_cfg = config.get("assistant", {})
        system_prompt = assistant_cfg.get(
            "system_prompt",
            "You are Jarvis, a helpful AI desktop assistant.",
        )
        fallback_cfg = config.get("fallback", {})
        self._max_retries = fallback_cfg.get("max_retries", 1)
        self._timeout = fallback_cfg.get("timeout", 10)
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Initialize persistent SQLite memory manager & inject context into system prompt
        self.memory_manager = get_memory_manager()
        memory_context = self.memory_manager.get_system_prompt_context()
        augmented_system_prompt = f"{system_prompt}\n\n{memory_context}"

        # Build provider chain from config
        provider_configs = config.get("providers", [])
        if not provider_configs:
            raise ValueError(
                "No providers configured in config.yaml. "
                "Add at least one provider under the 'providers' key."
            )

        self._providers: list[BaseProvider] = []
        for pc in provider_configs:
            provider = _create_provider(pc, augmented_system_prompt)
            if provider:
                self._providers.append(provider)

        if not self._providers:
            raise ValueError(
                "No providers could be initialized. "
                "Check your API keys in config.yaml."
            )

        # Initialize zero-latency local intent router (<1ms)
        self.intent_router = IntentRouter()

        # Start with the first provider
        self._active_index = 0
        logger.info(
            "Brain initialized with %d provider(s): %s",
            len(self._providers),
            " → ".join(str(p) for p in self._providers),
        )

    @property
    def active_provider(self) -> BaseProvider:
        """The currently active provider."""
        return self._providers[self._active_index]

    def think(self, user_text: str) -> str:
        """Process user input through the active provider, with fallback.

        Tries the active provider first. On failure, moves to the next
        provider in the chain. Wraps around if all providers have been tried.

        Args:
            user_text: The transcribed user speech.

        Returns:
            The final text response.
        """
        if not user_text.strip():
            return "I didn't catch that. Could you repeat?"

        logger.info("User said: %s", user_text)

        # ── 1. Zero-Latency Local Intent Fast Path (<1ms) ───────────
        fast_res = self.intent_router.match_and_execute(user_text)
        if fast_res is not None:
            logger.info("Fast-path local command handled: '%s' (%s)", fast_res.full_text[:80], fast_res.action_name)
            self.record_turn(user_text, fast_res.full_text)
            return fast_res.full_text

        # Try each provider starting from the active one
        attempts = 0
        total_providers = len(self._providers)

        while attempts < total_providers:
            provider = self._providers[self._active_index]

            for retry in range(self._max_retries + 1):
                try:
                    logger.info(
                        "Trying %s (attempt %d/%d, timeout %ds)...",
                        provider, retry + 1, self._max_retries + 1,
                        self._timeout,
                    )
                    # Run with timeout — skip provider if too slow
                    future = self._executor.submit(provider.think, user_text)
                    response = future.result(timeout=self._timeout)
                    logger.info("Response from %s: %s", provider, response[:100])
                    return response

                except TimeoutError:
                    logger.warning(
                        "%s timed out after %ds (attempt %d). Skipping.",
                        provider, self._timeout, retry + 1,
                    )
                    break  # Don't retry on timeout, move to next provider

                except Exception as e:
                    logger.warning(
                        "%s failed (attempt %d): %s",
                        provider, retry + 1, e,
                    )

            # This provider exhausted its retries — move to next
            logger.warning(
                "%s exhausted all retries. Falling back to next provider...",
                provider,
            )
            self._active_index = (self._active_index + 1) % total_providers
            attempts += 1

        # All providers failed
        logger.error("All providers failed.")
        return (
            "I'm having trouble connecting to my AI services right now, sir. "
            "Please check your internet connection and API keys."
        )

    def think_stream(self, user_text: str) -> BrainStreamResult:
        """Process user input through providers with streaming and fallback.

        Yields sentences one by one as they are synthesized, enabling
        sub-second perceived voice latency and follow-up intent detection.

        Args:
            user_text: The transcribed user speech.

        Returns:
            A BrainStreamResult iterable that yields clean sentences and tracks
            follow-up intent.
        """
        if not user_text.strip():
            def _empty_gen():
                yield "I didn't catch that. Could you repeat, sir?"
            return BrainStreamResult(_empty_gen(), provider_name="fallback")

        logger.info("User said (stream): %s", user_text)

        # ── 1. Zero-Latency Local Intent Fast Path (<1ms) ───────────
        fast_res = self.intent_router.match_and_execute(user_text)
        if fast_res is not None:
            logger.info("Fast-path local command handled: '%s' (%s)", fast_res.full_text[:80], fast_res.action_name)
            self.record_turn(user_text, fast_res.full_text)
            result = BrainStreamResult(fast_res.stream_generator(), provider_name="LocalFastPath")
            result.needs_follow_up = fast_res.needs_follow_up
            return result

        def _orchestrated_stream():
            attempts = 0
            total_providers = len(self._providers)

            while attempts < total_providers:
                provider = self._providers[self._active_index]
                for retry in range(self._max_retries + 1):
                    try:
                        logger.info(
                            "Trying %s stream (attempt %d/%d)...",
                            provider, retry + 1, self._max_retries + 1,
                        )
                        gen = provider.think_stream(user_text)
                        # Fetch the first sentence to verify connectivity and tool execution
                        first_sentence = next(gen, None)
                        if first_sentence is not None:
                            logger.info("First streamed sentence from %s: '%s'", provider, first_sentence[:60])
                            yield first_sentence
                            yield from gen
                            return
                        else:
                            logger.warning("%s produced empty stream.", provider)
                            break

                    except Exception as e:
                        logger.warning("%s stream failed (attempt %d): %s", provider, retry + 1, e)

                # Provider failed, try next
                logger.warning("%s stream exhausted. Falling back to next provider...", provider)
                self._active_index = (self._active_index + 1) % total_providers
                attempts += 1

            # All providers failed
            logger.error("All providers failed in stream mode.")
            yield "I am having trouble connecting to my AI services right now, sir. Please check your network connection."

        active_provider_name = str(self._providers[self._active_index]) if self._providers else "none"
        return BrainStreamResult(_orchestrated_stream(), provider_name=active_provider_name)

    def clear_history(self):
        """Clear conversation history on all providers."""
        for provider in self._providers:
            provider.clear_history()
        logger.info("Conversation history cleared on all providers.")

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        """Record a conversation turn into history across all providers and persistent memory."""
        # 1. Sync provider memory buffers
        for provider in self._providers:
            try:
                provider.record_turn(user_text, assistant_text)
            except Exception as e:
                logger.debug("Failed to record turn on provider %s: %s", provider, e)

        # 2. Persist to SQLite interaction history
        try:
            self.memory_manager.log_interaction("user", user_text)
            self.memory_manager.log_interaction("assistant", assistant_text)
        except Exception as e:
            logger.debug("Failed to log interaction to SQLite memory: %s", e)


    def get_status(self) -> str:
        """Return a string describing the current provider status."""
        lines = ["Provider chain:"]
        for i, p in enumerate(self._providers):
            marker = "→ " if i == self._active_index else "  "
            lines.append(f"  {marker}{p}")
        return "\n".join(lines)
