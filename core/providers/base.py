"""
base.py - Abstract base class for LLM providers and shared tool definitions.

All providers (Gemini, Groq, OpenAI) implement the same interface.
Tool definitions are stored here in a provider-agnostic format so each
provider can convert them to its native SDK format.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Generator, Iterable

from tools import system_ops, dev_ops, web_ops, memory_ops

logger = logging.getLogger(__name__)

SENTENCE_PATTERN = re.compile(r'((?:[0-9]+\.[0-9]+|[^.!?\n])+[.!?]+(?:\s+|\n+)|[^\n]+\n+)')


def split_sentences(token_generator: Iterable[str]) -> Generator[str, None, None]:
    """Buffer streamed tokens and yield complete sentences as soon as they form."""
    buffer = ""
    for token in token_generator:
        buffer += token
        while True:
            m = SENTENCE_PATTERN.match(buffer)
            if m:
                sentence = m.group(1).strip()
                buffer = buffer[m.end():]
                if sentence:
                    yield sentence
            else:
                break
    if buffer.strip():
        yield buffer.strip()


# ------------------------------------------------------------------ #
#  Tool Function Registry — name → callable                          #
# ------------------------------------------------------------------ #

TOOL_FUNCTIONS: dict[str, callable] = {
    # System operations
    "set_volume": system_ops.set_volume,
    "get_volume": system_ops.get_volume,
    "set_brightness": system_ops.set_brightness,
    "get_brightness": system_ops.get_brightness,
    "open_application": system_ops.open_application,
    "open_settings": system_ops.open_settings,
    "close_application": system_ops.close_application,
    "lock_screen": system_ops.lock_screen,
    "get_system_info": system_ops.get_system_info,
    # Developer operations
    "git_status": dev_ops.git_status,
    "git_commit": dev_ops.git_commit,
    "git_pull": dev_ops.git_pull,
    "run_command": dev_ops.run_command,
    # Web operations
    "web_search": web_ops.web_search,
    "get_weather": web_ops.get_weather,
    "get_time": web_ops.get_time,
    "open_website": web_ops.open_website,
    "search_youtube": web_ops.search_youtube,
    "search_google": web_ops.search_google,
    # Memory & Workspace operations
    "remember_fact": memory_ops.remember_fact,
    "recall_memory": memory_ops.recall_memory,
    "set_user_preference": memory_ops.set_user_preference,
    "get_user_preference": memory_ops.get_user_preference,
    "save_workspace": memory_ops.save_workspace,
    "launch_workspace": memory_ops.launch_workspace,
    "list_workspaces": memory_ops.list_workspaces,
}


# ------------------------------------------------------------------ #
#  Tool Definitions — provider-agnostic format (OpenAI-style JSON)   #
#  Each provider converts these to its native format.                #
# ------------------------------------------------------------------ #

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    # ---- System Operations ----
    {
        "name": "set_volume",
        "description": "Set the system audio volume on the user's Windows PC.",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "integer",
                    "description": "Volume level from 0 (mute) to 100 (max).",
                },
            },
            "required": ["level"],
        },
    },
    {
        "name": "get_volume",
        "description": "Get current Windows system master volume level percentage (0 to 100).",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "set_brightness",
        "description": "Set the screen brightness on the user's Windows PC.",
        "parameters": {
            "type": "object",
            "properties": {
                "level": {
                    "type": "integer",
                    "description": "Brightness level from 0 (darkest) to 100 (brightest).",
                },
            },
            "required": ["level"],
        },
    },
    {
        "name": "get_brightness",
        "description": "Get current screen brightness percentage (0 to 100).",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "open_application",
        "description": "Launch a desktop application or browser by name (e.g., 'edge' or 'microsoft edge', 'chrome', 'notepad', 'vscode', 'spotify').",
        "parameters": {

            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Name of the application to open.",
                },
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "open_settings",
        "description": "Open Windows Settings directly to a specific page (e.g. 'windowsupdate' for Windows Update or checking updates, 'bluetooth', 'display', 'sound', 'network', 'battery', 'apps'). Use this whenever the user asks about Windows Update, system updates, or adjusting system settings.",
        "parameters": {
            "type": "object",
            "properties": {
                "setting": {
                    "type": "string",
                    "description": "The specific settings page (e.g. 'windowsupdate', 'bluetooth', 'display', 'sound', 'network', 'battery').",
                },
            },
            "required": ["setting"],
        },
    },
    {
        "name": "close_application",
        "description": "Close / kill a running application by name.",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "Name of the application to close.",
                },
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "lock_screen",
        "description": "Lock the Windows workstation immediately.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_system_info",
        "description": "Get current system resource usage: CPU, RAM, disk, and battery.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    # ---- Developer Operations ----
    {
        "name": "git_status",
        "description": "Run 'git status' in a repository directory and return the output.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Absolute path to the git repository.",
                },
            },
            "required": ["repo_path"],
        },
    },
    {
        "name": "git_commit",
        "description": "Stage all changes and commit with a message in a git repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Absolute path to the git repository.",
                },
                "message": {
                    "type": "string",
                    "description": "The commit message.",
                },
            },
            "required": ["repo_path", "message"],
        },
    },
    {
        "name": "git_pull",
        "description": "Pull the latest changes from the remote in a git repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "repo_path": {
                    "type": "string",
                    "description": "Absolute path to the git repository.",
                },
            },
            "required": ["repo_path"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command on the user's system and return the output. Use with caution.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory. Defaults to current directory.",
                },
            },
            "required": ["command"],
        },
    },
    # ---- Web Operations ----
    {
        "name": "web_search",
        "description": "Search the web for information using a query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Name of the city.",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_time",
        "description": "Get the current date and time.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name (e.g., 'US/Eastern') or 'local' for system timezone.",
                },
            },
        },
    },
    {
        "name": "open_website",
        "description": "Open a URL in the user's default web browser.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to open (e.g., 'https://google.com').",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "search_youtube",
        "description": "Search for videos on YouTube and open the results in the default web browser.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search keywords, video title, topic, or channel name to search on YouTube.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_google",
        "description": "Perform a Google web search and open the search results in the default web browser.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query or keywords to search on Google.",
                },
            },
            "required": ["query"],
        },
    },
    # ---- Persistent Memory & Proactive Workspaces ----
    {
        "name": "remember_fact",
        "description": "Store a new personal fact, project note, or preference into persistent long-term memory across sessions.",
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The fact or information to remember.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category: 'personal', 'project', 'preference', 'tech', or 'general'.",
                },
            },
            "required": ["fact"],
        },
    },
    {
        "name": "recall_memory",
        "description": "Search and recall stored facts, notes, and user preferences from persistent long-term memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query or keyword. Leave empty to recall recent memories.",
                },
            },
        },
    },
    {
        "name": "set_user_preference",
        "description": "Save or update a user preference (e.g., default browser, volume, name, city) in persistent memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Preference key name (e.g. 'default_browser', 'default_city', 'theme').",
                },
                "value": {
                    "type": "string",
                    "description": "Preference value to store.",
                },
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "get_user_preference",
        "description": "Retrieve a specific user preference from persistent memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The preference key to look up.",
                },
            },
            "required": ["key"],
        },
    },
    {
        "name": "save_workspace",
        "description": "Save or update a proactive desktop workspace profile with specific applications, volume, and brightness.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the workspace profile (e.g. 'coding', 'study', 'focus', 'gaming').",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of the workspace purpose.",
                },
                "apps": {
                    "type": "string",
                    "description": "Comma-separated list of application names to launch (e.g. 'code, edge').",
                },
                "volume": {
                    "type": "integer",
                    "description": "Target volume level (0-100), or -1 to leave unchanged.",
                },
                "brightness": {
                    "type": "integer",
                    "description": "Target screen brightness level (0-100), or -1 to leave unchanged.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "launch_workspace",
        "description": "Execute and switch to a proactive desktop workspace profile (e.g., 'coding', 'study', 'focus', 'media', 'meeting').",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the workspace profile to launch.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "list_workspaces",
        "description": "List all configured proactive desktop workspaces and their settings.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
]


# ------------------------------------------------------------------ #
#  Shared Tool Execution                                             #
# ------------------------------------------------------------------ #

# Cache of recently executed tools to prevent duplicate side-effects during provider fallback cascades
# Key: (tool_name, serialized_args_tuple), Value: (timestamp, result_string)
_RECENT_TOOL_EXECUTIONS: dict[tuple, tuple[float, str]] = {}
_DEDUP_WINDOW_SECONDS = 15.0


def execute_tool(name: str, args: dict) -> str:
    """Look up and execute a registered tool function.

    Includes idempotency deduplication: if the same tool with identical arguments
    was executed successfully within the last 15 seconds (e.g. across a provider fallback cascade
    when Groq fails after executing a tool and Gemini is retried), the cached result is
    returned immediately to prevent opening duplicate browser windows or repeating actions.

    Args:
        name: The function name.
        args: The arguments dict.

    Returns:
        The string result, or an error message.
    """
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        logger.error("Unknown tool requested: %s", name)
        return f"Error: Unknown tool '{name}'"

    # Normalize arguments for deduplication check
    try:
        args_key = tuple(sorted((str(k), str(v)) for k, v in args.items()))
    except Exception:
        args_key = tuple()
    cache_key = (name, args_key)

    now = time.time()
    if cache_key in _RECENT_TOOL_EXECUTIONS:
        prev_time, prev_result = _RECENT_TOOL_EXECUTIONS[cache_key]
        if (now - prev_time) <= _DEDUP_WINDOW_SECONDS:
            logger.info(
                "Tool %s(%s) was already executed %.1fs ago. Returning cached result to prevent duplicate execution during provider fallback.",
                name, args, now - prev_time
            )
            return prev_result

    try:
        logger.info("Executing tool: %s(%s)", name, args)
        result = func(**args)
        str_result = str(result) if result else "Done."
        logger.info("Tool result: %s", str_result[:200])

        # Cache successful execution
        _RECENT_TOOL_EXECUTIONS[cache_key] = (now, str_result)

        # Clean old entries from cache (>60s)
        expired = [k for k, (t, _) in _RECENT_TOOL_EXECUTIONS.items() if (now - t) > 60.0]
        for k in expired:
            _RECENT_TOOL_EXECUTIONS.pop(k, None)

        return str_result
    except Exception as e:
        logger.error("Tool execution failed: %s — %s", name, e)
        return f"Error executing {name}: {e}"


# ------------------------------------------------------------------ #
#  Abstract Base Provider                                            #
# ------------------------------------------------------------------ #

class BaseProvider(ABC):
    """Abstract base class that all LLM providers must implement.

    Each provider manages its own conversation history and SDK-specific
    tool format. The Brain orchestrator calls `think()` and doesn't
    need to know which provider is handling the request.
    """

    def __init__(self, provider_config: dict, system_prompt: str):
        """Initialize the provider.

        Args:
            provider_config: Provider-specific config (name, model, api_key).
            system_prompt: The Jarvis system prompt.
        """
        self.name = provider_config.get("name", "unknown")
        self.model = provider_config.get("model", "")
        self.api_key = provider_config.get("api_key", "")
        self.system_prompt = system_prompt
        self._max_history = 20

    @abstractmethod
    def think(self, user_text: str) -> str:
        """Process user input and return a text response.

        Must handle tool calls internally (execute and loop until
        the model returns a final text response).

        Args:
            user_text: The transcribed user speech.

        Returns:
            The final text response.

        Raises:
            Exception: On API errors (rate limit, timeout, etc.)
                       so the Brain can fall back to the next provider.
        """
        ...

    def think_stream(self, user_text: str) -> Generator[str, None, None]:
        """Process user input and yield text sentence-by-sentence.

        Default implementation calls think() and splits the response.
        Providers with native token streaming can override this for sub-second latency.
        """
        response = self.think(user_text)
        yield from split_sentences([response])

    @abstractmethod
    def clear_history(self):
        """Reset conversation history."""
        ...

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        """Record an external/local conversation turn into provider history.

        Args:
            user_text: The user speech.
            assistant_text: The response text delivered to the user.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
