"""JARVIS Assistant - Tool functions exposed to Gemini via Function Calling."""

from .system_ops import (
    close_application,
    get_system_info,
    lock_screen,
    open_application,
    open_settings,
    set_brightness,
    set_volume,
)
from .dev_ops import (
    git_commit,
    git_pull,
    git_status,
    run_command,
)
from .web_ops import (
    get_time,
    get_weather,
    open_website,
    web_search,
)

__all__ = [
    # System Operations
    "set_volume",
    "set_brightness",
    "open_application",
    "open_settings",
    "close_application",
    "lock_screen",
    "get_system_info",
    # Developer Operations
    "git_status",
    "git_commit",
    "git_pull",
    "run_command",
    # Web & Information Retrieval
    "web_search",
    "get_weather",
    "get_time",
    "open_website",
]
