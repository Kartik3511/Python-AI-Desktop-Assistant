"""
JARVIS Assistant - Persistent Memory & Workspace Tools.

Provides functions for Gemini, Groq, and OpenAI to interact with
long-term persistent memory, user preferences, and proactive workspaces:
- remember_fact: Store facts and notes across sessions.
- recall_memory: Search stored facts and preferences.
- set_user_preference: Update user preferences.
- get_user_preference: Retrieve a user preference.
- save_workspace: Define or update a proactive desktop workspace profile.
- launch_workspace: Execute a workspace profile (apps, volume, brightness).
- list_workspaces: List all saved workspace profiles.
"""

import json
import logging
from typing import Optional

from core.memory_manager import get_memory_manager

logger = logging.getLogger(__name__)


def remember_fact(fact: str, category: str = "general") -> str:
    """
    Store a new fact, note, or piece of knowledge into persistent long-term memory.

    Args:
        fact: The factual information or instruction to remember (e.g. 'Kartik prefers dark mode').
        category: Optional category tag (e.g. 'personal', 'project', 'preference', 'tech'). Defaults to 'general'.

    Returns:
        Confirmation message that the information has been committed to long-term memory.
    """
    if not fact or not fact.strip():
        return "Error: Fact content cannot be empty."

    mm = get_memory_manager()
    fact_id = mm.remember_fact(fact=fact.strip(), category=category.strip())
    logger.info("Tool remember_fact committed ID %d: %s", fact_id, fact)
    return f"Information successfully committed to long-term memory (ID: {fact_id})."


def recall_memory(query: str = "") -> str:
    """
    Search and recall stored facts, notes, and user preferences from long-term memory.

    Args:
        query: Search keywords or topic to look up. Leave empty to retrieve the most recent memories.

    Returns:
        Formatted summary of relevant memories and preferences.
    """
    mm = get_memory_manager()
    facts = mm.recall_facts(query=query.strip() if query else "", limit=5)
    prefs = mm.get_all_preferences()

    output_lines = []

    # Include matching preferences if query is provided
    if query:
        q_lower = query.lower()
        matched_prefs = {k: v for k, v in prefs.items() if q_lower in k or q_lower in v.lower()}
        if matched_prefs:
            output_lines.append("Preferences:")
            for k, v in matched_prefs.items():
                output_lines.append(f"- {k}: {v}")

    if facts:
        output_lines.append("Stored Facts:")
        for f in facts:
            output_lines.append(f"- [{f['category'].title()}] {f['fact']}")

    if not output_lines:
        return f"No memories found matching '{query}'." if query else "No memories stored yet."

    return "\n".join(output_lines)


def set_user_preference(key: str, value: str) -> str:
    """
    Store or update a user preference (e.g., default browser, volume, name, city).

    Args:
        key: The preference key name (e.g., 'default_city', 'theme', 'default_browser').
        value: The preference value to assign (e.g., 'London', 'dark', 'edge').

    Returns:
        Confirmation message that the preference has been updated.
    """
    if not key or not value:
        return "Error: Preference key and value are both required."

    mm = get_memory_manager()
    mm.set_preference(key=key, value=value)
    return f"Preference '{key.strip()}' updated to '{value.strip()}'."


def get_user_preference(key: str) -> str:
    """
    Retrieve a specific user preference value from persistent memory.

    Args:
        key: The preference key to retrieve (e.g. 'user_name', 'default_browser', 'default_city').

    Returns:
        The stored value or an indication that it was not found.
    """
    if not key:
        return "Error: Preference key must be specified."

    mm = get_memory_manager()
    val = mm.get_preference(key)
    if val is not None:
        return f"Preference '{key}': {val}"
    return f"Preference '{key}' is not set."


def save_workspace(
    name: str,
    description: str = "",
    apps: str = "",
    volume: int = -1,
    brightness: int = -1,
) -> str:
    """
    Define or update a proactive desktop workspace profile.

    Args:
        name: Name of the workspace (e.g. 'coding', 'study', 'focus', 'gaming').
        description: Brief description of what this workspace is for.
        apps: Comma-separated list of applications to open (e.g. 'code, edge, spotify').
        volume: Optional target audio volume (0-100), or -1 to leave volume unchanged.
        brightness: Optional target screen brightness (0-100), or -1 to leave brightness unchanged.

    Returns:
        Confirmation message describing the saved workspace profile.
    """
    if not name or not name.strip():
        return "Error: Workspace name cannot be empty."

    app_list = [a.strip() for a in apps.split(",") if a.strip()] if apps else []
    vol_val = volume if 0 <= volume <= 100 else None
    bright_val = brightness if 0 <= brightness <= 100 else None

    mm = get_memory_manager()
    mm.save_workspace(
        name=name.strip(),
        description=description.strip() or f"{name.title()} workspace",
        apps=app_list,
        volume=vol_val,
        brightness=bright_val,
    )
    return f"Workspace '{name.strip()}' saved successfully with apps: {app_list or 'None'}."


def launch_workspace(name: str) -> str:
    """
    Execute and switch to a proactive desktop workspace profile.

    Launches configured applications, adjusts volume, and adjusts screen brightness
    as defined by the profile.

    Args:
        name: The name of the workspace to launch (e.g. 'coding', 'study', 'focus', 'media', 'meeting').

    Returns:
        Confirmation of executed workspace actions.
    """
    if not name or not name.strip():
        return "Error: Workspace name cannot be empty."

    mm = get_memory_manager()
    success, summary, actions = mm.execute_workspace(name.strip())
    if not success:
        return f"Failed to launch workspace: {summary}"
    return summary


def list_workspaces() -> str:
    """
    List all available proactive desktop workspaces with their configuration.

    Returns:
        Formatted summary of saved workspaces.
    """
    mm = get_memory_manager()
    workspaces = mm.list_workspaces()
    if not workspaces:
        return "No workspaces configured."

    lines = ["Available Workspaces:"]
    for w in workspaces:
        cfg = w.get("config", {})
        apps = ", ".join(cfg.get("apps", [])) or "None"
        vol = f", Vol: {cfg.get('volume')}%" if cfg.get("volume") is not None else ""
        bright = f", Bright: {cfg.get('brightness')}%" if cfg.get("brightness") is not None else ""
        lines.append(f"- {w['name'].title()}: {w['description']} (Apps: {apps}{vol}{bright})")

    return "\n".join(lines)
