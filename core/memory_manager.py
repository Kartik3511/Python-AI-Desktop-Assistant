"""
memory_manager.py - Persistent SQLite Memory, User Preferences, and Workspace Automation.

Provides ACID-compliant, thread-safe persistent memory for J.A.R.V.I.S.:
- User preferences (name, titles, browser, audio/display defaults)
- Proactive multi-app workspaces (coding, study, focus, media, meeting)
- Semantic knowledge and facts store (personal, project, habits)
- Conversational interaction history logging
- Dynamic system prompt context injection for cloud LLM providers
"""

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Tuple

from tools import system_ops, web_ops

logger = logging.getLogger(__name__)

# Default SQLite database path
_DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "jarvis_memory.db"


class MemoryManager:
    """Thread-safe SQLite persistent memory manager for JARVIS."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the database connection and schema.

        Args:
            db_path: Optional custom path to SQLite database. Defaults to data/jarvis_memory.db.
        """
        self.db_path = str(db_path) if db_path else str(_DEFAULT_DB_PATH)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=15.0)
        self._conn.row_factory = sqlite3.Row
        self._init_db()
        logger.info("MemoryManager initialized with database at: %s", self.db_path)

    # ── Database Initialization & Schema ───────────────────────────────

    def _init_db(self) -> None:
        """Create tables if they do not exist and seed default data."""
        with self._lock, self._conn:
            cursor = self._conn.cursor()

            # 1. Preferences Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 2. Workspaces Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    name TEXT PRIMARY KEY,
                    description TEXT,
                    config_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 3. Facts / Semantic Knowledge Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT DEFAULT 'general',
                    key TEXT,
                    fact TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # 4. Interaction History Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT DEFAULT 'default',
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Seed default preferences if table is empty
            cursor.execute("SELECT COUNT(*) FROM preferences")
            if cursor.fetchone()[0] == 0:
                defaults = [
                    ("user_name", "Kartik"),
                    ("user_title", "sir"),
                    ("default_browser", "edge"),
                    ("default_city", "London"),
                    ("preferred_volume", "60"),
                    ("preferred_brightness", "80"),
                ]
                cursor.executemany(
                    "INSERT INTO preferences (key, value) VALUES (?, ?)",
                    defaults,
                )
                logger.info("Seeded default preferences into memory.")

            # Seed default proactive workspaces if table is empty
            cursor.execute("SELECT COUNT(*) FROM workspaces")
            if cursor.fetchone()[0] == 0:
                default_workspaces = [
                    (
                        "coding",
                        "Development environment with Visual Studio Code, Terminal, and Git Bash",
                        json.dumps({
                            "apps": ["code", "terminal", "git bash"],
                            "volume": 50,
                            "brightness": 100,
                            "urls": [],
                        }),
                    ),
                    (
                        "study",
                        "Research and study environment with Microsoft Edge and Notepad",
                        json.dumps({
                            "apps": ["edge", "notepad"],
                            "volume": 20,
                            "brightness": 70,
                            "urls": [],
                        }),
                    ),
                    (
                        "focus",
                        "Deep distraction-free focus mode with dimmed display and muted audio",
                        json.dumps({
                            "apps": [],
                            "volume": 10,
                            "brightness": 50,
                            "urls": [],
                        }),
                    ),
                    (
                        "media",
                        "Entertainment workspace with Spotify and YouTube",
                        json.dumps({
                            "apps": ["spotify"],
                            "volume": 75,
                            "brightness": 90,
                            "urls": ["https://www.youtube.com"],
                        }),
                    ),
                    (
                        "meeting",
                        "Conference and meeting mode with balanced volume and Notepad",
                        json.dumps({
                            "apps": ["notepad"],
                            "volume": 50,
                            "brightness": 75,
                            "urls": [],
                        }),
                    ),
                ]
                cursor.executemany(
                    "INSERT INTO workspaces (name, description, config_json) VALUES (?, ?, ?)",
                    default_workspaces,
                )
                logger.info("Seeded default proactive workspaces into memory.")

            # Seed core facts if empty
            cursor.execute("SELECT COUNT(*) FROM facts")
            if cursor.fetchone()[0] == 0:
                default_facts = [
                    ("identity", "user_name", "User's name is Kartik, whom J.A.R.V.I.S. addresses as 'sir'."),
                    ("project", "architecture", "Kartik is developing J.A.R.V.I.S., an autonomous voice assistant with local-first fast-path routing, multi-provider fallback, and proactive desktop automation."),
                    ("preferences", "communication", "Kartik prefers refined British etiquette, concise spoken responses under 30 words, and zero markdown formatting."),
                ]
                cursor.executemany(
                    "INSERT INTO facts (category, key, fact) VALUES (?, ?, ?)",
                    default_facts,
                )
                logger.info("Seeded default facts into memory.")

    # ── User Preferences ───────────────────────────────────────────────

    def set_preference(self, key: str, value: str) -> bool:
        """Store or update a user preference."""
        clean_key = key.strip().lower().replace(" ", "_")
        clean_val = str(value).strip()
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT INTO preferences (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP;
            """, (clean_key, clean_val))
            logger.info("Preference set: %s = %s", clean_key, clean_val)
            return True

    def get_preference(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a user preference by key."""
        clean_key = key.strip().lower().replace(" ", "_")
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            cursor.execute("SELECT value FROM preferences WHERE key = ?", (clean_key,))
            row = cursor.fetchone()
            return row["value"] if row else default

    def get_all_preferences(self) -> Dict[str, str]:
        """Return all stored user preferences as a dictionary."""
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            cursor.execute("SELECT key, value FROM preferences ORDER BY key ASC")
            return {row["key"]: row["value"] for row in cursor.fetchall()}

    def delete_preference(self, key: str) -> bool:
        """Delete a preference key from memory."""
        clean_key = key.strip().lower().replace(" ", "_")
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM preferences WHERE key = ?", (clean_key,))
            return cursor.rowcount > 0

    # ── Proactive Workspaces ───────────────────────────────────────────

    def save_workspace(
        self,
        name: str,
        description: str,
        apps: List[str],
        volume: Optional[int] = None,
        brightness: Optional[int] = None,
        urls: Optional[List[str]] = None,
    ) -> bool:
        """Save or update a proactive desktop workspace profile."""
        clean_name = name.strip().lower().replace(" ", "_")
        config = {
            "apps": apps or [],
            "volume": volume,
            "brightness": brightness,
            "urls": urls or [],
        }
        config_json = json.dumps(config)
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT INTO workspaces (name, description, config_json, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(name) DO UPDATE SET
                    description = excluded.description,
                    config_json = excluded.config_json,
                    updated_at = CURRENT_TIMESTAMP;
            """, (clean_name, description.strip(), config_json))
            logger.info("Workspace saved: %s -> %s", clean_name, config)
            return True

    def get_workspace(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a workspace configuration by name."""
        clean_name = name.strip().lower().replace(" ", "_")
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            cursor.execute("SELECT name, description, config_json FROM workspaces WHERE name = ?", (clean_name,))
            row = cursor.fetchone()
            if not row:
                return None
            try:
                cfg = json.loads(row["config_json"])
            except Exception:
                cfg = {}
            return {
                "name": row["name"],
                "description": row["description"],
                "config": cfg,
            }

    def list_workspaces(self) -> List[Dict[str, Any]]:
        """List all available workspaces."""
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            cursor.execute("SELECT name, description, config_json FROM workspaces ORDER BY name ASC")
            results = []
            for row in cursor.fetchall():
                try:
                    cfg = json.loads(row["config_json"])
                except Exception:
                    cfg = {}
                results.append({
                    "name": row["name"],
                    "description": row["description"],
                    "config": cfg,
                })
            return results

    def delete_workspace(self, name: str) -> bool:
        """Delete a workspace by name."""
        clean_name = name.strip().lower().replace(" ", "_")
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM workspaces WHERE name = ?", (clean_name,))
            return cursor.rowcount > 0

    def execute_workspace(self, name: str) -> Tuple[bool, str, List[str]]:
        """Execute all proactive desktop actions defined in a workspace.

        Args:
            name: Workspace name (e.g. 'coding', 'study', 'focus', 'media', 'meeting').

        Returns:
            Tuple of (success, human_readable_summary, executed_actions_list).
        """
        clean_name = name.strip().lower().replace(" ", "_")
        ws = self.get_workspace(clean_name)
        if not ws:
            return False, f"Workspace '{name}' does not exist.", []

        config = ws.get("config", {})
        apps = config.get("apps", [])
        vol = config.get("volume")
        bright = config.get("brightness")
        urls = config.get("urls", [])

        actions_taken = []

        # 1. Adjust Volume
        if vol is not None and isinstance(vol, (int, float)) and 0 <= vol <= 100:
            try:
                system_ops.set_volume(int(vol))
                actions_taken.append(f"volume set to {int(vol)} percent")
            except Exception as e:
                logger.warning("Failed to set volume in workspace %s: %s", clean_name, e)

        # 2. Adjust Brightness
        if bright is not None and isinstance(bright, (int, float)) and 0 <= bright <= 100:
            try:
                system_ops.set_brightness(int(bright))
                actions_taken.append(f"brightness set to {int(bright)} percent")
            except Exception as e:
                logger.warning("Failed to set brightness in workspace %s: %s", clean_name, e)

        # 3. Launch Applications
        opened_apps = []
        for app in apps:
            if app and str(app).strip():
                try:
                    res = system_ops.open_application(str(app).strip())
                    app_lower = app.lower()
                    if "edge" in app_lower:
                        disp = "Microsoft Edge"
                    elif "code" in app_lower:
                        disp = "VS Code"
                    elif "terminal" in app_lower or app_lower == "wt":
                        disp = "Terminal"
                    elif "git" in app_lower or "bash" in app_lower:
                        disp = "Git Bash"
                    else:
                        disp = app.title()
                    opened_apps.append(disp)
                except Exception as e:
                    logger.warning("Failed to launch application '%s' in workspace %s: %s", app, clean_name, e)

        if opened_apps:
            actions_taken.append(f"launched {', '.join(opened_apps)}")

        # 4. Open URLs
        opened_urls = []
        for url in urls:
            if url and str(url).strip():
                try:
                    web_ops.open_website(str(url).strip())
                    opened_urls.append(url)
                except Exception as e:
                    logger.warning("Failed to open URL '%s' in workspace %s: %s", url, clean_name, e)

        if opened_urls:
            actions_taken.append(f"opened {len(opened_urls)} tab(s)")

        summary = f"{clean_name.title()} workspace initialized, sir: {'; '.join(actions_taken)}."
        logger.info("Executed workspace '%s': %s", clean_name, summary)
        return True, summary, actions_taken

    # ── Facts & Semantic Memory ────────────────────────────────────────

    def remember_fact(self, fact: str, category: str = "general", key: Optional[str] = None) -> int:
        """Store a new fact or piece of knowledge into persistent memory."""
        clean_fact = fact.strip()
        clean_cat = category.strip().lower() if category else "general"
        clean_key = key.strip().lower() if key else None

        with self._lock, self._conn:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT INTO facts (category, key, fact, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP);
            """, (clean_cat, clean_key, clean_fact))
            fact_id = cursor.lastrowid
            logger.info("Fact stored [ID %d, Cat '%s']: %s", fact_id, clean_cat, clean_fact)
            return fact_id

    def recall_facts(self, query: str = "", category: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Search and recall facts matching a query or category."""
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            if query and category:
                pattern = f"%{query.strip()}%"
                cursor.execute(
                    "SELECT id, category, key, fact, created_at FROM facts WHERE category = ? AND fact LIKE ? ORDER BY updated_at DESC LIMIT ?",
                    (category.strip().lower(), pattern, limit),
                )
            elif query:
                pattern = f"%{query.strip()}%"
                cursor.execute(
                    "SELECT id, category, key, fact, created_at FROM facts WHERE fact LIKE ? OR category LIKE ? ORDER BY updated_at DESC LIMIT ?",
                    (pattern, pattern, limit),
                )
            elif category:
                cursor.execute(
                    "SELECT id, category, key, fact, created_at FROM facts WHERE category = ? ORDER BY updated_at DESC LIMIT ?",
                    (category.strip().lower(), limit),
                )
            else:
                cursor.execute(
                    "SELECT id, category, key, fact, created_at FROM facts ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                )

            return [
                {
                    "id": row["id"],
                    "category": row["category"],
                    "key": row["key"],
                    "fact": row["fact"],
                    "created_at": row["created_at"],
                }
                for row in cursor.fetchall()
            ]

    def delete_fact(self, fact_id: int) -> bool:
        """Delete a fact by its ID."""
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            return cursor.rowcount > 0

    def get_all_facts(self) -> List[Dict[str, Any]]:
        """Retrieve all stored facts."""
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            cursor.execute("SELECT id, category, key, fact, created_at FROM facts ORDER BY category, id ASC")
            return [
                {
                    "id": row["id"],
                    "category": row["category"],
                    "key": row["key"],
                    "fact": row["fact"],
                    "created_at": row["created_at"],
                }
                for row in cursor.fetchall()
            ]

    # ── Interaction History ────────────────────────────────────────────

    def log_interaction(self, role: str, content: str, session_id: str = "default") -> None:
        """Record an interaction turn in SQLite for audit and context."""
        if not content or not content.strip():
            return
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            cursor.execute("""
                INSERT INTO interactions (session_id, role, content, timestamp)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP);
            """, (session_id, role.strip().lower(), content.strip()))

    def get_recent_interactions(self, limit: int = 10, session_id: str = "default") -> List[Dict[str, Any]]:
        """Retrieve recent interaction turns."""
        with self._lock, self._conn:
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT id, role, content, timestamp FROM interactions WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
            rows = cursor.fetchall()
            return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in reversed(rows)]

    # ── System Prompt Context Generation ───────────────────────────────

    def get_system_prompt_context(self) -> str:
        """Generate a compact, high-signal persistent memory block for LLM prompt injection."""
        prefs = self.get_all_preferences()
        user_name = prefs.get("user_name", "Kartik")
        user_title = prefs.get("user_title", "sir")

        pref_items = [f"{k}={v}" for k, v in prefs.items() if k not in ("user_name", "user_title")]
        pref_str = ", ".join(pref_items) if pref_items else "Standard defaults"

        workspaces = self.list_workspaces()
        ws_names = ", ".join(w["name"] for w in workspaces) if workspaces else "None"

        facts = self.recall_facts(limit=6)
        fact_lines = "\n".join(f"- {f['fact']}" for f in facts) if facts else "- None recorded yet."

        context = (
            "PERSISTENT USER MEMORY & PREFERENCES:\n"
            f"- User: {user_name} (addressed as '{user_title}')\n"
            f"- Configured Preferences: {pref_str}\n"
            f"- Available Workspaces: {ws_names}\n"
            "Key Remembered Facts:\n"
            f"{fact_lines}\n"
            "Use this memory naturally without explicitly reciting it unless asked."
        )
        return context


# Global singleton instance
_GLOBAL_MEMORY_MANAGER: Optional[MemoryManager] = None
_GLOBAL_LOCK = threading.Lock()


def get_memory_manager(db_path: Optional[str] = None) -> MemoryManager:
    """Retrieve or initialize the global MemoryManager singleton."""
    global _GLOBAL_MEMORY_MANAGER
    with _GLOBAL_LOCK:
        if _GLOBAL_MEMORY_MANAGER is None:
            _GLOBAL_MEMORY_MANAGER = MemoryManager(db_path=db_path)
        return _GLOBAL_MEMORY_MANAGER
