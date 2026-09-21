"""
intent_router.py - Zero-Latency Local Intent Classifier and Fast-Path Command Router.

Intercepts deterministic local system commands (volume, brightness, application launch/close,
workstation lock, time/date, hardware telemetry, dismissals, and compound/chained commands)
in <1ms on-device, bypassing cloud LLM inference entirely.

Supports multi-intent chained commands, such as:
- "Yes, set the brightness to 50% and turn the volume to 100%."
- "Dim the screen and mute audio."
- "Open Edge and set volume to 80%."
"""

from datetime import datetime
import logging
import random
import re
from typing import Callable, Generator, List, Optional

import urllib.parse

import psutil

from core.memory_manager import get_memory_manager
from tools import system_ops, web_ops

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Word to Number Mapping & Helper                                   #
# ------------------------------------------------------------------ #

WORD_NUMBERS = {
    "zero": 0, "none": 0, "mute": 0, "min": 0, "minimum": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "twenty-five": 25, "twenty five": 25,
    "thirty": 30, "thirty-five": 35, "thirty five": 35,
    "forty": 40, "forty-five": 45, "forty five": 45,
    "fifty": 50, "half": 50,
    "sixty": 60, "seventy": 70, "seventy-five": 75, "seventy five": 75,
    "eighty": 80, "ninety": 90,
    "hundred": 100, "one hundred": 100, "full": 100, "max": 100, "maximum": 100,
}

TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

ONES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}


def parse_spoken_number(text: str) -> Optional[int]:
    """Parse a spoken number string (digits or English words) into an integer."""
    clean = text.lower().replace("%", "").replace("percent", "").strip()

    if clean.isdigit():
        return int(clean)

    if clean in WORD_NUMBERS:
        return WORD_NUMBERS[clean]

    clean_compound = clean.replace("-", " ")
    parts = clean_compound.split()
    if len(parts) == 2 and parts[0] in TENS and parts[1] in ONES:
        return TENS[parts[0]] + ONES[parts[1]]

    match = re.search(r"\b(\d{1,3})\b", clean)
    if match:
        return int(match.group(1))

    return None


# ------------------------------------------------------------------ #
#  Fast-Path Result and Local Action Containers                       #
# ------------------------------------------------------------------ #

class LocalAction:
    """Represents a single parsed and executable local action."""

    def __init__(
        self,
        action_type: str,
        execute_fn: Callable[[], None],
        confirm_phrase: str,
        needs_follow_up: bool = True,
        standalone_response: Optional[str] = None,
    ):
        self.action_type = action_type
        self.execute_fn = execute_fn
        self.confirm_phrase = confirm_phrase
        self.needs_follow_up = needs_follow_up
        self.standalone_response = standalone_response


class FastPathResult:
    """Encapsulates the response from locally handled commands."""

    def __init__(self, sentences: list[str], needs_follow_up: bool, action_name: str = ""):
        self.sentences = sentences
        self.needs_follow_up = needs_follow_up
        self.action_name = action_name
        self.provider_name = "LocalFastPath"
        self.full_text = " ".join(sentences)

    def stream_generator(self) -> Generator[str, None, None]:
        """Yield sentences one by one for TTS streaming."""
        for s in self.sentences:
            yield s


# ------------------------------------------------------------------ #
#  Known Applications Map for Direct Launching                       #
# ------------------------------------------------------------------ #

KNOWN_APPS = {
    "edge": "msedge",
    "microsoft edge": "msedge",
    "msedge": "msedge",
    "chrome": "chrome",
    "google chrome": "chrome",
    "notepad": "notepad",
    "calculator": "calc",
    "calc": "calc",
    "code": "code",
    "vscode": "code",
    "visual studio code": "code",
    "spotify": "spotify",
    "cmd": "cmd",
    "command prompt": "cmd",
    "terminal": "terminal",
    "windows terminal": "terminal",
    "task manager": "taskmgr",
    "taskmgr": "taskmgr",
    "explorer": "explorer",
    "file explorer": "explorer",
    "settings": "settings",
    "windows update": "windowsupdate",
    "windows updates": "windowsupdate",
    "updates": "windowsupdate",
    "update": "windowsupdate",
}

# Common web portals for direct browser navigation
COMMON_WEBSITES = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "github": "https://github.com",
    "reddit": "https://reddit.com",
    "twitter": "https://twitter.com",
    "x": "https://x.com",
    "wikipedia": "https://wikipedia.org",
    "gmail": "https://mail.google.com",
    "chatgpt": "https://chatgpt.com",
}


# ------------------------------------------------------------------ #
#  Intent Router Implementation                                       #
# ------------------------------------------------------------------ #

class IntentRouter:
    """Zero-latency local command classifier and fast-path executor."""

    def __init__(self):
        self._previous_volume: int = 50
        logger.info("Local IntentRouter initialized with fast-path patterns.")

    def match_and_execute(self, user_text: str) -> Optional[FastPathResult]:
        """Classify user speech and execute matched local deterministic actions.

        Supports single commands, conversational prefix trimming ('Yes, ...'),
        and compound multi-intent commands ('... and ...').

        Args:
            user_text: Transcribed user input.

        Returns:
            FastPathResult if all clauses matched locally, None if fallthrough needed.
        """
        raw = user_text.strip()
        if not raw:
            return None

        # Normalize text: lowercase, strip punctuation except '%'
        clean = re.sub(r"[^\w\s%'-]", "", raw.lower()).strip()

        # ── 1. Immediate Standalone Dismissals & Standby ───────────────────
        dismiss_action = self._parse_dismissal(clean)
        if dismiss_action is not None:
            dismiss_action.execute_fn()
            return FastPathResult(
                [dismiss_action.standalone_response],
                needs_follow_up=False,
                action_name="dismissal",
            )

        # ── 2. Strip Conversational Leading Prefixes ───────────────────────
        # E.g., 'Yes, set the brightness...' -> 'set the brightness...'
        clean_stripped = re.sub(
            r"^(?:yes|yeah|sure|okay|ok|please|jarvis|hey\s+jarvis|also)\s*[,]?\s*",
            "",
            clean,
        ).strip()

        if not clean_stripped:
            return None

        # Check dismissal again on stripped text (e.g. 'No thanks' or 'Yes, no that is all')
        dismiss_action = self._parse_dismissal(clean_stripped)
        if dismiss_action is not None:
            dismiss_action.execute_fn()
            return FastPathResult(
                [dismiss_action.standalone_response],
                needs_follow_up=False,
                action_name="dismissal",
            )

        # ── 3. Split into Clauses for Chained / Compound Commands ──────────
        clauses = self._split_compound_clauses(clean_stripped)

        # Parse every clause to verify 100% of intents are local
        parsed_actions: List[LocalAction] = []
        for clause in clauses:
            action = self._parse_single_clause(clause)
            if action is None:
                # If ANY clause is not a deterministic local action, fall through to cloud LLM
                return None
            parsed_actions.append(action)

        if not parsed_actions:
            return None

        # ── 4. Execute All Matched Local Actions ───────────────────────────
        for action in parsed_actions:
            logger.info("Executing local action: %s", action.action_type)
            action.execute_fn()

        # ── 5. Compose Unified Butler Speech Response ──────────────────────
        if len(parsed_actions) == 1:
            single = parsed_actions[0]
            resp_text = single.standalone_response or f"Right away, sir. {single.confirm_phrase.capitalize()}."
            return FastPathResult(
                [resp_text],
                needs_follow_up=single.needs_follow_up,
                action_name=single.action_type,
            )

        # Multi-intent compound response
        confirmations = [a.confirm_phrase for a in parsed_actions if a.confirm_phrase]
        if len(confirmations) > 1:
            combined_body = ", and ".join([", ".join(confirmations[:-1]), confirmations[-1]])
        else:
            combined_body = confirmations[0] if confirmations else "Settings adjusted"

        # Check follow-up status (if any action requires standby/lock, cancel follow-up)
        all_follow_up = all(a.needs_follow_up for a in parsed_actions)
        if all_follow_up:
            speech = f"Right away, sir. {combined_body.capitalize()}. Is there anything else you require? [FOLLOW_UP]"
        else:
            speech = f"Right away, sir. {combined_body.capitalize()}. Standing by."

        action_names = "+".join(a.action_type for a in parsed_actions)
        return FastPathResult([speech], needs_follow_up=all_follow_up, action_name=action_names)

    # ── Clause Splitting ───────────────────────────────────────────────

    def _split_compound_clauses(self, text: str) -> List[str]:
        """Split a compound sentence into separate command clauses.

        Examples:
            'set brightness to 50% and turn volume to 100%'
            -> ['set brightness to 50%', 'turn volume to 100%']
        """
        # Split on coordinating conjunctions
        pattern = r"\s+(?:and\s+then|and\s+also|and|then)\s+|\s*,\s*(?=(?:and\s+|then\s+|set\s+|turn\s+|change\s+|put\s+|open\s+|launch\s+|close\s+|dim\s+|mute\s+|unmute\s+|lock\s+))"
        raw_parts = re.split(pattern, text)
        cleaned = [p.strip() for p in raw_parts if p.strip()]
        return cleaned if cleaned else [text]

    # ── Single Clause Parser ───────────────────────────────────────────

    def _parse_single_clause(self, clause: str) -> Optional[LocalAction]:
        """Attempt to match a single clause against deterministic local intents."""
        c = clause.strip()

        # 1. Volume
        vol_act = self._parse_volume(c)
        if vol_act is not None:
            return vol_act

        # 2. Brightness
        bright_act = self._parse_brightness(c)
        if bright_act is not None:
            return bright_act

        # 3. Workstation Lock
        lock_act = self._parse_lock(c)
        if lock_act is not None:
            return lock_act

        # 4. Telemetry / Battery / System
        sys_act = self._parse_system_telemetry(c)
        if sys_act is not None:
            return sys_act

        # 5. Time & Date
        time_act = self._parse_time_and_date(c)
        if time_act is not None:
            return time_act

        # 6. Web & Media Searches (YouTube, Google, Portals)
        web_act = self._parse_web_and_media(c)
        if web_act is not None:
            return web_act

        # 7. Proactive Desktop Workspaces & Modes
        ws_act = self._parse_workspace(c)
        if ws_act is not None:
            return ws_act

        # 8. Applications
        app_act = self._parse_application(c)
        if app_act is not None:
            return app_act

        # 9. Dismissal
        dis_act = self._parse_dismissal(c)
        if dis_act is not None:
            return dis_act

        return None

    # ── Detailed Parsers ───────────────────────────────────────────────

    def _parse_dismissal(self, clean: str) -> Optional[LocalAction]:
        """Parse conversational sign-offs, thanks, or standby orders."""
        dismissal_patterns = [
            r"^(?:no|nope)$",
            r"^(?:no\s+)?(?:thanks|thank\s+you)$",
            r"^(?:no\s+)?(?:that['’]?s|that\s+is|that\s+will\s+be|that\s+would\s+be)\s+all(?:\s+(?:thanks|thank\s+you))?$",
            r"^(?:no\s+)?(?:that['’]?s|that\s+is)\s+it(?:\s+(?:thanks|thank\s+you))?$",
            r"^(?:no\s+)?nothing\s+else(?:\s+(?:thanks|thank\s+you))?$",
            r"^(?:all\s+good|i['’]?m\s+good|im\s+good)$",
            r"^(?:standby|stand\s+by|dismissed|go\s+to\s+sleep|goodbye|bye|sleep|stop|cancel)$",
        ]
        for pat in dismissal_patterns:
            if re.match(pat, clean):
                ack = random.choice([
                    "Very good, sir. Standing by.",
                    "Understood, sir. Standing by.",
                    "As you wish, sir. Standing by.",
                ])
                return LocalAction(
                    action_type="dismissal",
                    execute_fn=lambda: None,
                    confirm_phrase="standing by",
                    needs_follow_up=False,
                    standalone_response=ack,
                )
        return None

    def _parse_lock(self, clean: str) -> Optional[LocalAction]:
        """Parse workstation lock command."""
        pattern = (
            r"\b(lock\s+(?:my\s+|the\s+)?(?:screen|workstation|computer|pc|desktop)|"
            r"lock\s+windows|secure\s+workstation)\b"
        )
        if re.search(pattern, clean):
            return LocalAction(
                action_type="lock_screen",
                execute_fn=system_ops.lock_screen,
                confirm_phrase="locking your workstation now",
                needs_follow_up=False,
                standalone_response="Locking your workstation now, sir. Have a pleasant day.",
            )
        return None

    def _parse_volume(self, clean: str) -> Optional[LocalAction]:
        """Parse volume adjustments."""
        # Mute
        if re.search(r"\b(mute(?:\s+(?:the\s+)?(?:audio|sound|volume|system))?|silence(?:\s+(?:the\s+)?(?:audio|sound|system))?|turn\s+off\s+(?:the\s+)?(?:sound|audio|volume))\b", clean):
            def _execute_mute():
                curr = system_ops.get_volume()
                if curr > 0:
                    self._previous_volume = curr
                system_ops.set_volume(0)

            return LocalAction(
                action_type="mute_volume",
                execute_fn=_execute_mute,
                confirm_phrase="audio muted",
                needs_follow_up=True,
                standalone_response="Audio muted, sir. Let me know when you wish to restore it. [FOLLOW_UP]",
            )

        # Unmute
        if re.search(r"\b(unmute(?:\s+(?:the\s+)?(?:audio|sound|volume|system))?|restore\s+(?:the\s+)?(?:audio|sound|volume)|turn\s+on\s+(?:the\s+)?(?:sound|audio|volume))\b", clean):
            target = self._previous_volume if self._previous_volume > 0 else 30
            return LocalAction(
                action_type="unmute_volume",
                execute_fn=lambda: system_ops.set_volume(target),
                confirm_phrase=f"audio unmuted to {target} percent",
                needs_follow_up=True,
                standalone_response=f"Audio unmuted, sir. Restored to {target} percent. Is there anything else you need? [FOLLOW_UP]",
            )

        # Relative Up
        if re.search(r"\b(volume\s+up|increase\s+(?:the\s+)?volume|turn\s+(?:the\s+)?volume\s+up|turn\s+it\s+up|louder|make\s+it\s+louder)\b", clean):
            delta = 15
            delta_match = re.search(r"\b(?:by\s+)?(\d{1,2})\s*(?:%|percent)?\b", clean)
            if delta_match:
                delta = int(delta_match.group(1))

            curr = system_ops.get_volume()
            curr = curr if curr >= 0 else 50
            new_vol = min(100, curr + delta)

            return LocalAction(
                action_type="volume_up",
                execute_fn=lambda: system_ops.set_volume(new_vol),
                confirm_phrase=f"volume increased to {new_vol} percent",
                needs_follow_up=True,
                standalone_response=f"Volume increased to {new_vol} percent, sir. Is that adequate? [FOLLOW_UP]",
            )

        # Relative Down
        if re.search(r"\b(volume\s+down|decrease\s+(?:the\s+)?volume|lower\s+(?:the\s+)?volume|turn\s+(?:the\s+)?volume\s+down|turn\s+it\s+down|quieter|make\s+it\s+quieter|softer)\b", clean):
            delta = 15
            delta_match = re.search(r"\b(?:by\s+)?(\d{1,2})\s*(?:%|percent)?\b", clean)
            if delta_match:
                delta = int(delta_match.group(1))

            curr = system_ops.get_volume()
            curr = curr if curr >= 0 else 50
            new_vol = max(0, curr - delta)

            return LocalAction(
                action_type="volume_down",
                execute_fn=lambda: system_ops.set_volume(new_vol),
                confirm_phrase=f"volume decreased to {new_vol} percent",
                needs_follow_up=True,
                standalone_response=f"Volume decreased to {new_vol} percent, sir. Is that better? [FOLLOW_UP]",
            )

        # Exact Volume: e.g. "turn the volume to 100%", "set volume to 50", "volume 80%"
        vol_pattern = r"(?:(?:set|turn|change|put)?\s*(?:the\s*)?volume\s*(?:to\s*)?|volume\s*(?:to\s*)?)([a-z0-9\s\-]+)"
        match = re.search(vol_pattern, clean)
        if match:
            raw_val = match.group(1).strip()
            level = parse_spoken_number(raw_val)
            if level is not None and 0 <= level <= 100:
                return LocalAction(
                    action_type="set_volume",
                    execute_fn=lambda: system_ops.set_volume(level),
                    confirm_phrase=f"master volume set to {level} percent",
                    needs_follow_up=True,
                    standalone_response=f"Right away, sir. Master volume set to {level} percent. Is there anything else you require? [FOLLOW_UP]",
                )

        return None

    def _parse_brightness(self, clean: str) -> Optional[LocalAction]:
        """Parse display brightness adjustments."""
        # Max Brightness
        if re.search(r"\b(?:(?:set|turn)\s+)?(?:max|maximum|full)\s+brightness\b|\bbrightness\s+(?:to\s+)?(?:max|maximum|full|100%?)\b", clean):
            return LocalAction(
                action_type="max_brightness",
                execute_fn=lambda: system_ops.set_brightness(100),
                confirm_phrase="screen brightness set to maximum",
                needs_follow_up=True,
                standalone_response="Screen brightness set to maximum, sir. Anything else? [FOLLOW_UP]",
            )

        # Dim Screen / Min Brightness
        if re.search(r"\b(dim\s+(?:the\s+)?(?:screen|display)|night\s+mode|(?:min|minimum|lowest)\s+brightness|brightness\s+(?:to\s+)?(?:min|minimum|lowest))\b", clean):
            return LocalAction(
                action_type="dim_brightness",
                execute_fn=lambda: system_ops.set_brightness(20),
                confirm_phrase="screen dimmed to twenty percent",
                needs_follow_up=True,
                standalone_response="Screen dimmed to twenty percent, sir. Is that comfortable? [FOLLOW_UP]",
            )

        # Relative Up
        if re.search(r"\b(brightness\s+up|increase\s+(?:the\s+)?brightness|turn\s+(?:the\s+)?brightness\s+up|brighter|make\s+it\s+brighter)\b", clean):
            curr = system_ops.get_brightness()
            curr = curr if curr >= 0 else 50
            new_bright = min(100, curr + 20)
            return LocalAction(
                action_type="brightness_up",
                execute_fn=lambda: system_ops.set_brightness(new_bright),
                confirm_phrase=f"brightness increased to {new_bright} percent",
                needs_follow_up=True,
                standalone_response=f"Brightness increased to {new_bright} percent, sir. Is that suitable? [FOLLOW_UP]",
            )

        # Relative Down
        if re.search(r"\b(brightness\s+down|decrease\s+(?:the\s+)?brightness|turn\s+(?:the\s+)?brightness\s+down|dimmer|make\s+it\s+dimmer)\b", clean):
            curr = system_ops.get_brightness()
            curr = curr if curr >= 0 else 50
            new_bright = max(10, curr - 20)
            return LocalAction(
                action_type="brightness_down",
                execute_fn=lambda: system_ops.set_brightness(new_bright),
                confirm_phrase=f"brightness reduced to {new_bright} percent",
                needs_follow_up=True,
                standalone_response=f"Brightness reduced to {new_bright} percent, sir. Is that comfortable? [FOLLOW_UP]",
            )

        # Exact Brightness: e.g. "set the brightness to 50%", "brightness 70"
        bright_pattern = r"(?:(?:set|turn|change|put)?\s*(?:the\s*)?(?:screen\s*|display\s*)?brightness\s*(?:to\s*)?|brightness\s*(?:to\s*)?)([a-z0-9\s\-]+)"
        match = re.search(bright_pattern, clean)
        if match:
            raw_val = match.group(1).strip()
            level = parse_spoken_number(raw_val)
            if level is not None and 0 <= level <= 100:
                return LocalAction(
                    action_type="set_brightness",
                    execute_fn=lambda: system_ops.set_brightness(level),
                    confirm_phrase=f"screen brightness adjusted to {level} percent",
                    needs_follow_up=True,
                    standalone_response=f"Screen brightness adjusted to {level} percent, sir. Shall I adjust anything else? [FOLLOW_UP]",
                )

        return None

    def _parse_system_telemetry(self, clean: str) -> Optional[LocalAction]:
        """Parse CPU, RAM, battery, and system hardware inquiries."""
        is_battery = bool(re.search(r"\b(battery\s+status|battery\s+percentage|battery\s+level|how\s+much\s+battery|battery\s+health)\b", clean))
        is_system = bool(re.search(r"\b(system\s+info(?:rmation)?|system\s+status|system\s+stats|cpu\s+usage|ram\s+usage|memory\s+usage|hardware\s+status|pc\s+stats)\b", clean))

        if is_battery:
            battery = psutil.sensors_battery()
            if battery is not None:
                plugged = "plugged in" if battery.power_plugged else "on battery power"
                ack = f"The battery is currently at {int(battery.percent)} percent and {plugged}, sir. Is there anything else you need? [FOLLOW_UP]"
                phrase = f"battery at {int(battery.percent)} percent"
            else:
                ack = "Your workstation is connected directly to AC power, sir. Anything else? [FOLLOW_UP]"
                phrase = "system on AC power"

            return LocalAction(
                action_type="battery_status",
                execute_fn=lambda: None,
                confirm_phrase=phrase,
                needs_follow_up=True,
                standalone_response=ack,
            )

        if is_system:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            battery = psutil.sensors_battery()
            if battery is not None:
                plugged = "plugged in" if battery.power_plugged else "on battery"
                ack = f"CPU utilization is at {int(cpu)} percent, memory is at {int(ram)} percent, and the battery is at {int(battery.percent)} percent ({plugged}), sir. Is there anything else you require? [FOLLOW_UP]"
                phrase = f"CPU at {int(cpu)} percent and RAM at {int(ram)} percent"
            else:
                ack = f"CPU utilization is at {int(cpu)} percent and memory usage is at {int(ram)} percent, sir. Is there anything else you require? [FOLLOW_UP]"
                phrase = f"CPU at {int(cpu)} percent and RAM at {int(ram)} percent"

            return LocalAction(
                action_type="system_info",
                execute_fn=lambda: None,
                confirm_phrase=phrase,
                needs_follow_up=True,
                standalone_response=ack,
            )

        return None

    def _parse_time_and_date(self, clean: str) -> Optional[LocalAction]:
        """Parse current time and date inquiries."""
        is_time = bool(re.search(r"\b(what\s+time\s+is\s+it|what['’]?s\s+the\s+time|tell\s+me\s+the\s+time|current\s+time|what\s+is\s+the\s+time)\b", clean))
        is_date = bool(re.search(r"\b(what\s+day\s+is\s+it|what['’]?s\s+the\s+date|what\s+is\s+the\s+date|what\s+is\s+today['’]?s\s+date|today['’]?s\s+date)\b", clean))

        if is_time or is_date:
            now = datetime.now()
            time_str = now.strftime("%I:%M %p").lstrip("0")
            day = now.day
            suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
            date_str = now.strftime(f"%A, %B {day}{suffix}")

            if is_time and is_date:
                ack = f"It is currently {time_str} on {date_str}, sir. Anything else I can assist with? [FOLLOW_UP]"
                phrase = f"it is {time_str} on {date_str}"
            elif is_time:
                ack = f"It is currently {time_str}, sir. Anything else I can assist with? [FOLLOW_UP]"
                phrase = f"it is {time_str}"
            else:
                ack = f"Today is {date_str}, sir. Is there anything else you require? [FOLLOW_UP]"
                phrase = f"today is {date_str}"

            return LocalAction(
                action_type="time_and_date",
                execute_fn=lambda: None,
                confirm_phrase=phrase,
                needs_follow_up=True,
                standalone_response=ack,
            )

        return None

    def _parse_application(self, clean: str) -> Optional[LocalAction]:
        """Parse application launch, termination, or launch retries."""
        # 1. Check for retry / negative feedback: e.g. "it is not open", "it didn't open", "open it again"
        retry_pattern = r"\b(it\s+is\s+not\s+open|it['’]?s\s+not\s+open|it\s+didn['’]?t\s+open|didn['’]?t\s+open|not\s+open|open\s+it\s+again|launch\s+it\s+again|try\s+again|try\s+opening\s+it\s+again)\b"
        if re.search(retry_pattern, clean):
            target = getattr(self, "_last_opened_app", None) or "edge"
            clean_disp = "Microsoft Edge" if "edge" in target else target.title()
            logger.info("Local Intent matched: Retry App Launch for '%s'", target)
            return LocalAction(
                action_type="retry_launch_app",
                execute_fn=lambda: system_ops.open_application(target),
                confirm_phrase=f"relaunching {clean_disp}",
                needs_follow_up=True,
                standalone_response=f"Apologies, sir. Relaunching {clean_disp} now. Is it appearing on your desktop? [FOLLOW_UP]",
            )

        # 2. Close application
        close_match = re.search(r"\b(?:close|quit|kill|terminate|exit)\s+(?:the\s+)?([a-z0-9\s\-]+)", clean)
        if close_match:
            candidate = close_match.group(1).strip()
            sorted_apps = sorted(KNOWN_APPS.keys(), key=len, reverse=True)
            for app_key in sorted_apps:
                if app_key in candidate or candidate in app_key:
                    return LocalAction(
                        action_type="close_app",
                        execute_fn=lambda ak=app_key: system_ops.close_application(ak),
                        confirm_phrase=f"closed {app_key.title()}",
                        needs_follow_up=True,
                        standalone_response=f"Closed {app_key.title()}, sir. Anything else I can assist with? [FOLLOW_UP]",
                    )

        # 3. Launch application: matches 'open edge', 'microsoft edge, open microsoft edge', 'can you launch spotify'
        app_target = None
        has_launch_verb = bool(re.search(r"\b(open|launch|start|run|bring\s+up)\b", clean))
        sorted_apps = sorted(KNOWN_APPS.keys(), key=len, reverse=True)

        for app_key in sorted_apps:
            if app_key in clean and (has_launch_verb or clean == app_key or clean.endswith(app_key) or clean.startswith(app_key)):
                app_target = app_key
                break

        if app_target is not None:
            self._last_opened_app = app_target
            if app_target in ("windows update", "windows updates", "updates", "update"):
                return LocalAction(
                    action_type="launch_settings",
                    execute_fn=lambda: system_ops.open_settings("windowsupdate"),
                    confirm_phrase="Windows Updates opened",
                    needs_follow_up=True,
                    standalone_response="Windows Updates opened, sir. Checking for updates. [FOLLOW_UP]",
                )
            elif app_target == "settings":
                return LocalAction(
                    action_type="launch_settings",
                    execute_fn=lambda: system_ops.open_settings("display"),
                    confirm_phrase="Settings opened",
                    needs_follow_up=True,
                    standalone_response="Windows Settings opened, sir. What else can I configure? [FOLLOW_UP]",
                )
            else:
                if "edge" in app_target:
                    ack = "Microsoft Edge is open, sir. Shall I navigate anywhere for you? [FOLLOW_UP]"
                    phrase = "Microsoft Edge is open"
                elif "chrome" in app_target:
                    ack = "Google Chrome is ready, sir. What shall we search? [FOLLOW_UP]"
                    phrase = "Google Chrome is ready"
                elif "code" in app_target or "vscode" in app_target:
                    ack = "Visual Studio Code is open, sir. Ready to write code. [FOLLOW_UP]"
                    phrase = "Visual Studio Code is open"
                elif "spotify" in app_target:
                    ack = "Spotify is ready, sir. Would you like me to play something? [FOLLOW_UP]"
                    phrase = "Spotify is ready"
                elif "notepad" in app_target:
                    ack = "Notepad is open, sir. Ready for your notes. [FOLLOW_UP]"
                    phrase = "Notepad is open"
                elif "calc" in app_target:
                    ack = "Calculator is ready, sir. What shall we calculate? [FOLLOW_UP]"
                    phrase = "Calculator is ready"
                elif "terminal" in app_target or "cmd" in app_target:
                    ack = "Terminal is open, sir. Ready for commands. [FOLLOW_UP]"
                    phrase = "Terminal is open"
                elif "task" in app_target:
                    ack = "Task Manager launched, sir. Monitoring system performance. [FOLLOW_UP]"
                    phrase = "Task Manager is open"
                else:
                    ack = f"{app_target.title()} has been launched, sir. Is there anything else you need? [FOLLOW_UP]"
                    phrase = f"{app_target.title()} launched"

                return LocalAction(
                    action_type="launch_app",
                    execute_fn=lambda: system_ops.open_application(app_target),
                    confirm_phrase=phrase,
                    needs_follow_up=True,
                    standalone_response=ack,
                )

        return None

    def _parse_web_and_media(self, clean: str) -> Optional[LocalAction]:
        """Parse web searches (YouTube, Google) and direct website navigation.

        Intercepts web and media searches locally in <1ms without consuming cloud LLM tokens.
        """
        # 1. YouTube Direct Navigation: e.g. "open youtube", "launch youtube", "go to youtube"
        if re.match(r"^(?:open|launch|go\s+to)?\s*youtube$", clean.strip(), re.IGNORECASE) or clean.strip().lower() == "youtube":
            url = "https://www.youtube.com"
            logger.info("Local Intent matched: Direct YouTube navigation")
            return LocalAction(
                action_type="open_youtube",
                execute_fn=lambda u=url: web_ops.open_website(u),
                confirm_phrase="YouTube opened",
                needs_follow_up=True,
                standalone_response="YouTube is open, sir. What shall we watch? [FOLLOW_UP]",
            )

        # 2. YouTube Search:
        # Matches queries ending with "on/in youtube" or starting with "youtube search/play/start"
        # Handles phonetic STT slips (e.g. "sart", "serch") and phrases like "Hanuman Chalisa on YouTube"
        yt_search = re.search(
            r"(?:(?:search(?:\s+for)?|sart|start|serch|play|find|look\s+up|put\s+on|stream|show|open)\s+)?(.+?)\s+(?:on|in)\s+youtube\b",
            clean,
            re.IGNORECASE,
        )
        if not yt_search:
            yt_search = re.search(
                r"\byoutube\s+(?:search(?:\s+for)?|sart|start|serch|play|find|show)\s+(.+)\b",
                clean,
                re.IGNORECASE,
            )

        if yt_search:
            raw_query = yt_search.group(1).strip()
            # Clean common conversational fillers and search prefixes
            clean_query = re.sub(
                r"^(?:(?:no|yes|please|can\s+you|could\s+you|would\s+you)\s*,?\s*)+",
                "",
                raw_query,
                flags=re.IGNORECASE,
            )
            clean_query = re.sub(
                r"^(?:search(?:\s+for)?|sart|start|serch|play|find|look\s+up|put\s+on|stream|show|open)\s+",
                "",
                clean_query,
                flags=re.IGNORECASE,
            )
            clean_query = re.sub(r"[?.!,]+$", "", clean_query).strip(" '\"")
            if clean_query:
                encoded = urllib.parse.quote_plus(clean_query)
                url = f"https://www.youtube.com/results?search_query={encoded}"
                logger.info("Local Intent matched: YouTube search for '%s'", clean_query)
                return LocalAction(
                    action_type="youtube_search",
                    execute_fn=lambda u=url: web_ops.open_website(u),
                    confirm_phrase=f"searching '{clean_query}' on YouTube",
                    needs_follow_up=True,
                    standalone_response=f"Searching for '{clean_query}' on YouTube, sir. [FOLLOW_UP]",
                )

        # 3. Google Search: e.g. "search sidemen on google", "google python documentation"
        g_search = re.search(
            r"\b(?:search\s+(?:for\s+)?|look\s+up\s+|find\s+)(.+?)\s+(?:on|in)\s+google\b",
            clean,
            re.IGNORECASE,
        )
        if not g_search:
            g_search = re.search(
                r"\bgoogle\s+(?:search\s+(?:for\s+)?|for\s+)?(.+)\b",
                clean,
                re.IGNORECASE,
            )

        if g_search:
            query = g_search.group(1).strip()
            query = re.sub(r"[?.!,]+$", "", query).strip(" '\"")
            if query and query.lower() not in ("chrome", "maps", "drive", "docs"):
                encoded = urllib.parse.quote_plus(query)
                url = f"https://www.google.com/search?q={encoded}"
                logger.info("Local Intent matched: Google search for '%s'", query)
                return LocalAction(
                    action_type="google_search",
                    execute_fn=lambda u=url: web_ops.open_website(u),
                    confirm_phrase=f"searching '{query}' on Google",
                    needs_follow_up=True,
                    standalone_response=f"Searching Google for '{query}', sir. [FOLLOW_UP]",
                )

        # 4. Common Portal Navigation: e.g. "open github", "open reddit", "go to wikipedia"
        for site_name, site_url in COMMON_WEBSITES.items():
            pattern = rf"\b(?:open|launch|go\s+to)\s+{re.escape(site_name)}\b"
            if re.search(pattern, clean, re.IGNORECASE) or clean.strip().lower() == site_name:
                disp_name = "GitHub" if site_name == "github" else site_name.title()
                logger.info("Local Intent matched: Direct portal navigation to %s", disp_name)
                return LocalAction(
                    action_type=f"open_{site_name}",
                    execute_fn=lambda u=site_url: web_ops.open_website(u),
                    confirm_phrase=f"{disp_name} opened",
                    needs_follow_up=True,
                    standalone_response=f"{disp_name} is open, sir. [FOLLOW_UP]",
                )

        # 5. Direct URL / Domain Navigation: e.g. "open https://news.ycombinator.com", "open stackoverflow.com"
        url_match = re.search(
            r"\b(?:open|launch|go\s+to|navigate\s+to)\s+(https?://\S+|\b[a-zA-Z0-9-]+\.(?:com|org|io|net|edu|dev|ai|gov)\b\S*)",
            clean,
            re.IGNORECASE,
        )
        if url_match:
            target_url = url_match.group(1).strip()
            logger.info("Local Intent matched: Direct URL navigation to %s", target_url)
            return LocalAction(
                action_type="open_url",
                execute_fn=lambda u=target_url: web_ops.open_website(u),
                confirm_phrase=f"navigating to {target_url}",
                needs_follow_up=True,
                standalone_response=f"Opening {target_url}, sir. [FOLLOW_UP]",
            )

        return None

    def _parse_workspace(self, clean: str) -> Optional[LocalAction]:
        """Parse proactive desktop workspace profile activation.

        Matches commands like:
        - "launch coding workspace" / "open coding mode" / "coding workspace"
        - "switch to study workspace" / "study mode"
        - "activate focus mode" / "focus workspace"
        - "launch media mode" / "meeting workspace"
        """
        ws_match = re.search(
            r"\b(?:launch|open|activate|switch\s+to|start|run|initialize|set\s+up)\s+(?:the\s+)?([a-z0-9\-_]+)\s+(?:workspace|mode|environment|profile)\b",
            clean,
            re.IGNORECASE,
        )
        if not ws_match:
            ws_match = re.search(
                r"\b([a-z0-9\-_]+)\s+(?:workspace|mode)\b",
                clean,
                re.IGNORECASE,
            )

        if ws_match:
            ws_candidate = ws_match.group(1).strip().lower()
            mm = get_memory_manager()
            ws_info = mm.get_workspace(ws_candidate)
            if ws_info:
                ws_name = ws_info["name"]
                cfg = ws_info.get("config", {})
                apps = cfg.get("apps", [])
                vol = cfg.get("volume")
                bright = cfg.get("brightness")

                details = []
                if apps:
                    app_names = [("VS Code" if a == "code" else ("Edge" if a == "edge" else a.title())) for a in apps]
                    details.append(f"{', '.join(app_names)} launched")
                if bright is not None:
                    details.append(f"brightness set to {bright} percent")
                if vol is not None:
                    details.append(f"volume at {vol} percent")

                detail_str = f": {', '.join(details)}" if details else ""
                ack = f"{ws_name.title()} workspace initialized, sir{detail_str}. Standing by for instructions. [FOLLOW_UP]"

                logger.info("Local Intent matched: Workspace activation '%s'", ws_name)
                return LocalAction(
                    action_type=f"workspace_{ws_name}",
                    execute_fn=lambda name=ws_name: mm.execute_workspace(name),
                    confirm_phrase=f"{ws_name.title()} workspace activated",
                    needs_follow_up=True,
                    standalone_response=ack,
                )

        return None
