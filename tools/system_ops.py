"""
JARVIS Assistant - System Operations Tools
Provides Windows system automation tools callable by Gemini function calling:
- Audio volume control
- Display brightness adjustment
- Application launching and termination
- Workstation lock
- Hardware resource statistics
"""

import ctypes
import logging
import os
import shutil
import subprocess
from typing import Dict, Optional

import psutil

# Optional pycaw import for direct Windows CoreAudio control
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    _PYCAW_AVAILABLE = True
except ImportError:
    _PYCAW_AVAILABLE = False

logger = logging.getLogger(__name__)

# Common application name to launch executable / command mapping
APP_LAUNCH_MAP: Dict[str, str] = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "notepad": "notepad",
    "calculator": "calc",
    "calc": "calc",
    "explorer": "explorer",
    "file explorer": "explorer",
    "cmd": "cmd",
    "command prompt": "cmd",
    "code": "code",
    "vscode": "code",
    "visual studio code": "code",
    "spotify": "spotify",
    "edge": "msedge",
    "msedge": "msedge",
    "microsoft edge": "msedge",
    "terminal": "wt",
    "powershell": "powershell",
    "taskmgr": "taskmgr",
    "task manager": "taskmgr",
    "settings": "start ms-settings:",
    "windows update": "start ms-settings:windowsupdate",
    "windows updates": "start ms-settings:windowsupdate",
    "update": "start ms-settings:windowsupdate",
    "updates": "start ms-settings:windowsupdate",
}

# Common application name to process image name for taskkill
APP_PROCESS_MAP: Dict[str, str] = {
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "notepad": "notepad.exe",
    "calculator": "CalculatorApp.exe",
    "calc": "CalculatorApp.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "cmd": "cmd.exe",
    "command prompt": "cmd.exe",
    "code": "Code.exe",
    "vscode": "Code.exe",
    "visual studio code": "Code.exe",
    "spotify": "Spotify.exe",
    "edge": "msedge.exe",
    "msedge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "terminal": "WindowsTerminal.exe",
    "powershell": "powershell.exe",
    "taskmgr": "Taskmgr.exe",
    "task manager": "Taskmgr.exe",
}


def set_volume(level: int) -> str:
    """
    Set the Windows system master audio volume level.

    Args:
        level: Target volume percentage, an integer between 0 and 100.

    Returns:
        A confirmation message indicating the new volume level or an error description.
    """
    try:
        level_int = max(0, min(100, int(level)))
    except (ValueError, TypeError):
        return f"Error: Invalid volume level '{level}'. Must be an integer between 0 and 100."

    logger.info("Setting system volume to %d%%", level_int)

    # Method 1: Use pycaw if available (precise and fast)
    if _PYCAW_AVAILABLE:
        try:
            device = AudioUtilities.GetSpeakers()
            if hasattr(device, "EndpointVolume"):
                endpoint_volume = device.EndpointVolume
            else:
                interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                endpoint_volume = cast(interface, POINTER(IAudioEndpointVolume))

            endpoint_volume.SetMasterVolumeLevelScalar(level_int / 100.0, None)
            logger.info("Master volume set to %d%% via pycaw", level_int)
            return f"System volume set to {level_int}%."
        except Exception as e:
            logger.warning("pycaw volume adjustment failed, falling back to PowerShell: %s", e)

    # Method 2: Fallback to NirCmd if installed
    nircmd_path = shutil.which("nircmd")
    if nircmd_path:
        try:
            nircmd_vol = int(level_int * 655.35)
            subprocess.run([nircmd_path, "setsysvolume", str(nircmd_vol)], check=True, timeout=5)
            logger.info("Master volume set to %d%% via nircmd", level_int)
            return f"System volume set to {level_int}%."
        except Exception as e:
            logger.warning("NirCmd volume adjustment failed: %s", e)

    # Method 3: PowerShell subprocess using WScript.Shell SendKeys
    try:
        steps_up = round(level_int / 2)
        ps_script = (
            "$wsh = New-Object -ComObject WScript.Shell; "
            "1..50 | ForEach-Object { $wsh.SendKeys([char]174) }; "
            f"1..{steps_up} | ForEach-Object {{ $wsh.SendKeys([char]175) }}"
        )
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0:
            logger.info("Master volume set to approx %d%% via PowerShell", level_int)
            return f"System volume set to approximately {level_int}%."
        else:
            logger.error("PowerShell volume command failed: %s", res.stderr)
            return f"Failed to set volume via PowerShell: {res.stderr.strip()}"
    except subprocess.TimeoutExpired:
        logger.error("PowerShell volume command timed out")
        return "Failed to set volume: operation timed out."
    except Exception as e:
        logger.error("Unexpected error setting volume: %s", e, exc_info=True)
        return f"Error setting volume to {level_int}%: {str(e)}"


def get_volume() -> int:
    """
    Get the current Windows system master audio volume percentage (0-100).

    Returns:
        The integer volume percentage (0-100), or -1 if unable to retrieve.
    """
    if _PYCAW_AVAILABLE:
        try:
            device = AudioUtilities.GetSpeakers()
            if hasattr(device, "EndpointVolume"):
                endpoint_volume = device.EndpointVolume
            else:
                interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                endpoint_volume = cast(interface, POINTER(IAudioEndpointVolume))

            scalar = endpoint_volume.GetMasterVolumeLevelScalar()
            vol = int(round(scalar * 100.0))
            logger.debug("Current master volume retrieved via pycaw: %d%%", vol)
            return vol
        except Exception as e:
            logger.warning("pycaw get_volume failed: %s", e)

    return -1



def set_brightness(level: int) -> str:
    """
    Set the screen brightness level using WMI via PowerShell.

    Args:
        level: Target brightness percentage, an integer between 0 and 100.

    Returns:
        A confirmation message indicating the new brightness level or an error description.
    """
    try:
        level_int = max(0, min(100, int(level)))
    except (ValueError, TypeError):
        return f"Error: Invalid brightness level '{level}'. Must be an integer between 0 and 100."

    logger.info("Setting screen brightness to %d%% via WMI", level_int)

    # Method 1: Get-WmiObject WmiMonitorBrightnessMethods
    ps_cmd_wmi = (
        f"(Get-WmiObject -Namespace root/wmi -Class WmiMonitorBrightnessMethods)"
        f".WmiSetBrightness(1, {level_int})"
    )
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd_wmi],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0 and not res.stderr.strip():
            logger.info("Screen brightness set to %d%% via WMI", level_int)
            return f"Screen brightness set to {level_int}%."
    except subprocess.TimeoutExpired:
        logger.error("PowerShell WMI brightness command timed out")
        return "Failed to set screen brightness: operation timed out."
    except Exception as e:
        logger.warning("WMI brightness call failed: %s", e)

    # Method 2: Fallback to Get-CimInstance
    ps_cmd_cim = (
        f"Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorBrightnessMethods | "
        f"Invoke-CimMethod -MethodName WmiSetBrightness -Arguments @{{Timeout=1; Brightness={level_int}}}"
    )
    try:
        res_cim = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd_cim],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res_cim.returncode == 0 and not res_cim.stderr.strip():
            logger.info("Screen brightness set to %d%% via CIM", level_int)
            return f"Screen brightness set to {level_int}%."

        err_msg = res_cim.stderr.strip() or "Display hardware does not support WMI brightness control."
        logger.error("CIM brightness call failed: %s", err_msg)
        return f"Failed to set screen brightness: {err_msg}"
    except subprocess.TimeoutExpired:
        logger.error("PowerShell CIM brightness command timed out")
        return "Failed to set screen brightness: operation timed out."
    except Exception as e:
        logger.error("Unexpected error setting brightness: %s", e, exc_info=True)
        return f"Error setting screen brightness to {level_int}%: {str(e)}"


def get_brightness() -> int:
    """
    Get the current screen brightness percentage (0-100) using WMI via PowerShell.

    Returns:
        The integer brightness percentage (0-100), or -1 if unable to retrieve.
    """
    try:
        ps_cmd_wmi = "(Get-WmiObject -Namespace root/wmi -Class WmiMonitorBrightness).CurrentBrightness"
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd_wmi],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip().isdigit():
            val = int(res.stdout.strip())
            logger.debug("Current brightness retrieved via WMI: %d%%", val)
            return val
    except Exception as e:
        logger.debug("WMI get_brightness failed: %s", e)

    try:
        ps_cmd_cim = "(Get-CimInstance -Namespace root/wmi -ClassName WmiMonitorBrightness).CurrentBrightness"
        res_cim = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd_cim],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res_cim.returncode == 0 and res_cim.stdout.strip().isdigit():
            val = int(res_cim.stdout.strip())
            logger.debug("Current brightness retrieved via CIM: %d%%", val)
            return val
    except Exception as e:
        logger.debug("CIM get_brightness failed: %s", e)

    return -1



def open_application(app_name: str) -> str:
    """
    Launch a Windows application by name.

    Maps common application names (e.g., chrome, notepad, calculator, explorer,
    cmd, code/vscode, spotify) to their corresponding executable or system command.

    Args:
        app_name: Name of the application to launch.

    Returns:
        A confirmation message if launched successfully, or an error description.
    """
    if not app_name or not app_name.strip():
        return "Error: Application name cannot be empty."

    raw_name = app_name.strip()
    clean_name = raw_name.lower().replace("application", "").replace("app", "").strip()

    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]

    target = None
    if "edge" in clean_name:
        for p in edge_paths:
            if os.path.exists(p):
                target = p
                break
        if not target:
            target = "microsoft-edge:"
    elif "chrome" in clean_name:
        target = "chrome"
    elif "code" in clean_name:
        target = "code"
    elif "notepad" in clean_name:
        target = "notepad"
    elif "calc" in clean_name:
        target = "calc"
    elif "spotify" in clean_name:
        target = "spotify"
    elif "terminal" in clean_name:
        target = "wt"
    else:
        target = APP_LAUNCH_MAP.get(clean_name, raw_name)

    logger.info("Opening application '%s' (resolved target: '%s')", raw_name, target)

    try:
        launched = False
        # If target is a valid file path, launch directly
        if target and os.path.isabs(str(target)) and os.path.exists(str(target)):
            try:
                subprocess.Popen([str(target)], shell=False)
                launched = True
                logger.info("Launched '%s' via direct process spawn", raw_name)
            except Exception as pe_err:
                logger.debug("Direct spawn failed for '%s': %s", target, pe_err)

        if not launched:
            if str(target).startswith("start "):
                subprocess.Popen(target, shell=True)
            elif "edge" in clean_name:
                subprocess.Popen("cmd /c start microsoft-edge:", shell=True)
            else:
                subprocess.Popen(f'cmd /c start "" "{target}"', shell=True)
            logger.info("Launched '%s' via desktop shell", raw_name)

        return f"Application '{raw_name}' launched successfully."
    except Exception as e:
        logger.error("Failed to launch application '%s': %s", raw_name, e, exc_info=True)
        return f"Failed to open application '{raw_name}': {str(e)}"



def open_settings(setting: str = "windowsupdate") -> str:
    """
    Open Windows Settings directly to a specific page.

    Args:
        setting: Specific page name, e.g. 'windowsupdate', 'bluetooth', 'display',
                 'sound', 'network', 'battery', 'apps'. Defaults to 'windowsupdate'.

    Returns:
        A confirmation message.
    """
    clean = (setting or "windowsupdate").lower().strip()
    page_map = {
        "windowsupdate": "windowsupdate",
        "windows update": "windowsupdate",
        "update": "windowsupdate",
        "updates": "windowsupdate",
        "bluetooth": "bluetooth",
        "display": "display",
        "sound": "sound",
        "audio": "sound",
        "network": "network",
        "wifi": "network-wifi",
        "battery": "batterysaver",
        "apps": "appsfeatures",
        "notifications": "notifications",
        "storage": "storagesense",
    }
    target_page = page_map.get(clean, clean.replace(" ", ""))
    cmd = f"start ms-settings:{target_page}"

    logger.info("Opening Windows Settings page: '%s' (command: '%s')", clean, cmd)
    try:
        subprocess.Popen(cmd, shell=True)
        return f"Opened Windows Settings ({clean}) successfully."
    except Exception as e:
        logger.error("Failed to open Windows Settings: %s", e)
        return f"Error opening Windows Settings: {e}"


def close_application(app_name: str) -> str:
    """
    Terminate a running Windows application by name using taskkill.

    Args:
        app_name: Name of the application or executable to terminate.

    Returns:
        A confirmation message if closed, or a message indicating the process was not found.
    """
    if not app_name or not app_name.strip():
        return "Error: Application name cannot be empty."

    raw_name = app_name.strip()
    clean_name = raw_name.lower()

    # Determine process image name
    process_image = APP_PROCESS_MAP.get(clean_name)
    if not process_image:
        process_image = raw_name if raw_name.lower().endswith(".exe") else f"{raw_name}.exe"

    logger.info("Closing application '%s' with process image '%s'", raw_name, process_image)

    # Possible candidate names (e.g. calculator might be CalculatorApp.exe or calc.exe)
    candidates = [process_image]
    if clean_name in ("calculator", "calc"):
        candidates = ["CalculatorApp.exe", "calc.exe", "Calculator.exe"]

    for proc in candidates:
        try:
            res = subprocess.run(
                ["taskkill", "/F", "/IM", proc],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode == 0:
                logger.info("Successfully terminated '%s' (process: %s)", raw_name, proc)
                return f"Application '{raw_name}' closed successfully."
        except subprocess.TimeoutExpired:
            logger.error("taskkill timed out for '%s'", proc)
            return f"Failed to close application '{raw_name}': operation timed out."
        except Exception as e:
            logger.error("Error running taskkill for '%s': %s", proc, e, exc_info=True)
            return f"Error closing application '{raw_name}': {str(e)}"

    logger.warning("No running process found for '%s' (checked %s)", raw_name, candidates)
    return f"No running instance of '{raw_name}' was found."


def lock_screen() -> str:
    """
    Lock the Windows workstation using user32.LockWorkStation.

    Returns:
        A confirmation message indicating whether the workstation was locked.
    """
    logger.info("Requesting workstation lock via user32.LockWorkStation")
    try:
        result = ctypes.windll.user32.LockWorkStation()
        if result != 0:
            logger.info("Workstation locked successfully")
            return "Windows workstation locked successfully."
        else:
            logger.error("user32.LockWorkStation returned code 0")
            return "Failed to lock the Windows workstation."
    except Exception as e:
        logger.error("Exception while locking workstation: %s", e, exc_info=True)
        return f"Error locking workstation: {str(e)}"


def get_system_info() -> str:
    """
    Retrieve current system resource statistics including CPU, RAM, and Disk usage.

    Returns:
        A formatted, human-readable summary of CPU, memory, and disk utilization.
    """
    logger.debug("Gathering system resource statistics...")
    try:
        # CPU statistics
        cpu_percent = psutil.cpu_percent(interval=0.5)
        cpu_count = psutil.cpu_count(logical=True)

        # Virtual memory statistics
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024 ** 3)
        mem_total_gb = mem.total / (1024 ** 3)

        # System disk statistics
        system_drive = os.getenv("SystemDrive", "C:") + "\\"
        disk = psutil.disk_usage(system_drive)
        disk_used_gb = disk.used / (1024 ** 3)
        disk_total_gb = disk.total / (1024 ** 3)
        disk_free_gb = disk.free / (1024 ** 3)

        lines = [
            "System Information:",
            f"- CPU Usage: {cpu_percent:.1f}% ({cpu_count} logical cores)",
            f"- RAM Usage: {mem.percent:.1f}% ({mem_used_gb:.1f} GB used / {mem_total_gb:.1f} GB total)",
            f"- Disk Usage ({system_drive}): {disk.percent:.1f}% ({disk_used_gb:.1f} GB used / {disk_total_gb:.1f} GB total, {disk_free_gb:.1f} GB free)",
        ]

        # Optional battery statistics if available
        battery = psutil.sensors_battery()
        if battery is not None:
            status = "Plugged in" if battery.power_plugged else "On battery"
            lines.append(f"- Battery: {battery.percent:.0f}% ({status})")

        output = "\n".join(lines)
        logger.info("System information retrieved successfully")
        return output
    except Exception as e:
        logger.error("Failed to gather system information: %s", e, exc_info=True)
        return f"Failed to retrieve system information: {str(e)}"
