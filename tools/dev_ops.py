"""
JARVIS Assistant - Developer Operations Tools
Provides developer workflow and shell automation tools callable by Gemini function calling:
- Git status
- Git stage & commit
- Git pull
- Arbitrary shell command execution
"""

import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum character count for returned command output to fit within context limits
MAX_OUTPUT_LENGTH: int = 2000
DEFAULT_TIMEOUT: int = 8


def git_status(repo_path: str) -> str:
    """
    Run 'git status' in the specified repository directory.

    Args:
        repo_path: Filepath to the git repository root or working directory.

    Returns:
        The text output of 'git status', or an error message if the operation failed.
    """
    if not repo_path or not repo_path.strip():
        return "Error: Repository path cannot be empty."

    clean_path = os.path.abspath(repo_path.strip())
    if not os.path.exists(clean_path):
        return f"Error: Repository path '{clean_path}' does not exist."
    if not os.path.isdir(clean_path):
        return f"Error: Path '{clean_path}' is not a directory."

    logger.info("Executing 'git status' in '%s'", clean_path)

    try:
        res = subprocess.run(
            ["git", "status"],
            cwd=clean_path,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT,
            check=True,
        )
        output = (res.stdout + res.stderr).strip()
        return output if output else "Working tree clean (no output)."
    except subprocess.CalledProcessError as e:
        logger.warning("git status failed in '%s' (exit code %d)", clean_path, e.returncode)
        err = (e.stderr or e.stdout).strip()
        return f"Git status failed (exit code {e.returncode}):\n{err}"
    except subprocess.TimeoutExpired:
        logger.error("git status timed out after %d seconds in '%s'", DEFAULT_TIMEOUT, clean_path)
        return f"Error: 'git status' operation timed out after {DEFAULT_TIMEOUT} seconds."
    except FileNotFoundError:
        logger.error("Git executable not found in system PATH")
        return "Error: 'git' command not found. Please ensure Git is installed and available in your PATH."
    except Exception as e:
        logger.error("Unexpected error during git status in '%s': %s", clean_path, e, exc_info=True)
        return f"Error running git status: {str(e)}"


def git_commit(repo_path: str, message: str) -> str:
    """
    Stage all changes and commit with the given message.

    Executes 'git add -A' followed by 'git commit -m "<message>"'.

    Args:
        repo_path: Filepath to the git repository.
        message: Commit message describing the changes.

    Returns:
        The output from the git add and git commit operations, or an error message.
    """
    if not repo_path or not repo_path.strip():
        return "Error: Repository path cannot be empty."
    if not message or not message.strip():
        return "Error: Commit message cannot be empty."

    clean_path = os.path.abspath(repo_path.strip())
    commit_msg = message.strip()

    if not os.path.exists(clean_path):
        return f"Error: Repository path '{clean_path}' does not exist."
    if not os.path.isdir(clean_path):
        return f"Error: Path '{clean_path}' is not a directory."

    logger.info("Staging and committing in '%s' with message: '%s'", clean_path, commit_msg)

    try:
        # Step 1: Stage all changes
        add_res = subprocess.run(
            ["git", "add", "-A"],
            cwd=clean_path,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT,
            check=True,
        )
        add_out = (add_res.stdout + add_res.stderr).strip()

        # Step 2: Commit staged changes
        commit_res = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=clean_path,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT,
            check=True,
        )
        commit_out = (commit_res.stdout + commit_res.stderr).strip()

        outputs = []
        if add_out:
            outputs.append(f"[git add]\n{add_out}")
        if commit_out:
            outputs.append(f"[git commit]\n{commit_out}")

        logger.info("Successfully committed changes in '%s'", clean_path)
        return "\n\n".join(outputs) if outputs else "Changes staged and committed successfully."

    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout).strip()
        logger.warning("Git commit operation failed in '%s': %s", clean_path, err)
        if "nothing to commit" in err.lower():
            return f"Nothing to commit: {err}"
        return f"Git commit failed (exit code {e.returncode}):\n{err}"
    except subprocess.TimeoutExpired:
        logger.error("Git operation timed out after %d seconds in '%s'", DEFAULT_TIMEOUT, clean_path)
        return f"Error: Git commit operation timed out after {DEFAULT_TIMEOUT} seconds."
    except FileNotFoundError:
        logger.error("Git executable not found in system PATH")
        return "Error: 'git' command not found. Please ensure Git is installed and available in your PATH."
    except Exception as e:
        logger.error("Unexpected error during git commit in '%s': %s", clean_path, e, exc_info=True)
        return f"Error executing git commit: {str(e)}"


def git_pull(repo_path: str) -> str:
    """
    Run 'git pull' in the specified repository directory to fetch and integrate remote changes.

    Args:
        repo_path: Filepath to the git repository.

    Returns:
        The output from the 'git pull' command, or an error message.
    """
    if not repo_path or not repo_path.strip():
        return "Error: Repository path cannot be empty."

    clean_path = os.path.abspath(repo_path.strip())
    if not os.path.exists(clean_path):
        return f"Error: Repository path '{clean_path}' does not exist."
    if not os.path.isdir(clean_path):
        return f"Error: Path '{clean_path}' is not a directory."

    logger.info("Executing 'git pull' in '%s'", clean_path)

    try:
        res = subprocess.run(
            ["git", "pull"],
            cwd=clean_path,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT,
            check=True,
        )
        output = (res.stdout + res.stderr).strip()
        logger.info("git pull completed successfully in '%s'", clean_path)
        return output if output else "Git pull completed successfully (no output)."
    except subprocess.CalledProcessError as e:
        logger.warning("git pull failed in '%s' (exit code %d)", clean_path, e.returncode)
        err = (e.stderr or e.stdout).strip()
        return f"Git pull failed (exit code {e.returncode}):\n{err}"
    except subprocess.TimeoutExpired:
        logger.error("git pull timed out after %d seconds in '%s'", DEFAULT_TIMEOUT, clean_path)
        return f"Error: 'git pull' operation timed out after {DEFAULT_TIMEOUT} seconds."
    except FileNotFoundError:
        logger.error("Git executable not found in system PATH")
        return "Error: 'git' command not found. Please ensure Git is installed and available in your PATH."
    except Exception as e:
        logger.error("Unexpected error during git pull in '%s': %s", clean_path, e, exc_info=True)
        return f"Error running git pull: {str(e)}"


def run_command(command: str, cwd: str = ".") -> str:
    """
    Run an arbitrary shell command on the host system.

    Captures stdout and stderr, times out after 30 seconds, and truncates
    output to 2000 characters if longer.

    Args:
        command: The shell command line string to execute.
        cwd: Directory from which to execute the command. Defaults to '.' (current directory).

    Returns:
        Combined stdout and stderr output from the command, truncated to 2000 characters if longer.
    """
    if not command or not command.strip():
        return "Error: Command string cannot be empty."

    cmd_str = command.strip()
    target_cwd = os.path.abspath(cwd.strip() if cwd else ".")

    if not os.path.exists(target_cwd):
        return f"Error: Working directory '{target_cwd}' does not exist."
    if not os.path.isdir(target_cwd):
        return f"Error: Path '{target_cwd}' is not a directory."

    logger.info("Executing shell command: '%s' in cwd: '%s'", cmd_str, target_cwd)

    try:
        res = subprocess.run(
            cmd_str,
            cwd=target_cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUT,
            check=True,
        )
        output = (res.stdout + res.stderr).strip()
        if not output:
            output = "Command executed successfully (exit code 0, no output)."
    except subprocess.CalledProcessError as e:
        logger.warning("Command '%s' returned non-zero exit code %d", cmd_str, e.returncode)
        err = (e.stderr or "").strip()
        out = (e.stdout or "").strip()
        output = f"Command failed with exit code {e.returncode}:\n{out}\n{err}".strip()
    except subprocess.TimeoutExpired:
        logger.warning("Command '%s' timed out after %d seconds", cmd_str, DEFAULT_TIMEOUT)
        output = f"Error: Command timed out after {DEFAULT_TIMEOUT} seconds. This operation takes too long for synchronous voice response. Suggest opening the app or running manually."
    except Exception as e:
        logger.error("Unexpected error running command '%s': %s", cmd_str, e, exc_info=True)
        return f"Error executing command: {str(e)}"

    # Truncate output to MAX_OUTPUT_LENGTH if needed
    if len(output) > MAX_OUTPUT_LENGTH:
        output = output[:MAX_OUTPUT_LENGTH] + f"\n... [Output truncated to {MAX_OUTPUT_LENGTH} characters]"

    return output
