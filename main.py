"""
main.py - JARVIS Assistant entry point and core execution loop.

State machine:
    IDLE  →  wake word detected  →  LISTENING  →  silence detected
      →  PROCESSING (STT + Gemini)  →  RESPONDING (TTS)  →  IDLE

Usage:
    python main.py
"""

import sys
import os
import time
import logging
import yaml
import numpy as np

# ------------------------------------------------------------------ #
#  Logging Setup                                                      #
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis")


# ------------------------------------------------------------------ #
#  State Definitions                                                  #
# ------------------------------------------------------------------ #

class State:
    """Simple state constants for the main loop."""
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    RESPONDING = "RESPONDING"


# ------------------------------------------------------------------ #
#  Config Loader                                                      #
# ------------------------------------------------------------------ #

def load_config(config_path: str = None) -> dict:
    """Load configuration from config.yaml.

    Args:
        config_path: Path to config file. Defaults to config.yaml in the
                     same directory as this script.

    Returns:
        Parsed config dictionary.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    if not os.path.exists(config_path):
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Config loaded from %s", config_path)
    return config


# ------------------------------------------------------------------ #
#  Startup Banner                                                     #
# ------------------------------------------------------------------ #

BANNER = r"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
"""


def print_banner():
    """Print the JARVIS startup banner."""
    print(BANNER)
    print("  Your AI Desktop Assistant")
    print("  Say 'Hey Jarvis' to begin.\n")
    print("─" * 50)


# ------------------------------------------------------------------ #
#  Main Loop                                                          #
# ------------------------------------------------------------------ #

def main():
    """Run the JARVIS assistant main loop."""

    print_banner()

    # 1. Load config
    config = load_config()

    # 2. Initialize all modules
    logger.info("Initializing modules...")

    from core.listener import Listener
    from core.transcriber import Transcriber
    from core.brain import Brain
    from core.speaker import Speaker

    listener = Listener(config)
    transcriber = Transcriber(config)
    brain = Brain(config)
    speaker = Speaker(config)

    logger.info("All modules initialized. Jarvis is ready.")
    speaker.speak_sync("Jarvis online. At your service, sir.")

    state = State.IDLE
    is_followup = False

    try:
        while True:
            # ── IDLE: Wait for wake word ──────────────────────────
            if state == State.IDLE:
                logger.info("State: IDLE — Listening for wake word...")
                detected = listener.listen_for_wake_word()

                if detected:
                    logger.info("Wake word detected!")
                    # Play instant local acknowledgement sound (zero cloud latency)
                    speaker.play_ack()
                    is_followup = False
                    state = State.LISTENING

            # ── LISTENING: Record user speech ─────────────────────
            elif state == State.LISTENING:
                logger.info(
                    "State: LISTENING — Recording speech%s...",
                    " (follow-up mode)" if is_followup else "",
                )
                # In follow-up mode, start fresh without pre-roll to avoid any residual audio
                audio_data = listener.record_until_silence(preroll_frames=0 if is_followup else 3)

                if audio_data is not None and len(audio_data) > 0:
                    state = State.PROCESSING
                else:
                    logger.info("No speech detected; returning to IDLE.")
                    is_followup = False
                    state = State.IDLE

            # ── PROCESSING: Transcribe + Brain Streaming ──────────
            elif state == State.PROCESSING:
                logger.info("State: PROCESSING — Transcribing audio...")
                start_time = time.time()

                # Speech-to-Text
                transcript = transcriber.transcribe(audio_data)
                stt_time = time.time() - start_time
                logger.info("Transcription (%.1fs): '%s'", stt_time, transcript)

                if not transcript.strip():
                    logger.info("Empty transcription, returning to IDLE.")
                    is_followup = False
                    state = State.IDLE
                    continue

                # Stream response from Brain
                logger.info("Sending to Brain for streaming inference...")
                stream_result = brain.think_stream(transcript)

                total_time = time.time() - start_time
                logger.info("Time to stream initiation: %.2fs", total_time)

                state = State.RESPONDING

            # ── RESPONDING: Stream-synthesize and speak response ──
            elif state == State.RESPONDING:
                logger.info("State: RESPONDING — Streaming playback...")
                speaker.speak_stream(stream_result)

                if not stream_result.full_text.strip():
                    logger.warning("Empty response received from stream. Speaking polite fallback.")
                    speaker.speak_sync("Apologies, sir. I experienced a momentary network delay.")

                logger.info("Response delivered: '%s'", stream_result.full_text[:120])


                # Autonomous Follow-Up Mode: check if user reply is expected
                if stream_result.needs_follow_up:
                    logger.info("Follow-up expected. Transitioning directly to LISTENING...")
                    is_followup = True
                    state = State.LISTENING
                else:
                    logger.info("No follow-up required. Returning to IDLE.")
                    is_followup = False
                    state = State.IDLE


    except KeyboardInterrupt:
        logger.info("\nShutdown requested (Ctrl+C).")
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
    finally:
        # Graceful cleanup
        logger.info("Cleaning up resources...")
        try:
            listener.cleanup()
        except Exception:
            pass
        try:
            speaker.cleanup()
        except Exception:
            pass
        logger.info("Jarvis shut down. Goodbye!")


if __name__ == "__main__":
    main()
