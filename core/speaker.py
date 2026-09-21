"""
JARVIS Assistant - Speaker Module
Handles Text-to-Speech (TTS) synthesis using edge-tts and audio playback using pygame.mixer.
"""

import asyncio
import concurrent.futures
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, Iterable, Optional

import edge_tts
import pygame


logger = logging.getLogger(__name__)


class Speaker:
    """
    Text-to-speech speaker engine for JARVIS.

    Synthesizes speech from text using Microsoft Edge TTS (edge-tts) and
    plays the generated audio through pygame.mixer.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the Speaker instance with configuration.

        Args:
            config: Application configuration dictionary containing the 'speaker' section.
        """
        speaker_config: Dict[str, Any] = (
            config.get("speaker", {}) if isinstance(config, dict) else {}
        )
        if not speaker_config and isinstance(config, dict):
            # In case the speaker section itself was passed directly
            speaker_config = config

        self.voice: str = speaker_config.get("voice", "en-GB-RyanNeural")
        self.rate: str = speaker_config.get("rate", "+0%")
        self.volume: str = speaker_config.get("volume", "+0%")

        # Create a dedicated temporary directory for audio files
        self.temp_dir: str = tempfile.mkdtemp(prefix="jarvis_tts_")
        logger.debug("Created temporary directory for TTS audio files: %s", self.temp_dir)

        # Initialize pygame.mixer for audio playback
        self._init_mixer()

        logger.info(
            "Speaker initialized (voice=%s, rate=%s, volume=%s)",
            self.voice,
            self.rate,
            self.volume,
        )

    def _init_mixer(self) -> None:
        """Initialize pygame.mixer safely if not already initialized."""
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
                logger.info("pygame.mixer initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize pygame.mixer: %s", e, exc_info=True)

    async def speak(self, text: str) -> None:
        """
        Synthesize text to speech and play it asynchronously.

        Blocks until playback is complete. Temporary audio files are cleaned up
        after playback.

        Args:
            text: The text to be spoken.
        """
        if not text or not text.strip():
            logger.debug("Empty or whitespace text received; skipping speech synthesis.")
            return

        # Ensure pygame.mixer is initialized
        if not pygame.mixer.get_init():
            logger.warning("pygame.mixer is not initialized. Attempting initialization...")
            self._init_mixer()
            if not pygame.mixer.get_init():
                logger.error("Cannot speak: pygame.mixer could not be initialized.")
                return

        # Ensure temp directory still exists
        if not os.path.exists(self.temp_dir):
            try:
                os.makedirs(self.temp_dir, exist_ok=True)
            except Exception as e:
                logger.error("Failed to create temporary directory %s: %s", self.temp_dir, e)
                return

        # Create unique temporary mp3 file path
        temp_filename = f"speech_{int(time.time() * 1000)}_{os.getpid()}.mp3"
        temp_file_path = os.path.join(self.temp_dir, temp_filename)

        try:
            logger.debug("Synthesizing audio for text: '%s...'", text[:50].replace("\n", " "))
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.voice,
                rate=self.rate,
                volume=self.volume,
            )
            await communicate.save(temp_file_path)

            if not os.path.exists(temp_file_path) or os.path.getsize(temp_file_path) == 0:
                logger.error("TTS synthesis produced an empty or missing audio file: %s", temp_file_path)
                return

            # Load and play audio via pygame.mixer.music
            pygame.mixer.music.load(temp_file_path)
            pygame.mixer.music.play()

            # Poll until playback finishes
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.05)

        except Exception as e:
            logger.error("Error during TTS synthesis or playback: %s", e, exc_info=True)
        finally:
            # Stop and unload music to release the file handle on Windows
            try:
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
                    if hasattr(pygame.mixer.music, "unload"):
                        pygame.mixer.music.unload()
            except Exception as unload_err:
                logger.debug("Error stopping/unloading pygame.mixer.music: %s", unload_err)

            # Clean up the temporary audio file
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug("Cleaned up temporary file: %s", temp_file_path)
                except Exception as cleanup_err:
                    logger.warning(
                        "Initial cleanup failed for %s (%s). Retrying after brief delay...",
                        temp_file_path,
                        cleanup_err,
                    )
                    try:
                        time.sleep(0.05)
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                            logger.debug("Cleaned up temporary file on retry: %s", temp_file_path)
                    except Exception as retry_err:
                        logger.warning(
                            "Failed to clean up temporary audio file %s: %s",
                            temp_file_path,
                            retry_err,
                        )

    def speak_sync(self, text: str) -> None:
        """
        Synchronously synthesize and speak text.

        Calls asyncio.run(self.speak(text)) or integrates with an existing event loop.

        Args:
            text: The text to be spoken.
        """
        if not text or not text.strip():
            logger.debug("Empty or whitespace text received; skipping speech synthesis.")
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # In an active event loop, dispatch to a worker thread to avoid blocking loop conflict
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.speak(text))
                future.result()
        else:
            try:
                asyncio.run(self.speak(text))
            except RuntimeError:
                # Handle cases where an inactive event loop exists in the thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    new_loop.run_until_complete(self.speak(text))
                finally:
                    new_loop.close()

    def speak_stream(self, sentence_generator: Iterable[str]) -> None:
        """
        Stream-synthesize and play sentences using a pipelined producer-consumer model.

        Sentence N+1 synthesizes concurrently in a background thread while Sentence N
        is playing through pygame.mixer, eliminating inter-sentence pauses.

        Args:
            sentence_generator: Iterable or generator yielding complete sentences.
        """
        self._init_mixer()
        if not pygame.mixer.get_init():
            logger.error("Cannot speak stream: pygame.mixer could not be initialized.")
            return

        audio_queue: queue.Queue = queue.Queue(maxsize=3)

        def producer():
            """Background worker thread: synthesizes sentences to temp mp3s."""
            try:
                for sentence in sentence_generator:
                    if not sentence or not sentence.strip():
                        continue
                    clean_sentence = sentence.strip()
                    temp_filename = f"stream_{int(time.time() * 1000)}_{os.getpid()}_{os.urandom(3).hex()}.mp3"
                    temp_path = os.path.join(self.temp_dir, temp_filename)

                    # Synthesize via edge_tts Communicate
                    communicate = edge_tts.Communicate(
                        text=clean_sentence,
                        voice=self.voice,
                        rate=self.rate,
                        volume=self.volume,
                    )
                    asyncio.run(communicate.save(temp_path))

                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                        audio_queue.put((temp_path, clean_sentence))
                    else:
                        logger.warning("Empty audio produced for sentence: '%s'", clean_sentence[:30])
            except Exception as e:
                logger.error("Error in TTS stream producer: %s", e, exc_info=True)
            finally:
                # Sentinel to signal end of stream
                audio_queue.put((None, None))

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            while True:
                try:
                    # Timeout to avoid hanging indefinitely if producer dies
                    temp_path, sent_text = audio_queue.get(timeout=30.0)
                except queue.Empty:
                    logger.warning("TTS audio queue timed out waiting for next sentence.")

                    break

                if temp_path is None:
                    # End of stream sentinel reached
                    break

                try:
                    logger.debug("Playing streamed sentence: '%s...'", sent_text[:40])
                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()

                    while pygame.mixer.music.get_busy():
                        time.sleep(0.02)

                except Exception as play_err:
                    logger.error("Error during playback of '%s': %s", temp_path, play_err)
                finally:
                    # Release file handle on Windows
                    try:
                        if pygame.mixer.get_init():
                            pygame.mixer.music.stop()
                            if hasattr(pygame.mixer.music, "unload"):
                                pygame.mixer.music.unload()
                    except Exception:
                        pass

                    # Clean up the played temp file
                    if os.path.exists(temp_path):
                        for _ in range(3):
                            try:
                                os.remove(temp_path)
                                break
                            except Exception:
                                time.sleep(0.05)

        finally:
            producer_thread.join(timeout=1.0)
            # Drain remaining items and delete unplayed temp files
            while not audio_queue.empty():
                try:
                    rem_path, _ = audio_queue.get_nowait()
                    if rem_path and os.path.exists(rem_path):
                        try:
                            os.remove(rem_path)
                        except Exception:
                            pass
                except Exception:
                    break

    def play_ack(self) -> None:

        """Play immediate local acknowledgment sound with zero cloud latency (~40ms)."""
        ack_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources", "yes.mp3")
        if os.path.exists(ack_path):
            try:
                self._init_mixer()
                pygame.mixer.music.load(ack_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.02)
                if hasattr(pygame.mixer.music, "unload"):
                    pygame.mixer.music.unload()
                return
            except Exception as e:
                logger.warning("Error playing local ack sound: %s", e)

        # Fallback to online TTS if local file is missing
        self.speak_sync("Yes, sir?")

    def cleanup(self) -> None:
        """
        Stop pygame.mixer and remove temporary audio directory and files.
        """
        logger.info("Cleaning up Speaker resources...")

        # Stop and quit pygame mixer
        try:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                if hasattr(pygame.mixer.music, "unload"):
                    pygame.mixer.music.unload()
                pygame.mixer.quit()
                logger.debug("pygame.mixer stopped and quit successfully.")
        except Exception as e:
            logger.warning("Error stopping pygame.mixer during cleanup: %s", e)

        # Remove temporary directory and remaining files
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.debug("Removed temporary directory: %s", self.temp_dir)
            except Exception as e:
                logger.warning("Error removing temporary directory %s: %s", self.temp_dir, e)

    def __enter__(self) -> "Speaker":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit, cleans up resources."""
        self.cleanup()

    def __del__(self) -> None:
        """Destructor to ensure cleanup if explicitly missed."""
        try:
            self.cleanup()
        except Exception:
            pass
