"""
JARVIS Assistant - Listener Module
Handles continuous microphone audio capture, wake word detection,
and voice activity detection (VAD) for dynamic end-of-speech.
"""

from collections import deque
import logging
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
from openwakeword.model import Model
import pyaudio
import webrtcvad

logger = logging.getLogger(__name__)

# Supported audio format mappings from config string to PyAudio format constants
AUDIO_FORMAT_MAP: Dict[str, int] = {
    "int16": pyaudio.paInt16,
    "int32": pyaudio.paInt32,
    "float32": pyaudio.paFloat32,
    "int8": pyaudio.paInt8,
    "uint8": pyaudio.paUInt8,
}


class Listener:
    """Handles continuous audio capture, wake word detection, and dynamic speech recording."""

    def __init__(self, config: dict) -> None:
        """Initialize the Listener instance with configuration settings.

        Sets up audio stream parameters, initializes PyAudio, loads the openWakeWord
        model, and initializes WebRTC VAD.

        Args:
            config: Full application configuration dictionary containing:
                    - 'listener': sample_rate, channels, chunk_size, format
                    - 'wake_word': model_name, threshold
                    - 'vad': aggressiveness, silence_duration, max_recording_duration
        """
        if not isinstance(config, dict):
            config = {}

        # --------------------------------------------------------
        # 1. Parse configuration sections with defaults
        # --------------------------------------------------------
        listener_cfg: Dict[str, Any] = config.get("listener", {})
        if not isinstance(listener_cfg, dict):
            listener_cfg = {}

        wake_word_cfg: Dict[str, Any] = config.get("wake_word", {})
        if not isinstance(wake_word_cfg, dict):
            wake_word_cfg = {}

        vad_cfg: Dict[str, Any] = config.get("vad", {})
        if not isinstance(vad_cfg, dict):
            vad_cfg = {}

        # Audio stream settings (default: 16kHz, mono, 1280 samples / 80ms, int16)
        self.sample_rate: int = int(listener_cfg.get("sample_rate", 16000))
        self.channels: int = int(listener_cfg.get("channels", 1))
        self.chunk_size: int = int(listener_cfg.get("chunk_size", 1280))
        format_name: str = str(listener_cfg.get("format", "int16")).lower()
        self.audio_format: int = AUDIO_FORMAT_MAP.get(format_name, pyaudio.paInt16)

        # Duration of a single chunk in seconds (e.g. 1280 / 16000 = 0.08s / 80ms)
        self.frame_duration: float = self.chunk_size / self.sample_rate

        # Wake word settings (default: model 'hey_jarvis', threshold 0.6)
        self.wake_word_model: str = str(
            wake_word_cfg.get("model_name") or wake_word_cfg.get("model_path") or "hey_jarvis"
        )
        self.wake_word_threshold: float = float(wake_word_cfg.get("threshold", 0.6))

        # Voice Activity Detection (VAD) settings
        self.vad_aggressiveness: int = max(0, min(3, int(vad_cfg.get("aggressiveness", 3))))
        self.silence_duration: float = float(vad_cfg.get("silence_duration", 1.5))
        self.max_recording_duration: float = float(vad_cfg.get("max_recording_duration", 30.0))

        # Ring buffer for recent audio chunks (~2.0 seconds history = 25 chunks of 80ms)
        ring_buffer_chunks: int = int(listener_cfg.get("ring_buffer_size", 25))
        self.ring_buffer: deque = deque(maxlen=ring_buffer_chunks)

        # Internal state
        self.audio: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None
        self._is_running: bool = False

        # --------------------------------------------------------
        # 2. Initialize WebRTC VAD
        # --------------------------------------------------------
        try:
            self.vad = webrtcvad.Vad(self.vad_aggressiveness)
            logger.info("WebRTC VAD initialized with aggressiveness level %d.", self.vad_aggressiveness)
        except Exception as e:
            logger.error("Failed to initialize WebRTC VAD: %s", e, exc_info=True)
            raise

        # --------------------------------------------------------
        # 3. Load openWakeWord model
        # --------------------------------------------------------
        logger.info(
            "Loading openWakeWord model '%s' (threshold: %.2f)...",
            self.wake_word_model,
            self.wake_word_threshold,
        )
        try:
            start_load = time.perf_counter()
            self.oww_model = Model(
                wakeword_models=[self.wake_word_model],
                inference_framework="onnx",
            )
            elapsed = time.perf_counter() - start_load
            logger.info("openWakeWord model loaded successfully in %.2f seconds.", elapsed)
        except Exception as e:
            logger.error(
                "Failed to load openWakeWord model '%s': %s",
                self.wake_word_model,
                e,
                exc_info=True,
            )
            raise

        # --------------------------------------------------------
        # 4. Initialize PyAudio input stream
        # --------------------------------------------------------
        self._init_stream()

    def _init_stream(self) -> None:
        """Initialize or re-initialize the PyAudio microphone input stream."""
        try:
            if self.stream is not None:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

            if self.audio is None:
                self.audio = pyaudio.PyAudio()

            logger.info(
                "Opening PyAudio stream (rate=%d, channels=%d, chunk=%d, format=%s)...",
                self.sample_rate,
                self.channels,
                self.chunk_size,
                self.audio_format,
            )
            self.stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
            self._is_running = True
            logger.info("PyAudio microphone stream opened successfully.")
        except Exception as e:
            logger.error("Failed to initialize PyAudio stream: %s", e, exc_info=True)
            self.cleanup()
            raise

    def _read_chunk(self) -> bytes:
        """Read a single raw audio chunk from the microphone stream.

        Returns:
            Raw audio byte string of length chunk_size * sample_width.

        Raises:
            RuntimeError: If the audio stream is closed or uninitialized.
            IOError: If reading from the audio stream fails.
        """
        if self.stream is None or not self._is_running:
            raise RuntimeError("Audio stream is not running or has been closed.")

        try:
            return self.stream.read(self.chunk_size, exception_on_overflow=False)
        except IOError as e:
            # Underflow/overflow or hardware hiccup
            logger.warning("PyAudio stream read IOError: %s", e)
            raise
        except Exception as e:
            logger.error("Unexpected error reading from PyAudio stream: %s", e)
            raise

    def _is_speech(self, chunk: bytes) -> bool:
        """Determine if an audio chunk contains active speech using WebRTC VAD.

        WebRTC VAD strictly accepts frames of 10, 20, or 30 ms (320, 640, or 960 bytes
        for 16kHz 16-bit mono). Since each microphone chunk is 80 ms (2560 bytes),
        the chunk is divided into 20 ms sub-frames (640 bytes each). If any sub-frame
        contains speech, the whole chunk is considered active speech.

        Args:
            chunk: Raw 16-bit PCM mono audio bytes.

        Returns:
            True if voice activity is detected, False otherwise.
        """
        # 20ms sub-frame size in bytes at sample_rate, 16-bit mono (2 bytes per sample)
        subframe_bytes = int(self.sample_rate * 0.02 * 2)

        # Check if the chunk is already a valid standalone WebRTC VAD frame size
        valid_single_sizes = {
            int(self.sample_rate * d * 2) for d in (0.01, 0.02, 0.03)
        }
        if len(chunk) in valid_single_sizes:
            try:
                return bool(self.vad.is_speech(chunk, self.sample_rate))
            except Exception as e:
                logger.debug("Direct VAD check error: %s", e)
                return False

        # Split the chunk into 20ms sub-frames and evaluate each
        if len(chunk) >= subframe_bytes and len(chunk) % subframe_bytes == 0:
            try:
                for offset in range(0, len(chunk), subframe_bytes):
                    subframe = chunk[offset : offset + subframe_bytes]
                    if self.vad.is_speech(subframe, self.sample_rate):
                        return True
                return False
            except Exception as e:
                logger.debug("Sub-frame VAD check error: %s", e)

        # Fallback to direct call (handles mocked VAD in unit tests or irregular chunk sizes)
        try:
            return bool(self.vad.is_speech(chunk, self.sample_rate))
        except Exception as e:
            logger.debug("Fallback VAD check error: %s", e)
            return False

    def listen_for_wake_word(self) -> bool:
        """Continuously capture microphone audio and listen for the wake word.

        Blocks and reads audio chunks, stores them in the ring buffer, and feeds
        them to the openWakeWord model. When wake word score meets or exceeds
        the configured threshold, logs detection and returns True.

        Returns:
            True when the wake word is detected above threshold.
        """
        logger.info(
            "Listening for wake word '%s' (threshold: %.2f)...",
            self.wake_word_model,
            self.wake_word_threshold,
        )

        # Flush residual audio from ring buffer, reset model state, and drain hardware buffer
        self.ring_buffer.clear()
        try:
            self.oww_model.reset()
        except Exception as e:
            logger.debug("Error resetting model on listen start: %s", e)

        if self.stream is not None and self.stream.is_active():
            try:
                available = self.stream.get_read_available()
                if available > 0:
                    self.stream.read(available, exception_on_overflow=False)
            except Exception as e:
                logger.debug("Error draining audio stream: %s", e)

        while self._is_running:
            try:
                chunk = self._read_chunk()
            except IOError:
                # Sleep briefly on I/O overflow to allow hardware buffer recovery
                time.sleep(0.01)
                continue
            except Exception as e:
                logger.error("Audio capture failed during wake word listening: %s", e)
                break

            # Store in ring buffer
            self.ring_buffer.append(chunk)

            # Convert raw bytes to 16-bit signed numpy array for openWakeWord
            pcm_data = np.frombuffer(chunk, dtype=np.int16)

            try:
                prediction: Dict[str, float] = self.oww_model.predict(pcm_data)
            except Exception as e:
                logger.warning("Error running openWakeWord model inference: %s", e)
                continue

            # Extract model prediction score
            score: float = 0.0
            if self.wake_word_model in prediction:
                score = float(prediction[self.wake_word_model])
            else:
                # Match partial or versioned model names (e.g. 'hey_jarvis_v0.1')
                for model_key, val in prediction.items():
                    if self.wake_word_model.lower() in model_key.lower():
                        score = float(val)
                        break

            # Check if threshold is reached
            if score >= self.wake_word_threshold:
                logger.info(
                    "Wake word '%s' detected! (Score: %.3f >= Threshold: %.2f)",
                    self.wake_word_model,
                    score,
                    self.wake_word_threshold,
                )
                # Reset model internal activation state after detection
                try:
                    self.oww_model.reset()
                except Exception as e:
                    logger.debug("Error resetting openWakeWord model state: %s", e)
                return True

        return False

    def record_until_silence(self, preroll_frames: int = 0) -> np.ndarray:
        """Record audio until silence is detected via WebRTC VAD or timeout.

        After wake word detection, captures audio frames and evaluates voice activity.
        Stops recording when consecutive silent frames equal or exceed silence_duration
        seconds, or when max_recording_duration seconds is reached.

        Args:
            preroll_frames: Optional number of historical chunks from the ring buffer
                            to prepend to the recording (default 0).

        Returns:
            The full recorded audio as a 1-D numpy int16 array.
        """
        # Calculate thresholds in terms of frame counts
        max_silent_frames: int = max(1, int(math.ceil(self.silence_duration / self.frame_duration)))
        max_total_frames: int = max(1, int(math.ceil(self.max_recording_duration / self.frame_duration)))

        logger.info(
            "Recording started (dynamic VAD: silence_duration=%.1fs [%d frames], max_duration=%.1fs [%d frames])...",
            self.silence_duration,
            max_silent_frames,
            self.max_recording_duration,
            max_total_frames,
        )

        # Drain any residual audio samples sitting in hardware buffer (e.g. from speaker playback)
        if self.stream is not None and self.stream.is_active():
            try:
                available = self.stream.get_read_available()
                if available > 0:
                    self.stream.read(available, exception_on_overflow=False)
            except Exception as e:
                logger.debug("Error draining audio stream: %s", e)

        recorded_frames: List[bytes] = []

        # Prepend requested preroll frames from the ring buffer
        if preroll_frames > 0 and len(self.ring_buffer) > 0:
            history = list(self.ring_buffer)
            preroll = history[-preroll_frames:]
            recorded_frames.extend(preroll)

            logger.debug("Prepended %d pre-roll frames to recording.", len(preroll))

        consecutive_silent_frames: int = 0
        total_frames: int = 0
        speech_frames_count: int = 0
        speech_detected_once: bool = False
        start_time = time.perf_counter()

        initial_wait_frames: int = max(1, int(math.ceil(4.0 / self.frame_duration)))

        while self._is_running and total_frames < max_total_frames:
            try:
                chunk = self._read_chunk()
            except IOError:
                time.sleep(0.01)
                continue
            except Exception as e:
                logger.error("Audio capture failed during silence recording: %s", e)
                break

            # Update ring buffer and recording accumulator
            self.ring_buffer.append(chunk)
            recorded_frames.append(chunk)
            total_frames += 1

            # Check for speech activity in current 80ms chunk
            has_speech = self._is_speech(chunk)

            if has_speech:
                consecutive_silent_frames = 0
                speech_frames_count += 1
                # Require at least 2 speech chunks (~160ms) to confirm genuine human speech
                if speech_frames_count >= 2:
                    speech_detected_once = True
            else:
                consecutive_silent_frames += 1

            # Check termination conditions:
            if not speech_detected_once:
                # Before user starts speaking: give up to 4.0s to begin
                if consecutive_silent_frames >= initial_wait_frames:
                    logger.debug("No speech detected within initial wait window (4.0s).")
                    break
            else:
                # After user has started speaking: stop after silence_duration of silence
                if consecutive_silent_frames >= max_silent_frames:
                    logger.debug(
                        "Silence detected for %.2f seconds (%d consecutive silent frames).",
                        consecutive_silent_frames * self.frame_duration,
                        consecutive_silent_frames,
                    )
                    break

        elapsed_time = time.perf_counter() - start_time

        # Convert recorded frames to numpy int16 array (return empty if no speech occurred)
        if recorded_frames and speech_detected_once:
            full_audio = np.frombuffer(b"".join(recorded_frames), dtype=np.int16)
            # Energy gate: check RMS energy to verify human voice vs background room noise
            if len(full_audio) > 0:
                rms = float(np.sqrt(np.mean(full_audio.astype(np.float32) ** 2)))
                if rms < 180.0:
                    logger.info(
                        "Recorded audio below voice energy floor (RMS: %.1f < 180.0). Discarding as silence/noise.",
                        rms,
                    )
                    full_audio = np.empty(0, dtype=np.int16)
                    speech_detected_once = False
        else:
            full_audio = np.empty(0, dtype=np.int16)

        audio_duration = len(full_audio) / self.sample_rate if self.sample_rate > 0 else 0.0

        logger.info(
            "Recording stopped. Total duration: %.2f seconds (%d samples, speech_detected=%s, elapsed=%.2fs).",
            audio_duration,
            len(full_audio),
            speech_detected_once,
            elapsed_time,
        )

        return full_audio

    def cleanup(self) -> None:
        """Stop and close the PyAudio input stream and terminate PyAudio."""
        logger.info("Cleaning up Listener resources...")
        self._is_running = False

        if self.stream is not None:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
                logger.debug("PyAudio stream stopped and closed successfully.")
            except Exception as e:
                logger.warning("Error stopping/closing PyAudio stream: %s", e)
            finally:
                self.stream = None

        if self.audio is not None:
            try:
                self.audio.terminate()
                logger.debug("PyAudio terminated successfully.")
            except Exception as e:
                logger.warning("Error terminating PyAudio: %s", e)
            finally:
                self.audio = None

    def __enter__(self) -> "Listener":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit, ensures cleanup is called."""
        self.cleanup()

    def __del__(self) -> None:
        """Destructor to ensure cleanup if explicitly missed."""
        try:
            self.cleanup()
        except Exception:
            pass
