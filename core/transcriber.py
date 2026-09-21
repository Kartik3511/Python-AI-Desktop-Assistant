"""Speech-to-text transcriber module using faster-whisper."""

import logging
import time
from typing import Any, Dict, Optional

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class Transcriber:
    """Handles speech-to-text transcription using faster-whisper."""

    def __init__(self, config: dict) -> None:
        """Initialize the WhisperModel with settings from configuration.

        Args:
            config: Configuration dictionary containing transcriber settings.
                    Expected keys under 'transcriber':
                    - model_size (default: 'base.en')
                    - device (default: 'cpu')
                    - compute_type (default: 'int8')
        """
        transcriber_cfg = config.get("transcriber", config) if isinstance(config, dict) else {}
        if not isinstance(transcriber_cfg, dict):
            transcriber_cfg = {}

        self.model_size: str = transcriber_cfg.get("model_size") or "base.en"
        self.device: str = transcriber_cfg.get("device") or "cpu"
        self.compute_type: str = transcriber_cfg.get("compute_type") or "int8"

        logger.info(
            "Loading Whisper model '%s' (device: %s, compute_type: %s)...",
            self.model_size,
            self.device,
            self.compute_type,
        )

        start_time = time.perf_counter()
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            load_time = time.perf_counter() - start_time
            logger.info(
                "Whisper model '%s' loaded successfully in %.2f seconds.",
                self.model_size,
                load_time,
            )
        except Exception as e:
            logger.error(
                "Failed to load Whisper model '%s': %s",
                self.model_size,
                e,
                exc_info=True,
            )
            raise

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a numpy int16 audio array to text.

        Args:
            audio: Numpy int16 audio array.

        Returns:
            Transcribed text with whitespace stripped, or empty string on failure.
        """
        if audio is None or audio.size == 0:
            logger.warning("Empty or None audio array provided for transcription.")
            return ""

        try:
            # Convert int16 audio array to float32 normalized to [-1, 1]
            if audio.dtype == np.int16:
                audio_float = audio.astype(np.float32) / 32768.0
            elif np.issubdtype(audio.dtype, np.floating):
                audio_float = audio.astype(np.float32)
                max_val = np.max(np.abs(audio_float)) if audio_float.size > 0 else 0.0
                if max_val > 1.0:
                    audio_float = audio_float / 32768.0
            else:
                audio_float = audio.astype(np.float32) / 32768.0

            # Ensure mono 1D contiguous array for WhisperModel
            if audio_float.ndim > 1:
                audio_float = np.mean(audio_float, axis=-1)
            audio_float = np.ascontiguousarray(audio_float.flatten(), dtype=np.float32)

            start_time = time.perf_counter()
            whisper_prompt = (
                "JARVIS desktop assistant. Windows commands, apps, Edge, YouTube, "
                "GTA 6, Minecraft, Discord, Spotify, GitHub, volume, brightness."
            )
            segments, _ = self.model.transcribe(
                audio_float,
                beam_size=5,
                initial_prompt=whisper_prompt,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                no_speech_threshold=0.6,
            )

            # Extract text from all segments and concatenate
            text_segments = [segment.text.strip() for segment in segments]
            text = " ".join(seg for seg in text_segments if seg).strip()

            elapsed_time = time.perf_counter() - start_time

            if not text:
                logger.debug(
                    "Transcription completed in %.2fs: [No speech detected]",
                    elapsed_time,
                )
                return ""

            # Filter known Whisper silence/background noise hallucinations
            clean_check = text.lower().strip(" .!?,")
            silence_hallucinations = {
                "thank you",
                "thank you very much",
                "thanks for watching",
                "very easy",
                "very easy thank you very much",
                "you",
                "bye",
                "goodbye",
            }
            if clean_check in silence_hallucinations:
                logger.info("Filtered silence hallucination: '%s' (treating as silence)", text)
                return ""

            logger.info(
                "Transcription completed in %.2fs: '%s'",
                elapsed_time,
                text,
            )
            return text

        except Exception as e:
            logger.error("Error during transcription: %s", e, exc_info=True)
            return ""
