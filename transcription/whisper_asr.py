"""
Whisper ASR Module
===================
Wraps faster-whisper for report mode only: accumulates audio for an entire
song and transcribes the full recording at once when the song ends, for
higher accuracy than any live-streaming recognizer can offer. Live-mode
vocalist tracking is handled separately by transcription/vosk_asr.py, which
can produce partial results fast enough to be useful in real time.
"""

import threading
import time
from typing import List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("whisper")


class WhisperASR:
    def __init__(
        self,
        model_size: str = "small",
        language: Optional[str] = "en",
        device: str = "cpu",
        compute_type: str = "int8",
        sample_rate: int = 16000,
    ):
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.sample_rate = sample_rate

        self._model = None
        self._model_lock = threading.Lock()

        # Report mode: accumulate full song audio
        self._recording: List[np.ndarray] = []
        self._recording_started_at: Optional[float] = None
        self._recording_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self):
        """Load the Whisper model (blocking, call once at startup)."""
        logger.info(f"Loading faster-whisper model '{self.model_size}' on {self.device} ({self.compute_type}) ...")
        start = time.time()
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            elapsed = time.time() - start
            logger.info(f"Whisper model loaded in {elapsed:.1f}s")
        except ImportError:
            logger.error("faster-whisper not installed. Run: pip install faster-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    # ------------------------------------------------------------------
    # Report mode
    # ------------------------------------------------------------------

    def start_recording(self):
        """Begin accumulating audio for report-mode transcription."""
        with self._recording_lock:
            self._recording = []
            self._recording_started_at = None
        logger.debug("Report recording started")

    def accumulate(self, audio: np.ndarray, timestamp: float):
        """
        Called by AudioCapture (on_chunk) — append audio to the current song's
        recording. `timestamp` is when this chunk was flushed (its end); on the
        first chunk after a reset this derives the recording's true start time
        from the chunk's own length, rather than relying on wall-clock time at
        whenever start_recording() happened to be called (which could be off by
        up to one chunk_duration and throw off report-mode timestamp alignment).
        """
        with self._recording_lock:
            if not self._recording:
                self._recording_started_at = timestamp - (len(audio) / self.sample_rate)
            self._recording.append(audio.copy())

    def pop_recording(self) -> Optional[tuple]:
        """
        Grab and clear the accumulated audio (fast, thread-safe) — call this
        synchronously as soon as the song ends, before the next song's
        start_recording() resets the buffer. Returns (audio, recording_started_at)
        or None if nothing was recorded. The actual transcription (slow) is
        meant to happen afterward, off this call.
        """
        with self._recording_lock:
            if not self._recording:
                return None
            full_audio = np.concatenate(self._recording)
            started_at = self._recording_started_at
            self._recording = []
            self._recording_started_at = None
        return full_audio, started_at

    def transcribe_segments(self, audio: np.ndarray) -> List[tuple]:
        """
        Transcribe a full recording and return per-segment results as
        [(text, start_offset_sec, end_offset_sec), ...], offsets relative to
        the start of `audio`. Used for post-hoc, timestamp-aware alignment
        (report mode) rather than a single joined string.
        """
        duration = len(audio) / self.sample_rate
        logger.info(f"Report-mode transcription: {duration:.1f}s of audio")

        if self._model is None:
            logger.error("Model not loaded — call load_model() first")
            return []

        audio = audio.astype(np.float32)
        if audio.max() > 1.0:
            audio = audio / 32768.0

        if duration < 0.5:
            return []

        try:
            with self._model_lock:
                segments, info = self._model.transcribe(
                    audio,
                    language=self.language,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
                    word_timestamps=False,
                )
                results = [
                    (seg.text.strip(), seg.start, seg.end)
                    for seg in segments
                    if seg.text.strip()
                ]
            logger.debug(f"Report transcription: {len(results)} segment(s)")
            return results
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return []

    def transcribe_words(
        self,
        audio: np.ndarray,
        initial_prompt: Optional[str] = None,
        vad_filter: bool = False,
    ) -> List[tuple]:
        """
        Transcribe a recording and return word-level results as
        [(word, start_sec, end_sec, probability), ...], offsets relative to the start of `audio`.

        `initial_prompt` (the song's lyrics) biases Whisper towards the right
        words, which helps a lot on sung vocals. VAD is off by default because
        speech-detection filters tend to cut out singing.
        """
        duration = len(audio) / self.sample_rate
        logger.info(f"Transcribing {duration:.1f}s of audio (word timestamps) ...")

        if self._model is None:
            logger.error("Model not loaded — call load_model() first")
            return []
        if duration < 0.5:
            return []

        audio = audio.astype(np.float32)
        try:
            with self._model_lock:
                segments, _info = self._model.transcribe(
                    audio,
                    language=self.language,
                    beam_size=5,
                    vad_filter=vad_filter,
                    word_timestamps=True,
                    condition_on_previous_text=False,
                    initial_prompt=initial_prompt or None,
                    # Whisper discards a whole 30 s window it judges "probably no speech". Singing over a band
                    # often looks like that, which silently deleted entire verses (a 30 s hole in the transcript,
                    # slides reported "not heard"). Never skip: junk over real silence is filtered later by
                    # word confidence, but lost singing can't be recovered.
                    no_speech_threshold=None,
                )
                words = []
                for seg in segments:
                    for w in (seg.words or []):
                        text = w.word.strip()
                        if text:
                            prob = getattr(w, "probability", None)
                            words.append((text, float(w.start), float(w.end),
                                          1.0 if prob is None else round(float(prob), 3)))
            logger.info(f"Heard {len(words)} word(s)")
            return words
        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return []


def segments_to_text(segments: List[tuple]) -> str:
    """Join transcribe_segments() output back into a single readable string."""
    return " ".join(text for text, _start, _end in segments).strip()
