"""
Vosk Streaming ASR Module
==========================
Offline, low-latency streaming recognizer used for live vocalist-position
tracking. Unlike WhisperASR (which decodes fixed multi-second blocks),
Vosk consumes raw audio continuously and emits partial hypotheses every
~100-300ms plus a confirmed final result whenever it detects an utterance
boundary.

Partial results are intentionally forwarded to the alignment engine too —
they're noisier than finals but get the vocalist position moving faster,
which is the whole point of switching off chunked Whisper for the live
path. rapidfuzz matching in AlignmentEngine already tolerates ASR noise.
"""

import json
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("vosk")


class VoskStreamASR:
    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000,
        on_partial: Optional[Callable[[str, float], None]] = None,
        on_final: Optional[Callable[[str, float], None]] = None,
    ):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.on_partial = on_partial
        self.on_final = on_final

        self._model = None
        self._recognizer = None
        self._queue: "queue.Queue" = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_partial = ""

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self):
        """Load the Vosk model (blocking, call once at startup)."""
        logger.info(f"Loading Vosk model from '{self.model_path}' ...")
        start = time.time()
        try:
            import vosk
            vosk.SetLogLevel(-1)  # suppress Kaldi's own stderr logging
            self._model = vosk.Model(self.model_path)
            elapsed = time.time() - start
            logger.info(f"Vosk model loaded in {elapsed:.1f}s")
        except ImportError:
            logger.error("vosk not installed. Run: pip install vosk")
            raise
        except Exception as e:
            logger.error(
                f"Failed to load Vosk model from '{self.model_path}': {e}. "
                "Download a model from https://alphacephei.com/vosk/models "
                "and set vosk.model_path in config.yaml."
            )
            raise

    # ------------------------------------------------------------------
    # Streaming control
    # ------------------------------------------------------------------

    def start(self):
        """Create the recognizer and start the worker thread."""
        import vosk
        if self._model is None:
            raise RuntimeError("Model not loaded — call load_model() first")

        self._recognizer = vosk.KaldiRecognizer(self._model, self.sample_rate)
        self._recognizer.SetWords(False)
        self._last_partial = ""

        self._stop.clear()
        self._worker_thread = threading.Thread(
            target=self._worker, daemon=True, name="Vosk-Stream"
        )
        self._worker_thread.start()
        logger.info("Vosk streaming worker started")

    def stop(self):
        """Stop the worker thread."""
        self._stop.set()
        self._queue.put(None)  # sentinel
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        self._recognizer = None

    def feed(self, audio: np.ndarray, timestamp: float):
        """
        Called from the audio callback thread — must stay cheap.
        Queues a raw float32 mono block for the worker to process.
        """
        self._queue.put((audio, timestamp))

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _worker(self):
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:
                break
            audio, timestamp = item
            self._process(audio, timestamp)

    def _process(self, audio: np.ndarray, timestamp: float):
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        try:
            if self._recognizer.AcceptWaveform(pcm16):
                result = json.loads(self._recognizer.Result())
                text = result.get("text", "").strip()
                self._last_partial = ""
                if text and self.on_final:
                    logger.debug(f"Vosk final → '{text}'")
                    self.on_final(text, timestamp)
            else:
                result = json.loads(self._recognizer.PartialResult())
                partial = result.get("partial", "").strip()
                if partial and partial != self._last_partial:
                    self._last_partial = partial
                    if self.on_partial:
                        self.on_partial(partial, timestamp)
        except Exception as e:
            logger.error(f"Vosk processing error: {e}")
