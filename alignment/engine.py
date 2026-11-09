"""
Alignment Engine
=================
Compares vocalist position (from ASR) against slide position (from ProPresenter)
and generates lag events when slides are behind what is being sung.

Approach:
  1. When a new song loads, store all slide texts as an ordered list.
  2. `slide_position`   = index of the slide currently displayed (from Pro7 WS)
  3. `vocalist_position`= index of the slide the vocalist appears to be singing
                          (fuzzy-matched from ASR transcription against known lyrics)
  4. lag = vocalist_position - slide_position
  5. If lag >= threshold → emit a LagEvent

Fuzzy matching uses rapidfuzz token_set_ratio which handles:
  - Word order variations
  - Partial lyric snippets (vocalist may sing only part of a slide)
  - Minor ASR errors (misheard words)
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from rapidfuzz import fuzz

from propresenter.client import Presentation, Slide
from utils.logger import get_logger

logger = get_logger("alignment")

# Minimum fuzzy match score (0-100) to accept a lyric match
MIN_MATCH_SCORE = 45


@dataclass
class LagEvent:
    timestamp: float
    song_name: str
    slide_index: int
    slide_text: str
    vocalist_index: int
    vocalist_matched_text: str
    lag_phrases: int            # how many phrases ahead vocalist is
    lag_seconds: float          # how long this lag has been active
    asr_text: str               # raw transcription that triggered this

    def summary(self) -> str:
        lag_str = f"{self.lag_phrases} phrase(s) ahead, ~{self.lag_seconds:.1f}s"
        return (
            f"[LAG] '{self.song_name}' | "
            f"Slide {self.slide_index + 1}: '{self.slide_text[:40]}' | "
            f"Vocalist at slide {self.vocalist_index + 1} | {lag_str}"
        )


class AlignmentEngine:
    def __init__(
        self,
        lag_threshold_phrases: int = 1,
        lag_threshold_seconds: float = 4.0,
        lookback_window: int = 3,
        on_lag_event: Optional[Callable[[LagEvent], None]] = None,
    ):
        self.lag_threshold_phrases = lag_threshold_phrases
        self.lag_threshold_seconds = lag_threshold_seconds
        self.lookback_window = lookback_window
        self.on_lag_event = on_lag_event

        self._presentation: Optional[Presentation] = None
        self._slide_position: int = 0
        self._vocalist_position: int = 0

        # Track when the lag started (to calculate lag_seconds)
        self._lag_started_at: Optional[float] = None
        self._lag_active: bool = False

        # Events collected this song
        self._lag_events: List[LagEvent] = []
        # All events this session
        self._session_events: List[LagEvent] = []

    # ------------------------------------------------------------------
    # State updates (called by session manager)
    # ------------------------------------------------------------------

    def set_presentation(self, presentation: Presentation):
        """Called when Pro7 loads a new song."""
        self._presentation = presentation
        self._slide_position = 0
        self._vocalist_position = 0
        self._lag_started_at = None
        self._lag_active = False
        self._lag_events = []
        logger.info(f"Alignment engine: loaded '{presentation.name}' ({len(presentation.slides)} slides)")

    def update_slide_position(self, index: int, timestamp: Optional[float] = None):
        """
        Called when Pro7 advances to a new slide.
        `timestamp` defaults to now (live use); pass an explicit value when
        replaying recorded history (report mode) so lag_seconds is computed
        against the original event times, not whenever the replay happens to run.
        """
        self._slide_position = index
        logger.debug(f"Slide position → {index}")
        self._evaluate(timestamp if timestamp is not None else time.time())

    def update_from_transcription(self, text: str, timestamp: float):
        """
        Called when ASR produces new transcribed text.
        Attempts to locate the transcribed text within the known song lyrics
        and updates vocalist_position accordingly.
        """
        if not self._presentation or not self._presentation.slides:
            return
        if not text.strip():
            return

        new_pos = self._find_vocalist_position(text)
        if new_pos is not None and new_pos != self._vocalist_position:
            logger.debug(f"Vocalist position → slide {new_pos} (was {self._vocalist_position})")
            self._vocalist_position = new_pos

        self._evaluate_with_asr(text, timestamp)

    # ------------------------------------------------------------------
    # Matching logic
    # ------------------------------------------------------------------

    def _find_vocalist_position(self, asr_text: str) -> Optional[int]:
        """
        Find the slide index in the current presentation that best matches
        the ASR transcription.

        Search window: start from current vocalist position, look ahead
        (singers don't go backward) and slightly behind (ASR repeat/echo).
        """
        slides = self._presentation.slides
        n = len(slides)
        if n == 0:
            return None

        # Search window: don't jump too far backward, allow jumping ahead
        window_start = max(0, self._vocalist_position - self.lookback_window)
        window_end = min(n, self._vocalist_position + n)  # can advance to end of song

        best_score = 0
        best_idx = None

        asr_lower = asr_text.lower()

        for i in range(window_start, window_end):
            slide_text = slides[i].text.lower()
            if not slide_text:
                continue
            score = fuzz.token_set_ratio(asr_lower, slide_text)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= MIN_MATCH_SCORE:
            logger.debug(f"ASR match: slide {best_idx} score={best_score:.0f} '{asr_text[:40]}'")
            return best_idx
        else:
            logger.debug(f"ASR no match (best score={best_score:.0f}): '{asr_text[:40]}'")
            return None

    # ------------------------------------------------------------------
    # Lag evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, timestamp: float):
        """Evaluate lag based on current positions (no ASR text available)."""
        self._check_lag("", timestamp)

    def _evaluate_with_asr(self, asr_text: str, timestamp: float):
        """Evaluate lag after an ASR update."""
        self._check_lag(asr_text, timestamp)

    def _check_lag(self, asr_text: str, timestamp: float):
        if not self._presentation:
            return

        lag = self._vocalist_position - self._slide_position

        if lag >= self.lag_threshold_phrases:
            if not self._lag_active:
                self._lag_active = True
                self._lag_started_at = timestamp
                logger.debug(f"Lag started: vocalist={self._vocalist_position} slide={self._slide_position}")

            lag_seconds = timestamp - (self._lag_started_at or timestamp)

            if lag_seconds >= self.lag_threshold_seconds or lag > self.lag_threshold_phrases:
                slides = self._presentation.slides
                slide_text = slides[self._slide_position].text if self._slide_position < len(slides) else ""
                vocalist_text = slides[self._vocalist_position].text if self._vocalist_position < len(slides) else ""

                event = LagEvent(
                    timestamp=timestamp,
                    song_name=self._presentation.name,
                    slide_index=self._slide_position,
                    slide_text=slide_text,
                    vocalist_index=self._vocalist_position,
                    vocalist_matched_text=vocalist_text,
                    lag_phrases=lag,
                    lag_seconds=lag_seconds,
                    asr_text=asr_text,
                )

                # Deduplicate: don't fire the same event twice in quick succession
                if not self._lag_events or self._lag_events[-1].slide_index != self._slide_position:
                    self._lag_events.append(event)
                    self._session_events.append(event)
                    logger.warning(f"  *** {event.summary()}")
                    if self.on_lag_event:
                        self.on_lag_event(event)
        else:
            if self._lag_active:
                logger.debug(f"Lag resolved at slide {self._slide_position}")
            self._lag_active = False
            self._lag_started_at = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_song_events(self) -> List[LagEvent]:
        return list(self._lag_events)

    def get_session_events(self) -> List[LagEvent]:
        return list(self._session_events)

    def get_current_lag(self) -> int:
        return max(0, self._vocalist_position - self._slide_position)

    def get_vocalist_position(self) -> int:
        return self._vocalist_position

    def get_presentation(self) -> Optional[Presentation]:
        return self._presentation

    def get_slide_text(self, index: int) -> str:
        """Safe bounds-checked lookup of a slide's text in the current presentation."""
        if not self._presentation:
            return ""
        slides = self._presentation.slides
        return slides[index].text if 0 <= index < len(slides) else ""


# ──────────────────────────────────────────────────────────────────────
# Post-hoc replay (report mode)
# ──────────────────────────────────────────────────────────────────────

def replay_song(
    presentation: Presentation,
    slide_history: List[tuple],
    segments: List[tuple],
    recording_start: float,
    lag_threshold_phrases: int = 1,
    lag_threshold_seconds: float = 4.0,
    lookback_window: int = 3,
) -> List[LagEvent]:
    """
    Reconstruct lag events for a finished song from recorded history, using
    a fresh AlignmentEngine so the exact same fuzzy-matching and lag-timing
    logic as the live path applies — just driven by recorded timestamps
    instead of real-time events.

    `slide_history`: chronological [(timestamp, slide_index), ...] recorded
    while the song was live.
    `segments`: Whisper output as [(text, start_offset_sec, end_offset_sec), ...],
    offsets relative to when recording started.
    `recording_start`: wall-clock time (time.time()) when recording began —
    used to convert segment offsets back to absolute timestamps so they line
    up with slide_history's timestamps.
    """
    events: List[LagEvent] = []
    engine = AlignmentEngine(
        lag_threshold_phrases=lag_threshold_phrases,
        lag_threshold_seconds=lag_threshold_seconds,
        lookback_window=lookback_window,
        on_lag_event=events.append,
    )
    engine.set_presentation(presentation)

    timeline = [(ts, "slide", idx) for ts, idx in slide_history]
    timeline += [(recording_start + start, "asr", text) for text, start, _end in segments]
    timeline.sort(key=lambda e: e[0])

    for ts, kind, payload in timeline:
        if kind == "slide":
            engine.update_slide_position(payload, timestamp=ts)
        else:
            engine.update_from_transcription(payload, ts)

    return events
