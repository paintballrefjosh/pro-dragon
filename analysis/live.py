"""
Live (in-song) delay estimates
===============================
While a song is being recorded, the final analysis can't run yet (it needs the whole song
transcribed). The LiveAnalyzer gives *provisional* delays in the meantime:

  * every few seconds the newest slice of audio is transcribed (with a few seconds of
    overlap before it, for context) — never the whole song again;
  * words that are settled (they end a couple of seconds before "now") are committed;
    words at the very edge of the window may still change, so they wait for the next pass;
  * the same word-matching analysis as the final report is re-run on the words committed
    so far, so each slide change gets an estimated delay once the singing around it is known.

Estimates can still move a little (more words may re-pair a slide) and slide changes whose
words haven't been transcribed yet are shown as "pending". The final analysis after the song
stops replaces all of this.
"""

from typing import Callable, List, Optional, Sequence

import numpy as np

from analysis.song_timing import SlideEvent, SlideTiming, Word, analyze_song, tokenize

SAMPLE_RATE = 16000
PENDING_NOTE = "waiting for the singing to be transcribed"


class LiveAnalyzer:
    def __init__(
        self,
        transcribe: Callable[[np.ndarray, float], Sequence[tuple]],
        overlap_s: float = 3.0,
        tail_guard_s: float = 2.0,
        min_new_s: float = 4.0,
        max_window_s: float = 40.0,
        pending_horizon_s: float = 8.0,
    ):
        """
        transcribe(window_audio, window_start_s) -> [(text, start, end, prob), ...] with times
        relative to the START OF THE WINDOW.
        """
        self.transcribe = transcribe
        self.overlap_s = overlap_s
        self.tail_guard_s = tail_guard_s
        self.min_new_s = min_new_s
        self.max_window_s = max_window_s
        self.pending_horizon_s = pending_horizon_s

        self.words: List[Word] = []        # committed words, seconds from the start of the recorded audio
        self.covered_s = 0.0               # everything that starts before this is final

    # ------------------------------------------------------------------

    def update(self, audio: np.ndarray) -> int:
        """Transcribe what's new in `audio` (the whole recording so far). Returns the number of words committed."""
        total_s = len(audio) / SAMPLE_RATE
        if total_s - self.covered_s < self.min_new_s:
            return 0
        start_s = max(0.0, self.covered_s - self.overlap_s, total_s - self.max_window_s)
        window = audio[int(start_s * SAMPLE_RATE):]
        heard = self.transcribe(window, start_s)

        limit = total_s - self.tail_guard_s            # words ending after this may still change
        committed = []
        seen = {(w.text.lower(), round(w.start, 1)) for w in self.words[-40:]}
        for item in heard:
            text, s, e = item[0], start_s + float(item[1]), start_s + float(item[2])
            prob = float(item[3]) if len(item) > 3 else 1.0
            if s < self.covered_s - 0.02:              # already final (it's in the overlap region)
                continue
            if e > limit:                              # too close to the edge: re-transcribe next time
                continue
            if any(w.text.lower() == text.lower() and abs(w.start - s) < 0.5 for w in self.words[-12:]):
                continue                               # a re-detection of a word we already have
            committed.append(Word(text, s, e, prob))
        self.words.extend(committed)
        if committed:
            self.covered_s = max(self.covered_s, committed[-1].end)
        else:
            self.covered_s = max(self.covered_s, limit - self.overlap_s)    # silence/no words: don't let the window grow forever
        return len(committed)

    # ------------------------------------------------------------------

    def timings(
        self,
        slides: Sequence,
        events: Sequence[SlideEvent],
        song_start_ts: float,
        audio_start_ts: float,
        offset_s: float = 0.0,
        min_confidence: float = 0.25,
        max_window_s: float = 20.0,
        late_margin_s: float = 0.75,
    ) -> List[SlideTiming]:
        """
        Provisional per-slide timings from the words committed so far.

        A slide whose own words haven't been transcribed yet can latch onto a NEIGHBOURING line that shares
        many small words ("I can see the light in the darkness" / "I can hear the roar in the heavens"), which
        always looks like a LATE slide change (those words were sung earlier). So a late estimate for a recent
        slide change is held back as pending until the transcript has run `pending_horizon_s` past the change;
        by then the slide's real words are in and the pairing is right. On-time/early estimates show at once.
        """
        res = analyze_song(slides, events, self.words, song_start_ts, audio_start_ts,
                           offset_s=offset_s, min_confidence=min_confidence, max_window_s=max_window_s)
        known_until = (audio_start_ts - song_start_ts) + self.covered_s - offset_s   # song time the singing is known up to
        out = []
        for t in res.timings:
            too_recent = known_until < t.change_time + self.pending_horizon_s
            if t.delay is None and t.text.strip() and too_recent:
                t.note = PENDING_NOTE                  # its words may simply not have been transcribed yet
            elif t.delay is not None and not t.initial and t.delay > late_margin_s and too_recent:
                t.onset_time, t.delay, t.confidence = None, None, 0.0      # probably a neighbouring line: wait and see
                t.note = PENDING_NOTE
            out.append(t)
        return out

    @staticmethod
    def is_pending(t: SlideTiming) -> bool:
        return t.delay is None and t.note == PENDING_NOTE
