"""LiveAnalyzer: provisional delays computed while the song is still playing, from short overlapping
transcription windows, using a scripted transcriber that (like real Whisper) cannot see words cut
off at the edges of the window."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.live import PENDING_NOTE, SAMPLE_RATE, LiveAnalyzer  # noqa: E402
from analysis.song_timing import SlideEvent, tokenize  # noqa: E402

LYRICS = ["Holy holy holy Lord God almighty", "Early in the morning our song shall rise to thee",
          "Holy holy holy merciful and mighty", "God in three persons blessed trinity",
          "Only thou art holy there is none beside thee"]
SLIDES = [SimpleNamespace(index=i, text=t, group="Verse") for i, t in enumerate(LYRICS)]
T0 = 5000.0


def make_script(onsets, gap=0.4):
    """Absolute word script: (text, start, end) for each lyric line sung at its onset."""
    words = []
    for idx, onset in enumerate(onsets):
        t = onset
        for tok in tokenize(LYRICS[idx]):
            words.append((tok, t, t + 0.3))
            t += gap
    return words


class FakeTranscriber:
    """Returns the script's words that lie fully inside the window (relative times), and logs window sizes."""
    def __init__(self, script):
        self.script, self.windows, self.calls = script, [], 0

    def __call__(self, window, start_s):
        self.calls += 1
        end_s = start_s + len(window) / SAMPLE_RATE
        self.windows.append(len(window) / SAMPLE_RATE)
        return [(t, s - start_s, e - start_s, 0.9) for t, s, e in self.script if s >= start_s and e <= end_s]


def run_live(onsets, delays, total_s, step=6.0):
    script = make_script(onsets)
    tr = FakeTranscriber(script)
    live = LiveAnalyzer(tr)
    events = [SlideEvent(T0 + on + d, i) for i, (on, d) in enumerate(zip(onsets, delays))]
    snapshots = []
    t = step
    while t <= total_s:
        live.update(np.zeros(int(t * SAMPLE_RATE), dtype=np.float32))
        seen_events = [e for e in events if e.ts - T0 <= t]
        snapshots.append((t, live.timings(SLIDES, seen_events, T0, T0)))
        t += step
    return live, tr, snapshots, script


def test_live_estimates_converge_to_the_true_delays_and_never_duplicate_words():
    onsets = [8.0, 16.0, 26.0, 35.0, 44.0]
    delays = [1.2, -0.8, 0.4, 2.0, -0.3]
    live, tr, snaps, script = run_live(onsets, delays, total_s=70)
    final = snaps[-1][1]
    for t, d in zip(final, delays):
        assert t.delay is not None and abs(t.delay - d) < 0.3, (t.slide_index, t.delay, d)
    assert len(live.words) == len(script)                                  # every word once: no loss, no duplicates
    assert [w.text for w in live.words] == [s[0] for s in script]
    assert max(tr.windows) <= 40.5                                          # windows stay bounded (max_window_s)


def test_slides_whose_singing_isnt_transcribed_yet_are_pending_not_missing():
    onsets = [8.0, 16.0, 26.0]
    script = make_script(onsets)
    live = LiveAnalyzer(FakeTranscriber(script))
    events = [SlideEvent(T0 + on + 0.5, i) for i, on in enumerate(onsets)]
    audio = lambda secs: np.zeros(int(secs * SAMPLE_RATE), dtype=np.float32)

    live.update(audio(20))                                   # transcribed up to ~18 s; the 3rd slide changes at 26.5 s
    third = live.timings(SLIDES, events, T0, T0)[2]
    assert third.delay is None and third.note == PENDING_NOTE and LiveAnalyzer.is_pending(third)
    assert "not heard" not in third.note                     # not mistaken for a skipped slide

    live.update(audio(32))                                   # a few seconds later its singing is known
    third = live.timings(SLIDES, events, T0, T0)[2]
    assert third.delay is not None and abs(third.delay - 0.5) < 0.3 and not LiveAnalyzer.is_pending(third)
    # slides that were never sung become "not heard" only after the pending horizon has passed
    skipped = [SlideEvent(T0 + 5.0, 3)]
    live.update(audio(60))
    assert live.timings(SLIDES, skipped, T0, T0)[0].note != PENDING_NOTE


def test_long_silence_does_not_make_the_window_grow_without_bound():
    live, tr, snaps, _ = run_live([100.0], [0.2], total_s=96)               # 96 s of nothing
    assert max(tr.windows) <= 40.5 and live.words == []
    assert tr.calls >= 10


def test_too_little_new_audio_skips_the_transcription():
    tr = FakeTranscriber([])
    live = LiveAnalyzer(tr)
    assert live.update(np.zeros(int(2 * SAMPLE_RATE), dtype=np.float32)) == 0
    assert tr.calls == 0


# --- a slide whose own words aren't transcribed yet must not latch onto the NEIGHBOURING line -----------------

SIMILAR = ["And I can see the light in the darkness As the darkness blocks the sun",
           "I can hear the roar in the heavens As the space between us falls"]
SIM_SLIDES = [SimpleNamespace(index=i, text=t, group="Bridge") for i, t in enumerate(SIMILAR)]


def _sung(line, onset, gap=0.4):
    from analysis.live import Word
    t, out = onset, []
    for tok in tokenize(SIMILAR[line]):
        out.append(Word(tok, t, t + 0.3, 1.0))
        t += gap
    return out


def test_a_late_estimate_for_a_just_changed_slide_waits_for_its_own_words():
    live = LiveAnalyzer(lambda *a: [], pending_horizon_s=8.0)
    events = [SlideEvent(T0 + 10.0, 0), SlideEvent(T0 + 22.0, 1)]          # line 0 shown at 10 s, line 1 at 22 s
    live.words = _sung(0, 10.0)                                            # line 0 sung at 10 s; line 1 not sung yet (it starts at 22 s)
    live.covered_s = 20.0                                                  # transcript known up to 20 s: before the 2nd change

    ts = live.timings(SIM_SLIDES, events, T0, T0, late_margin_s=0.25)
    assert ts[0].delay is not None and abs(ts[0].delay) < 0.5              # line 0: matched to its own words
    # line 1 would pair with line 0's words (shared "I can … the … in the …") and look ~12 s late: it must be held back
    assert ts[1].delay is None and ts[1].note == PENDING_NOTE

    live.words += _sung(1, 22.4)                                           # now line 1 really is sung, 0.4 s after the change
    live.covered_s = 31.0                                                  # and the transcript has run 8 s+ past the change
    ts = live.timings(SIM_SLIDES, events, T0, T0, late_margin_s=0.25)
    assert ts[1].delay is not None and abs(ts[1].delay - (-0.4)) < 0.5     # its own words, correctly early


def test_on_time_and_early_estimates_still_show_immediately():
    live = LiveAnalyzer(lambda *a: [], pending_horizon_s=8.0)
    live.words = _sung(0, 10.4)                                            # sung 0.4 s AFTER the change at 10.0 → early
    live.covered_s = 14.0
    t = live.timings(SIM_SLIDES, [SlideEvent(T0 + 10.0, 0)], T0, T0, late_margin_s=0.25)[0]
    assert t.delay is not None and t.delay < 0 and t.note != PENDING_NOTE  # not held back: only LATE ones wait
