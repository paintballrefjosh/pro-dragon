"""A weak partial match far from the slide change is a coincidence, not a measurement (the "18.5 s early" bug)."""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.song_timing import SlideEvent, Word, analyze_song, tokenize  # noqa: E402

VERSE = "So come what may in the space between us all"           # the slide shown at 20 s
CHORUS = "There will be another in the fire between standing next to me"   # sung at 40 s; shares "in the … between" with the verse
SLIDES = [SimpleNamespace(index=0, text=VERSE, group="Verse 3"), SimpleNamespace(index=1, text=CHORUS, group="Chorus 3")]
T0 = 1000.0


def sung(text, onset, gap=0.4):
    out, t = [], onset
    for tok in tokenize(text):
        out.append(Word(tok, t, t + 0.3, 1.0))
        t += gap
    return out


def test_a_weak_match_far_away_is_not_reported_as_a_delay():
    words = sung(CHORUS, 40.0)                                   # the verse itself was never transcribed
    res = analyze_song(SLIDES, [SlideEvent(T0 + 20.0, 0), SlideEvent(T0 + 40.5, 1)], words, T0)
    verse, chorus = res.timings
    assert verse.delay is None and "not clearly heard" in verse.note        # no invented "-20 s"
    assert chorus.delay is not None and abs(chorus.delay - 0.5) < 0.4        # the real one is untouched


def test_a_weak_match_close_to_the_change_is_still_measured():
    # only a few of the slide's words are heard, but they are right where the change happened
    words = sung("come what may in the space", 20.3)[:3] + sung("space between", 22.6)
    res = analyze_song(SLIDES[:1], [SlideEvent(T0 + 20.0, 0)], words, T0)
    t = res.timings[0]
    assert t.delay is not None and abs(t.delay) < 1.5
