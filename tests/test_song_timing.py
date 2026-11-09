"""Synthetic checks of the per-slide timing analysis. A simulated singer (with ASR
errors) sings sections in ANY order; a simulated operator changes slides at known
delays. The analysis must match by words (not slide position) and recover the delays."""

import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.song_timing import SlideEvent, Word, analyze_song, summarize, tokenize  # noqa: E402

# ProPresenter's arrangement: slide index -> (section, words)
SLIDES = [
    ("Verse 1", "For my waking breath for my daily bread"),          # 0
    ("Verse 1", "I depend on You I depend on You"),                  # 1
    ("Verse 2", "Where the spirit leads as I'm following"),          # 2
    ("Verse 2", "Every step I take is in Your hands"),               # 3
    ("Chorus", "You're the way the truth and the life"),             # 4
    ("Chorus", "You're the well that never runs dry"),               # 5
    ("Blank", ""),                                                   # 6
]


def make_slides():
    return [SimpleNamespace(index=i, text=t, group=g) for i, (g, t) in enumerate(SLIDES)]


def sing(plan, drop=0.12, typo=0.08, jitter=0.12, seed=3, line_gap=1.2):
    """
    plan: [(slide_index_sung, onset_seconds)] in the order the band sings them.
    Returns heard Words (with ASR noise).
    """
    rng = random.Random(seed)
    words = []
    for idx, onset in plan:
        t = onset
        for tok in tokenize(SLIDES[idx][1]):
            if rng.random() > drop:
                heard = tok
                if rng.random() < typo and len(tok) > 3:
                    heard = tok[:-1] + "x"
                s = t + rng.uniform(-jitter, jitter)
                words.append(Word(heard, max(0.0, s), s + 0.3))
            t += 0.4
    return words


def ev(song_start, index, onset, delay, initial=False):
    return SlideEvent(ts=song_start + onset + delay, index=index, initial=initial)


T0 = 1000.0


def test_follows_the_band_in_order():
    plan = [(0, 5.0), (1, 9.0), (2, 13.0), (3, 17.0)]
    delays = [-1.0, 0.0, 2.0, 3.5]
    events = [ev(T0, i, on, d) for (i, on), d in zip(plan, delays)]
    res = analyze_song(make_slides(), events, sing(plan), T0)
    for t, d in zip(res.timings, delays):
        assert t.delay is not None, t
        assert abs(t.delay - d) < 0.6, f"slide {t.ordinal}: measured {t.delay:.2f}, true {d}"
        assert t.group
    assert res.unshown == []


def test_different_arrangement_and_jumping_around():
    """Band: V1a V1b | Chorus a b | V2a V2b | Chorus a b (chorus twice, verse 2 AFTER the chorus).
    ProPresenter's order is V1, V2, Chorus — so the operator jumps 1 -> 4 -> 2 -> 4."""
    plan = [(0, 5.0), (1, 9.0), (4, 14.0), (5, 18.0), (2, 24.0), (3, 28.0), (4, 34.0), (5, 38.0)]
    delays = [0.5, -0.5, 1.5, 0.0, 2.5, -1.0, 0.8, 0.3]
    events = [ev(T0, i, on, d) for (i, on), d in zip(plan, delays)]
    res = analyze_song(make_slides(), events, sing(plan), T0)
    assert [t.slide_index for t in res.timings] == [0, 1, 4, 5, 2, 3, 4, 5]
    for t, d in zip(res.timings, delays):
        assert t.delay is not None, t
        assert abs(t.delay - d) < 0.6, f"event {t.ordinal} (slide {t.slide_index + 1}): measured {t.delay:.2f}, true {d}"
    # the 2nd time the operator shows the chorus pairs with the 2nd time it is sung (not the 1st)
    assert res.timings[6].onset_time > 30 and res.timings[2].onset_time < 20
    assert res.unshown == []


def test_slide_shown_but_band_skipped_it():
    plan = [(0, 5.0), (1, 9.0), (4, 24.0)]               # band never sings verse 2
    events = [ev(T0, 0, 5.0, 0.4), ev(T0, 1, 9.0, 0.4),
              SlideEvent(ts=T0 + 16.0, index=2),            # operator clicks through verse 2 anyway
              ev(T0, 4, 24.0, 0.4)]
    res = analyze_song(make_slides(), events, sing(plan), T0)
    assert res.timings[2].delay is None and "not heard" in res.timings[2].note
    assert abs(res.timings[3].delay - 0.4) < 0.6          # later slides still measured correctly


def test_sung_but_never_shown():
    plan = [(0, 5.0), (1, 9.0), (4, 14.0), (5, 18.0)]     # band sings the chorus...
    events = [ev(T0, 0, 5.0, 0.2), ev(T0, 1, 9.0, 0.2)]   # ...but the operator never shows it
    res = analyze_song(make_slides(), events, sing(plan, drop=0, typo=0), T0)
    assert [t.slide_index for t in res.timings] == [0, 1]
    assert {u.slide_index for u in res.unshown} == {4, 5}
    assert all(u.group == "Chorus" for u in res.unshown)
    assert abs(res.unshown[0].onset_time - 14.0) < 0.5


def test_operator_far_too_late_is_not_paired():
    plan = [(0, 5.0), (1, 9.0)]
    events = [ev(T0, 0, 5.0, 0.5), ev(T0, 1, 9.0, 40.0)]  # 40 s late: outside the 20 s window
    res = analyze_song(make_slides(), events, sing(plan, drop=0, typo=0), T0)
    assert res.timings[0].delay is not None
    note = res.timings[1].note
    assert res.timings[1].delay is None
    assert "didn't match this slide" in note and "sung elsewhere" in note       # not "sung near this change"
    assert "nearest: 0:09" in note and "earlier" in note                         # points at where it was sung
    assert "within" not in note


def test_offset_shifts_delay():
    plan = [(0, 5.0), (1, 9.0), (4, 14.0)]
    events = [ev(T0, i, on, 0.0) for i, on in plan]
    words = sing(plan, drop=0, typo=0, jitter=0)
    base = analyze_song(make_slides(), events, words, T0).timings
    pos = analyze_song(make_slides(), events, words, T0, offset_s=0.25).timings
    neg = analyze_song(make_slides(), events, words, T0, offset_s=-0.25).timings
    for b, p, n in zip(base, pos, neg):
        assert abs((p.delay - b.delay) - 0.25) < 1e-6      # audio late -> vocals earlier -> slide later
        assert abs((n.delay - b.delay) + 0.25) < 1e-6


def test_blank_and_undetected_slides():
    events = [SlideEvent(ts=10.0, index=0), SlideEvent(ts=12.0, index=6)]
    res = analyze_song(make_slides(), events, [], 0.0)       # nothing heard
    assert res.timings[0].delay is None and "not heard" in res.timings[0].note
    assert res.timings[1].delay is None and "no lyrics" in res.timings[1].note


def test_initial_slide_flagged_and_excluded_from_summary():
    plan = [(0, 30.0), (1, 34.0), (4, 40.0)]                 # long instrumental intro
    events = [SlideEvent(ts=T0, index=0, initial=True), ev(T0, 1, 34.0, 0.5), ev(T0, 4, 40.0, 0.5)]
    res = analyze_song(make_slides(), events, sing(plan, drop=0, typo=0, jitter=0), T0)
    first = res.timings[0]
    assert first.initial and first.delay is not None and first.delay < -25      # on screen ~30 s before vocals
    s = summarize(res.timings)
    assert s["measured"] == 2 and abs(s["mean_delay"] - 0.5) < 0.1


def test_similar_slides_dont_cause_false_unshown():
    # slide 1 ("I depend on You x2") and a near-duplicate must not yield phantom "never shown"
    slides = make_slides() + [SimpleNamespace(index=7, text="I depend on You yes I depend on You", group="Verse 3")]
    plan = [(0, 5.0), (1, 9.0)]
    events = [ev(T0, 0, 5.0, 0.2), ev(T0, 1, 9.0, 0.2)]
    res = analyze_song(slides, events, sing(plan, drop=0, typo=0), T0)
    assert res.unshown == []


def _spoken(text, start, prob=0.9, gap=0.35):
    out, t = [], start
    for w in text.split():
        out.append(Word(w, t, t + 0.3, prob))
        t += gap
    return out


def test_talking_between_songs_is_reported_as_words_heard_no_slide():
    plan = [(0, 5.0), (1, 9.0), (4, 40.0)]                           # 30 s gap: the pastor talks in it
    words = sing(plan, drop=0, typo=0, jitter=0) + _spoken("Welcome everyone please grab a seat and turn to your neighbor", 20.0)
    events = [ev(T0, i, on, 0.3) for i, on in plan]
    res = analyze_song(make_slides(), events, words, T0)
    assert len(res.heard_other) == 1
    h = res.heard_other[0]
    assert h.text.startswith("Welcome everyone") and abs(h.onset_time - 20.0) < 0.1 and h.end_time > h.onset_time
    assert res.unshown == []                                          # not confused with unshown lyrics
    assert all(t.delay is not None for t in res.timings)


def test_other_words_respect_offset():
    words = _spoken("Welcome everyone please grab a seat", 20.0)
    res = analyze_song(make_slides(), [], words, T0, offset_s=0.25)
    assert abs(res.heard_other[0].onset_time - 19.75) < 1e-6


def test_phantom_words_are_filtered():
    base = sing([(0, 5.0)], drop=0, typo=0, jitter=0)
    cases = {
        "low confidence": _spoken("Welcome everyone please grab a seat", 20.0, prob=0.2),
        "too short": _spoken("Thanks for watching", 20.0),
        "recognizer loop": _spoken("you you you you you you you", 20.0),
    }
    for name, extra in cases.items():
        res = analyze_song(make_slides(), [ev(T0, 0, 5.0, 0.0)], base + extra, T0)
        assert res.heard_other == [], name


def test_normal_singing_yields_no_other_words():
    plan = [(0, 5.0), (1, 9.0), (4, 14.0), (5, 18.0), (2, 24.0), (3, 28.0)]
    events = [ev(T0, i, on, 0.2) for i, on in plan]
    for seed in range(5):
        res = analyze_song(make_slides(), events, sing(plan, drop=0.25, typo=0.15, seed=seed), T0)
        assert res.heard_other == [], (seed, res.heard_other)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
