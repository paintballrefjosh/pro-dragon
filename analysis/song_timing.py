"""
Per-slide timing analysis
==========================
Given, for one song:
  * the slide events  — when ProPresenter put each slide on screen
  * the heard words   — ASR output with timestamps (what the vocalist sang)
  * the lyrics        — the text of each slide

work out, for every slide change, how far it was from the moment the vocalist
started singing *that slide's words*:

    delay = slide_change_time - vocal_onset_time

    delay > 0  -> slide changed AFTER the singing started (LATE: congregation
                  had to sing without words)
    delay < 0  -> slide changed BEFORE the singing started (EARLY: ideal)

Matching is by WORDS, never by slide position
---------------------------------------------
The band may sing a different arrangement than the one in ProPresenter, skip a
section, or repeat one — and the operator may follow along by jumping around.
So nothing here assumes slides are shown in index order, or even in song order:

  1. For every distinct slide text, find each *occurrence* in the audio where
     those words were sung (local alignment of the heard words against the
     slide's words, repeated to find every occurrence — a chorus may be sung
     several times).
  2. Pair each slide change with an occurrence of ITS OWN words that is close in
     time, keeping the pairing in chronological order (the 2nd time the operator
     shows the chorus pairs with the 2nd time it is sung).
  3. A slide shown with no matching sung words nearby is reported as "not heard"
     (band skipped it, or the recognizer missed it); sung lyrics with no slide
     change are reported as "sung but never shown".

Offset
------
`offset_s` corrects for a known audio delay (e.g. NDI): positive means the
audio arrives LATE relative to the room/slides, so vocal times are moved
EARLIER by that amount before the delay is computed.
"""

import math
import re
import statistics
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from rapidfuzz import fuzz

DEFAULT_WORD_DURATION = 0.4   # seconds between sung words, if it can't be measured
MIN_WORD_DURATION = 0.15
MAX_WORD_DURATION = 0.8
FUZZY_MATCH_MIN = 0.8         # similarity needed for two different spellings to match
GAP_PENALTY = 0.5             # local-alignment penalty for a skipped slide word / extra heard word
MAX_INTRA_GAP_S = 3.0         # a silence this long between heard words ends an occurrence
UNSHOWN_MIN_COVERAGE = 0.6    # only report "sung but never shown" for well-heard lyrics
WEAK_MATCH_COVERAGE = 0.4     # a match this poor (fewer than 40% of the slide's words) …
WEAK_MATCH_MAX_DELAY_S = 8.0  # … more than this many seconds from the slide change is a coincidence, not a measurement
OTHER_PAUSE_S = 2.5           # a pause this long splits unmatched words into separate stretches
OTHER_MIN_DISTINCT = 3        # ignore stretches like "you you you you" (recognizer loops)


# ──────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SlideEvent:
    ts: float                 # wall-clock time the slide became live
    index: int                # Pro7 slide index
    initial: bool = False     # True = slide was already live when the song was started
    # What the slide was when it went live (a snapshot taken at the time). Present for songs recorded
    # since the slide list could change mid-song (an operator editing the presentation shifts every
    # later index); older songs have None and are looked up by `index` in the saved slide list.
    text: Optional[str] = None
    group: Optional[str] = None


class _SnapshotSlide:
    """A slide as it was when an event happened (see SlideEvent.text)."""
    def __init__(self, ev: "SlideEvent"):
        self.index = ev.index
        self.text = ev.text or ""
        self.group = ev.group or ""


@dataclass
class Word:
    text: str
    start: float              # seconds from the start of the recorded audio
    end: float
    prob: float = 1.0         # recognizer confidence 0-1 (1.0 when unknown, e.g. older saved songs)


@dataclass
class SlideTiming:
    ordinal: int              # 1-based position in the order slides were shown
    slide_index: int          # Pro7 slide index
    group: str
    text: str
    change_time: float        # seconds from song start when the slide went live
    initial: bool
    onset_time: Optional[float]   # seconds from song start the vocalist began this slide's words
    delay: Optional[float]        # change_time - onset_time (+ = late, - = early)
    confidence: float             # fraction of this slide's words heard in the matched occurrence
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UnshownLyric:
    """Lyrics that were sung (and recognized) but for which no slide change was matched."""
    onset_time: float         # seconds from song start
    slide_index: int          # first slide in the presentation containing these words
    group: str
    text: str
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HeardWords:
    """Recognized words that no slide's lyrics explain (talking, ad-libs, lyrics not on any slide)."""
    onset_time: float         # seconds from song start
    end_time: float
    text: str                 # as heard
    confidence: float         # mean recognizer confidence

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalysisResult:
    timings: List[SlideTiming]
    unshown: List[UnshownLyric]
    heard_other: List[HeardWords]


@dataclass
class _Occurrence:
    key: tuple                # slide-text class (token tuple)
    onset: float              # seconds from song start (offset already applied)
    first_heard: float        # time of the first matched heard word (song time)
    last_heard: float         # end of the last matched heard word (song time)
    matched: int
    coverage: float
    pairs: List[Tuple[int, int]]   # (slide word position, heard token index)


# ──────────────────────────────────────────────────────────────────────
# Text helpers
# ──────────────────────────────────────────────────────────────────────

_NON_WORD = re.compile(r"[^a-z0-9']+")


def tokenize(text: str) -> List[str]:
    """Lowercase words, punctuation stripped, apostrophes dropped ("don't" -> "dont")."""
    out = []
    for tok in _NON_WORD.split(text.lower().replace("’", "'")):
        tok = tok.replace("'", "")
        if tok:
            out.append(tok)
    return out


@lru_cache(maxsize=200_000)
def word_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if len(a) < 4 or len(b) < 4:
        return 0.0
    r = fuzz.ratio(a, b) / 100.0
    return r if r >= FUZZY_MATCH_MIN else 0.0


# ──────────────────────────────────────────────────────────────────────
# Finding where each slide's words were sung
# ──────────────────────────────────────────────────────────────────────

def _best_local_alignment(
    slide_tokens: Sequence[str],
    heard: Sequence[str],
    chain_break: Sequence[bool],
    used: Sequence[bool],
) -> Tuple[float, List[Tuple[int, int]]]:
    """
    Smith-Waterman-style local alignment of slide words against the heard-word
    stream. `chain_break[j]` = True when a long silence separates heard word j-1
    from j (an occurrence can't span it); `used[j]` = heard word already claimed
    by an earlier occurrence. Returns (score, [(slide_pos, heard_idx), ...]).
    """
    k, m = len(slide_tokens), len(heard)
    H = [[0.0] * (k + 1) for _ in range(m + 1)]
    ptr = [[0] * (k + 1) for _ in range(m + 1)]    # 1=diag(match) 2=skip heard 3=skip slide word
    best, best_j, best_i = 0.0, 0, 0
    zeros = [0.0] * (k + 1)

    for j in range(1, m + 1):
        if used[j - 1]:
            continue
        prev = zeros if chain_break[j - 1] else H[j - 1]
        row, hj = H[j], heard[j - 1]
        for i in range(1, k + 1):
            sim = word_similarity(slide_tokens[i - 1], hj)
            v, p = 0.0, 0
            if sim > 0 and prev[i - 1] + 2.0 * sim > v:
                v, p = prev[i - 1] + 2.0 * sim, 1
            if prev[i] - GAP_PENALTY > v:
                v, p = prev[i] - GAP_PENALTY, 2
            if row[i - 1] - GAP_PENALTY > v:
                v, p = row[i - 1] - GAP_PENALTY, 3
            row[i] = v
            ptr[j][i] = p
            if v > best:
                best, best_j, best_i = v, j, i

    pairs: List[Tuple[int, int]] = []
    j, i = best_j, best_i
    while j > 0 and i > 0 and H[j][i] > 0:
        p = ptr[j][i]
        if p == 1:
            pairs.append((i - 1, j - 1))
            j -= 1
            i -= 1
        elif p == 2:
            j -= 1
        elif p == 3:
            i -= 1
        else:
            break
    pairs.reverse()
    return best, pairs


def _find_occurrences(
    slide_tokens: Sequence[str],
    heard: Sequence[str],
    chain_break: Sequence[bool],
    min_confidence: float,
) -> List[List[Tuple[int, int]]]:
    """All distinct places the slide's words were sung, as lists of (slide_pos, heard_idx)."""
    k = len(slide_tokens)
    need = min(k, max(2, math.ceil(min_confidence * k)))
    used = [False] * len(heard)
    found: List[List[Tuple[int, int]]] = []
    while True:
        _score, pairs = _best_local_alignment(slide_tokens, heard, chain_break, used)
        if len(pairs) < need:
            break
        found.append(pairs)
        for j in range(pairs[0][1], pairs[-1][1] + 1):   # claim the whole stretch
            used[j] = True
    return found


# ──────────────────────────────────────────────────────────────────────
# Pairing slide changes with occurrences
# ──────────────────────────────────────────────────────────────────────

def _pair_events(
    event_times: List[float],
    event_keys: List[Optional[tuple]],
    event_initial: List[bool],
    occs: List[_Occurrence],
    window_s: float,
) -> Dict[int, int]:
    """
    Order-preserving assignment of events -> occurrences of the same slide text,
    maximizing the number of pairs, then minimizing |delay|. Returns {event_idx: occ_idx}.
    """
    n, m = len(event_times), len(occs)
    order = sorted(range(m), key=lambda j: occs[j].onset)       # occurrences by time
    ev_order = sorted(range(n), key=lambda i: event_times[i])   # events by time (already, but be safe)

    def gain(i: int, j: int) -> Optional[float]:
        if event_keys[i] is None or occs[j].key != event_keys[i]:
            return None
        delay = event_times[i] - occs[j].onset
        if event_initial[i]:
            if delay > window_s:             # initial slide may be on screen long before the vocals
                return None
            if delay <= 0:       # on screen before the vocals: any earlier sung occurrence is possible,
                return 1.0 - 1e-4 * min(abs(delay), 1e4)     # so just prefer the nearest one
            mag = min(delay, window_s)
        else:
            if abs(delay) > window_s:
                return None
            mag = abs(delay)
        return 1.0 - 0.5 * (mag / window_s)

    NEG = -1e18
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]       # 0 = skip event, 1 = skip occ, 2 = pair
    for a in range(1, n + 1):
        for b in range(1, m + 1):
            best, mv = dp[a - 1][b], 0
            if dp[a][b - 1] > best:
                best, mv = dp[a][b - 1], 1
            g = gain(ev_order[a - 1], order[b - 1])
            if g is not None and dp[a - 1][b - 1] + g > best:
                best, mv = dp[a - 1][b - 1] + g, 2
            dp[a][b] = best
            back[a][b] = mv

    pairing: Dict[int, int] = {}
    a, b = n, m
    while a > 0 and b > 0:
        mv = back[a][b]
        if mv == 2:
            pairing[ev_order[a - 1]] = order[b - 1]
            a -= 1
            b -= 1
        elif mv == 1:
            b -= 1
        else:
            a -= 1
    return pairing


# ──────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────

def analyze_song(
    slides: Sequence,                 # objects with .index .text .group (Pro7 slides)
    events: Sequence[SlideEvent],
    words: Sequence[Word],
    song_start_ts: float,
    audio_start_ts: Optional[float] = None,
    offset_s: float = 0.0,
    min_confidence: float = 0.25,
    max_window_s: float = 20.0,
    other_min_words: int = 4,
    other_min_prob: float = 0.5,
) -> AnalysisResult:
    """
    `words` are timed relative to the start of the recorded audio, which began
    at `audio_start_ts` (defaults to song_start_ts). Returns one SlideTiming per
    slide event (in the order shown) plus any lyrics sung but never shown.
    `max_window_s`: how far (seconds) a sung occurrence may be from a slide change
    and still be paired with it. `other_min_words` / `other_min_prob`: how many
    words, at what mean recognizer confidence, an unexplained stretch needs to be
    reported as "words heard, no slide matched".
    """
    if audio_start_ts is None:
        audio_start_ts = song_start_ts
    audio_shift = audio_start_ts - song_start_ts      # audio t=0 in song time

    by_index = {s.index: s for s in slides}
    # Each event's slide: the snapshot taken when it happened if there is one (robust to the slide list
    # changing mid-song), else the saved list entry for its index.
    ev_slide = [(_SnapshotSlide(ev) if ev.text is not None else by_index.get(ev.index)) for ev in events]
    slides = list(slides) + [s for ev, s in zip(events, ev_slide) if ev.text is not None]   # their words can be sung too

    # Heard tokens (a recognized "word" may contain several tokens, or none)
    heard: List[str] = []
    heard_src: List[Word] = []
    for w in words:
        for tok in tokenize(w.text):
            heard.append(tok)
            heard_src.append(w)
    chain_break = [False] * len(heard)
    for j in range(1, len(heard)):
        chain_break[j] = heard_src[j].start - heard_src[j - 1].end > MAX_INTRA_GAP_S

    # Distinct slide texts (classes) -> where they were sung
    classes: Dict[tuple, dict] = {}
    for s in slides:
        toks = tuple(tokenize(s.text))
        if toks and toks not in classes:
            classes[toks] = {"first": s, "pairs_list": []}
    for key, info in classes.items():
        info["pairs_list"] = _find_occurrences(key, heard, chain_break, min_confidence) if heard else []

    # Typical gap between sung words, measured from consecutive matches in occurrences
    gaps = []
    for key, info in classes.items():
        for pairs in info["pairs_list"]:
            for (p0, h0), (p1, h1) in zip(pairs, pairs[1:]):
                dt = heard_src[h1].start - heard_src[h0].start
                if 0 < p1 - p0 <= 2 and dt > 0:
                    gaps.append(dt / (p1 - p0))
    word_dur = statistics.median(gaps) if gaps else DEFAULT_WORD_DURATION
    word_dur = min(MAX_WORD_DURATION, max(MIN_WORD_DURATION, word_dur))

    # Build occurrences with onsets in song time
    occs: List[_Occurrence] = []
    shift = audio_shift - offset_s
    for key, info in classes.items():
        k = len(key)
        for pairs in info["pairs_list"]:
            implied = [heard_src[h].start - p * word_dur for p, h in pairs[:3]]
            occs.append(_Occurrence(
                key=key,
                onset=statistics.median(implied) + shift,
                first_heard=heard_src[pairs[0][1]].start + shift,
                last_heard=heard_src[pairs[-1][1]].end + shift,
                matched=len(pairs),
                coverage=len(pairs) / k,
                pairs=pairs,
            ))

    # Pair slide changes with occurrences
    ev_keys: List[Optional[tuple]] = []
    for ev, s in zip(events, ev_slide):
        toks = tuple(tokenize(s.text)) if s else ()
        ev_keys.append(toks or None)
    ev_times = [ev.ts - song_start_ts for ev in events]
    pairing = _pair_events(ev_times, ev_keys, [ev.initial for ev in events], occs, max_window_s)

    timings: List[SlideTiming] = []
    for ordinal, (ev, key, change_time, slide) in enumerate(zip(events, ev_keys, ev_times, ev_slide), start=1):
        text = slide.text if slide else ""
        group = getattr(slide, "group", "") if slide else ""
        onset = delay = None
        confidence = 0.0
        note = ""

        if slide is None:
            note = "slide not found in presentation"
        elif key is None:
            note = "no lyrics on this slide"
        elif ordinal - 1 in pairing and (
                occs[pairing[ordinal - 1]].coverage < WEAK_MATCH_COVERAGE
                and abs(change_time - occs[pairing[ordinal - 1]].onset) > WEAK_MATCH_MAX_DELAY_S):
            # e.g. 3 of 12 words happen to match a different line sung 18 s later (small words like "I", "the",
            # "in"): reporting "18 s early" would be an invention. Treat the slide's own words as not heard.
            far = occs[pairing[ordinal - 1]].onset - change_time
            note = (f"these words were not clearly heard — only a weak partial match {abs(far):.0f}s "
                    f"{'later' if far > 0 else 'earlier'} was found, and it was ignored")
        elif ordinal - 1 in pairing:
            occ = occs[pairing[ordinal - 1]]
            onset = occ.onset
            delay = change_time - onset
            confidence = occ.coverage
            if ev.initial:
                note = "already live when song started (true delay is this early or earlier)"
            elif confidence < 0.5:
                note = "low confidence (few of this slide's words heard)"
        elif any(o.key == key for o in occs):
            # Its words are sung somewhere in the song, but not at this moment: what was sung
            # here didn't match this slide. Say where they are sung, to help find the cause.
            nearest = min((o for o in occs if o.key == key), key=lambda o: abs(o.onset - change_time))
            gap = nearest.onset - change_time
            when = f"{int(nearest.onset // 60)}:{int(nearest.onset % 60):02d}"
            note = (f"the words sung at this point didn't match this slide — its words are sung elsewhere in "
                    f"the song (nearest: {when}, {abs(gap):.0f}s {'later' if gap > 0 else 'earlier'})")
        else:
            note = "these words were not heard in the audio (band skipped them, or not recognized)"

        timings.append(SlideTiming(
            ordinal=ordinal, slide_index=ev.index, group=group, text=text,
            change_time=change_time, initial=ev.initial,
            onset_time=onset, delay=delay, confidence=confidence, note=note,
        ))

    # Sung but never shown: unmatched, well-heard occurrences that don't just
    # duplicate (overlap in time with) an occurrence that WAS paired.
    paired = set(pairing.values())
    paired_spans = [(occs[j].first_heard, occs[j].last_heard) for j in paired]

    def overlaps_paired(o: _Occurrence) -> bool:
        length = max(1e-6, o.last_heard - o.first_heard)
        for a, b in paired_spans:
            if min(b, o.last_heard) - max(a, o.first_heard) > 0.4 * length:
                return True
        return False

    candidates = sorted(
        (o for j, o in enumerate(occs) if j not in paired and o.coverage >= UNSHOWN_MIN_COVERAGE and not overlaps_paired(o)),
        key=lambda o: (-o.coverage, o.onset),
    )
    kept: List[_Occurrence] = []
    for o in candidates:      # among overlapping candidates (near-duplicate slides) keep the best one
        length = max(1e-6, o.last_heard - o.first_heard)
        if not any(min(k_.last_heard, o.last_heard) - max(k_.first_heard, o.first_heard) > 0.4 * length for k_ in kept):
            kept.append(o)
    unshown = []
    for o in sorted(kept, key=lambda o: o.onset):
        first = classes[o.key]["first"]
        unshown.append(UnshownLyric(
            onset_time=o.onset, slide_index=first.index, group=getattr(first, "group", ""),
            text=first.text, confidence=o.coverage,
        ))

    # Words heard that NO slide explains (talking, ad-libs, lyrics not on any slide).
    # A heard word is "explained" if it falls inside any occurrence of any slide's words.
    explained = [False] * len(heard)
    for o in occs:
        for j in range(o.pairs[0][1], o.pairs[-1][1] + 1):
            explained[j] = True

    heard_other: List[HeardWords] = []
    stretch: List[int] = []

    def flush():
        if not stretch:
            return
        toks = [heard[j] for j in stretch]
        wds = list(dict.fromkeys(id(heard_src[j]) for j in stretch))         # unique Word objects, in order
        by_id = {id(heard_src[j]): heard_src[j] for j in stretch}
        objs = [by_id[i] for i in wds]
        prob = statistics.mean(w.prob for w in objs)
        if len(toks) >= other_min_words and len(set(toks)) >= OTHER_MIN_DISTINCT and prob >= other_min_prob:
            heard_other.append(HeardWords(
                onset_time=objs[0].start + shift,
                end_time=objs[-1].end + shift,
                text=" ".join(w.text for w in objs),
                confidence=prob,
            ))
        stretch.clear()

    for j in range(len(heard)):
        if explained[j]:
            flush()
            continue
        if stretch and heard_src[j].start - heard_src[stretch[-1]].end > OTHER_PAUSE_S:
            flush()
        stretch.append(j)
    flush()

    return AnalysisResult(timings=timings, unshown=unshown, heard_other=heard_other)


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

def margins_from_cfg(timing_cfg: dict) -> Tuple[float, float]:
    """(early_s, late_s) margin of error from config. `on_time_window_s` sets both;
    `on_time_early_s` / `on_time_late_s` override one side."""
    base = float(timing_cfg.get("on_time_window_s", 0.75))
    early = timing_cfg.get("on_time_early_s")
    late = timing_cfg.get("on_time_late_s")
    return (base if early is None else float(early)), (base if late is None else float(late))


def classify(delay: float, late_s: float, early_s: Optional[float] = None) -> str:
    """ON TIME if the slide changed no more than `late_s` after, and no more than
    `early_s` before, the vocal start (early_s defaults to late_s)."""
    if early_s is None:
        early_s = late_s
    if delay > late_s:
        return "LATE"
    if delay < -early_s:
        return "EARLY"
    return "ON TIME"


def summarize(timings: Sequence[SlideTiming], window_s: float = 0.75, early_s: Optional[float] = None) -> dict:
    """Aggregate stats over the transitions that have a measured delay
    (excludes slides that were already live when the song started).
    `window_s` is the late margin (and the early margin unless `early_s` is given)."""
    measured = [t for t in timings if t.delay is not None and not t.initial]
    out = {
        "transitions": len(timings),
        "measured": len(measured),
        "window_s": window_s,
        "early_window_s": window_s if early_s is None else early_s,
    }
    if not measured:
        out.update(mean_delay=None, median_delay=None, mean_abs_delay=None, worst_late=None,
                   worst_early=None, pct_late=None, pct_on_time=None, pct_early=None)
        return out

    delays = [t.delay for t in measured]
    labels = [classify(d, window_s, early_s) for d in delays]
    n = len(delays)
    out.update(
        mean_delay=statistics.mean(delays),
        median_delay=statistics.median(delays),
        mean_abs_delay=statistics.mean(abs(d) for d in delays),
        worst_late=max(delays),
        worst_early=min(delays),
        pct_late=100.0 * labels.count("LATE") / n,
        pct_on_time=100.0 * labels.count("ON TIME") / n,
        pct_early=100.0 * labels.count("EARLY") / n,
    )
    return out
