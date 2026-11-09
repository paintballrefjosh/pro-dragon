"""
Per-slide timing report
========================
Formats the result of analysis/song_timing.py: one row per slide transition,
with the signed delay and the words on the slide so it is obvious *where* the
operator was slow (or early).

    Delay:  + = slide changed after the vocalist started singing it (LATE)
            - = slide changed before the vocalist started (EARLY)
"""

from datetime import datetime
from typing import List, Optional, Sequence

from analysis.song_timing import SlideTiming, classify, summarize

WORDS_WIDTH = 60


def fmt_clock(seconds: float) -> str:
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    return f"{sign}{int(seconds // 60)}:{seconds % 60:04.1f}"


def fmt_delay(delay: float) -> str:
    return f"{delay:+.1f}s"


def one_line(text: str, width: int = WORDS_WIDTH) -> str:
    flat = " / ".join(part.strip() for part in text.splitlines() if part.strip())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def format_song_report(
    song_name: str,
    started_at: float,
    duration_s: float,
    timings: Sequence[SlideTiming],
    offset_ms: float,
    window_s: float,
    unshown: Sequence = (),
    heard_other: Sequence = (),
    early_s: Optional[float] = None,
    operator: str = "",
    notes: Sequence[str] = (),
) -> str:
    """`window_s` = late margin of error (also the early margin unless `early_s` is given)."""
    if early_s is None:
        early_s = window_s
    L: List[str] = []
    L.append("=" * 100)
    L.append(f"  {song_name}")
    L.append(
        f"  {datetime.fromtimestamp(started_at).strftime('%A %b %d, %I:%M:%S %p')}  |  "
        f"length {fmt_clock(duration_s)}  |  audio offset applied: {offset_ms:+.0f} ms"
    )
    if operator:
        L.append(f"  Operator: {operator}")
    for n in notes:
        L.append(f"  NOTE: {n}")
    L.append("=" * 100)
    L.append("  Delay: + = slide came AFTER the singing started (late)   - = slide came BEFORE (early)")
    L.append("  Each slide is matched to where ITS WORDS are sung, so a different arrangement or skipped sections are fine.")
    if abs(early_s - window_s) < 1e-9:
        L.append(f"  ON TIME = within ±{window_s:.2f}s")
    else:
        L.append(f"  ON TIME = up to {window_s:.2f}s late, or up to {early_s:.2f}s early")
    L.append("")
    L.append(f"  {'#':>3}  {'At':>6}  {'Section':<10} {'Delay':>7}  {'Verdict':<8} Words on slide")
    L.append("  " + "-" * 96)

    for t in timings:
        section = (t.group or "")[:10]
        words = one_line(t.text) if t.text else "(blank slide)"
        if t.delay is None:
            delay, verdict = "—", "n/a"
        else:
            delay = fmt_delay(t.delay)
            verdict = classify(t.delay, window_s, early_s)
            if t.initial:
                delay = "≤" + delay
        L.append(f"  {t.slide_index + 1:>3}  {fmt_clock(t.change_time):>6}  {section:<10} {delay:>7}  {verdict:<8} {words}")
        if t.note:
            L.append(f"  {'':>3}  {'':>6}  {'':<10} {'':>7}  {'':<8}   ↳ {t.note}")

    s = summarize(timings, window_s, early_s)
    L.append("  " + "-" * 96)
    if s["measured"] == 0:
        L.append("  No slide transitions could be measured (were vocals audible / did the lyrics match?).")
    else:
        L.append(
            f"  {s['measured']} transitions measured  |  average {fmt_delay(s['mean_delay'])}  |  "
            f"median {fmt_delay(s['median_delay'])}  |  avg |delay| {s['mean_abs_delay']:.1f}s"
        )
        L.append(
            f"  Late {s['pct_late']:.0f}%   On time {s['pct_on_time']:.0f}%   Early {s['pct_early']:.0f}%   |  "
            f"worst late {fmt_delay(s['worst_late'])}   most early {fmt_delay(s['worst_early'])}"
        )
    if unshown:
        L.append("")
        L.append("  Sung, but no slide change was matched (the slide was skipped, or the band went off-arrangement):")
        for u in unshown:
            sect = (u.group or "")[:10]
            L.append(f"  {'':>3}  {fmt_clock(u.onset_time):>6}  {sect:<10} {'':>7}  {'NOT SHOWN':<8} {one_line(u.text)}")
    if heard_other:
        L.append("")
        L.append("  Words heard, but no slide matched (speaking, ad-lib, or lyrics that aren't on any slide):")
        for h in heard_other:
            L.append(f"  {'':>3}  {fmt_clock(h.onset_time):>6}  {'':<10} {'':>7}  {'NO SLIDE':<8} \"{one_line(h.text, 70)}\"")
    L.append("")
    return "\n".join(L)
