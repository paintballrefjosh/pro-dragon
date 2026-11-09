"""
Per-operator statistics
========================
Pools the measured slide changes of every saved song by operator (the name typed
in when the song was recorded) and summarizes how early/late they run. Pure
functions over the song dicts the library loads; margins are applied here, at
read time, so they can be changed freely without re-analyzing anything.
"""

import statistics
from typing import Dict, List, Optional, Sequence

from analysis.song_timing import classify

UNASSIGNED = "(no operator)"


def measured_delays(timings: Sequence[dict]) -> List[float]:
    """Delays of slide changes that were actually measured (not 'not heard', not already-live-at-start)."""
    return [t["delay"] for t in timings if t.get("delay") is not None and not t.get("initial")]


def delay_stats(delays: Sequence[float], late_s: float, early_s: Optional[float] = None) -> dict:
    n = len(delays)
    if n == 0:
        return {"measured": 0, "mean_delay": None, "median_delay": None, "mean_abs_delay": None,
                "pct_on_time": None, "pct_late": None, "pct_early": None, "worst_late": None, "most_early": None,
                "n_late": 0, "n_early": 0, "n_on_time": 0}
    labels = [classify(d, late_s, early_s) for d in delays]
    n_late, n_early, n_on = labels.count("LATE"), labels.count("EARLY"), labels.count("ON TIME")
    return {
        "measured": n,
        "mean_delay": statistics.mean(delays),
        "median_delay": statistics.median(delays),
        "mean_abs_delay": statistics.mean(abs(d) for d in delays),
        "pct_on_time": 100.0 * n_on / n, "pct_late": 100.0 * n_late / n, "pct_early": 100.0 * n_early / n,
        "worst_late": max(delays), "most_early": min(delays),
        "n_late": n_late, "n_early": n_early, "n_on_time": n_on,
    }


def operator_name(song: dict) -> str:
    return (song.get("operator") or "").strip() or UNASSIGNED


def aggregate(songs: Sequence[dict], late_s: float, early_s: Optional[float] = None) -> List[dict]:
    """
    `songs`: dicts with id, name, started_at, operator, timings, unshown, heard_other.
    Returns one entry per operator (alphabetical, unassigned last), each with pooled
    stats over all their measured slide changes and a per-song breakdown (oldest first).
    """
    groups: Dict[str, List[dict]] = {}
    for s in songs:
        groups.setdefault(operator_name(s), []).append(s)

    out = []
    for name, group in groups.items():
        group = sorted(group, key=lambda s: s.get("started_at", 0))
        pooled: List[float] = []
        per_song = []
        for s in group:
            d = measured_delays(s.get("timings", []))
            pooled.extend(d)
            st = delay_stats(d, late_s, early_s)
            per_song.append({
                "id": s["id"], "name": s["name"], "started_at": s.get("started_at", 0),
                "duration": s.get("duration", 0),
                "not_heard": sum(1 for t in s.get("timings", []) if t.get("delay") is None and (t.get("text") or "").strip()),
                "not_shown": len(s.get("unshown", [])), "no_slide": len(s.get("heard_other", [])),
                **st,
            })
        entry = {
            "name": name, "unassigned": name == UNASSIGNED, "songs": len(group),
            "first": group[0].get("started_at", 0), "last": group[-1].get("started_at", 0),
            "not_heard": sum(p["not_heard"] for p in per_song),
            "not_shown": sum(p["not_shown"] for p in per_song),
            "no_slide": sum(p["no_slide"] for p in per_song),
            "per_song": per_song,
            **delay_stats(pooled, late_s, early_s),
        }
        out.append(entry)
    out.sort(key=lambda e: (e["unassigned"], e["name"].lower()))
    return out


def format_operator_report(entries: Sequence[dict], late_s: float, early_s: Optional[float] = None) -> str:
    """Plain-text report across all operators (CLI: python main.py --operators)."""
    from datetime import datetime
    if early_s is None:
        early_s = late_s
    L: List[str] = []
    L.append("=" * 100)
    L.append("  OPERATOR REPORT")
    L.append("=" * 100)
    if abs(early_s - late_s) < 1e-9:
        L.append(f"  ON TIME = within ±{late_s:.2f}s of the vocalist.  Delay: + = late, − = early.")
    else:
        L.append(f"  ON TIME = up to {late_s:.2f}s late or {early_s:.2f}s early.  Delay: + = late, − = early.")
    L.append("")
    if not entries:
        L.append("  No saved songs found.")
        return "\n".join(L) + "\n"
    L.append(f"  {'Operator':<22} {'Songs':>5} {'Changes':>8} {'Avg':>7} {'On time':>8} {'Late':>6} {'Early':>6} "
             f"{'Worst late':>11} {'Most early':>11}  {'Not heard':>9}")
    L.append("  " + "-" * 98)

    def f(v, fmt):
        return "—" if v is None else fmt.format(v)

    for e in entries:
        L.append(f"  {e['name'][:22]:<22} {e['songs']:>5} {e['measured']:>8} {f(e['mean_delay'], '{:+.1f}s'):>7} "
                 f"{f(e['pct_on_time'], '{:.0f}%'):>8} {f(e['pct_late'], '{:.0f}%'):>6} {f(e['pct_early'], '{:.0f}%'):>6} "
                 f"{f(e['worst_late'], '{:+.1f}s'):>11} {f(e['most_early'], '{:+.1f}s'):>11}  {e['not_heard']:>9}")
    for e in entries:
        L.append("")
        L.append(f"  {e['name']}")
        L.append("  " + "-" * 60)
        for p in e["per_song"]:
            when = datetime.fromtimestamp(p["started_at"]).strftime("%b %d %I:%M %p")
            L.append(f"    {when:<16} {p['name'][:30]:<30} {p['measured']:>3} changes  avg {f(p['mean_delay'], '{:+.1f}s'):>6}  "
                     f"on time {f(p['pct_on_time'], '{:.0f}%'):>4}")
    L.append("")
    return "\n".join(L)
