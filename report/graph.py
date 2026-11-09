"""
Song timing graph (SVG)
========================
A boxy, stock-chart-style step graph of each slide's delay over the song:

  * x axis: time in the song          * y axis: delay in seconds
  * up = EARLY, down = LATE, middle band = ON TIME (the margin of error), so
    green/yellow sit above the 0s line and red below it. Axis numbers are the
    delay itself (negative = early, up top; positive = late, down below).
  * each slide holds its delay as a flat "box" until the next slide change, then
    steps to the next level (no diagonal shark-teeth)
  * line colour: green = on time, yellow = early, red = late
  * a "sections" strip under the time axis names each part of the song (Verse 1, Chorus …, from
    ProPresenter's slide groups), tinted by that section's average delay, so it is easy to see
    which parts the operator is too slow or too early on
  * a "details" lane under the chart prints every slide's delay value at its exact time
    (coloured like the line, staggered so labels never overlap) and calls out problems:
      ✕ slide shown but its words weren't heard      (grey)
      ◆ lyrics sung but the slide was never shown     (orange)
      ▬ words heard that no slide matched (talking…)  (blue)

Pure function returning an SVG string (colours are explicit so the saved .svg
looks the same standalone and embedded in the web page).
"""

import math
from typing import List, Optional, Sequence
from xml.sax.saxutils import escape

from analysis.live import PENDING_NOTE
from analysis.song_timing import SlideTiming, classify

GREEN = "#2fbf71"
YELLOW = "#e6b422"
RED = "#e5484d"
GREY = "#8b98a7"
ORANGE = "#f08a3c"
BLUE = "#4aa3ff"
CARD = "#0f1620"
GRID = "#243040"
TEXT = "#cbd5df"
MUTED = "#7f8d9c"

COLOR = {"ON TIME": GREEN, "EARLY": YELLOW, "LATE": RED}


def _clock(seconds: float) -> str:
    seconds = max(0.0, seconds)
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _flat(text: str, width: int = 70) -> str:
    flat = " / ".join(p.strip() for p in (text or "").splitlines() if p.strip())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _nice_step(span: float, target: int) -> float:
    for step in (1, 2, 5, 10, 15, 30, 60, 120, 300):
        if span / step <= target:
            return step
    return 600


def render_song_graph(
    song_name: str,
    duration_s: float,
    timings: Sequence[SlideTiming],
    unshown: Sequence = (),
    heard_other: Sequence = (),
    late_s: float = 0.75,
    early_s: Optional[float] = None,
    width: int = 1000,
    height: int = 432,
    operator: str = "",
) -> str:
    if early_s is None:
        early_s = late_s

    ml, mr, mt = 58, 18, 54                    # plot margins (left, right, top)
    axis_h = 26                                # room for the time labels under the chart
    lane_h = 66                                # details lane under the axis (values + problem markers)
    legend_h = 22                              # lane legend at the very bottom
    has_groups = any((t.group or "").strip() for t in timings)
    sect_h = 26 if has_groups else 0           # "sections" strip (Verse 1, Chorus …) under the time axis
    height += sect_h                           # extra room, so the chart itself keeps its size
    plot_h = height - mt - axis_h - sect_h - lane_h - legend_h
    plot_w = width - ml - mr
    sect_top = mt + plot_h + axis_h
    lane_top = sect_top + sect_h

    # ---- scales -----------------------------------------------------------
    last_t = max([t.change_time for t in timings] + [0.0])
    T = max(duration_s, last_t + 3.0, 10.0)
    steady = [abs(t.delay) for t in timings if t.delay is not None and not t.initial]
    yr = max(3.0, math.ceil(max(steady + [0.0]) + 0.5), 2.0 * late_s, 2.0 * early_s)

    def X(t: float) -> float:
        return ml + (min(max(t, 0.0), T) / T) * plot_w

    def Y(d: float) -> float:                  # early (negative) = up, late (positive) = down
        d = max(-yr, min(yr, d))
        return mt + (yr + d) / (2.0 * yr) * plot_h

    out: List[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'font-family="Segoe UI, Roboto, Helvetica, Arial, sans-serif" role="img" '
        f'aria-label="Slide timing graph for {escape(song_name)}">'
    )
    out.append(f'<rect width="{width}" height="{height}" rx="10" fill="{CARD}"/>')

    # ---- title + legend -----------------------------------------------------
    out.append(f'<text x="{ml}" y="24" fill="{TEXT}" font-size="15" font-weight="700">{escape(song_name)}</text>')
    margin_txt = (f"on time = within ±{late_s:.2f}s" if abs(early_s - late_s) < 1e-9
                  else f"on time = ≤{late_s:.2f}s late, ≤{early_s:.2f}s early")
    who = f"Operator: {operator} · " if operator else ""
    out.append(f'<text x="{ml}" y="42" fill="{MUTED}" font-size="11">{escape(who)}Delay of each slide change vs. when its words were sung · {escape(margin_txt)}</text>')
    lx = width - mr
    for label, col in (("Late", RED), ("Early", YELLOW), ("On time", GREEN)):
        lx -= 12 + len(label) * 6.6 + 14
        out.append(f'<rect x="{lx:.1f}" y="15" width="10" height="10" rx="2" fill="{col}"/>'
                   f'<text x="{lx + 15:.1f}" y="24" fill="{TEXT}" font-size="11">{label}</text>')

    # ---- zones, grid ----------------------------------------------------------
    y_early, y_late = Y(-early_s), Y(late_s)        # y_early is the upper edge of the on-time band
    out.append(f'<rect x="{ml}" y="{mt}" width="{plot_w}" height="{y_early - mt:.1f}" fill="{YELLOW}" opacity="0.08"/>')
    out.append(f'<rect x="{ml}" y="{y_early:.1f}" width="{plot_w}" height="{y_late - y_early:.1f}" fill="{GREEN}" opacity="0.13"/>')
    out.append(f'<rect x="{ml}" y="{y_late:.1f}" width="{plot_w}" height="{mt + plot_h - y_late:.1f}" fill="{RED}" opacity="0.08"/>')

    ystep = 1 if yr <= 6 else (2 if yr <= 12 else 5)
    d = -math.floor(yr / ystep) * ystep
    while d <= yr + 1e-9:
        y = Y(d)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + plot_w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" fill="{MUTED}" font-size="11" text-anchor="end">{d:+g}s</text>')
        d += ystep
    y0 = Y(0)
    out.append(f'<line x1="{ml}" y1="{y0:.1f}" x2="{ml + plot_w}" y2="{y0:.1f}" stroke="{TEXT}" stroke-width="1" stroke-dasharray="5 4" opacity="0.55"/>')
    out.append(f'<text x="{ml + 6}" y="{mt + 14}" fill="{YELLOW}" font-size="10" font-weight="700" opacity="0.9">▲ EARLY (slide before singing started)</text>')
    out.append(f'<text x="{ml + 6}" y="{mt + plot_h - 6}" fill="{RED}" font-size="10" font-weight="700" opacity="0.9">▼ LATE (slide after singing started)</text>')

    xstep = _nice_step(T, 10)
    t = 0.0
    while t <= T + 1e-9:
        x = X(t)
        out.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + plot_h + 4}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{mt + plot_h + 17}" fill="{MUTED}" font-size="11" text-anchor="middle">{_clock(t)}</text>')
        t += xstep

    ordered = sorted(timings, key=lambda s: s.change_time)

    # ---- sections strip: which part of the song (Verse 1, Chorus …) each stretch is --------------
    # A section runs from the first slide change into a group until the next group's first change.
    # It is tinted by that section's average delay so the slow/early parts of a song stand out, and
    # a faint dashed line carries each boundary up through the chart.
    if has_groups:
        runs: List[dict] = []
        for tm in ordered:
            g = (tm.group or "").strip()
            if not runs or runs[-1]["group"] != g:
                runs.append({"group": g, "start": tm.change_time, "delays": []})
            if tm.delay is not None and not tm.initial:
                runs[-1]["delays"].append(tm.delay)
        out.append(f'<text x="{ml - 8}" y="{sect_top + 15}" fill="{MUTED}" font-size="10" text-anchor="end">sections</text>')
        for k, run in enumerate(runs):
            end_t = runs[k + 1]["start"] if k + 1 < len(runs) else T
            x1, x2 = X(run["start"]), X(end_t)
            w = x2 - x1
            if w < 1.5:
                continue
            ds = run["delays"]
            mean = sum(ds) / len(ds) if ds else None
            col = COLOR[classify(mean, late_s, early_s)] if mean is not None else GREY
            name = run["group"] or "(no section)"
            tip = f"{name} · {_clock(run['start'])}–{_clock(end_t)} · " + (
                f"average {mean:+.1f}s ({classify(mean, late_s, early_s).lower()}) over {len(ds)} slide change(s)"
                if mean is not None else "no slide changes measured in this section")
            if k > 0:
                out.append(f'<line x1="{x1:.1f}" y1="{mt}" x2="{x1:.1f}" y2="{sect_top + sect_h - 4}" stroke="{TEXT}" stroke-width="1" '
                           f'stroke-dasharray="2 5" opacity="0.28"/>')
            label = ""
            avg = (f"{'−' if mean < 0 else '+'}{abs(mean):.1f}s" if mean is not None else "")
            for cand in (f"{name}  {avg}".strip(), name):          # full label if it fits, else the name alone
                if len(cand) * 6.3 + 10 <= w:
                    label = cand
                    break
            else:
                room = int((w - 10) / 6.3)
                label = (name[: room - 1] + "…") if room >= 4 else ""
            out.append(f'<g><title>{escape(tip)}</title>'
                       f'<rect x="{x1 + 1:.1f}" y="{sect_top + 2}" width="{max(w - 2, 1):.1f}" height="{sect_h - 6}" rx="4" '
                       f'fill="{col}" opacity="0.22"/>'
                       f'<rect x="{x1 + 1:.1f}" y="{sect_top + 2}" width="{max(w - 2, 1):.1f}" height="3" rx="1.5" fill="{col}"/>'
                       + (f'<text x="{x1 + 6:.1f}" y="{sect_top + 18}" fill="{TEXT}" font-size="11" font-weight="700">{escape(label)}</text>' if label else "")
                       + '</g>')

    # ---- the step line -------------------------------------------------------
    holds = []        # (timing, x1, x2, y, colour)
    for i, tm in enumerate(ordered):
        if tm.delay is None:
            continue
        end_t = ordered[i + 1].change_time if i + 1 < len(ordered) else T
        label = classify(tm.delay, late_s, early_s)
        holds.append((i, tm, X(tm.change_time), X(max(end_t, tm.change_time + 0.4)), Y(tm.delay), COLOR[label], label))

    defs, body = [], []
    for k, (i, tm, x1, x2, y, col, label) in enumerate(holds):
        # vertical riser from the previous box (only when this slide directly follows a measured one)
        if k > 0 and holds[k - 1][0] == i - 1:
            py, pcol = holds[k - 1][4], holds[k - 1][5]
            if abs(py - y) > 0.6:
                gid = f"g{k}"
                defs.append(f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="0" y1="{py:.1f}" x2="0" y2="{y:.1f}">'
                            f'<stop offset="0" stop-color="{pcol}"/><stop offset="1" stop-color="{col}"/></linearGradient>')
                body.append(f'<line x1="{x1:.1f}" y1="{py:.1f}" x2="{x1:.1f}" y2="{y:.1f}" stroke="url(#{gid})" stroke-width="3"/>')
        body.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="4" stroke-linecap="butt"/>')

    for (i, tm, x1, x2, y, col, label) in holds:
        shown = f"{tm.delay:+.1f}s"
        clamped = abs(tm.delay) > yr
        tip = (f"Slide {tm.slide_index + 1}" + (f" · {tm.group}" if tm.group else "") +
               f" · at {_clock(tm.change_time)} · {shown} {label.lower()}" +
               (" (off the chart)" if clamped else "") +
               (" · already on screen when the song started" if tm.initial else "") +
               f"\n{_flat(tm.text)}")
        fill = CARD if tm.initial else col
        body.append(f'<g><title>{escape(tip)}</title><circle cx="{x1:.1f}" cy="{y:.1f}" r="5" fill="{fill}" stroke="{col}" stroke-width="2.5"/></g>')

    out.append("<defs>" + "".join(defs) + "</defs>")
    out.extend(body)

    # ---- details lane ----------------------------------------------------------
    # Every slide change gets its delay value printed at its exact time (coloured like the line),
    # so values, gaps and problems can be read straight off the timeline. Neighbouring labels are
    # staggered over several rows (with a thin guide line up to the exact time) so they never overlap.
    out.append(f'<rect x="{ml}" y="{lane_top}" width="{plot_w}" height="{lane_h - 4}" rx="4" fill="#131c28"/>')
    out.append(f'<text x="{ml - 8}" y="{lane_top + 14}" fill="{MUTED}" font-size="10" text-anchor="end">details</text>')
    marker_cy = lane_top + 11                                   # row for ◆ and ▬ markers
    row_ys = [lane_top + 29, lane_top + 43, lane_top + 57]      # text baselines of the value rows

    labels = []        # (x, text, colour, tooltip)
    for tm in ordered:
        if not tm.text.strip():
            continue                                            # blank slides have nothing to measure
        x = X(tm.change_time)
        if tm.delay is None and tm.note == PENDING_NOTE:         # live view: its singing isn't transcribed yet
            tip = f"Slide {tm.slide_index + 1} shown at {_clock(tm.change_time)}: waiting for the singing to be transcribed\n{_flat(tm.text)}"
            labels.append((x, "…", MUTED, tip))
        elif tm.delay is None:
            tip = f"Slide {tm.slide_index + 1} shown at {_clock(tm.change_time)}: {tm.note or 'no timing'}\n{_flat(tm.text)}"
            labels.append((x, "✕", GREY, tip))
        else:
            label = classify(tm.delay, late_s, early_s)
            sign = "−" if tm.delay < 0 else "+"
            text = ("≤" if tm.initial else "") + f"{sign}{abs(tm.delay):.1f}"
            tip = (f"Slide {tm.slide_index + 1}" + (f" · {tm.group}" if tm.group else "") +
                   f" · at {_clock(tm.change_time)} · {tm.delay:+.1f}s {label.lower()}\n{_flat(tm.text)}")
            labels.append((x, text, COLOR[label], tip))

    row_right = [-1e9] * len(row_ys)
    for x, text, col, tip in sorted(labels, key=lambda l: l[0]):
        w = len(text) * 6.4 + 4
        left = max(ml + 2, x - w / 2)
        row = next((i for i, r in enumerate(row_right) if left >= r + 2), None)
        if row is None:                                         # crowded: use the row that frees up soonest
            row = min(range(len(row_ys)), key=lambda i: row_right[i])
        row_right[row] = left + w
        ty = row_ys[row]
        out.append(f'<g><title>{escape(tip)}</title>'
                   f'<line x1="{x:.1f}" y1="{lane_top + 2}" x2="{x:.1f}" y2="{ty - 10}" stroke="{col}" stroke-width="1" opacity="0.45"/>'
                   f'<text x="{left + w / 2:.1f}" y="{ty}" fill="{col}" font-size="10.5" font-weight="700" text-anchor="middle">{escape(text)}</text></g>')

    for u in unshown:
        x = X(u.onset_time)
        tip = f"Sung at {_clock(u.onset_time)} but this slide was never shown ({u.group or 'slide'})\n{_flat(u.text)}"
        out.append(f'<g><title>{escape(tip)}</title>'
                   f'<path d="M{x:.1f} {marker_cy - 6:.1f}L{x + 6:.1f} {marker_cy:.1f}L{x:.1f} {marker_cy + 6:.1f}L{x - 6:.1f} {marker_cy:.1f}Z" fill="{ORANGE}"/></g>')
    for h in heard_other:
        x1, x2 = X(h.onset_time), max(X(h.end_time), X(h.onset_time) + 8)
        tip = f"Words heard at {_clock(h.onset_time)} that no slide matched\n“{_flat(h.text)}”"
        out.append(f'<g><title>{escape(tip)}</title><rect x="{x1:.1f}" y="{marker_cy - 4:.1f}" width="{x2 - x1:.1f}" height="8" rx="3" fill="{BLUE}"/></g>')

    # lane legend
    lgx = ml + plot_w
    items = [("values = delay in seconds (− early, + late)", MUTED), ("✕ words not heard", GREY),
             ("◆ sung, slide not shown", ORANGE), ("▬ words heard, no slide", BLUE)]
    for label, col in reversed(items):
        lgx -= len(label) * 6.0 + 16
        out.append(f'<text x="{lgx:.1f}" y="{height - 8}" fill="{col}" font-size="10.5">{escape(label)}</text>')

    out.append("</svg>")
    return "\n".join(out)


def render_operator_trend(
    operator: str,
    per_song: Sequence[dict],
    late_s: float = 0.75,
    early_s: Optional[float] = None,
    width: int = 1000,
    height: int = 340,
) -> str:
    """
    One operator over time: a boxy step graph with one box per recorded song (oldest first)
    at that song's AVERAGE delay — same colours and orientation as the song graph (early up,
    late down). The lane under it prints each song's average delay and % on time.
    `per_song`: dicts with name, started_at, measured, mean_delay, pct_on_time.
    """
    from datetime import datetime

    if early_s is None:
        early_s = late_s
    ml, mr, mt = 58, 18, 54
    axis_h, lane_h, legend_h = 30, 40, 20
    plot_h = height - mt - axis_h - lane_h - legend_h
    plot_w = width - ml - mr
    lane_top = mt + plot_h + axis_h

    n = max(1, len(per_song))
    vals = [abs(p["mean_delay"]) for p in per_song if p.get("mean_delay") is not None]
    yr = max(3.0, math.ceil(max(vals + [0.0]) + 0.5), 2.0 * late_s, 2.0 * early_s)
    slot = plot_w / n

    def Y(d: float) -> float:
        d = max(-yr, min(yr, d))
        return mt + (yr + d) / (2.0 * yr) * plot_h

    out: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'font-family="Segoe UI, Roboto, Helvetica, Arial, sans-serif" role="img" aria-label="Trend for {escape(operator)}">',
        f'<rect width="{width}" height="{height}" rx="10" fill="{CARD}"/>',
        f'<text x="{ml}" y="24" fill="{TEXT}" font-size="15" font-weight="700">{escape(operator)} — average delay per song</text>',
        f'<text x="{ml}" y="42" fill="{MUTED}" font-size="11">Each box is one song (oldest first) at its average delay · up = early, down = late · '
        f'on time = {"within ±%.2fs" % late_s if abs(early_s - late_s) < 1e-9 else "≤%.2fs late, ≤%.2fs early" % (late_s, early_s)}</text>',
    ]
    y_early, y_late = Y(-early_s), Y(late_s)
    out.append(f'<rect x="{ml}" y="{mt}" width="{plot_w}" height="{y_early - mt:.1f}" fill="{YELLOW}" opacity="0.08"/>')
    out.append(f'<rect x="{ml}" y="{y_early:.1f}" width="{plot_w}" height="{y_late - y_early:.1f}" fill="{GREEN}" opacity="0.13"/>')
    out.append(f'<rect x="{ml}" y="{y_late:.1f}" width="{plot_w}" height="{mt + plot_h - y_late:.1f}" fill="{RED}" opacity="0.08"/>')
    ystep = 1 if yr <= 6 else (2 if yr <= 12 else 5)
    d = -math.floor(yr / ystep) * ystep
    while d <= yr + 1e-9:
        y = Y(d)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + plot_w}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" fill="{MUTED}" font-size="11" text-anchor="end">{d:+g}s</text>')
        d += ystep
    y0 = Y(0)
    out.append(f'<line x1="{ml}" y1="{y0:.1f}" x2="{ml + plot_w}" y2="{y0:.1f}" stroke="{TEXT}" stroke-width="1" stroke-dasharray="5 4" opacity="0.55"/>')
    out.append(f'<text x="{ml + 6}" y="{mt + 14}" fill="{YELLOW}" font-size="10" font-weight="700" opacity="0.9">▲ EARLY</text>')
    out.append(f'<text x="{ml + 6}" y="{mt + plot_h - 6}" fill="{RED}" font-size="10" font-weight="700" opacity="0.9">▼ LATE</text>')

    out.append(f'<rect x="{ml}" y="{lane_top}" width="{plot_w}" height="{lane_h - 4}" rx="4" fill="#131c28"/>')
    out.append(f'<text x="{ml - 8}" y="{lane_top + 15}" fill="{MUTED}" font-size="10" text-anchor="end">avg</text>')
    out.append(f'<text x="{ml - 8}" y="{lane_top + 30}" fill="{MUTED}" font-size="10" text-anchor="end">on time</text>')

    prev = None            # (y, colour, was_measured) of the previous box, for the risers
    label_every = 1 if n <= 14 else int(math.ceil(n / 14))
    for i, p in enumerate(per_song):
        x1, x2 = ml + i * slot + 3, ml + (i + 1) * slot - 3
        cx = (x1 + x2) / 2
        when = datetime.fromtimestamp(p.get("started_at", 0)).strftime("%b %d")
        tip = f"{p.get('name', '')} · {when} · "
        if p.get("mean_delay") is None:
            tip += "no slide changes could be measured"
            out.append(f'<g><title>{escape(tip)}</title><path d="M{cx - 6:.1f} {y0 - 6:.1f}L{cx + 6:.1f} {y0 + 6:.1f}M{cx + 6:.1f} {y0 - 6:.1f}L{cx - 6:.1f} {y0 + 6:.1f}" '
                       f'stroke="{GREY}" stroke-width="2.5" stroke-linecap="round"/></g>')
            out.append(f'<text x="{cx:.1f}" y="{lane_top + 15}" fill="{GREY}" font-size="10.5" font-weight="700" text-anchor="middle">✕</text>')
            prev = None
        else:
            cls = classify(p["mean_delay"], late_s, early_s)
            col, y = COLOR[cls], Y(p["mean_delay"])
            tip += f"avg {p['mean_delay']:+.1f}s ({cls.lower()}) · {p['pct_on_time']:.0f}% on time · {p['measured']} slide changes"
            if prev is not None and abs(prev[0] - y) > 0.6:
                gid = f"t{i}"
                out.append(f'<defs><linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="0" y1="{prev[0]:.1f}" x2="0" y2="{y:.1f}">'
                           f'<stop offset="0" stop-color="{prev[1]}"/><stop offset="1" stop-color="{col}"/></linearGradient></defs>')
                out.append(f'<line x1="{x1:.1f}" y1="{prev[0]:.1f}" x2="{x1:.1f}" y2="{y:.1f}" stroke="url(#{gid})" stroke-width="3"/>')
            out.append(f'<g><title>{escape(tip)}</title><line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="5"/>'
                       f'<circle cx="{x1:.1f}" cy="{y:.1f}" r="5" fill="{col}" stroke="{CARD}" stroke-width="1.5"/></g>')
            sign = "−" if p["mean_delay"] < 0 else "+"
            out.append(f'<text x="{cx:.1f}" y="{lane_top + 15}" fill="{col}" font-size="10.5" font-weight="700" text-anchor="middle">{sign}{abs(p["mean_delay"]):.1f}</text>')
            out.append(f'<text x="{cx:.1f}" y="{lane_top + 30}" fill="{TEXT}" font-size="10" text-anchor="middle">{p["pct_on_time"]:.0f}%</text>')
            prev = (y, col, True)
        if i % label_every == 0:
            out.append(f'<text x="{cx:.1f}" y="{mt + plot_h + 18}" fill="{MUTED}" font-size="10.5" text-anchor="middle">{escape(when)}</text>')
    out.append(f'<text x="{ml + plot_w}" y="{height - 6}" fill="{MUTED}" font-size="10.5" text-anchor="end">values = average delay in seconds (− early, + late) · hover a box for the song</text>')
    out.append("</svg>")
    return "\n".join(out)
