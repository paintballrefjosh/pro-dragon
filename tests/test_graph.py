import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.song_timing import HeardWords, SlideTiming, UnshownLyric, classify, margins_from_cfg, summarize  # noqa: E402
from report.graph import render_song_graph, GREEN, YELLOW, RED  # noqa: E402

NS = {"s": "http://www.w3.org/2000/svg"}


def tm(i, change, delay, text="Some words\nmore words", initial=False, note=""):
    return SlideTiming(ordinal=i + 1, slide_index=i, group="Verse", text=text, change_time=change, initial=initial,
                       onset_time=None if delay is None else change - delay, delay=delay,
                       confidence=1.0 if delay is not None else 0.0, note=note)


def test_classify_and_summary_with_separate_margins():
    assert classify(-0.7, 0.75) == "ON TIME" and classify(-1.0, 0.75) == "EARLY" and classify(1.0, 0.75) == "LATE"
    # early is the ideal: tolerate up to 2s early but only 0.5s late
    assert classify(-1.5, 0.5, 2.0) == "ON TIME" and classify(0.6, 0.5, 2.0) == "LATE" and classify(-2.1, 0.5, 2.0) == "EARLY"
    assert margins_from_cfg({"on_time_window_s": 0.75}) == (0.75, 0.75)
    assert margins_from_cfg({"on_time_window_s": 0.75, "on_time_early_s": 2}) == (2.0, 0.75)
    ts = [tm(0, 5, -1.5), tm(1, 10, 0.6), tm(2, 15, 0.0), tm(3, 20, None)]
    s = summarize(ts, 0.5, 2.0)
    assert s["measured"] == 3 and abs(s["pct_late"] - 100 / 3) < 1e-6 and abs(s["pct_on_time"] - 200 / 3) < 1e-6
    assert s["early_window_s"] == 2.0


def test_graph_is_valid_svg_with_colors_and_problem_markers():
    timings = [tm(0, 5, 0.1), tm(1, 15, -2.0), tm(2, 25, 3.5), tm(3, 35, None, note="not heard"),
               tm(4, 45, 0.3), tm(5, 55, None, text="", note="no lyrics")]
    svg = render_song_graph("Test <Song> & co", 70, timings,
                            unshown=[UnshownLyric(30, 2, "Chorus", "Sung line", 0.9)],
                            heard_other=[HeardWords(50, 54, "Welcome everyone please sit", 0.9)],
                            late_s=0.75)
    root = ET.fromstring(svg)                                   # well-formed (names/text escaped)
    lines = root.findall(".//s:line", NS)
    colors = [l.get("stroke") for l in lines if l.get("stroke-width") == "4"]      # the flat "boxes"
    assert colors == [GREEN, YELLOW, RED, GREEN]                                    # one box per measured slide
    # boxes are flat (boxy), not diagonal
    for l in lines:
        if l.get("stroke-width") == "4":
            assert l.get("y1") == l.get("y2")
    assert "Test &lt;Song&gt;" in svg
    assert ">details<" in svg and ">problems<" not in svg and "words not heard" in svg   # lane renamed + legend present
    titles = [t.text for t in root.findall(".//s:title", NS)]
    assert any("never shown" in t for t in titles) and any("Welcome everyone" in t for t in titles)
    assert any("not heard" in t for t in titles)                                    # unmeasured slide called out
    assert "nan" not in svg.lower()


def test_early_is_above_the_zero_line_and_late_below():
    svg = render_song_graph("Orientation", 60, [tm(0, 5, -2.0), tm(1, 15, 0.0), tm(2, 25, 2.5)])
    root = ET.fromstring(svg)
    boxes = {l.get("stroke"): float(l.get("y1")) for l in root.findall(".//s:line", NS) if l.get("stroke-width") == "4"}
    zero = next(float(l.get("y1")) for l in root.findall(".//s:line", NS) if l.get("stroke-dasharray") == "5 4")
    assert boxes[YELLOW] < zero            # early: above the 0s line (smaller y = higher on screen)
    assert abs(boxes[GREEN] - zero) < 0.5  # on time: on the line
    assert boxes[RED] > zero               # late: below the line
    assert svg.index("▲ EARLY") < svg.index("▼ LATE")


def _lane_labels(svg):
    """(x, text, fill, y) of the value labels in the details lane."""
    root = ET.fromstring(svg)
    out = []
    for t in root.findall(".//s:text", NS):
        if t.get("font-weight") == "700" and t.get("font-size") == "10.5" and t.get("text-anchor") == "middle":
            out.append((float(t.get("x")), t.text, t.get("fill"), float(t.get("y"))))
    return out


def test_details_lane_lists_every_delay_at_its_time_without_overlap():
    # 12 slide changes only 1.5 s apart (~17 px on the chart): labels are wider than that, so they must be staggered
    timings = [tm(i, 10 + 1.5 * i, d) for i, d in enumerate([0.1, -1.0, 2.0, 0.4, -0.2, 1.6, 0.0, -2.2, 0.3, 0.9, -0.6, 1.2])]
    timings.append(tm(12, 40, None, note="not heard"))
    svg = render_song_graph("Dense", 80, timings, late_s=0.75)
    labels = _lane_labels(svg)
    texts = [t for _, t, _, _ in labels]
    assert texts == ["+0.1", "−1.0", "+2.0", "+0.4", "−0.2", "+1.6", "+0.0", "−2.2", "+0.3", "+0.9", "−0.6", "+1.2", "✕"]
    colors = {t: c for _, t, c, _ in labels}
    assert colors["+0.1"] == GREEN and colors["−1.0"] == YELLOW and colors["+2.0"] == RED
    assert [x for x, *_ in labels] == sorted(x for x, *_ in labels)            # in time order
    # no two labels on the same row overlap horizontally
    by_row = {}
    for x, t, _, y in labels:
        by_row.setdefault(y, []).append((x - (len(t) * 6.4 + 4) / 2, x + (len(t) * 6.4 + 4) / 2))
    for spans in by_row.values():
        spans.sort()
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert a1 <= b0 + 0.01
    assert len(by_row) >= 2                                                    # actually staggered
    # all labels sit inside the lane, below the time axis
    root = ET.fromstring(svg)
    assert all(y > 300 for *_, y in labels)


def test_details_lane_skips_blank_slides_and_flags_initial():
    svg = render_song_graph("Mixed", 60, [tm(0, 1, -30.0, initial=True), tm(1, 20, 0.2), tm(2, 30, None, text="", note="no lyrics")])
    assert [t for _, t, *_ in _lane_labels(svg)] == ["≤−30.0", "+0.2"]         # blank slide: no label


def test_graph_handles_empty_and_extreme_values():
    render_song_graph("Empty", 0, [])
    ET.fromstring(render_song_graph("Far", 60, [tm(0, 1, -40.0, initial=True), tm(1, 20, 0.2)]))
    ET.fromstring(render_song_graph("Asym", 60, [tm(0, 5, -1.0), tm(1, 20, 0.2)], late_s=0.5, early_s=2.0))


def test_sections_strip_names_each_part_of_the_song_and_tints_by_average_delay():
    def gm(i, change, delay, group):
        t = tm(i, change, delay)
        t.group = group
        return t
    timings = [gm(0, 5, 0.0, "Verse 1"), gm(1, 15, 0.1, "Verse 1"),      # on time
               gm(2, 40, 3.0, "Chorus 1"), gm(3, 50, 2.5, "Chorus 1"),   # late
               gm(4, 70, -3.0, "Verse 2"),                               # early
               gm(5, 90, None, "Bridge")]                                # nothing measured
    svg = render_song_graph("Sections", 110, timings, late_s=0.75, early_s=1.25)
    root = ET.fromstring(svg)
    assert ">sections<" in svg
    groups = {t.text.split(" · ")[0]: t.text for t in root.findall(".//s:title", NS) if " · " in (t.text or "") and "–" in t.text}
    assert set(groups) == {"Verse 1", "Chorus 1", "Verse 2", "Bridge"}
    assert "on time" in groups["Verse 1"] and "late" in groups["Chorus 1"] and "early" in groups["Verse 2"]
    assert "no slide changes measured" in groups["Bridge"]
    assert "Chorus 1" in "".join(t.text or "" for t in root.findall(".//s:text", NS))        # the name is drawn on the strip
    # the strip adds height without squeezing the chart; songs with no group names keep the old size
    plain = [tm(i, 5 + 10 * i, 0.0) for i in range(3)]
    for t in plain:
        t.group = ""
    assert ">sections<" not in render_song_graph("Plain", 60, plain)
    h = lambda s: float(ET.fromstring(s).get("height"))
    assert h(render_song_graph("A", 60, [tm(0, 5, 0.0)])) > h(render_song_graph("B", 60, plain))
    ET.fromstring(render_song_graph("Narrow <&>", 600, [gm(i, 3 * i, 0.0, f"G{i}") for i in range(40)]))   # many tiny sections stay valid
