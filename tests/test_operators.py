"""Per-operator tracking: pooled statistics, re-assigning an operator on a saved song, the
operator API, the trend graph, and the text report."""

import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.operators import UNASSIGNED, aggregate, delay_stats, format_operator_report  # noqa: E402
from analysis.song_runner import SongData, save_song  # noqa: E402
from analysis.song_timing import SlideEvent, SlideTiming, UnshownLyric  # noqa: E402
from app import Controller  # noqa: E402
from gui.server import DashboardServer  # noqa: E402
from library import LibraryError, SongLibrary  # noqa: E402
from settings import SettingsStore  # noqa: E402
from test_gui_api import free_port  # noqa: E402

H = {"X-ProDragon": "1"}
NS = {"s": "http://www.w3.org/2000/svg"}


def make_song(root: Path, session: str, number: int, name: str, operator: str, delays, start=1_000_000.0,
              not_heard: int = 0, unshown: int = 0):
    """Save a real song (json/txt/svg) with the given per-slide delays."""
    timings = []
    for i, d in enumerate(delays):
        timings.append(SlideTiming(ordinal=i + 1, slide_index=i, group="Verse", text=f"line {i}", change_time=10.0 * (i + 1),
                                   initial=False, onset_time=None if d is None else 10.0 * (i + 1) - d, delay=d,
                                   confidence=1.0, note="" if d is not None else "not heard"))
    for k in range(not_heard):
        timings.append(SlideTiming(ordinal=len(timings) + 1, slide_index=50 + k, group="X", text="never heard", change_time=999.0,
                                   initial=False, onset_time=None, delay=None, confidence=0.0, note="not heard"))
    song = SongData(name=name, presentation_uuid="u", slides=[{"index": i, "text": f"line {i}", "group": "Verse", "label": ""}
                                                              for i in range(len(delays))],
                    song_start_ts=start, song_end_ts=start + 200, audio_start_ts=start,
                    events=[SlideEvent(start + 10.0 * (i + 1), i) for i in range(len(delays))],
                    words=[], timings=timings,
                    unshown=[UnshownLyric(5.0, 1, "Chorus", "sung line", 0.9) for _ in range(unshown)], operator=operator)
    return save_song(song, root / session, number, 0, 0.75, 0.75)


def test_pooled_stats_margins_and_grouping():
    s = delay_stats([-1.5, 0.0, 0.6, 2.0], 0.5, 2.0)             # 0.5 late / 2.0 early margins
    assert s["measured"] == 4 and s["n_early"] == 0 and s["n_on_time"] == 2 and s["n_late"] == 2
    assert delay_stats([], 0.75)["mean_delay"] is None

    songs = [
        {"id": "a", "name": "A", "started_at": 3, "operator": "Sam", "duration": 100,
         "timings": [{"delay": 0.0, "text": "x"}, {"delay": 2.0, "text": "y"}, {"delay": None, "text": "z"},
                     {"delay": -9.0, "text": "w", "initial": True}], "unshown": [{}], "heard_other": []},
        {"id": "b", "name": "B", "started_at": 1, "operator": "Sam", "duration": 100,
         "timings": [{"delay": -2.0, "text": "x"}], "unshown": [], "heard_other": [{}, {}]},
        {"id": "c", "name": "C", "started_at": 2, "operator": "", "duration": 100, "timings": [{"delay": 0.1, "text": "x"}],
         "unshown": [], "heard_other": []},
        {"id": "d", "name": "D", "started_at": 4, "operator": "alex", "duration": 100, "timings": [], "unshown": [], "heard_other": []},
    ]
    out = aggregate(songs, 0.75)
    assert [e["name"] for e in out] == ["alex", "Sam", UNASSIGNED]            # alphabetical (case-insensitive), unassigned last
    sam = out[1]
    assert sam["songs"] == 2 and sam["measured"] == 3                         # initial + not-heard excluded from the pool
    assert abs(sam["mean_delay"] - 0.0) < 1e-9 and sam["not_heard"] == 1 and sam["not_shown"] == 1 and sam["no_slide"] == 2
    assert [p["id"] for p in sam["per_song"]] == ["b", "a"]                    # oldest first
    assert out[0]["measured"] == 0 and out[0]["mean_delay"] is None            # operator with nothing measurable
    assert aggregate(songs, 3.0)[1]["pct_on_time"] == 100.0                    # margins applied at read time


def test_report_text_lists_every_operator():
    songs = [{"id": "a", "name": "Song A", "started_at": 1_000_000, "operator": "Sam", "duration": 1,
              "timings": [{"delay": 1.2, "text": "x"}], "unshown": [], "heard_other": []}]
    text = format_operator_report(aggregate(songs, 0.75), 0.75)
    assert "OPERATOR REPORT" in text and "Sam" in text and "Song A" in text and "+1.2s" in text
    assert "No saved songs" in format_operator_report([], 0.75)


def test_library_set_operator_regenerates_files_and_validates():
    tmp = Path(tempfile.mkdtemp())
    try:
        jp = make_song(tmp, "session_x", 1, "Abide", "", [0.1, -1.0, 2.0])
        lib = SongLibrary(lambda: str(tmp))
        sid = lib.id_for(jp)
        assert lib.detail(sid)["operator"] == ""
        d = lib.set_operator(sid, "  Sam   Smith ")
        assert d["operator"] == "Sam Smith"                                    # whitespace normalized
        assert [t["delay"] for t in d["timings"]] == [0.1, -1.0, 2.0]          # analysis untouched
        assert "Operator: Sam Smith" in jp.with_suffix(".txt").read_text(encoding="utf-8")
        assert "Operator: Sam Smith" in jp.with_suffix(".svg").read_text(encoding="utf-8")
        assert json.loads(jp.read_text(encoding="utf-8"))["operator"] == "Sam Smith"
        assert lib.list()[0]["operator"] == "Sam Smith" and lib.operator_names() == ["Sam Smith"]
        for bad in ("x" * 61, "bad\x00name", "esc\x1bname"):                  # too long / control characters
            try:
                lib.set_operator(sid, bad)
                assert False, bad
            except LibraryError:
                pass
        assert lib.set_operator(sid, "tab\tname")["operator"] == "tab name"    # plain whitespace is just normalized
        try:
            lib.set_operator("../evil.json", "x")
            assert False
        except LibraryError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_operator_api_and_trend_graph():
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "config.yaml").write_text(f"report:\n  output_dir: {(tmp / 'reports').as_posix()}\n", encoding="utf-8")
        root = tmp / "reports"
        make_song(root, "session_a", 1, "Song 1", "Sam", [0.2, 0.5, -0.3], start=1_000_000)
        make_song(root, "session_a", 2, "Song 2", "Sam", [2.0, 1.5, 2.5], start=1_001_000, not_heard=1, unshown=2)
        make_song(root, "session_b", 1, "Song 3", "Alex", [-1.8, -2.0], start=1_002_000)
        jp4 = make_song(root, "session_b", 2, "Song 4", "", [0.0, 0.1], start=1_003_000)

        settings = SettingsStore(str(tmp / "config.yaml"))
        dash = DashboardServer("127.0.0.1", free_port())
        ctrl = Controller(settings, dash)
        dash.start()
        base = f"http://127.0.0.1:{dash.port}"
        try:
            ops = requests.get(base + "/api/operators", headers=H).json()["operators"]
            by = {o["name"]: o for o in ops}
            assert list(by) == ["Alex", "Sam", UNASSIGNED]
            assert by["Sam"]["songs"] == 2 and by["Sam"]["measured"] == 6 and by["Sam"]["not_heard"] == 1 and by["Sam"]["not_shown"] == 2
            assert by["Alex"]["pct_early"] == 100.0 and by["Sam"]["pct_late"] > 0

            # margins in the query change the labels, server-side
            wide = {o["name"]: o for o in requests.get(base + "/api/operators", params={"late": 3, "early": 3}, headers=H).json()["operators"]}
            assert wide["Sam"]["pct_on_time"] == 100.0 and wide["Alex"]["pct_on_time"] == 100.0
            assert requests.get(base + "/api/operators", params={"late": "x"}, headers=H).status_code == 400
            assert requests.get(base + "/api/operators", params={"late": 99}, headers=H).status_code == 400

            g = requests.get(base + "/api/operator/graph.svg", params={"name": "Sam"}, headers=H)
            assert g.status_code == 200 and g.headers["Content-Type"].startswith("image/svg+xml")
            root_el = ET.fromstring(g.text)
            assert len([l for l in root_el.findall(".//s:line", NS) if l.get("stroke-width") == "5"]) == 2     # one box per song
            assert requests.get(base + "/api/operator/graph.svg", params={"name": "Nobody"}, headers=H).status_code == 404

            # assigning the unassigned song (note: this route must not be mistaken for /api/song/{cmd})
            r = requests.post(base + "/api/song/operator", json={"id": ctrl.library.id_for(jp4), "operator": "Pat"}, headers=H)
            assert r.status_code == 200 and r.json()["detail"]["operator"] == "Pat"
            names = [o["name"] for o in requests.get(base + "/api/operators", headers=H).json()["operators"]]
            assert names == ["Alex", "Pat", "Sam"]
            assert requests.post(base + "/api/song/operator", json={"id": "../x.json", "operator": "Z"}, headers=H).status_code == 400
            assert requests.post(base + "/api/song/operator", json={"id": ctrl.library.id_for(jp4), "operator": "y" * 99}, headers=H).status_code == 400
            assert requests.post(base + "/api/song/operator", json={}, headers={}).status_code == 403     # CSRF guard

            # current operator
            assert requests.post(base + "/api/operator", json={"name": " Dana "}, headers=H).json()["message"] == "Dana"
            assert ctrl.operator == "Dana" and "Dana" in ctrl.snapshot()["operators"]
            assert requests.post(base + "/api/operator", json={"name": "z" * 70}, headers=H).status_code == 400
        finally:
            ctrl.shutdown()
            dash.stop()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
