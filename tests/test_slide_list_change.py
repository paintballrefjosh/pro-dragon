"""
ProPresenter's slide list can change while a song is being recorded (an operator deletes a blank slide
or a group). Every slide index after the edit then means a different slide. Pro Dragon must notice,
reload the list, keep what each earlier slide change showed, and measure the right words.
"""

import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.song_runner import SongData, analyze_songdata, load_song  # noqa: E402
from analysis.song_timing import SlideEvent, Word, analyze_song  # noqa: E402
from audio.capture import FileSource  # noqa: E402
from library import SongLibrary  # noqa: E402
from propresenter.client import ProPresenterClient  # noqa: E402
from session import Session  # noqa: E402
from test_session_e2e import LYRICS, FakeWhisper, write_test_wav  # noqa: E402

S = {"slides": [], "index": 0, "cues": None}          # the fake ProPresenter's live state; cues=None -> len(slides)


class Pro7(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        pid = {"uuid": "song-1", "name": "Edited Song", "index": 0}
        if self.path == "/version":
            self._send({"name": "ProPresenter", "host_description": "7.fake"})
        elif self.path == "/v1/presentation/active":
            self._send({"presentation": {"id": pid, "groups": [
                {"name": "Song", "slides": [{"text": t, "label": ""} for t in S["slides"]]}]}})
        elif self.path == "/v1/presentation/slide_index":
            cues = len(S["slides"]) if S["cues"] is None else S["cues"]
            self._send({"presentation_index": {"index": S["index"], "presentation_id": pid, "total_cues": cues}})
        else:
            self.send_response(404)
            self.end_headers()


def start_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Pro7)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def wait_for(cond, timeout=6.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


# --------------------------------------------------------------------------- analysis

class _Slide:
    def __init__(self, index, text, group=""):
        self.index, self.text, self.group = index, text, group


def test_event_snapshots_beat_a_stale_slide_list_and_survive_saving():
    # saved list says index 1 is LYRICS[0]; but the event recorded that the slide was really LYRICS[1]
    slides = [_Slide(0, ""), _Slide(1, LYRICS[0], "Verse 1"), _Slide(2, LYRICS[1], "Verse 1")]
    words = [Word(t, 10.0 + 0.4 * i, 10.3 + 0.4 * i) for i, t in enumerate(LYRICS[1].lower().split())]
    ev = SlideEvent(ts=100.0 + 10.0, index=1, text=LYRICS[1], group="Chorus")
    res = analyze_song(slides, [ev], words, song_start_ts=100.0)
    t = res.timings[0]
    assert t.text == LYRICS[1] and t.group == "Chorus"          # the snapshot, not slides[1]
    assert t.delay is not None and abs(t.delay) < 0.4
    # the same event without a snapshot is looked up in the list (older songs): wrong words -> not measured
    old = analyze_song(slides, [SlideEvent(ts=110.0, index=1)], words, song_start_ts=100.0).timings[0]
    assert old.text == LYRICS[0] and old.delay is None

    # snapshots and notes round-trip through the saved json
    song = SongData("S", "u", [{"index": 1, "text": LYRICS[0], "group": "", "label": ""}], 100.0, 120.0, 100.0,
                    [ev, SlideEvent(105.0, 0)], notes=["the list changed"])
    data = song.to_json(0, 0.75)
    assert data["events"][0]["text"] == LYRICS[1] and "text" not in data["events"][1] and data["notes"] == ["the list changed"]
    back = SongData.from_json(data, Path("."))
    assert back.events[0].text == LYRICS[1] and back.events[1].text is None and back.notes == ["the list changed"]


# --------------------------------------------------------------------------- client detection

def test_client_notices_a_changed_slide_count_once_and_reloads():
    server = start_server()
    S.update(slides=["", "", LYRICS[0], LYRICS[1]], index=0, cues=None)
    updates, mismatches = [], []
    c = ProPresenterClient("127.0.0.1", server.server_address[1], poll_interval=0.05)
    try:
        assert c.check_connection()
        c.on_slides_updated = lambda old, new, since: updates.append((len(old.slides), len(new.slides), since))
        c.on_slides_mismatch = lambda cues, loaded: mismatches.append((cues, loaded))
        c.start_polling()
        assert wait_for(lambda: c.current_presentation is not None and len(c.current_presentation.slides) == 4)
        time.sleep(0.3)
        assert updates == [] and mismatches == []                        # nothing happens while the list is stable

        t_edit = time.time()
        S["slides"] = ["", LYRICS[0], LYRICS[1]]                         # someone deletes a blank slide
        assert wait_for(lambda: len(updates) == 1)
        old_n, new_n, since = updates[0]
        assert (old_n, new_n) == (4, 3) and abs(since - t_edit) < 1.0
        assert [s.text for s in c.current_presentation.slides] == ["", LYRICS[0], LYRICS[1]]
        time.sleep(0.5)
        assert len(updates) == 1 and mismatches == []                    # reloaded once, not on every poll

        S["cues"] = 9                                                    # now ProPresenter's count can't be reconciled
        assert wait_for(lambda: len(mismatches) == 1)
        time.sleep(0.6)
        assert mismatches == [(9, 3)] and len(updates) == 1              # warned once, no reload storm
    finally:
        c.disconnect()
        server.shutdown()


# --------------------------------------------------------------------------- the whole thing

def test_song_recorded_across_an_edit_is_measured_against_the_right_words(tmp_path):
    server = start_server()
    # blank, blank(an extra one), line0, line1, line2  -> the operator deletes one blank during the song
    S.update(slides=["", "", LYRICS[0], LYRICS[1], LYRICS[2]], index=0, cues=None)    # live on the title slide
    wav = tmp_path / "in.wav"
    write_test_wav(wav, seconds=14)
    # the singer starts line0 at 3.0 s, line1 at 6.5 s, line2 at 10.0 s of the song
    whisper = FakeWhisper([(0, 3.0), (1, 6.5), (2, 10.0)])
    cfg = {"report": {"output_dir": str(tmp_path / "reports")}, "timing": {"on_time_window_s": 0.75, "min_match_confidence": 0.25}}
    pro7 = ProPresenterClient("127.0.0.1", server.server_address[1], poll_interval=0.05)
    session = None
    try:
        assert pro7.check_connection()
        session = Session(cfg, pro7, lambda cb: FileSource(str(wav), on_frame=cb), lambda: whisper, dashboard=None,
                          offset_ms=0, console=False)
        pro7.on_slide_changed = session._on_slide_changed
        pro7.on_presentation_changed = session._on_presentation_changed
        pro7.on_slides_updated = session._on_slides_updated
        pro7.on_slides_mismatch = session._on_slides_mismatch
        pro7.start_polling()
        session.start()
        time.sleep(0.5)
        assert session.start_song()
        t0 = time.time()

        time.sleep(max(0, t0 + 1.5 - time.time()))
        S["slides"] = ["", LYRICS[0], LYRICS[1], LYRICS[2]]               # the edit: indexes after 0 now mean something else
        time.sleep(max(0, t0 + 3.2 - time.time()))
        S["index"] = 1                                                    # operator goes to line0 (0.2 s late)
        time.sleep(max(0, t0 + 6.5 - time.time()))
        S["index"] = 2                                                    # line1, on time
        time.sleep(max(0, t0 + 9.6 - time.time()))
        S["index"] = 3                                                    # line2, 0.4 s early
        time.sleep(max(0, t0 + 12.0 - time.time()))
        assert session.stop_song()
        for t in session._analysis_threads:
            t.join(timeout=20)

        path = sorted(session.out_dir.glob("song_*.json"))[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        notes = " ".join(data["notes"])
        assert "5 → 4" in notes and "edited while recording" in notes
        assert path.with_suffix(".txt").read_text(encoding="utf-8").count("NOTE:") == 1

        by_text = {t["text"]: t for t in data["timings"] if t["text"]}
        assert set(by_text) == {LYRICS[0], LYRICS[1], LYRICS[2]}
        for lyric, want in ((LYRICS[0], 0.2), (LYRICS[1], 0.0), (LYRICS[2], -0.4)):
            d = by_text[lyric]["delay"]
            assert d is not None and abs(d - want) < 0.35, (lyric, d, want)
        # the first event (blank slide live at the start) kept its meaning from before the edit
        assert data["events"][0]["text"] == ""
        assert data["events"][-1]["text"] == LYRICS[2]
    finally:
        pro7.disconnect()
        server.shutdown()


# --------------------------------------------------------------------------- repairing an old song

def test_fixing_an_old_song_with_the_correct_slide_list(tmp_path):
    """The user's real case: a song saved with an extra blank slide in its list (no snapshots)."""
    root = tmp_path / "reports"
    d = root / "session_1"
    d.mkdir(parents=True)
    wrong = ["", "", LYRICS[0], LYRICS[1], LYRICS[2]]                     # list as loaded at Start (5), 2 blanks
    right = ["", LYRICS[0], LYRICS[1], LYRICS[2]]                         # what ProPresenter really had after the edit
    start = 1000.0
    words = []
    for lyric, onset in ((0, 3.0), (1, 6.5), (2, 10.0)):
        words += [(t, onset + 0.4 * i, onset + 0.4 * i + 0.3, 1.0) for i, t in enumerate(LYRICS[lyric].lower().split())]
    song = SongData("Old", "song-1", [{"index": i, "text": t, "group": "", "label": ""} for i, t in enumerate(wrong)],
                    start, start + 14, start, [SlideEvent(start, 0, True), SlideEvent(start + 3.2, 1),
                                               SlideEvent(start + 6.5, 2), SlideEvent(start + 9.6, 3)], words=words)
    cfg = {"on_time_window_s": 0.75, "min_match_confidence": 0.25, "max_match_window_s": 20.0}
    analyze_songdata(song, None, cfg, 0.0)
    from analysis.song_runner import save_song
    save_song(song, d, 1, 0.0, 0.75, stem="song_01_old")
    before = {t.text: t.delay for t in song.timings if t.text}
    assert all(v is None or abs(v) > 3 for v in before.values())          # wrong list: nothing lines up

    lib = SongLibrary(lambda: str(root))
    out = lib.remap_slides("session_1/song_01_old.json", [{"index": i, "text": t, "group": "", "label": ""} for i, t in enumerate(right)],
                           cfg, 0.0)
    got = {t["text"]: t["delay"] for t in out["timings"] if t["text"]}
    for lyric, want in ((LYRICS[0], 0.2), (LYRICS[1], 0.0), (LYRICS[2], -0.4)):
        assert got[lyric] is not None and abs(got[lyric] - want) < 0.35, (lyric, got)
    saved = json.loads((d / "song_01_old.json").read_text(encoding="utf-8"))
    assert len(saved["slides_before_remap"]) == 5 and len(saved["slides"]) == 4        # the original list is kept
    assert "5 → 4" in saved["notes"][0] and out["slides_remapped"]
    again = lib.remap_slides("session_1/song_01_old.json", saved["slides"], cfg, 0.0)  # fixing twice keeps the FIRST original
    assert len(json.loads((d / "song_01_old.json").read_text(encoding="utf-8"))["slides_before_remap"]) == 5
    assert len(again["notes"]) == 2
