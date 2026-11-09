"""Deleting saved songs: their files go, nothing else is touched, bad ids are refused, and the
GUI API behaves (header guard, validation, state clean-up)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from library import SongLibrary  # noqa: E402
from test_gui_api import Env, H  # noqa: E402


def make_song(root: Path, session: str, stem: str, duration=10.0, operator="Sam"):
    d = root / session
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.json").write_text(json.dumps({
        "name": stem, "operator": operator, "song_start_ts": 1000.0, "song_end_ts": 1000.0 + duration,
        "audio_file": f"{stem}.wav", "timings": [], "summary": {"measured": 0}}), encoding="utf-8")
    for ext in (".txt", ".svg", ".wav"):
        (d / f"{stem}{ext}").write_bytes(b"x")
    return f"{session}/{stem}.json"


def test_delete_removes_one_songs_files_and_tidies_empty_sessions(tmp_path):
    root = tmp_path / "reports"
    a = make_song(root, "session_1", "song_01_a")
    b = make_song(root, "session_1", "song_02_b")
    c = make_song(root, "session_2", "song_01_c")
    (root / "session_2" / "session_report.txt").write_text("report", encoding="utf-8")
    outside = tmp_path / "keep.txt"
    outside.write_text("mine", encoding="utf-8")
    lib = SongLibrary(lambda: str(root))

    res = lib.delete([a])
    assert res["deleted"] == [a] and res["failed"] == {} and res["files"] == 4
    assert sorted(p.name for p in (root / "session_1").iterdir()) == ["song_02_b.json", "song_02_b.svg", "song_02_b.txt", "song_02_b.wav"]
    assert {s["id"] for s in lib.list()} == {b, c}

    res = lib.delete([c, c])                                   # duplicates are only handled once
    assert res["deleted"] == [c]
    assert not (root / "session_2").exists()                   # no songs left: folder (and its report) removed
    assert (root / "session_1").exists() and outside.read_text(encoding="utf-8") == "mine"


def test_delete_refuses_bad_ids_and_never_leaves_the_reports_folder(tmp_path):
    root = tmp_path / "reports"
    good = make_song(root, "session_1", "song_01_a")
    victim = tmp_path / "victim.json"
    victim.write_text("{}", encoding="utf-8")
    secret = tmp_path / "secret.wav"
    secret.write_bytes(b"keep me")
    lib = SongLibrary(lambda: str(root))

    res = lib.delete(["../victim.json", "session_1/nope.json", "", "session_1/song_01_a.txt"])
    assert res["deleted"] == [] and len(res["failed"]) == 4 and victim.exists()

    # a song whose json points its audio_file outside the folder must only lose a file named in ITS folder
    p = root / "session_1" / "song_01_a.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    data["audio_file"] = "../../secret.wav"
    p.write_text(json.dumps(data), encoding="utf-8")
    assert lib.delete([good])["deleted"] == [good]
    assert secret.exists() and victim.exists()


def test_gui_api_delete_flow():
    with Env() as e:
        root = e.tmp / "reports"
        keep = make_song(root, "session_1", "song_01_keep", duration=120)
        tiny = make_song(root, "session_1", "song_02_tiny", duration=3)
        assert {s["id"] for s in e.get("/api/songs").json()["songs"]} == {keep, tiny}

        assert e.post("/api/songs/delete", {"ids": [tiny]}, headers={}).status_code == 403     # same guard as other writes
        assert e.post("/api/songs/delete", {"ids": "nope"}).status_code == 400
        assert e.post("/api/songs/delete", {"ids": []}).status_code == 400
        assert e.post("/api/songs/delete", {"ids": ["../../etc/passwd.json"]}).status_code == 400

        ok = e.post("/api/songs/delete", {"ids": [tiny, "session_1/ghost.json"]})
        assert ok.status_code == 200
        body = ok.json()
        assert body["deleted"] == [tiny] and "session_1/ghost.json" in body["failed"]
        assert [s["id"] for s in e.get("/api/songs").json()["songs"]] == [keep]
        assert not list((root / "session_1").glob("song_02_tiny*")) and (root / "session_1" / "song_01_keep.wav").exists()
