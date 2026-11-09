"""
The GUI backend end to end over real HTTP: settings (validation, overrides file),
the anti-CSRF header, connecting to a (fake) ProPresenter, a monitoring session fed
by a wav file, start/stop song, song list/detail/graph/files, path-traversal
protection, re-analysis with a different offset, and the WebSocket.
"""

import asyncio
import json
import shutil
import socket
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from http.server import ThreadingHTTPServer
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_session_e2e import STATE, FakePro7, FakeWhisper, write_test_wav  # noqa: E402

from app import Controller  # noqa: E402
from gui.server import DashboardServer  # noqa: E402
from settings import SettingsStore  # noqa: E402

H = {"X-ProDragon": "1"}


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Env:
    def __enter__(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pro7_server = ThreadingHTTPServer(("127.0.0.1", 0), FakePro7)
        threading.Thread(target=self.pro7_server.serve_forever, daemon=True).start()
        self.pro7_port = self.pro7_server.server_address[1]

        (self.tmp / "config.yaml").write_text(
            f"report:\n  output_dir: {(self.tmp / 'reports').as_posix()}\npropresenter:\n  poll_interval: 0.05\n", encoding="utf-8")
        self.settings = SettingsStore(str(self.tmp / "config.yaml"))
        self.port = free_port()
        self.dash = DashboardServer("127.0.0.1", self.port)
        script = [(1, 1.5, +1.0), (2, 4.5, -0.6), (3, 7.0, +0.3)]
        self.script = script
        whisper = FakeWhisper([(slide - 1, onset) for slide, onset, _ in script])
        self.ctrl = Controller(self.settings, self.dash, whisper_factory=lambda wcfg: whisper, console=False, auto=False)
        self.dash.start()
        self.ctrl.start_ticker()
        self.base = f"http://127.0.0.1:{self.port}"
        STATE["index"] = None
        return self

    def get(self, path, **kw):
        return requests.get(self.base + path, headers=H, timeout=15, **kw)

    def post(self, path, body=None, headers=H):
        return requests.post(self.base + path, json=body if body is not None else {}, headers=headers, timeout=30)

    def __exit__(self, *exc):
        try:
            self.ctrl.shutdown()
        finally:
            self.dash.stop()
            self.pro7_server.shutdown()
            shutil.rmtree(self.tmp, ignore_errors=True)


def test_settings_validation_overrides_and_csrf_guard():
    with Env() as e:
        r = e.get("/api/settings").json()
        keys = {f["key"] for f in r["fields"]}
        assert {"timing.offset_ms", "timing.on_time_late_s", "timing.on_time_early_s", "whisper.model"} <= keys
        assert r["values"]["timing.on_time_late_s"] == 0.75 and r["overridden"] == []

        # a POST without the custom header (what a hostile web page would send) is refused
        assert e.post("/api/settings", {"patch": {"timing.offset_ms": 5}}, headers={}).status_code == 403

        bad = e.post("/api/settings", {"patch": {"propresenter.port": 99999, "whisper.model": "huge", "nope": 1}})
        assert bad.status_code == 422
        assert set(bad.json()["errors"]) == {"propresenter.port", "whisper.model", "nope"}
        assert not (e.tmp / "user_settings.json").exists()                 # nothing saved on error

        ok = e.post("/api/settings", {"patch": {"timing.on_time_late_s": "0.5", "timing.on_time_early_s": 2,
                                                 "timing.offset_ms": 150}})
        assert ok.status_code == 200
        v = ok.json()["settings"]
        assert v["values"]["timing.on_time_late_s"] == 0.5 and v["values"]["timing.on_time_early_s"] == 2.0
        assert "timing.offset_ms" in v["overridden"]
        saved = json.loads((e.tmp / "user_settings.json").read_text(encoding="utf-8"))
        assert saved["timing.offset_ms"] == 150
        assert (e.tmp / "config.yaml").read_text(encoding="utf-8").startswith("report:")   # YAML untouched

        # a fresh store (like a restart) sees the overrides; reset removes them
        assert SettingsStore(str(e.tmp / "config.yaml")).cfg["timing"]["offset_ms"] == 150
        assert e.post("/api/settings/reset").json()["settings"]["overridden"] == []
        assert not (e.tmp / "user_settings.json").exists()


def test_websocket_origin_check_and_initial_state():
    with Env() as e:
        async def go():
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.ws_connect(e.base + "/ws") as ws:
                    state = json.loads((await ws.receive(timeout=5)).data)
                    assert "pro7" in state and "whisper" in state and state["monitoring"] is False
                try:
                    async with s.ws_connect(e.base + "/ws", headers={"Origin": "http://evil.example"}):
                        return False
                except aiohttp.WSServerHandshakeError as err:
                    return err.status == 403
        assert asyncio.run(go())


def test_equalizer_frames_stream_to_browsers_while_monitoring():
    with Env() as e:
        wav = e.tmp / "in.wav"
        write_test_wav(wav)
        assert e.post("/api/propresenter/connect", {"host": "127.0.0.1", "port": e.pro7_port}).json()["ok"]
        STATE["index"] = 0

        async def go():
            import aiohttp
            frames, states = [], 0
            async with aiohttp.ClientSession() as s:
                async with s.ws_connect(e.base + "/ws") as ws:
                    await ws.receive(timeout=5)                                       # initial state
                    assert e.post("/api/monitoring/start", {"source": {"kind": "file", "file": str(wav)}}).status_code == 200
                    end = time.time() + 4
                    while time.time() < end and len(frames) < 8:
                        try:
                            msg = await ws.receive(timeout=1)
                        except asyncio.TimeoutError:
                            continue
                        data = json.loads(msg.data)
                        if "_spectrum" in data:
                            assert set(data) == {"_spectrum"}, "equalizer frames are their own small messages, not part of the state"
                            frames.append(data["_spectrum"])
                        else:
                            states += 1
            return frames, states

        frames, states = asyncio.run(go())
        assert len(frames) >= 5, "expected a steady stream of equalizer frames"
        f = frames[-1]
        assert len(f["db"]) == 24 and len(f["edges"]) == 25 and f["edges"][0] == 60.0
        assert max(max(x["db"]) for x in frames) > -60          # the test wav has sound in it
        assert states >= 0                                       # (state pushes also arrive; how many depends on machine load)
        assert "_spectrum" not in e.dash._state                 # not stored, so new browsers don't replay it
        e.post("/api/monitoring/stop")


def test_full_gui_session_songs_graph_files_reanalysis():
    with Env() as e:
        wav = e.tmp / "in.wav"
        write_test_wav(wav)

        # monitoring can't start without a source
        assert e.post("/api/monitoring/start", {"source": {}}).status_code == 400
        # can't start a song before monitoring
        assert e.post("/api/song/start").status_code == 400

        # connect to the fake ProPresenter via the API (also saves host/port)
        c = e.post("/api/propresenter/connect", {"host": "127.0.0.1", "port": e.pro7_port})
        assert c.status_code == 200 and c.json()["ok"]
        assert e.get("/api/settings").json()["values"]["propresenter.port"] == e.pro7_port
        bad = e.post("/api/propresenter/connect", {"host": "127.0.0.1", "port": free_port()})
        assert bad.status_code == 502 and not bad.json()["ok"]
        assert e.post("/api/propresenter/connect", {"host": "127.0.0.1", "port": e.pro7_port}).json()["ok"]

        # never enumerate the machine's real audio devices from a test (it can disturb a Pro Dragon someone is running)
        e.ctrl.list_audio_sources = lambda include_ndi=False: [
            {"kind": "input", "index": 0, "name": "Fake mic", "channels": 2, "sample_rate": 44100, "host_api": "test", "num": 1}]
        devs = e.get("/api/devices").json()
        assert devs["ok"] and isinstance(devs["devices"], list)

        # start monitoring from a wav file
        STATE["index"] = 0
        r = e.post("/api/monitoring/start", {"source": {"kind": "file", "file": str(wav)}, "operator": "  Sam  "})
        assert r.status_code == 200, r.text
        assert e.ctrl.operator == "Sam"                                          # whitespace trimmed
        assert e.post("/api/monitoring/start", {"source": {"kind": "file", "file": str(wav)}}).status_code == 400  # already on
        time.sleep(0.6)
        assert e.settings.cfg["audio"]["last"]["kind"] == "file"                 # remembered for next time

        assert e.post("/api/song/start").json()["ok"]
        t0 = time.time()
        for slide, onset, delay in e.script:
            time.sleep(max(0, t0 + onset + delay - time.time()))
            STATE["index"] = slide
        time.sleep(max(0, t0 + 9.5 - time.time()))
        live = json.loads(requests.get(e.base + "/api/songs", headers=H).text)    # exercise the endpoint mid-session
        assert live["ok"]
        assert e.post("/api/song/stop").json()["ok"]

        # wait for the background analysis to land in the library
        song = None
        for _ in range(80):
            songs = e.get("/api/songs").json()["songs"]
            if songs and songs[0]["analyzed"]:
                song = songs[0]
                break
            time.sleep(0.25)
        assert song, "song never appeared as analyzed"
        assert song["name"] == "Holy Holy Holy" and song["measured"] == 3
        assert song["operator"] == "Sam"                                          # stamped on the song

        d = e.get("/api/song", params={"id": song["id"]}).json()["song"]
        delays = [t["delay"] for t in d["timings"] if t["delay"] is not None]
        for got, (_, _, want) in zip(delays, e.script):
            assert abs(got - want) < 0.3
        assert d["has_words"] and d["has_audio"] and d["files"]["svg"] and d["files"]["txt"]
        assert d["operator"] == "Sam"
        ops = e.get("/api/operators").json()["operators"]
        assert [o["name"] for o in ops] == ["Sam"] and ops[0]["songs"] == 1 and ops[0]["measured"] == 3
        assert "Operator: Sam" in e.get("/api/song/file", params={"id": song["id"], "kind": "txt"}).text

        # graph: valid SVG; margins respected
        g1 = e.get("/api/graph.svg", params={"id": song["id"], "late": 0.75, "early": 0.75})
        assert g1.status_code == 200 and g1.headers["Content-Type"].startswith("image/svg+xml")
        ET.fromstring(g1.text)
        g2 = e.get("/api/graph.svg", params={"id": song["id"], "late": 0.2, "early": 5})
        assert "≤0.20s late, ≤5.00s early" in g2.text
        assert e.get("/api/graph.svg", params={"id": song["id"], "late": "x"}).status_code == 400

        # files + path traversal
        f = e.get("/api/song/file", params={"id": song["id"], "kind": "txt"})
        assert f.status_code == 200 and "Holy Holy Holy" in f.text
        assert e.get("/api/song/file", params={"id": song["id"], "kind": "wav"}).status_code == 200
        for evil in ("../config.yaml", "..\\..\\x.json", "/etc/passwd", "session_x/../../config.json", "a.json"):
            assert e.get("/api/song", params={"id": evil}).status_code == 404, evil
        assert e.get("/api/song/file", params={"id": song["id"], "kind": "exe"}).status_code == 404

        # re-analysis with a different offset shifts every delay by exactly that amount
        r2 = e.post("/api/song/reanalyze", {"id": song["id"], "offset_ms": 250})
        assert r2.status_code == 200 and r2.json()["ok"]
        d2 = r2.json()["detail"]
        for a, b in zip(d["timings"], d2["timings"]):
            if a["delay"] is not None:
                assert abs((b["delay"] - a["delay"]) - 0.25) < 1e-6
        assert d2["settings"]["offset_ms"] == 250
        assert e.post("/api/song/reanalyze", {"id": "../nope.json"}).status_code == 400

        # live state reached browsers' snapshot
        snap = e.ctrl.snapshot()
        assert snap["monitoring"] and snap["last_song_id"] == song["id"]
        assert any("analyzed" in m["text"] for m in snap["messages"])

        # stop monitoring
        assert e.post("/api/monitoring/stop").json()["ok"]
        for _ in range(40):
            if not e.ctrl.snapshot()["monitoring"]:
                break
            time.sleep(0.25)
        assert not e.ctrl.snapshot()["monitoring"]
        assert (Path(e.settings.cfg["report"]["output_dir"]) / Path(song["id"]).parent / "session_report.txt").exists()
