"""The live in-song estimates through the real Session/Controller/HTTP stack: while a song is being
recorded, provisional delays appear in the live state and in the live graph; when the song stops
they disappear and the final analysis takes over. The wav is a ramp (sample value = time) so the
scripted transcriber can tell where each window starts."""

import shutil
import sys
import tempfile
import threading
import time
import wave
import xml.etree.ElementTree as ET
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.song_timing import tokenize  # noqa: E402
from app import Controller  # noqa: E402
from gui.server import DashboardServer  # noqa: E402
from settings import SettingsStore  # noqa: E402
from test_gui_api import free_port  # noqa: E402
from test_session_e2e import LYRICS, STATE, FakePro7  # noqa: E402

H = {"X-ProDragon": "1"}


def write_ramp_wav(path, seconds=40, rate=44100):
    t = np.arange(int(seconds * rate)) / rate
    pcm = np.repeat((t / 100.0 * 32767)[:, None], 2, axis=1).astype(np.int16)        # value encodes time (0..0.4)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


class WindowAwareWhisper:
    """Knows the song's words in song time; for a window it works out the window's start from the
    ramp in the audio itself, and returns only the words that lie fully inside it."""
    def __init__(self, onsets_by_lyric):
        self.script = []
        for idx, onset in onsets_by_lyric:
            t = onset
            for tok in tokenize(LYRICS[idx]):
                self.script.append((tok, t, t + 0.3))
                t += 0.4
        self.windows = []
        self.base = None            # ramp value at the song's first sample (the file started playing slightly earlier)

    def transcribe_words(self, audio, initial_prompt=None, vad_filter=False):
        if self.base is None:       # the first window always begins at the song's start
            self.base = float(audio[0]) * 100.0
        start = float(audio[0]) * 100.0 - self.base
        end = start + len(audio) / 16000
        self.windows.append(len(audio) / 16000)
        return [(t, s - start, e - start, 0.9) for t, s, e in self.script if s >= start - 0.05 and e <= end + 0.05]


def test_live_estimates_appear_while_recording_and_final_report_follows():
    tmp = Path(tempfile.mkdtemp())
    pro7_server = ThreadingHTTPServer(("127.0.0.1", 0), FakePro7)
    threading.Thread(target=pro7_server.serve_forever, daemon=True).start()
    try:
        (tmp / "config.yaml").write_text(
            f"report:\n  output_dir: {(tmp / 'reports').as_posix()}\npropresenter:\n  poll_interval: 0.05\n"
            "live:\n  interval_s: 2\n", encoding="utf-8")
        wav = tmp / "ramp.wav"
        write_ramp_wav(wav)
        script = [(1, 2.0, +1.0), (2, 6.0, -0.5), (3, 10.0, +0.4)]      # (pro7 slide, onset, true delay)
        whisper = WindowAwareWhisper([(s - 1, on) for s, on, _ in script])

        settings = SettingsStore(str(tmp / "config.yaml"))
        dash = DashboardServer("127.0.0.1", free_port())
        ctrl = Controller(settings, dash, whisper_factory=lambda w: whisper)
        dash.start()
        ctrl.start_ticker()
        base = f"http://127.0.0.1:{dash.port}"
        STATE["index"] = 0
        try:
            assert requests.post(base + "/api/propresenter/connect", json={"host": "127.0.0.1", "port": pro7_server.server_address[1]},
                                 headers=H).json()["ok"]
            assert requests.post(base + "/api/monitoring/start", json={"source": {"kind": "file", "file": str(wav)}},
                                 headers=H).json()["ok"]
            time.sleep(0.5)

            # no live graph before a song is being recorded
            assert requests.get(base + "/api/live/graph.svg", headers=H).status_code == 404
            assert ctrl.snapshot()["live"]["enabled"] is False

            assert requests.post(base + "/api/song/start", json={}, headers=H).json()["ok"]
            assert ctrl.snapshot()["live"]["enabled"] is True
            t0 = time.time()
            for slide, onset, delay in script:
                time.sleep(max(0, t0 + onset + delay - time.time()))
                STATE["index"] = slide

            # wait (song still recording) for the provisional delays to appear
            rows = []
            for _ in range(100):
                rows = [r for r in ctrl.snapshot()["live"]["rows"] if not r["blank"]]
                if len(rows) == 4 and all(r["delay"] is not None for r in rows[1:]):
                    break
                time.sleep(0.25)
            real = [r for r in rows if not r["initial"]]
            assert len(real) == 3 and all(r["delay"] is not None and not r["pending"] for r in real), rows
            for r, (_, _, want) in zip(real, script):
                assert abs(r["delay"] - want) < 0.35, (r, want)
            assert max(whisper.windows) <= 40.5                        # only short windows, never the whole song again

            # the provisional graph is real SVG with our three delays
            g = requests.get(base + "/api/live/graph.svg", headers=H)
            assert g.status_code == 200 and g.headers["Content-Type"].startswith("image/svg+xml")
            ET.fromstring(g.text)
            assert "live estimate" in g.text and ">+1.0<" in g.text or ">+0.9<" in g.text or ">+1.1<" in g.text

            # stop: live data disappears, the final analysis takes over and lands in the library
            assert requests.post(base + "/api/song/stop", json={}, headers=H).json()["ok"]
            assert ctrl.snapshot()["live"]["rows"] == []
            assert requests.get(base + "/api/live/graph.svg", headers=H).status_code == 404
            song = None
            for _ in range(80):
                songs = requests.get(base + "/api/songs", headers=H).json()["songs"]
                if songs and songs[0]["analyzed"]:
                    song = songs[0]
                    break
                time.sleep(0.25)
            assert song and song["measured"] == 3
        finally:
            ctrl.shutdown()
            dash.stop()
    finally:
        pro7_server.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)
