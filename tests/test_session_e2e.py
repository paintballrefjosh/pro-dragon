"""
End-to-end check of the real pipeline with fakes at the edges:
  * a fake ProPresenter HTTP server (version / active presentation / slide_index)
  * a 44.1 kHz stereo .wav played in real time through FileSource (exercises resampling)
  * a scripted stand-in for Whisper that "hears" the singer at known times
The operator (this test) changes slides at known delays; Session must record the
window, save the files, and the report must recover those delays.
"""

import json
import shutil
import sys
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.song_runner import analyze_songdata, load_song  # noqa: E402
from analysis.song_timing import tokenize  # noqa: E402
from audio.capture import FileSource, StreamResampler  # noqa: E402
from propresenter.client import ProPresenterClient  # noqa: E402
from session import Session  # noqa: E402

LYRICS = ["Holy holy holy Lord God almighty", "Early in the morning our song shall rise to thee",
          "Holy holy holy merciful and mighty", "God in three persons blessed trinity"]

STATE = {"index": None, "uuid": "abc-123"}


class FakePro7(BaseHTTPRequestHandler):
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
        if self.path == "/version":      # real Pro7 serves this at the root, not under /v1
            self._send({"name": "ProPresenter", "platform": "win", "host_description": "7.fake"})
        elif self.path == "/v1/presentation/active":
            self._send({"presentation": {
                "id": {"uuid": STATE["uuid"], "name": "Holy Holy Holy", "index": 0},
                "groups": [
                    {"name": "Title", "slides": [{"text": "", "label": ""}]},          # blank slide -> index 0
                    {"name": "Verse 1", "slides": [{"text": t, "label": ""} for t in LYRICS[:2]]},
                    {"name": "Verse 2", "slides": [{"text": t, "label": ""} for t in LYRICS[2:]]},
                ]}})
        elif self.path == "/v1/presentation/slide_index":
            idx = STATE["index"]
            self._send({"presentation_index": None if idx is None else
                        {"index": idx, "presentation_id": {"uuid": STATE["uuid"], "name": "Holy Holy Holy", "index": 0}}})
        else:
            self.send_response(404)
            self.end_headers()


class FakeWhisper:
    """'Hears' each lyric line at scripted times (seconds from the start of the audio)."""
    def __init__(self, onsets_by_slide):
        self.onsets = onsets_by_slide

    def transcribe_words(self, audio, initial_prompt=None, vad_filter=False):
        words = []
        for lyric_idx, onset in self.onsets:
            for k, tok in enumerate(tokenize(LYRICS[lyric_idx])):
                words.append((tok, onset + 0.4 * k, onset + 0.4 * k + 0.3))
        return words


def write_test_wav(path, seconds=12, rate=44100):
    rng = np.random.default_rng(0)
    pcm = (rng.normal(0, 0.1, (int(seconds * rate), 2)) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def test_resampler_length_and_continuity():
    r = StreamResampler(44100, 16000)
    x = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100).astype(np.float32)
    out = np.concatenate([r.process(x[i:i + 2205]) for i in range(0, len(x), 2205)])
    assert abs(len(out) - 16000) <= 2
    ref = np.sin(2 * np.pi * 440 * np.arange(len(out)) / 16000)
    assert np.corrcoef(out[100:-100], ref[100:-100])[0, 1] > 0.99     # no seams between blocks


def test_full_session_recovers_delays():
    tmp = Path(tempfile.mkdtemp())
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakePro7)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        wav = tmp / "in.wav"
        write_test_wav(wav)

        # singer starts lyric lines at these times (song-relative); operator's slides are off by `delays`
        # Pro7 slide indexes: 0=blank title, 1..4 = lyrics 0..3
        script = [(1, 1.5, +1.0), (2, 4.5, -0.6), (3, 7.0, +0.3)]   # (pro7 slide, onset, true delay)
        whisper = FakeWhisper([(slide - 1, onset) for slide, onset, _ in script])

        cfg = {"report": {"output_dir": str(tmp / "reports")},
               "timing": {"on_time_window_s": 0.75, "min_match_confidence": 0.25}}
        pro7 = ProPresenterClient("127.0.0.1", port, poll_interval=0.05)
        assert pro7.check_connection()
        session = Session(cfg, pro7, lambda cb: FileSource(str(wav), on_frame=cb), lambda: whisper, dashboard=None, offset_ms=0,
                          console=False)
        pro7.on_slide_changed = session._on_slide_changed
        pro7.on_presentation_changed = session._on_presentation_changed
        pro7.start_polling()
        STATE["index"] = 0
        session.start()
        time.sleep(0.5)

        assert session.start_song()
        t0 = time.time()
        for slide, onset, delay in script:
            time.sleep(max(0, t0 + onset + delay - time.time()))
            STATE["index"] = slide
        time.sleep(max(0, t0 + 9.5 - time.time()))
        assert session.stop_song()
        for t in session._analysis_threads:
            t.join(timeout=20)

        songs = sorted(session.out_dir.glob("song_*.json"))
        assert len(songs) == 1
        data = json.loads(songs[0].read_text(encoding="utf-8"))
        timings = data["timings"]
        # events: initial blank slide (index 0, live at start) + 3 changes
        assert [t["slide_index"] for t in timings] == [0, 1, 2, 3]
        assert timings[0]["initial"] and timings[0]["delay"] is None      # blank slide: nothing to measure
        for t, (_, _, true_delay) in zip(timings[1:], script):
            assert t["delay"] is not None
            assert abs(t["delay"] - true_delay) < 0.3, (t, true_delay)
        assert (session.out_dir / songs[0].name.replace(".json", ".wav")).exists()
        assert "Verse 1" in songs[0].with_suffix(".txt").read_text(encoding="utf-8")

        # saved audio is ~9.5s at 16 kHz even though the source was 44.1 kHz stereo
        with wave.open(str(songs[0].with_suffix(".wav")), "rb") as w:
            assert w.getframerate() == 16000
            assert abs(w.getnframes() / 16000 - 9.5) < 0.5

        # offline re-analysis from the saved files with a different offset
        song = load_song(songs[0])
        analyze_songdata(song, None, cfg["timing"], offset_ms=250)
        for t_new, t_old in zip(song.timings[1:], timings[1:]):
            assert abs((t_new.delay - t_old["delay"]) - 0.25) < 1e-6

        session.finish()
    finally:
        server.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
