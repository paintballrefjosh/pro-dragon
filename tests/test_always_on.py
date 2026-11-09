"""
The GUI's always-on behaviour: ProPresenter connects (and reconnects) by itself, audio monitoring starts
by itself from the last audio source, and switching sources doesn't need Stop/Start buttons.
"""

import socket
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import Controller  # noqa: E402
from propresenter.client import ProPresenterClient  # noqa: E402
from settings import SettingsStore  # noqa: E402
from test_session_e2e import STATE, FakePro7, FakeWhisper, write_test_wav  # noqa: E402


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_for(cond, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def serve(port=0):
    server = ThreadingHTTPServer(("127.0.0.1", port), FakePro7)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_client_connects_when_proPresenter_appears_and_notices_when_it_goes_away():
    port = free_port()
    changes = []
    c = ProPresenterClient("127.0.0.1", port, poll_interval=0.05)
    c.on_connection_changed = changes.append
    c.start_polling()                                   # nothing is listening yet — no connect step, no error
    server = None
    try:
        time.sleep(0.4)
        assert c.connected is False and changes == []

        STATE["index"] = None
        server = serve(port)                            # ProPresenter starts later
        assert wait_for(lambda: c.connected), "should connect by itself"
        assert changes == [True]

        server.shutdown()
        server.server_close()                           # ProPresenter goes away
        assert wait_for(lambda: not c.connected, timeout=8)
        assert wait_for(lambda: changes == [True, False])

        server = serve(port)                            # …and comes back
        assert wait_for(lambda: c.connected)
        assert wait_for(lambda: changes == [True, False, True])
    finally:
        c.disconnect()
        if server:
            server.shutdown()


def test_reconfigure_follows_a_new_address():
    a, b = serve(), serve()
    try:
        c = ProPresenterClient("127.0.0.1", a.server_address[1], poll_interval=0.05)
        c.start_polling()
        assert wait_for(lambda: c.connected)
        c.reconfigure("127.0.0.1", b.server_address[1])
        assert c.connected is False and c.current_uuid is None          # forgets the old one at once
        assert wait_for(lambda: c.connected)
        assert c.base_url.endswith(str(b.server_address[1]))
        c.disconnect()
    finally:
        a.shutdown()
        b.shutdown()


def test_controller_in_auto_mode_connects_monitors_and_switches_sources():
    tmp = Path(tempfile.mkdtemp())
    server = serve()
    wav1, wav2 = tmp / "first.wav", tmp / "second.wav"
    write_test_wav(wav1, seconds=60)
    write_test_wav(wav2, seconds=60)
    (tmp / "config.yaml").write_text(
        f"report:\n  output_dir: {(tmp / 'reports').as_posix()}\npropresenter:\n  port: {server.server_address[1]}\n  poll_interval: 0.05\n",
        encoding="utf-8")
    settings = SettingsStore(str(tmp / "config.yaml"))
    settings.set_runtime("audio.last", {"kind": "file", "file": str(wav1)})        # what the last run used
    STATE["index"] = 0
    ctrl = Controller(settings, None, whisper_factory=lambda w: FakeWhisper([]), console=False, auto=True)
    try:
        assert ctrl.pro7 is None and ctrl.session is None                          # nothing happens until it starts
        ctrl.start_ticker()
        # no Connect press, no Start monitoring press
        assert wait_for(lambda: ctrl.pro7 is not None and ctrl.pro7.connected and ctrl.session is not None, timeout=12)
        snap = ctrl.snapshot()
        assert snap["pro7"]["connected"] is True and snap["monitoring"] is True
        assert "first.wav" in ctrl.session.source.describe()

        # switch source (no stop/start buttons): the old session ends, a new one starts on the new source
        ok, msg = ctrl.set_audio_source({"kind": "file", "file": str(wav2)})
        assert ok and msg == "switching"
        assert wait_for(lambda: ctrl.session is not None and not ctrl._switching and "second.wav" in ctrl.session.source.describe(), timeout=15)
        assert settings.cfg["audio"]["last"]["file"] == str(wav2)                    # remembered for the next launch

        # bad requests are refused up front
        assert ctrl.set_audio_source({"kind": "file", "file": str(tmp / "missing.wav")})[0] is False
        assert ctrl.set_audio_source({})[0] is False

        # not while a song is recording
        assert ctrl.song_command("start")[0]
        ok, msg = ctrl.set_audio_source({"kind": "file", "file": str(wav1)})
        assert ok is False and "Stop the song" in msg
        assert "second.wav" in ctrl.session.source.describe()
        time.sleep(1.5)                                  # let some audio arrive, or there is nothing to stop and analyze
        assert ctrl.song_command("stop")[0]
    finally:
        ctrl.shutdown()
        server.shutdown()


def test_auto_mode_waits_quietly_when_there_is_no_audio_source_yet():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "config.yaml").write_text(f"report:\n  output_dir: {(tmp / 'reports').as_posix()}\npropresenter:\n  port: {free_port()}\n", encoding="utf-8")
    ctrl = Controller(SettingsStore(str(tmp / "config.yaml")), None, whisper_factory=lambda w: FakeWhisper([]), console=False, auto=True)
    try:
        ctrl.start_ticker()
        assert wait_for(lambda: ctrl.pro7 is not None)                               # the client exists and keeps trying…
        time.sleep(0.5)
        snap = ctrl.snapshot()
        assert ctrl.session is None and snap["monitoring"] is False and snap["pro7"]["connected"] is False
        assert not [m for m in snap["messages"] if m["kind"] == "error"]            # …without nagging
    finally:
        ctrl.shutdown()
