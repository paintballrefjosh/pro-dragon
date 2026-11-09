"""NdiSource against a real NDI stream: a sender on this machine broadcasts a 48 kHz stereo
tone (left channel only), and the source must discover it, receive it, pick channels,
resample to 16 kHz and stamp blocks with sensible times."""

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NDI = pytest.importorskip("NDIlib", reason="ndi-python not installed")

from audio.capture import NdiSource, list_ndi_sources  # noqa: E402

NAME = f"ProDragonTest{os.getpid()}"


class Sender:
    """Real-time 48 kHz stereo sender: 440 Hz at amplitude 0.3 on the LEFT channel, silence on the right."""

    def __enter__(self):
        assert NDI.initialize()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        assert self.ready.wait(5)
        return self

    def _run(self):
        cs = NDI.SendCreate()
        cs.ndi_name = NAME
        s = NDI.send_create(cs)
        self.ready.set()
        rate, n, pos, t0 = 48000, 1024, 0, time.time()
        while not self.stop.is_set():
            tone = (0.3 * np.sin(2 * np.pi * 440 * (np.arange(n) + pos) / rate)).astype(np.float32)
            f = NDI.AudioFrameV2()
            f.sample_rate, f.no_channels, f.no_samples = rate, 2, n
            f.data = np.stack([tone, np.zeros(n, np.float32)])
            f.channel_stride_in_bytes = n * 4
            NDI.send_send_audio_v2(s, f)
            pos += n
            delay = t0 + pos / rate - time.time()
            if delay > 0:
                time.sleep(delay)
        NDI.send_destroy(s)

    def __exit__(self, *exc):
        self.stop.set()
        self.t.join(timeout=3)


def capture(channels, seconds=2.5):
    blocks = []
    src = NdiSource(NAME, channels, on_frame=lambda a, ts: blocks.append((a.copy(), ts)))
    t_start = time.time()
    src.start()
    for _ in range(80):                    # (re)connecting can take a couple of seconds
        if blocks:
            break
        time.sleep(0.1)
    time.sleep(seconds)
    desc = src.describe()
    src.stop()
    return blocks, t_start, desc


def rms(blocks):
    return float(np.sqrt(np.mean(np.concatenate([b for b, _ in blocks]) ** 2)))


def test_discovery_receive_channels_resample_and_timestamps():
    with Sender():
        found = list_ndi_sources(timeout_s=6)
        assert any(NAME in s["name"] for s in found) and all(s["kind"] == "ndi" for s in found)

        blocks, t_start, desc = capture(None)                 # all channels mixed to mono
        assert "NDI" in desc and "48000" in desc and "2ch" in desc
        assert blocks and all(b.dtype == np.float32 for b, _ in blocks)
        total = sum(len(b) for b, _ in blocks)
        assert 16000 * 1.5 < total < 16000 * 3.2               # ≈ the captured duration at 16 kHz, not 48 kHz
        assert abs(rms(blocks) - 0.3 / np.sqrt(2) / 2) < 0.02  # mean of (tone, silence)

        # timestamps: contiguous, monotonic, and near wall-clock "now"
        ts = [t for _, t in blocks]
        assert all(b > a for a, b in zip(ts, ts[1:]))
        gaps = [(t2 - t1) - len(b1) / 16000 for (b1, t1), (_b2, t2) in zip(blocks, blocks[1:])]
        assert max(abs(g) for g in gaps) < 0.15
        assert t_start - 1 < ts[0] < t_start + 6 and ts[-1] < time.time() + 0.1

        left, _, _ = capture([1], 1.5)
        right, _, _ = capture([2], 1.5)
        assert abs(rms(left) - 0.3 / np.sqrt(2)) < 0.02 and rms(right) < 0.005


def test_source_not_found_gives_a_clear_error():
    class Quick(NdiSource):
        DISCOVER_TIMEOUT_S = 2.0
    with pytest.raises(RuntimeError, match="not found"):
        Quick("definitely-not-a-real-source-xyz").start()
    with pytest.raises(RuntimeError, match="at 127.0.0.1"):          # the message names the static address
        Quick("definitely-not-a-real-source-xyz", addresses="127.0.0.1").start()


def test_static_addresses_are_validated_and_normalised():
    from audio.capture import parse_ndi_addresses
    assert parse_ndi_addresses(None) == "" and parse_ndi_addresses("  ") == ""
    assert parse_ndi_addresses("192.168.20.15, mixer.local:5959;192.168.20.15") == "192.168.20.15,mixer.local:5959"
    assert parse_ndi_addresses(["10.0.0.5", " 10.0.0.6 "]) == "10.0.0.5,10.0.0.6"
    for bad in ("http://10.0.0.5", "10.0.0.5:99999", "10.0.0.5:0", "bad address!", "-x"):
        with pytest.raises(ValueError):
            parse_ndi_addresses(bad)


def test_source_is_found_with_a_static_address_set():
    """With extra_ips set the finder still lists local sources, and a capture works end to end."""
    with Sender():
        found = list_ndi_sources(timeout_s=6, extra_ips="127.0.0.1")
        assert any(NAME in s["name"] for s in found)
        blocks = []
        src = NdiSource(NAME, None, on_frame=lambda a, ts: blocks.append((a.copy(), ts)), addresses="127.0.0.1")
        src.start()
        time.sleep(2.0)
        src.stop()
        assert blocks and rms(blocks) > 0.05


def test_ndi_address_setting_is_validated_and_reaches_the_source():
    from settings import FIELD_BY_KEY, SettingsStore, coerce
    import tempfile
    f = FIELD_BY_KEY["audio.ndi_address"]
    assert coerce(f, "") is None and coerce(f, "10.0.0.5 , 10.0.0.6:5959") == "10.0.0.5,10.0.0.6:5959"
    with pytest.raises(ValueError):
        coerce(f, "not an address!")
    from app import Controller
    tmp = Path(tempfile.mkdtemp())
    c = Controller(SettingsStore(str(tmp / "config.yaml")))
    assert c.settings.update({"audio.ndi_address": "10.0.0.5"}) == {}
    assert c.settings.update({"audio.ndi_address": "nope!"}) != {}
    assert c.settings.cfg["audio"]["ndi_address"] == "10.0.0.5"        # a rejected edit saves nothing
    assert callable(c._source_factory({"kind": "ndi", "name": "X (Audio)"}))


def test_controller_builds_ndi_sources():
    from app import Controller
    from settings import SettingsStore
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    c = Controller(SettingsStore(str(tmp / "config.yaml")))
    assert callable(c._source_factory({"kind": "ndi", "name": "X (Audio)", "channels": "1,2"}))
    with pytest.raises(ValueError):
        c._source_factory({"kind": "ndi", "name": ""})
    assert c.ndi_available()
