"""Finding the loopback for the default speakers when Windows names the same USB device two ways."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio.capture import _pick_loopback  # noqa: E402

LOOPBACKS = [
    {"index": 22, "name": "LS32A70 (NVIDIA High Definition Audio) [Loopback]"},
    {"index": 23, "name": "Speakers (Realtek(R) Audio) [Loopback]"},
    {"index": 24, "name": "Speakers (2- USB Audio CODEC ) [Loopback]"},
]


def test_exact_name_still_matches():
    assert _pick_loopback("Speakers (Realtek(R) Audio)", LOOPBACKS)["index"] == 23


def test_the_input_side_name_of_a_usb_device_finds_its_speakers():
    # what Windows reported at launch: the microphone half of the USB codec, name cut at 31 characters
    assert _pick_loopback("Microphone (2- USB Audio CODEC ", LOOPBACKS)["index"] == 24
    assert _pick_loopback("Microphone (2- USB Audio CODEC )", LOOPBACKS)["index"] == 24


def test_guess_prefers_a_speakers_device_over_a_monitor_and_handles_none():
    from audio.capture import _guess_loopback
    assert _guess_loopback(LOOPBACKS)["index"] == 23                       # 'Speakers (Realtek…)', not the NVIDIA monitor
    assert _guess_loopback(LOOPBACKS[:1])["index"] == 22                   # nothing called Speakers: first available
    assert _guess_loopback([]) is None


class _NoDefaultPA:
    """PortAudio when Windows has no usable default output: asking for it raises 'Invalid device info'."""
    def __init__(self, loopbacks):
        self._lb = loopbacks

    def get_loopback_device_info_generator(self):
        return iter(self._lb)

    def get_host_api_info_by_type(self, _t):
        return {"defaultOutputDevice": -1}

    def get_device_info_by_index(self, idx):
        raise OSError(-9996, "Invalid device info")

    def get_default_wasapi_loopback(self):
        raise OSError(-9996, "Invalid device info")


def test_resolving_the_default_never_raises_and_the_guess_then_takes_over():
    from audio.capture import LoopbackSource, _guess_loopback
    src = LoopbackSource.__new__(LoopbackSource)            # no real PortAudio involved
    src._pa = _NoDefaultPA(LOOPBACKS)
    src._pa_mod = type("PA", (), {"paWASAPI": 13})
    info, name, loopbacks = src._resolve_default_loopback()
    assert info is None and "Invalid device info" in name and loopbacks == LOOPBACKS
    assert _guess_loopback(loopbacks)["index"] == 23         # what the app then listens to


def test_unknown_device_matches_nothing():
    assert _pick_loopback("Headphones (Something Else)", LOOPBACKS) is None
    assert _pick_loopback("", LOOPBACKS) is None
    assert _pick_loopback("Speakers (Realtek(R) Audio)", []) is None
