"""
Audio Sources
==============
Everything that produces audio for Pro Dragon sits behind one small interface,
`AudioSource`, so new inputs (NDI, network streams, ...) can be added later
without touching the rest of the app:

    source = SomeSource(..., on_frame=callback)
    source.start() ... source.stop()

`on_frame(audio, timestamp)` is called with:
  audio     — mono float32 at exactly `TARGET_RATE` (16 kHz, what Whisper wants)
  timestamp — wall-clock `time.time()` of the FIRST sample in `audio`
              (the capture time of the sound, not the time the callback ran)

Sources shipped today:
  DeviceSource    — any input device (microphone, line-in, Dante Virtual
                    Soundcard). Channels are chosen by the user and averaged
                    to mono.
  LoopbackSource  — Windows only: whatever is playing out of the computer's
                    speakers (WASAPI loopback). For testing against YouTube or
                    a media player. Needs the PyAudioWPatch package.
  NdiSource       — audio from an NDI source on the network (mixer, camera, PC
                    running NDI Tools). Needs ndi-python; network/mix delay is
                    handled with the offset setting (timing.offset_ms).
  FileSource      — plays a 16-bit PCM .wav file in real time. Handy for
                    repeatable tests.

Each source records its native sample rate and resamples to 16 kHz itself
(opening hardware at 16 kHz is often rejected, especially WASAPI loopback).

To add a source (e.g. NDI): subclass AudioSource, call `self._emit(block,
native_rate, capture_ts)` from your capture thread/callback, implement
start()/stop()/describe().
"""

import platform
import re
import threading
import time
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("audio")

TARGET_RATE = 16000


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

class StreamResampler:
    """
    Causal streaming resampler (area averaging, handles non-integer ratios such
    as 44.1k -> 16k). Averaging each output sample over its input window doubles
    as a gentle low-pass, which is plenty for speech recognition.
    """

    def __init__(self, in_rate: float, out_rate: float = TARGET_RATE):
        self.ratio = float(in_rate) / float(out_rate)
        self._passthrough = abs(self.ratio - 1.0) < 1e-9
        self._buf = np.zeros(0, dtype=np.float64)
        self._pos = 0.0  # where the next output window starts, in `_buf` coordinates

    def process(self, x: np.ndarray) -> np.ndarray:
        if self._passthrough:
            return x.astype(np.float32, copy=False)
        self._buf = np.concatenate([self._buf, x.astype(np.float64)])
        n = len(self._buf)
        k = int(np.floor((n - self._pos) / self.ratio))
        if k <= 0:
            return np.zeros(0, dtype=np.float32)
        edges = self._pos + np.arange(k + 1) * self.ratio
        cs = np.concatenate(([0.0], np.cumsum(self._buf)))
        c = np.interp(edges, np.arange(n + 1), cs)  # integral of the signal at each edge
        out = np.diff(c) / self.ratio
        end = edges[-1]
        drop = int(np.floor(end))
        self._buf = self._buf[drop:]
        self._pos = end - drop
        return out.astype(np.float32)


def _db(block: np.ndarray) -> float:
    if block is None or len(block) == 0:
        return -96.0
    rms = float(np.sqrt(np.mean(np.square(block))))
    return -96.0 if rms < 1e-9 else float(20 * np.log10(rms))


# Spectrum ("graphical EQ") display: log-spaced bands over the 16 kHz mono signal that Whisper
# hears (so nothing above 8 kHz), from the newest ~128 ms of audio.
SPECTRUM_BANDS = 24
SPECTRUM_FFT = 2048
SPECTRUM_F_LO = 60.0
SPECTRUM_F_HI = 8000.0
_SPECTRUM_FLOOR_DB = -96.0


def spectrum_edges(n: int = SPECTRUM_BANDS, f_lo: float = SPECTRUM_F_LO, f_hi: float = SPECTRUM_F_HI) -> List[float]:
    """The n+1 band edge frequencies in Hz (log-spaced)."""
    return [float(f_lo * (f_hi / f_lo) ** (i / n)) for i in range(n + 1)]


def band_levels_db(samples: np.ndarray, rate: float = 16000.0, n: int = SPECTRUM_BANDS) -> List[float]:
    """
    Level of each log-spaced band in dB, 0 dB = a full-scale sine wave (so a sine of amplitude A
    shows as 20·log10(A) in its band). Uses the loudest FFT bin in each band. Silence -> -96.
    """
    x = np.asarray(samples, dtype=np.float32)
    if len(x) < SPECTRUM_FFT:
        x = np.concatenate((np.zeros(SPECTRUM_FFT - len(x), np.float32), x))
    x = x[-SPECTRUM_FFT:]
    win = np.hanning(SPECTRUM_FFT).astype(np.float32)
    amp = np.abs(np.fft.rfft(x * win)) / (win.sum() / 2.0)        # sine amplitude A -> peak bin ≈ A
    bin_hz = rate / SPECTRUM_FFT
    edges = spectrum_edges(n)
    out: List[float] = []
    for i in range(n):
        lo = int(edges[i] / bin_hz)
        hi = max(lo + 1, int(np.ceil(edges[i + 1] / bin_hz)))
        peak = float(amp[lo:hi].max()) if lo < len(amp) else 0.0
        out.append(_SPECTRUM_FLOOR_DB if peak < 1e-9 else max(_SPECTRUM_FLOOR_DB, float(20 * np.log10(peak))))
    return out


# ──────────────────────────────────────────────────────────────────────
# Interface
# ──────────────────────────────────────────────────────────────────────

class AudioSource(ABC):
    def __init__(self, on_frame: Optional[Callable[[np.ndarray, float], None]] = None):
        self.on_frame = on_frame
        self._resampler: Optional[StreamResampler] = None
        self._resample_rate: Optional[float] = None
        self._level_db = -96.0
        self._ring = np.zeros(SPECTRUM_FFT, np.float32)      # newest audio, for the spectrum display
        self._last_emit = 0.0

    @abstractmethod
    def start(self): ...

    @abstractmethod
    def stop(self): ...

    @abstractmethod
    def describe(self) -> str: ...

    def get_level_db(self) -> float:
        return self._level_db

    def get_bands_db(self) -> List[float]:
        """Per-band levels (dB) of the newest audio for the GUI's equalizer; silent if nothing arrived lately."""
        if time.time() - self._last_emit > 0.6:
            return [_SPECTRUM_FLOOR_DB] * SPECTRUM_BANDS
        return band_levels_db(self._ring, TARGET_RATE)

    def _emit(self, mono_native: np.ndarray, native_rate: float, capture_ts: float):
        """Resample a native-rate mono block to 16 kHz and hand it to the consumer."""
        if self._resampler is None or self._resample_rate != native_rate:
            self._resampler = StreamResampler(native_rate, TARGET_RATE)
            self._resample_rate = native_rate
        block = self._resampler.process(mono_native)
        if len(block) == 0:
            return
        self._level_db = _db(block)
        self._ring = np.concatenate((self._ring, block.astype(np.float32, copy=False)))[-SPECTRUM_FFT:]
        self._last_emit = time.time()
        if self.on_frame:
            try:
                self.on_frame(block, capture_ts)
            except Exception as e:
                logger.error(f"Error in on_frame callback: {e}", exc_info=True)


# ──────────────────────────────────────────────────────────────────────
# Device discovery
# ──────────────────────────────────────────────────────────────────────

def _loopback_available() -> bool:
    if platform.system() != "Windows":
        return False
    try:
        import pyaudiowpatch  # noqa: F401
        return True
    except ImportError:
        return False


def _ndi_module():
    """The NDI bindings (ndi-python), or None if not installed."""
    try:
        import NDIlib
        return NDIlib
    except ImportError:
        return None


_ADDR_RE = re.compile(r"^(?:[A-Za-z0-9]([A-Za-z0-9.\-]*[A-Za-z0-9])?)(?::(\d{1,5}))?$")


def parse_ndi_addresses(text) -> str:
    """
    Normalise a list of static NDI addresses ("192.168.20.15, mixer.local:5959" or a list)
    into the comma-separated form NDI's finder wants. Each entry is an IPv4 address or host
    name with an optional :port. Empty -> "". Raises ValueError for anything else.
    """
    if text is None:
        return ""
    parts = text if isinstance(text, (list, tuple)) else re.split(r"[,;\s]+", str(text))
    out: List[str] = []
    for p in parts:
        p = str(p).strip()
        if not p:
            continue
        m = _ADDR_RE.match(p)
        if not m or (m.group(2) and not 1 <= int(m.group(2)) <= 65535):
            raise ValueError(f"'{p}' isn't a valid address — use an IP address or host name, optionally with :port "
                             f"(e.g. 192.168.20.15 or 192.168.20.15:5959)")
        if p not in out:
            out.append(p)
    return ",".join(out)


def _make_finder(ndi, extra_ips: str = ""):
    """An NDI finder. `extra_ips` = comma-separated machines to query directly (unicast), so sources on
    another subnet/VLAN — where the normal local-network discovery can't reach — are found too."""
    extra = parse_ndi_addresses(extra_ips)
    if not extra:
        return ndi.find_create_v2()
    settings = ndi.FindCreate()
    settings.show_local_sources = True
    settings.extra_ips = extra
    return ndi.find_create_v2(settings)


def list_ndi_sources(timeout_s: float = 3.0, extra_ips: str = "") -> List[dict]:
    """NDI sources currently visible on the network (empty if ndi-python isn't installed)."""
    ndi = _ndi_module()
    if ndi is None or not ndi.initialize():
        return []
    finder = _make_finder(ndi, extra_ips)
    try:
        names: List[str] = []
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            changed = ndi.find_wait_for_sources(finder, 500)
            names = [s.ndi_name for s in ndi.find_get_current_sources(finder)]
            if names and not changed:
                break               # something found and the list has settled
        return [{"kind": "ndi", "index": None, "name": n, "channels": 0, "sample_rate": 0, "host_api": "NDI"}
                for n in sorted(names)]
    finally:
        ndi.find_destroy(finder)


def list_devices(include_loopback: bool = True, include_ndi: bool = False, ndi_addresses: str = "") -> List[dict]:
    """
    All capturable devices. Each dict: num (menu number), kind ('input' |
    'loopback' | 'ndi'), index (backend device index; None for NDI), name,
    channels, sample_rate, host_api. NDI discovery takes a few seconds, so it
    is opt-in.
    """
    import sounddevice as sd

    devices: List[dict] = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append({
                "kind": "input", "index": i, "name": dev["name"],
                "channels": dev["max_input_channels"],
                "sample_rate": int(dev["default_samplerate"]),
                "host_api": sd.query_hostapis(dev["hostapi"])["name"],
            })

    if include_loopback and _loopback_available():
        import pyaudiowpatch as pa
        p = pa.PyAudio()
        try:
            for d in p.get_loopback_device_info_generator():
                devices.append({
                    "kind": "loopback", "index": d["index"], "name": d["name"],
                    "channels": d["maxInputChannels"],
                    "sample_rate": int(d["defaultSampleRate"]),
                    "host_api": "WASAPI loopback",
                })
        finally:
            p.terminate()
    elif include_loopback and platform.system() == "Windows":
        logger.warning("Speaker (loopback) capture needs the PyAudioWPatch package: pip install PyAudioWPatch")

    if include_ndi:
        devices.extend(list_ndi_sources(extra_ips=ndi_addresses))

    for n, d in enumerate(devices, start=1):
        d["num"] = n
    return devices


def print_devices(include_loopback: bool = True, include_ndi: bool = True, ndi_addresses: str = ""):
    if include_ndi and _ndi_module() is not None:
        print("(looking for NDI sources on the network …)")
    devices = list_devices(include_loopback=include_loopback, include_ndi=include_ndi, ndi_addresses=ndi_addresses)
    if not devices:
        logger.error("No audio devices found.")
        return
    print("\nAvailable audio sources:")
    print(f"{'#':<4} {'Type':<10} {'Ch':<4} {'Rate':<8} {'Host API':<18} Name")
    print("-" * 90)
    for d in devices:
        kind = {"loopback": "Speakers", "ndi": "NDI"}.get(d["kind"], "Input")
        ch = d["channels"] or "-"
        rate = d["sample_rate"] or "-"
        print(f"{d['num']:<4} {kind:<10} {ch!s:<4} {rate!s:<8} {d['host_api']:<18} {d['name']}")
    if include_loopback and platform.system() == "Windows" and _loopback_available():
        try:                                          # what the "Default speakers" choice will actually capture right now
            probe = LoopbackSource(None)
            print(f"\n'Default speakers' resolves to: {probe.describe()}")
            probe._pa.terminate()
        except Exception as e:
            print(f"\n'Default speakers' can't be resolved right now: {e}")
    print("\n'Speakers' = capture what is playing out of that output device (test mode).")
    print("'NDI' = audio from an NDI source on the network (use the offset setting for its delay).\n")


def select_device_interactive(include_loopback: bool = True, include_ndi: bool = True, ndi_addresses: str = "") -> dict:
    devices = list_devices(include_loopback=include_loopback, include_ndi=include_ndi, ndi_addresses=ndi_addresses)
    by_num = {d["num"]: d for d in devices}
    while True:
        raw = input("Enter device number: ").strip()
        try:
            n = int(raw)
            if n in by_num:
                return by_num[n]
            print(f"  Invalid number. Choose from: {sorted(by_num)}")
        except ValueError:
            matches = [d for d in devices if raw.lower() in d["name"].lower()]
            if len(matches) == 1:
                return matches[0]
            print("  Ambiguous — multiple matches." if matches else "  No match found. Try again.")


def select_channels_interactive(device: dict) -> List[int]:
    max_ch = device["channels"]
    print(f"\nDevice '{device['name']}' has {max_ch} channel(s).")
    print("Enter channel number(s) to capture (comma-separated, 1-based; several are averaged to mono).")
    while True:
        raw = input("Channels: ").strip()
        try:
            channels = [int(c.strip()) for c in raw.split(",") if c.strip()]
            invalid = [c for c in channels if c < 1 or c > max_ch]
            if invalid:
                print(f"  Invalid channels: {invalid}. Must be 1-{max_ch}.")
            elif not channels:
                print("  Please enter at least one channel.")
            else:
                return channels
        except ValueError:
            print("  Enter numbers only, separated by commas.")


# ──────────────────────────────────────────────────────────────────────
# Sources
# ──────────────────────────────────────────────────────────────────────

class DeviceSource(AudioSource):
    """Microphone / line-in / Dante Virtual Soundcard via sounddevice."""

    def __init__(self, device_index: int, channels: List[int], on_frame=None):
        super().__init__(on_frame)
        import sounddevice as sd
        self._sd = sd
        self.device_index = device_index
        self.channels = channels
        dev = sd.query_devices(device_index)
        self._name = dev["name"]
        self._native_rate = int(dev["default_samplerate"])
        self._device_channels = dev["max_input_channels"]
        self._stream = None

    def describe(self) -> str:
        return f"input '{self._name}' ch={self.channels} @ {self._native_rate} Hz"

    def start(self):
        self._stream = self._sd.InputStream(
            device=self.device_index,
            channels=self._device_channels,
            samplerate=self._native_rate,
            dtype="float32",
            blocksize=0,
            callback=self._callback,
        )
        self._stream.start()
        logger.info(f"Audio capture started — {self.describe()}")

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Audio capture stopped")

    def _callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio status: {status}")
        sel = [indata[:, c - 1] for c in self.channels if 0 < c <= indata.shape[1]]
        if not sel:
            return
        mono = np.mean(np.stack(sel, axis=1), axis=1)
        # Time of the first sample: now, minus how long ago the driver captured it.
        age = time_info.currentTime - time_info.inputBufferAdcTime
        if not (0.0 <= age < 1.0):
            age = frames / self._native_rate  # driver gave nonsense: assume one block old
        self._emit(mono, self._native_rate, time.time() - age)


def _pick_loopback(default_name: str, loopbacks: List[dict]) -> Optional[dict]:
    """
    The loopback entry that belongs to the default output device. Windows names one USB device
    'Speakers (X)' on its output side and 'Microphone (X)' on its input side (and the names are cut
    to 31 characters), so a whole-name match can fail; compare the part in brackets as well.
    """
    def core(name: str) -> str:
        name = name.replace("[Loopback]", "").strip()
        i = name.find("(")
        return (name[i:] if i >= 0 else name).strip().rstrip(") ").lower()

    if not default_name.strip():
        return None
    for lb in loopbacks:
        if default_name in lb["name"]:
            return lb
    want = core(default_name)
    for lb in loopbacks:
        if want and core(lb["name"]) == want:
            return lb
    return None


def _guess_loopback(loopbacks: List[dict]) -> Optional[dict]:
    """Best available loopback when the default speakers can't be identified: a device called 'Speakers…'
    (not a monitor's HDMI audio), else the first one."""
    for lb in loopbacks:
        if lb["name"].strip().lower().startswith("speakers"):
            return lb
    return loopbacks[0] if loopbacks else None


class LoopbackSource(AudioSource):
    """Captures what is playing through a Windows output device (WASAPI loopback)."""

    def __init__(self, device_index: Optional[int] = None, on_frame=None):
        super().__init__(on_frame)
        if platform.system() != "Windows":
            raise RuntimeError("Speaker capture is Windows-only. On macOS, install BlackHole and pick it as an input device.")
        try:
            import pyaudiowpatch as pa
        except ImportError as e:
            raise RuntimeError("Speaker capture needs PyAudioWPatch: pip install PyAudioWPatch") from e
        self._pa_mod = pa
        self._pa = pa.PyAudio()
        self._guessed = False

        if device_index is None:
            info = None
            default_name, loopbacks = "?", []
            for attempt in range(3):
                info, default_name, loopbacks = self._resolve_default_loopback()
                if info is not None:
                    break
                if attempt < 2:                       # Windows may be mid-change (a USB device just came or went): look again
                    self._pa.terminate()
                    time.sleep(0.7)
                    self._pa = pa.PyAudio()
            if info is None:                          # still unidentifiable: use the best available instead of failing
                info = _guess_loopback(loopbacks)
                if info is None:
                    raise RuntimeError(f"No loopback device found for the default speakers '{default_name}' "
                                       f"(no speaker outputs are available) — pick an audio source under Setup")
                self._guessed = True
                logger.warning(f"Windows reported the default speakers as '{default_name}', which has no loopback; "
                               f"using '{info['name']}' instead")
        else:
            info = self._pa.get_device_info_by_index(device_index)

        self._info = info
        self._channels = int(info["maxInputChannels"])
        self._native_rate = int(info["defaultSampleRate"])
        self._stream = None

    def _resolve_default_loopback(self):
        """(loopback info or None, default output's name, all loopbacks) for the current Windows default speakers."""
        pa = self._pa_mod
        loopbacks: List[dict] = []
        try:
            loopbacks = list(self._pa.get_loopback_device_info_generator())
        except Exception as e:
            logger.debug(f"listing loopback devices failed: {e}")
        info, default_name = None, "no default output device"
        try:
            wasapi = self._pa.get_host_api_info_by_type(pa.paWASAPI)
            default_out = self._pa.get_device_info_by_index(wasapi["defaultOutputDevice"])   # raises 'Invalid device info' if none
            default_name = default_out["name"]
            if default_out.get("isLoopbackDevice"):
                return default_out, default_name, loopbacks
            info = _pick_loopback(default_name, loopbacks)
        except Exception as e:
            default_name = f"no usable default output device ({e})"
        if info is None:                              # let PyAudioWPatch work out the default speakers' loopback itself
            try:
                info = self._pa.get_default_wasapi_loopback()
            except Exception:
                info = None
        return info, default_name, loopbacks

    def describe(self) -> str:
        note = " — the default speakers couldn't be identified, so this one was picked" if self._guessed else ""
        return f"speakers '{self._info['name']}' ({self._channels}ch @ {self._native_rate} Hz, all channels mixed){note}"

    def start(self):
        pa = self._pa_mod
        self._stream = self._pa.open(
            format=pa.paInt16,
            channels=self._channels,
            rate=self._native_rate,
            input=True,
            input_device_index=self._info["index"],
            frames_per_buffer=int(self._native_rate * 0.05),
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        logger.info(f"Audio capture started — {self.describe()}")

    def stop(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        self._pa.terminate()
        logger.info("Audio capture stopped")

    def _callback(self, in_data, frame_count, time_info, status):
        try:
            pcm = np.frombuffer(in_data, dtype=np.int16).reshape(-1, self._channels)
            mono = pcm.astype(np.float32).mean(axis=1) / 32768.0
            age = 0.0
            try:
                age = time_info["current_time"] - time_info["input_buffer_adc_time"]
            except Exception:
                pass
            if not (0.0 <= age < 1.0):
                age = frame_count / self._native_rate
            self._emit(mono, self._native_rate, time.time() - age)
        except Exception as e:
            logger.error(f"Loopback callback error: {e}", exc_info=True)
        return (None, self._pa_mod.paContinue)


class NdiSource(AudioSource):
    """
    Receives the audio of an NDI source on the network (a mixer, a camera, a PC running
    NDI Tools / OBS, ...). Needs the `ndi-python` package (the NDI runtime ships in its wheel).

    Frames arrive as planar float32 (channels, samples) at the sender's rate (usually 48 kHz).
    Timestamps: NDI's own timestamps use the *sender's* clock, which isn't synchronised with
    this computer, so each frame is stamped with its arrival time minus its duration. Network
    and mixing delay (often 100–300 ms) isn't visible to us — set `timing.offset_ms` for it.
    """

    DISCOVER_TIMEOUT_S = 8.0

    def __init__(self, source_name: str, channels: Optional[List[int]] = None, on_frame=None, addresses: str = ""):
        super().__init__(on_frame)
        if _ndi_module() is None:
            raise RuntimeError("NDI audio needs the ndi-python package: pip install ndi-python")
        self.addresses = parse_ndi_addresses(addresses)      # static unicast addresses to query (other subnets)
        self.source_name = source_name
        self.channels = channels
        self._ndi = _ndi_module()
        self._recv = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._rate = 0
        self._n_channels = 0
        self._found_name = source_name

    def describe(self) -> str:
        info = f" ({self._rate} Hz, {self._n_channels}ch)" if self._rate else ""
        return f"NDI '{self._found_name}'{info}"

    def start(self):
        ndi = self._ndi
        if not ndi.initialize():
            raise RuntimeError("Could not start the NDI runtime")
        finder = _make_finder(ndi, self.addresses)
        target = None
        try:
            deadline = time.time() + self.DISCOVER_TIMEOUT_S
            wanted = self.source_name.lower()
            while time.time() < deadline and target is None:
                ndi.find_wait_for_sources(finder, 1000)
                for s in ndi.find_get_current_sources(finder):
                    if wanted in s.ndi_name.lower():
                        target = s
                        break
            if target is None:
                where = (f"at {self.addresses} or on the local network" if self.addresses
                         else "on the local network (if it is on another subnet/VLAN, set its IP address in Settings → NDI)")
                raise RuntimeError(f"NDI source '{self.source_name}' not found {where} "
                                   f"(waited {self.DISCOVER_TIMEOUT_S:.0f}s). Check it is running and reachable.")
            settings = ndi.RecvCreateV3()
            settings.source_to_connect_to = target
            settings.color_format = ndi.RECV_COLOR_FORMAT_FASTEST
            settings.bandwidth = ndi.RECV_BANDWIDTH_AUDIO_ONLY
            self._recv = ndi.recv_create_v3(settings)
            if self._recv is None:
                raise RuntimeError("Could not connect to the NDI source")
            self._found_name = target.ndi_name
        finally:
            ndi.find_destroy(finder)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="NdiSource")
        self._thread.start()
        logger.info(f"Audio capture started — {self.describe()}")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        if self._recv is not None:
            self._ndi.recv_destroy(self._recv)
            self._recv = None
        logger.info("Audio capture stopped")

    def _run(self):
        ndi = self._ndi
        last_audio = time.time()
        warned = False
        while not self._stop.is_set():
            try:
                ftype, _v, a, _m = ndi.recv_capture_v2(self._recv, 100)
            except Exception as e:
                logger.error(f"NDI receive error: {e}", exc_info=True)
                break
            if ftype != ndi.FRAME_TYPE_AUDIO:
                if time.time() - last_audio > 5 and not warned:
                    logger.warning("No NDI audio received for 5 s (source stopped or network problem)")
                    warned = True
                continue
            try:
                data = np.asarray(a.data, dtype=np.float32)          # (channels, samples), planar
                rate = int(a.sample_rate)
                if data.ndim == 1:
                    data = data[None, :]
                self._rate, self._n_channels = rate, data.shape[0]
                chans = self.channels or list(range(1, data.shape[0] + 1))
                sel = [data[c - 1] for c in chans if 0 < c <= data.shape[0]]
                if sel and data.shape[1] > 0:
                    mono = np.mean(np.stack(sel, axis=0), axis=0)
                    self._emit(mono, rate, time.time() - data.shape[1] / rate)
                last_audio, warned = time.time(), False
            finally:
                ndi.recv_free_audio_v2(self._recv, a)


class FileSource(AudioSource):
    """Plays a 16-bit PCM .wav file as if it were live audio (real time)."""

    def __init__(self, path: str, channels: Optional[List[int]] = None, on_frame=None, realtime: bool = True):
        super().__init__(on_frame)
        self.path = Path(path)
        self.channels = channels
        self.realtime = realtime
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        with wave.open(str(self.path), "rb") as w:
            if w.getsampwidth() != 2:
                raise RuntimeError(f"{self.path.name}: only 16-bit PCM .wav files are supported (convert with ffmpeg)")
            self._native_rate = w.getframerate()
            self._file_channels = w.getnchannels()
            self._duration = w.getnframes() / self._native_rate
        self.finished = threading.Event()

    def describe(self) -> str:
        return f"file '{self.path.name}' ({self._duration:.0f}s @ {self._native_rate} Hz)"

    def start(self):
        self._stop.clear()
        self.finished.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="FileSource")
        self._thread.start()
        logger.info(f"Audio capture started — {self.describe()}")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        block = int(self._native_rate * 0.05)
        t0 = time.time()
        sent = 0
        with wave.open(str(self.path), "rb") as w:
            while not self._stop.is_set():
                raw = w.readframes(block)
                if not raw:
                    break
                pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, self._file_channels).astype(np.float32) / 32768.0
                chans = self.channels or list(range(1, self._file_channels + 1))
                sel = [pcm[:, c - 1] for c in chans if 0 < c <= self._file_channels]
                mono = np.mean(np.stack(sel, axis=1), axis=1) if sel else pcm[:, 0]
                self._emit(mono, self._native_rate, t0 + sent / self._native_rate)
                sent += len(pcm)
                if self.realtime:
                    delay = (t0 + sent / self._native_rate) - time.time()
                    if delay > 0:
                        self._stop.wait(delay)
        self.finished.set()
        logger.info("Audio file finished")
