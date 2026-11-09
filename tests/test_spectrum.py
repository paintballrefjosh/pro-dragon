"""The equalizer bands: a known tone must land in the right band at the right level."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio.capture import (SPECTRUM_BANDS, AudioSource, band_levels_db, spectrum_edges)  # noqa: E402

RATE = 16000


def tone(freq, amp, seconds=0.5):
    t = np.arange(int(RATE * seconds)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def band_of(freq):
    e = spectrum_edges()
    return next(i for i in range(SPECTRUM_BANDS) if e[i] <= freq < e[i + 1])


def test_edges_are_log_spaced_and_cover_the_voice_range():
    e = spectrum_edges()
    assert len(e) == SPECTRUM_BANDS + 1 and abs(e[0] - 60) < 1e-6 and abs(e[-1] - 8000) < 1e-6
    ratios = [b / a for a, b in zip(e, e[1:])]
    assert max(ratios) - min(ratios) < 1e-9


def test_a_tone_shows_in_its_band_at_its_level():
    for f in (150, 440, 1000, 3000, 6000):
        db = band_levels_db(tone(f, 0.3), RATE)
        top = int(np.argmax(db))
        assert abs(top - band_of(f)) <= 1, (f, top)                       # leakage may touch a neighbour, no more
        assert abs(db[band_of(f)] - 20 * np.log10(0.3)) < 1.5, (f, db[band_of(f)])
        far = [v for i, v in enumerate(db) if abs(i - band_of(f)) > 3]
        assert max(far) < db[top] - 30                                    # the rest of the spectrum stays low


def test_two_tones_light_two_bands_and_silence_is_floor():
    x = tone(250, 0.2) + tone(2500, 0.2)
    db = band_levels_db(x, RATE)
    assert db[band_of(250)] > -20 and db[band_of(2500)] > -20
    assert db[band_of(800)] < -50
    assert set(band_levels_db(np.zeros(4096, np.float32), RATE)) == {-96.0}
    assert len(band_levels_db(np.zeros(10, np.float32), RATE)) == SPECTRUM_BANDS       # short input is padded


class Dummy(AudioSource):
    def start(self): pass
    def stop(self): pass
    def describe(self): return "dummy"


def test_source_reports_bands_then_goes_quiet_when_audio_stops():
    src = Dummy()
    assert set(src.get_bands_db()) == {-96.0}                              # nothing received yet
    for i in range(0, 8000, 1600):                                          # 0.5 s of 1 kHz in 100 ms blocks (already 16 kHz)
        src._emit(tone(1000, 0.3, 0.1), RATE, time.time())
    db = src.get_bands_db()
    assert int(np.argmax(db)) in (band_of(1000) - 1, band_of(1000), band_of(1000) + 1)
    src._last_emit -= 5                                                     # pretend the stream stalled 5 s ago
    assert set(src.get_bands_db()) == {-96.0}
