"""
Song data: save / load / analyze
=================================
A `SongData` is everything captured for one song — the slides, the slide-change
events, and the recorded audio — plus (once analyzed) the words heard and the
per-slide timings. It is saved as a .wav + .json pair so a song can be
re-analyzed later (different offset, bigger Whisper model, ...) without
replaying the service:

    python main.py --analyze reports/session_.../song_01_xxx.json --offset-ms 200
"""

import json
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from analysis.song_timing import HeardWords, SlideEvent, SlideTiming, UnshownLyric, Word, analyze_song, summarize
from audio.capture import StreamResampler, TARGET_RATE
from report.graph import render_song_graph
from report.timing_report import format_song_report
from utils.logger import get_logger

logger = get_logger("song")


@dataclass
class SongData:
    name: str
    presentation_uuid: str
    slides: List[dict]                       # [{"index","text","group","label"}]
    song_start_ts: float
    song_end_ts: float
    audio_start_ts: float
    events: List[SlideEvent]
    audio: Optional[np.ndarray] = None       # float32 mono @16k (not kept after saving)
    audio_file: str = ""
    words: Optional[List[tuple]] = None      # [(text, start, end)] relative to audio start
    timings: List[SlideTiming] = field(default_factory=list)
    unshown: List[UnshownLyric] = field(default_factory=list)   # sung, but no slide change matched
    heard_other: List[HeardWords] = field(default_factory=list)  # words heard that no slide explains
    directory: Optional[Path] = None        # where the .wav lives (set when loaded from disk)
    operator: str = ""                       # who ran the slides (typed in when the song was recorded)
    notes: List[str] = field(default_factory=list)   # things to know about this recording (e.g. the slide list changed mid-song)
    slides_before_remap: Optional[List[dict]] = None  # the original slide list, kept when it was replaced by a corrected one

    @property
    def duration(self) -> float:
        return max(0.0, self.song_end_ts - self.song_start_ts)

    # --- (de)serialization ------------------------------------------------

    def to_json(self, offset_ms: float, window_s: float, early_s: Optional[float] = None) -> dict:
        return {
            "name": self.name,
            "presentation_uuid": self.presentation_uuid,
            "slides": self.slides,
            "song_start_ts": self.song_start_ts,
            "song_end_ts": self.song_end_ts,
            "audio_start_ts": self.audio_start_ts,
            "audio_file": self.audio_file,
            "operator": self.operator,
            "notes": self.notes,
            **({"slides_before_remap": self.slides_before_remap} if self.slides_before_remap is not None else {}),
            "events": [self._event_json(e) for e in self.events],
            "settings": {"offset_ms": offset_ms, "on_time_window_s": window_s,
                         "on_time_late_s": window_s, "on_time_early_s": window_s if early_s is None else early_s},
            "words": self.words,
            "timings": [t.to_dict() for t in self.timings],
            "unshown": [u.to_dict() for u in self.unshown],
            "heard_other": [h.to_dict() for h in self.heard_other],
            "summary": summarize(self.timings, window_s, early_s),
        }

    @staticmethod
    def _event_json(e: SlideEvent) -> dict:
        d = {"ts": e.ts, "index": e.index, "initial": e.initial}
        if e.text is not None:                       # snapshot of the slide at that moment
            d["text"], d["group"] = e.text, e.group or ""
        return d

    @classmethod
    def from_json(cls, data: dict, directory: Path) -> "SongData":
        """Rebuild a saved song, including its saved analysis (so files can be regenerated, e.g. after
        changing the operator, without re-running the analysis)."""
        return cls(
            name=data["name"],
            presentation_uuid=data.get("presentation_uuid", ""),
            slides=data["slides"],
            song_start_ts=data["song_start_ts"],
            song_end_ts=data["song_end_ts"],
            audio_start_ts=data.get("audio_start_ts", data["song_start_ts"]),
            events=[SlideEvent(e["ts"], e["index"], e.get("initial", False), e.get("text"), e.get("group"))
                    for e in data["events"]],
            audio_file=data.get("audio_file", ""),
            words=[tuple(w) for w in data["words"]] if data.get("words") is not None else None,
            timings=[SlideTiming(**t) for t in data.get("timings", [])],
            unshown=[UnshownLyric(**u) for u in data.get("unshown", [])],
            heard_other=[HeardWords(**h) for h in data.get("heard_other", [])],
            directory=directory,
            operator=data.get("operator", "") or "",
            notes=list(data.get("notes") or []),
            slides_before_remap=data.get("slides_before_remap"),
        )


class _SlideView:
    """Adapter so plain slide dicts look like Pro7 Slide objects to analyze_song()."""
    def __init__(self, d: dict):
        self.index = d["index"]
        self.text = d.get("text", "")
        self.group = d.get("group", "")


# ──────────────────────────────────────────────────────────────────────
# WAV helpers
# ──────────────────────────────────────────────────────────────────────

def write_wav(path: Path, audio: np.ndarray, rate: int = TARGET_RATE):
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise RuntimeError(f"{path.name}: only 16-bit PCM wav is supported")
        rate, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch).astype(np.float32) / 32768.0
    mono = pcm.mean(axis=1)
    return mono if rate == TARGET_RATE else StreamResampler(rate, TARGET_RATE).process(mono)


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:40] or "song"


# ──────────────────────────────────────────────────────────────────────
# Analysis + output
# ──────────────────────────────────────────────────────────────────────

def lyrics_prompt(slides: List[dict], max_chars: int = 700) -> str:
    text = " ".join(" ".join(s.get("text", "").split()) for s in slides if s.get("text"))
    return text[:max_chars]


def analyze_songdata(song: SongData, whisper, timing_cfg: dict, offset_ms: float, force_transcribe: bool = False):
    """Fill song.words (transcribing if needed) and song.timings."""
    if song.words is None or force_transcribe:
        if whisper is None:
            raise RuntimeError("This song has no transcript yet and no Whisper model is loaded")
        if song.audio is None:
            if song.directory is None or not song.audio_file:
                raise RuntimeError("No audio available for this song")
            song.audio = read_wav(song.directory / song.audio_file)
        prompt = lyrics_prompt(song.slides) if timing_cfg.get("use_lyrics_prompt", True) else None
        song.words = whisper.transcribe_words(
            song.audio, initial_prompt=prompt, vad_filter=timing_cfg.get("vad_filter", False)
        )

    result = analyze_song(
        [_SlideView(s) for s in song.slides],
        song.events,
        [Word(*w) for w in song.words],
        song_start_ts=song.song_start_ts,
        audio_start_ts=song.audio_start_ts,
        offset_s=offset_ms / 1000.0,
        min_confidence=timing_cfg.get("min_match_confidence", 0.25),
        max_window_s=timing_cfg.get("max_match_window_s", 20.0),
        other_min_words=timing_cfg.get("other_min_words", 4),
        other_min_prob=timing_cfg.get("other_min_confidence", 0.5),
    )
    song.timings, song.unshown, song.heard_other = result.timings, result.unshown, result.heard_other


def save_song(song: SongData, directory: Path, number: int, offset_ms: float, window_s: float,
              early_s: Optional[float] = None, stem: Optional[str] = None) -> Path:
    """Write the .wav (if audio is in memory), .json, .txt report and .svg graph for a song.
    `window_s` = late margin (and early margin unless `early_s` given). Returns the json path."""
    directory.mkdir(parents=True, exist_ok=True)
    stem = stem or f"song_{number:02d}_{slugify(song.name)}"
    if song.audio is not None and not song.audio_file:
        song.audio_file = f"{stem}.wav"
        write_wav(directory / song.audio_file, song.audio)

    json_path = directory / f"{stem}.json"
    json_path.write_text(json.dumps(song.to_json(offset_ms, window_s, early_s), indent=2), encoding="utf-8")
    report = format_song_report(song.name, song.song_start_ts, song.duration, song.timings, offset_ms, window_s,
                                unshown=song.unshown, heard_other=song.heard_other, early_s=early_s,
                                operator=song.operator, notes=song.notes)
    (directory / f"{stem}.txt").write_text(report, encoding="utf-8")
    (directory / f"{stem}.svg").write_text(
        render_song_graph(song.name, song.duration, song.timings, song.unshown, song.heard_other, window_s, early_s,
                          operator=song.operator),
        encoding="utf-8")
    return json_path


def load_song(json_path: Path) -> SongData:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return SongData.from_json(data, json_path.parent)
