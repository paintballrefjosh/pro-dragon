"""
Report Generator
=================
Collects lag events throughout a session and writes a formatted report
at the end of the session (or on demand).

Text report format:
  - Session header (date, mode, config)
  - Per-song breakdown with lag events
  - Session summary (total events, worst offender)

JSON format: machine-readable for future tooling / dashboards.
"""

import json
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List

from alignment.engine import LagEvent
from utils.logger import get_logger

logger = get_logger("report")


class SongRecord:
    """Aggregates all lag events for a single song in the session."""

    def __init__(self, name: str, slide_count: int):
        self.name = name
        self.slide_count = slide_count
        self.start_time = time.time()
        self.end_time: float = 0.0
        self.lag_events: List[LagEvent] = []
        # Chronological [(timestamp, slide_index), ...], recorded live —
        # used for post-hoc (report-mode) replay against the full-song transcript.
        self.slide_history: List[tuple] = []
        # Full-song transcript text, if a report-mode Whisper pass produced one.
        self.full_transcript: str = ""

    def finish(self):
        self.end_time = time.time()

    def total_lag_seconds(self) -> float:
        return sum(e.lag_seconds for e in self.lag_events)

    def worst_event(self) -> "LagEvent | None":
        if not self.lag_events:
            return None
        return max(self.lag_events, key=lambda e: e.lag_seconds)


class ReportGenerator:
    def __init__(self, output_dir: str = "./reports", fmt: str = "text", auto_open: bool = False):
        self.output_dir = Path(output_dir)
        self.fmt = fmt.lower()
        self.auto_open = auto_open
        self._songs: List[SongRecord] = []
        self._current_song: SongRecord | None = None
        self._session_start = time.time()
        self._session_label = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Event intake
    # ------------------------------------------------------------------

    def start_song(self, name: str, slide_count: int) -> "SongRecord | None":
        """
        Start tracking a new song. If a previous song was still open (Pro7
        loaded a new presentation without an intervening clearAll), it's
        finalized here instead — returned so the caller can still flush its
        recorded audio for report-mode transcription.
        """
        finished = None
        if self._current_song:
            self._current_song.finish()
            finished = self._current_song
            self._songs.append(self._current_song)
        self._current_song = SongRecord(name, slide_count)
        logger.info(f"Report: tracking song '{name}'")
        return finished

    def record_lag_event(self, event: LagEvent):
        if self._current_song:
            self._current_song.lag_events.append(event)

    def record_slide_change(self, index: int, timestamp: float):
        if self._current_song:
            self._current_song.slide_history.append((timestamp, index))

    def end_song(self) -> "SongRecord | None":
        """Finalize the current song and return its record (e.g. so a
        background report-mode transcription can attach results to it after
        it's no longer the 'current' song)."""
        record = None
        if self._current_song:
            self._current_song.finish()
            record = self._current_song
            self._songs.append(self._current_song)
            self._current_song = None
        return record

    def end_session(self) -> str:
        """Finalize and write the report. Returns the file path."""
        if self._current_song:
            self.end_song()
        path = self._write()
        if self.auto_open:
            self._open_file(path)
        return path

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def _write(self) -> str:
        if self.fmt == "json":
            return self._write_json()
        return self._write_text()

    def _write_text(self) -> str:
        path = self.output_dir / f"session_{self._session_label}.txt"
        session_duration = time.time() - self._session_start
        total_events = sum(len(s.lag_events) for s in self._songs)

        lines = []
        lines.append("=" * 65)
        lines.append("  PRO DRAGON — SESSION REPORT")
        lines.append("=" * 65)
        lines.append(f"  Date    : {datetime.now().strftime('%A, %B %d, %Y')}")
        lines.append(f"  Time    : {datetime.now().strftime('%I:%M %p')}")
        lines.append(f"  Duration: {_fmt_duration(session_duration)}")
        lines.append(f"  Songs   : {len(self._songs)}")
        lines.append(f"  Total Lag Events: {total_events}")
        lines.append("=" * 65)
        lines.append("")

        worst_song = None
        worst_seconds = 0.0
        for song in self._songs:
            s = self._format_song(song)
            lines.extend(s)
            lines.append("")
            if song.total_lag_seconds() > worst_seconds:
                worst_seconds = song.total_lag_seconds()
                worst_song = song

        lines.append("─" * 65)
        lines.append("  SESSION SUMMARY")
        lines.append("─" * 65)
        if total_events == 0:
            lines.append("  No significant slide lag detected this session.  ✓")
        else:
            lines.append(f"  Total lag events : {total_events}")
            if worst_song:
                lines.append(f"  Most lag         : '{worst_song.name}' ({worst_song.total_lag_seconds():.1f}s total)")
                worst = worst_song.worst_event()
                if worst:
                    lines.append(f"  Worst single lag : slide {worst.slide_index + 1} — {worst.lag_seconds:.1f}s")
        lines.append("─" * 65)

        content = "\n".join(lines)
        path.write_text(content, encoding="utf-8")
        logger.info(f"Report written to: {path}")
        print(f"\n{'=' * 65}")
        print(content)
        print(f"{'=' * 65}\n")
        return str(path)

    def _format_song(self, song: SongRecord) -> List[str]:
        lines = []
        status = "✓" if not song.lag_events else "⚠"
        duration = _fmt_duration(song.end_time - song.start_time) if song.end_time else "?"
        lines.append(f"  {status}  {song.name}  ({duration}, {song.slide_count} slides)")
        lines.append("  " + "─" * 55)

        if not song.lag_events:
            lines.append("     No lag events detected.")
        else:
            for ev in song.lag_events:
                ts = datetime.fromtimestamp(ev.timestamp).strftime("%H:%M:%S")
                lines.append(f"     [{ts}]  Slide {ev.slide_index + 1}:")
                lines.append(f"       Display : \"{_truncate(ev.slide_text, 50)}\"")
                lines.append(f"       Vocalist: singing slide {ev.vocalist_index + 1}")
                lines.append(
                    f"       Lag     : {ev.lag_phrases} phrase(s) ahead  "
                    f"({ev.lag_seconds:.1f}s)"
                )
                if ev.asr_text:
                    lines.append(f"       ASR text: \"{_truncate(ev.asr_text, 50)}\"")
                lines.append("")

        if song.full_transcript:
            lines.append("     Full transcript:")
            lines.append(f"       \"{_truncate(song.full_transcript, 200)}\"")
            lines.append("")
        return lines

    def _write_json(self) -> str:
        path = self.output_dir / f"session_{self._session_label}.json"
        data = {
            "session": {
                "date": self._session_label,
                "duration_seconds": time.time() - self._session_start,
            },
            "songs": [
                {
                    "name": s.name,
                    "slide_count": s.slide_count,
                    "duration_seconds": s.end_time - s.start_time if s.end_time else 0,
                    "total_lag_seconds": s.total_lag_seconds(),
                    "full_transcript": s.full_transcript,
                    "lag_events": [
                        {
                            "timestamp": e.timestamp,
                            "slide_index": e.slide_index,
                            "slide_text": e.slide_text,
                            "vocalist_index": e.vocalist_index,
                            "lag_phrases": e.lag_phrases,
                            "lag_seconds": e.lag_seconds,
                            "asr_text": e.asr_text,
                        }
                        for e in s.lag_events
                    ],
                }
                for s in self._songs
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"JSON report written to: {path}")
        return str(path)

    def _open_file(self, path: str):
        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.run(["open", path])
            elif system == "Windows":
                os.startfile(path)
            else:
                subprocess.run(["xdg-open", path])
        except Exception as e:
            logger.warning(f"Could not auto-open report: {e}")


def _fmt_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s:02d}s"


def _truncate(text: str, length: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= length else text[: length - 1] + "…"
