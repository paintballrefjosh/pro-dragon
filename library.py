"""
Song library
=============
Finds the songs saved under the reports folder (reports/session_<date>/song_NN_*.json),
loads them for the GUI, and re-analyzes them with different settings.

Song ids are paths relative to the reports folder ("session_x/song_01_foo.json");
every id is resolved and checked to stay inside the reports folder.
"""

import json
from pathlib import Path
from typing import Callable, List, Optional

from analysis.song_runner import analyze_songdata, load_song, save_song
from analysis.operators import aggregate
from analysis.song_timing import HeardWords, SlideTiming, UnshownLyric, margins_from_cfg, summarize
from report.graph import render_operator_trend, render_song_graph
from utils.logger import get_logger

logger = get_logger("library")

FILE_KINDS = {"json": ".json", "txt": ".txt", "svg": ".svg", "wav": ".wav"}


class LibraryError(Exception):
    pass


class SongLibrary:
    def __init__(self, get_output_dir: Callable[[], str]):
        self._get_output_dir = get_output_dir

    @property
    def root(self) -> Path:
        return Path(self._get_output_dir()).resolve()

    # --- ids -------------------------------------------------------------

    def resolve(self, song_id: str) -> Path:
        if not song_id or not song_id.endswith(".json"):
            raise LibraryError("bad song id")
        path = (self.root / song_id).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            raise LibraryError("bad song id")
        if not path.is_file():
            raise LibraryError("song not found")
        return path

    def id_for(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    # --- listing -----------------------------------------------------------

    def list(self) -> List[dict]:
        root = self.root
        if not root.exists():
            return []
        songs = []
        for p in root.glob("session_*/song_*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Skipping unreadable {p.name}: {e}")
                continue
            summary = data.get("summary") or {}
            songs.append({
                "id": self.id_for(p),
                "name": data.get("name", p.stem),
                "operator": data.get("operator", "") or "",
                "session": p.parent.name,
                "started_at": data.get("song_start_ts", 0),
                "duration": max(0.0, data.get("song_end_ts", 0) - data.get("song_start_ts", 0)),
                "transitions": summary.get("transitions", len(data.get("timings", []))),
                "measured": summary.get("measured", 0),
                "pct_on_time": summary.get("pct_on_time"),
                "pct_late": summary.get("pct_late"),
                "pct_early": summary.get("pct_early"),
                "analyzed": bool(data.get("timings")),
                "problems": {
                    "not_heard": sum(1 for t in data.get("timings", [])
                                     if t.get("delay") is None and (t.get("text") or "").strip()),
                    "not_shown": len(data.get("unshown", [])),
                    "no_slide": len(data.get("heard_other", [])),
                },
            })
        songs.sort(key=lambda s: s["started_at"], reverse=True)
        return songs

    # --- one song ----------------------------------------------------------

    def detail(self, song_id: str) -> dict:
        path = self.resolve(song_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "id": song_id,
            "name": data.get("name", path.stem),
            "operator": data.get("operator", "") or "",
            "started_at": data.get("song_start_ts", 0),
            "duration": max(0.0, data.get("song_end_ts", 0) - data.get("song_start_ts", 0)),
            "settings": data.get("settings", {}),
            "timings": data.get("timings", []),
            "unshown": data.get("unshown", []),
            "heard_other": data.get("heard_other", []),
            "summary": data.get("summary", {}),
            "notes": data.get("notes") or [],
            "presentation_uuid": data.get("presentation_uuid", ""),
            "slides_remapped": data.get("slides_before_remap") is not None,
            "has_words": data.get("words") is not None,
            "has_audio": bool(data.get("audio_file")) and (path.parent / data["audio_file"]).is_file(),
            "files": {k: (path.with_suffix(ext)).is_file() for k, ext in FILE_KINDS.items() if k != "wav"},
        }

    def graph_svg(self, song_id: str, late_s: float, early_s: float) -> str:
        d = self.detail(song_id)
        return render_song_graph(
            d["name"], d["duration"],
            [SlideTiming(**t) for t in d["timings"]],
            [UnshownLyric(**u) for u in d["unshown"]],
            [HeardWords(**h) for h in d["heard_other"]],
            late_s, early_s, operator=d["operator"],
        )

    # --- deleting ---------------------------------------------------------------

    def delete(self, song_ids: List[str]) -> dict:
        """
        Permanently delete saved songs: each one's .json/.txt/.svg and its audio .wav. A session
        folder left with no songs is cleaned up too (its session_report.txt describes songs that
        no longer exist). Returns {"deleted": [ids], "failed": {id: reason}, "files": n}.
        Every id is path-checked, so nothing outside the reports folder can be touched.
        """
        deleted: List[str] = []
        failed = {}
        files = 0
        folders = set()
        for song_id in dict.fromkeys(song_ids or []):         # de-duplicated, order kept
            try:
                path = self.resolve(song_id)
                try:
                    audio = json.loads(path.read_text(encoding="utf-8")).get("audio_file") or ""
                except Exception:
                    audio = ""
                targets = [path.with_suffix(ext) for ext in (".txt", ".svg")] + [path.with_suffix(".wav")]
                if audio:
                    targets.append(path.parent / Path(audio).name)       # name only: never follow a path out of the folder
                for t in targets:
                    if t.is_file():
                        t.resolve().relative_to(self.root)
                        t.unlink()
                        files += 1
                path.unlink()                                             # the .json last: it is what lists the song
                files += 1
                folders.add(path.parent)
                deleted.append(song_id)
            except (LibraryError, OSError, ValueError) as e:
                failed[song_id] = str(e) or "could not delete"
        for folder in folders:
            try:
                if folder != self.root and not any(folder.glob("song_*.json")):
                    report = folder / "session_report.txt"
                    if report.is_file():
                        report.unlink()
                    folder.rmdir()                                        # only succeeds if it is now empty
            except OSError:
                pass
        if deleted:
            logger.info(f"Deleted {len(deleted)} song(s) ({files} files)")
        return {"deleted": deleted, "failed": failed, "files": files}

    # --- operators -------------------------------------------------------------

    @staticmethod
    def clean_operator(name) -> str:
        """Trim and validate an operator name (no control characters, max 60 chars)."""
        s = " ".join(str(name or "").split())
        if len(s) > 60:
            raise LibraryError("operator name is too long (60 characters max)")
        if any(ord(c) < 32 for c in s):
            raise LibraryError("operator name has invalid characters")
        return s

    def set_operator(self, song_id: str, name: str) -> dict:
        """Change who ran a saved song; regenerates its .json/.txt/.svg (the analysis itself is untouched)."""
        name = self.clean_operator(name)
        path = self.resolve(song_id)
        song = load_song(path)
        song.operator = name
        saved = json.loads(path.read_text(encoding="utf-8")).get("settings", {})
        late = saved.get("on_time_late_s", saved.get("on_time_window_s", 0.75))
        early = saved.get("on_time_early_s", late)
        save_song(song, path.parent, 0, saved.get("offset_ms", 0), late, early, stem=path.stem)
        return self.detail(song_id)

    def _all_details(self) -> List[dict]:
        out = []
        for s in self.list():
            try:
                out.append(self.detail(s["id"]))
            except Exception as e:                      # one broken file must not hide the rest
                logger.warning(f"Skipping {s['id']}: {e}")
        return out

    def operators(self, late_s: float, early_s: float) -> List[dict]:
        return aggregate(self._all_details(), late_s, early_s)

    def operator_names(self) -> List[str]:
        return sorted({s["operator"] for s in self.list() if s["operator"]}, key=str.lower)

    def operator_graph_svg(self, name: str, late_s: float, early_s: float) -> str:
        for e in self.operators(late_s, early_s):
            if e["name"] == name:
                return render_operator_trend(name, e["per_song"], late_s, early_s)
        raise LibraryError("operator not found")

    def file_path(self, song_id: str, kind: str) -> Path:
        path = self.resolve(song_id)
        if kind not in FILE_KINDS:
            raise LibraryError("unknown file type")
        if kind == "wav":
            name = json.loads(path.read_text(encoding="utf-8")).get("audio_file", "")
            target = (path.parent / name).resolve()
            try:
                target.relative_to(self.root)
            except ValueError:
                raise LibraryError("bad file")
        else:
            target = path.with_suffix(FILE_KINDS[kind])
        if not target.is_file():
            raise LibraryError("file not found")
        return target

    # --- fixing the slide list ------------------------------------------------------

    def remap_slides(self, song_id: str, new_slides: List[dict], timing_cfg: dict, offset_ms: float) -> dict:
        """
        Replace a saved song's slide list (e.g. with ProPresenter's current list, after the presentation
        was edited mid-song) and re-read every slide change through it, using the words already
        transcribed. The original list is kept in the file (`slides_before_remap`) and a note is added.
        """
        from datetime import datetime
        path = self.resolve(song_id)
        song = load_song(path)
        if song.words is None:
            raise LibraryError("this song has no transcript yet — re-analyze it first")
        if not new_slides:
            raise LibraryError("the new slide list is empty")
        old_n = len(song.slides)
        if song.slides_before_remap is None:
            song.slides_before_remap = song.slides
        song.slides = new_slides
        for e in song.events:
            e.text = e.group = None                     # resolve through the corrected list, not an old snapshot
        song.notes.append(f"Slide list replaced with ProPresenter's current list ({old_n} → {len(new_slides)} slides) on "
                          f"{datetime.now().strftime('%b %d %I:%M %p')}; every slide change was re-read with it. "
                          f"This assumes the new list is right for the whole song.")
        analyze_songdata(song, None, timing_cfg, offset_ms)
        early, late = margins_from_cfg(timing_cfg)
        save_song(song, path.parent, 0, offset_ms, late, early, stem=path.stem)
        return self.detail(song_id)

    # --- re-analysis ---------------------------------------------------------

    def reanalyze(self, song_id: str, timing_cfg: dict, offset_ms: float,
                  whisper_provider: Optional[Callable] = None, retranscribe: bool = False) -> dict:
        """Re-run the analysis on a saved song (cached words unless `retranscribe`)
        with the given settings, rewrite its files, and return the new detail."""
        path = self.resolve(song_id)
        song = load_song(path)
        whisper = None
        if retranscribe or song.words is None:
            if whisper_provider is None:
                raise LibraryError("this song needs transcribing but speech recognition isn't available")
            whisper = whisper_provider()
        analyze_songdata(song, whisper, timing_cfg, offset_ms, force_transcribe=retranscribe)
        early, late = margins_from_cfg(timing_cfg)
        save_song(song, path.parent, 0, offset_ms, late, early, stem=path.stem)   # keeps song.operator
        return self.detail(song_id)
