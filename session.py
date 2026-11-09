"""
Session
========
One monitoring session: an audio source is running, and the operator starts/stops
songs (terminal key `S`, or the GUI buttons).

  * ProPresenter client  -> slide-change events (with timestamps)
  * Audio source         -> 16 kHz mono blocks (with capture timestamps)
  * Start/Stop a song    -> only audio and slide changes inside that window are kept
  * When a song stops, its audio is transcribed with word timestamps (Whisper, in a
    background thread), matched to the lyrics, and a per-slide delay report +
    graph is written.

The Session does not own the ProPresenter client, the Whisper model or the web
server — whoever creates it (the terminal runner or the GUI controller) does.
"""

import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, List, Optional

import numpy as np

from analysis.live import LiveAnalyzer
from analysis.song_runner import SongData, analyze_songdata, lyrics_prompt, save_song
from analysis.song_timing import SlideEvent, margins_from_cfg, summarize
from report.timing_report import fmt_clock, format_song_report, one_line
from utils.logger import get_logger

logger = get_logger("session")

SAMPLE_RATE = 16000
TAIL_WAIT_S = 0.3        # let the last in-flight audio blocks arrive after Stop


class _Recording:
    """State for the song currently being recorded."""

    def __init__(self, name: str, presentation, start_ts: float, initial_index: Optional[int]):
        self.name = name
        self.presentation = presentation
        self.start_ts = start_ts
        self.end_ts: Optional[float] = None
        self.events: List[SlideEvent] = []
        if initial_index is not None:
            self.events.append(SlideEvent(ts=start_ts, index=initial_index, initial=True))
        self.blocks: List[np.ndarray] = []
        self.audio_start_ts: Optional[float] = None
        self.live: Optional[LiveAnalyzer] = None          # provisional in-song delays (if enabled)
        self.notes: List[str] = []                        # saved with the song (e.g. the slide list changed mid-song)


class Session:
    def __init__(self, cfg: dict, pro7, audio_source_factory: Callable, whisper_provider: Callable,
                 dashboard=None, offset_ms: Optional[float] = None, console: bool = True, operator: str = ""):
        """
        whisper_provider: callable returning a loaded WhisperASR (may block while it loads);
                          called from the analysis thread.
        offset_ms: fixed override of timing.offset_ms (e.g. from --offset-ms); None = follow config.
        """
        self.cfg = cfg
        self.pro7 = pro7
        self.whisper_provider = whisper_provider
        self.dashboard = dashboard
        self.console = console
        self._offset_override = offset_ms
        self.operator = operator          # who is running the slides; stamped on each song when it stops

        self.source = audio_source_factory(self._on_audio_frame)

        self._lock = threading.RLock()
        self._rec: Optional[_Recording] = None
        self._song_count = 0
        self._analysis_lock = threading.Lock()      # one Whisper job at a time
        self._analysis_threads: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._finished_reports: List[str] = []
        self._messages: Deque[dict] = deque(maxlen=30)
        self.last_song_id: Optional[str] = None
        # live (in-song) estimates
        self._live_thread: Optional[threading.Thread] = None
        self._live_stop = threading.Event()
        self._live_timings: list = []                    # provisional SlideTiming list for the current song
        self._live_announced: set = set()

        label = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_dir = Path(cfg["report"]["output_dir"]) / f"session_{label}"
        self.status = "idle"        # idle | recording | analyzing
        self._last_song_name = ""

    # ------------------------------------------------------------------
    # Live configuration (re-read at analysis time, so GUI changes apply)
    # ------------------------------------------------------------------

    def update_config(self, cfg: dict):
        self.cfg = cfg

    @property
    def timing_cfg(self) -> dict:
        return self.cfg.get("timing", {})

    @property
    def offset_ms(self) -> float:
        if self._offset_override is not None:
            return self._offset_override
        return float(self.timing_cfg.get("offset_ms", 0))

    @property
    def margins(self):
        """(early_s, late_s)"""
        return margins_from_cfg(self.timing_cfg)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _on_audio_frame(self, block: np.ndarray, ts: float):
        with self._lock:
            rec = self._rec
            if rec is None:
                return
            block_end = ts + len(block) / SAMPLE_RATE
            if block_end <= rec.start_ts:
                return                                   # entirely before Start
            if rec.end_ts is not None and ts >= rec.end_ts:
                return                                   # entirely after Stop
            if ts < rec.start_ts:                        # straddles Start: trim the head
                cut = int(round((rec.start_ts - ts) * SAMPLE_RATE))
                block, ts = block[cut:], ts + cut / SAMPLE_RATE
            if rec.end_ts is not None and block_end > rec.end_ts:   # straddles Stop: trim the tail
                keep = int(round((rec.end_ts - ts) * SAMPLE_RATE))
                block = block[:max(0, keep)]
            if len(block) == 0:
                return
            if rec.audio_start_ts is None:
                rec.audio_start_ts = ts
            rec.blocks.append(block.copy())

    # ------------------------------------------------------------------
    # Pro7 callbacks
    # ------------------------------------------------------------------

    def _on_slide_changed(self, index: int, ts: float):
        with self._lock:
            rec = self._rec
            if rec is None:
                return
            if rec.events and rec.events[-1].index == index:
                return                                   # same slide we already have (incl. live-at-Start)
            ev = SlideEvent(ts=max(ts, rec.start_ts), index=index)
            rec.events.append(ev)
            slide = self._snapshot(rec, ev)               # remember WHAT this slide was: the list can change mid-song
            elapsed = ts - rec.start_ts
        text = one_line(slide.text, 50) if slide and slide.text else "(blank slide)"
        self.log(f"▶ Slide {index + 1:<3} at {fmt_clock(elapsed)}   {text}", kind="slide")
        self._push_live()

    def _on_presentation_changed(self, pres):
        with self._lock:
            rec = self._rec
            changed = rec is not None and pres.uuid != rec.presentation.uuid
        if changed:
            self.log(f"⚠ ProPresenter switched to '{pres.name}' — ending the current song and starting a new one", kind="warn")
            self.stop_song()
            self.start_song()

    @staticmethod
    def _slide(rec: _Recording, index: int):
        slides = rec.presentation.slides
        return slides[index] if 0 <= index < len(slides) else None

    def _snapshot(self, rec: _Recording, ev: SlideEvent):
        """Store the slide's text/group on the event, from the slide list as it is right now."""
        slide = self._slide(rec, ev.index)
        if slide is not None:
            ev.text, ev.group = slide.text, slide.group
        return slide

    def _on_slides_updated(self, old_pres, new_pres, since_ts: float):
        """The live presentation was edited mid-song (its slide count changed): switch to the reloaded list.
        Earlier slide changes keep the text they had; changes since the edit are re-read with the new list."""
        with self._lock:
            rec = self._rec
            if rec is None or new_pres.uuid != rec.presentation.uuid:
                return
            before, after = len(rec.presentation.slides), len(new_pres.slides)
            rec.presentation = new_pres
            for ev in rec.events:
                if ev.ts >= since_ts:
                    self._snapshot(rec, ev)
            note = (f"ProPresenter's slide list changed {fmt_clock(max(0.0, since_ts - rec.start_ts))} into the song "
                    f"({before} → {after} slides) — the presentation was edited while recording. Slide changes before that "
                    f"point use the old list, later ones the new list. The slide on screen at the moment of the edit "
                    f"may not be measured.")
            rec.notes.append(note)
        self.log("⚠ " + note, kind="warn")
        self._push_live()

    def _on_slides_mismatch(self, cues: int, loaded: int):
        with self._lock:
            rec = self._rec
            if rec is None:
                return
            note = (f"ProPresenter now reports {cues} slides live but only {loaded} could be loaded — slide numbers "
                    f"after the edit may map to the wrong words in this song's report.")
            rec.notes.append(note)
        self.log("⚠ " + note, kind="warn")

    # ------------------------------------------------------------------
    # Song control
    # ------------------------------------------------------------------

    def start_song(self) -> bool:
        with self._lock:
            if self._rec is not None:
                self.log("A song is already being recorded.", kind="warn")
                return False
        if self.pro7 is not None and not getattr(self.pro7, "connected", True):
            self.log("✗ ProPresenter isn't connected yet — check the address under Setup. It keeps retrying by itself.", kind="error")
            return False
        pres = self.pro7.fetch_active_presentation() if self.pro7 else None
        if pres is None or not pres.slides:
            self.log("✗ No live presentation found in ProPresenter — put the song live first, then start the song.", kind="error")
            return False
        cues = self.pro7.live_total_cues
        if cues is not None and cues != len(pres.slides):
            self.log(f"⚠ ProPresenter reports {cues} slides live but {len(pres.slides)} were loaded — "
                     f"slide numbering/words in the report may be off for '{pres.name}'.", kind="warn")
        with self._lock:
            now = time.time()
            rec = self._rec = _Recording(pres.name, pres, now, self.pro7.current_slide_index)
            for ev in rec.events:                         # the slide already live when the song starts
                self._snapshot(rec, ev)
            if cues is not None and cues != len(pres.slides):
                rec.notes.append(f"ProPresenter reported {cues} slides live but {len(pres.slides)} were loaded when the "
                                 f"song started — slide numbers may map to the wrong words in this report.")
            self.status = "recording"
        self.log(f"● Recording '{pres.name}' ({len(pres.slides)} slides) — stop the song when it ends", kind="info")
        self._start_live(rec)
        self._push_live()
        return True

    def stop_song(self) -> bool:
        with self._lock:
            rec = self._rec
            if rec is None:
                self.log("No song is being recorded.", kind="warn")
                return False
            rec.end_ts = time.time()
        time.sleep(TAIL_WAIT_S)                      # collect audio still in flight
        self._stop_live()                            # the final analysis takes over from the live estimates
        with self._lock:
            self._rec = None
            self._song_count += 1
            number = self._song_count
            self.status = "analyzing"

        if not rec.blocks or rec.audio_start_ts is None:
            self.log("✗ No audio was captured for that song — nothing to analyze.", kind="error")
            self._refresh_status()
            self._push_live()
            return False

        audio = np.concatenate(rec.blocks)
        song = SongData(
            name=rec.name,
            presentation_uuid=rec.presentation.uuid,
            slides=[{"index": s.index, "text": s.text, "group": s.group, "label": s.label}
                    for s in rec.presentation.slides],
            song_start_ts=rec.start_ts,
            song_end_ts=rec.end_ts,
            audio_start_ts=rec.audio_start_ts,
            events=rec.events,
            audio=audio,
            operator=self.operator,
            notes=list(rec.notes),
        )
        self.log(f"■ Stopped '{rec.name}'" + (f" (operator: {self.operator})" if self.operator else "") +
                 f" ({fmt_clock(song.duration)}, {len(rec.events)} slide changes) — analyzing …", kind="info")

        # Save the raw capture immediately so nothing is lost if analysis fails.
        early, late = self.margins
        try:
            save_song(song, self.out_dir, number, self.offset_ms, late, early)
        except Exception as e:
            logger.error(f"Could not save raw song data: {e}", exc_info=True)

        t = threading.Thread(target=self._analyze, args=(song, number), daemon=True, name=f"Analyze-{number}")
        self._analysis_threads.append(t)
        t.start()
        self._push_live()
        return True

    # ------------------------------------------------------------------
    # Live (in-song) delay estimates
    # ------------------------------------------------------------------

    @property
    def live_cfg(self) -> dict:
        return self.cfg.get("live", {}) or {}

    def _start_live(self, rec: _Recording):
        cfg = self.live_cfg
        if not cfg.get("enabled", True):
            return
        prompt = (lyrics_prompt([{"text": s.text} for s in rec.presentation.slides])
                  if self.timing_cfg.get("use_lyrics_prompt", True) else None)
        vad = self.timing_cfg.get("vad_filter", False)

        def transcribe(window, start_s):
            whisper = self.whisper_provider()                   # may block while the model loads
            return whisper.transcribe_words(window, initial_prompt=prompt, vad_filter=vad)

        rec.live = LiveAnalyzer(transcribe, max_window_s=float(cfg.get("max_window_s", 40)))
        with self._lock:
            self._live_timings = []
            self._live_announced = set()
        self._live_stop.clear()
        self._live_thread = threading.Thread(target=self._live_loop, args=(rec,), daemon=True, name="LiveDelays")
        self._live_thread.start()

    def _stop_live(self):
        self._live_stop.set()
        t = self._live_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=20)                                   # at most one in-flight window
        self._live_thread = None
        with self._lock:
            self._live_timings = []

    def _live_loop(self, rec: _Recording):
        while not self._live_stop.wait(max(2.0, float(self.live_cfg.get("interval_s", 6)))):
            with self._lock:
                if self._rec is not rec:
                    return
            try:
                self._live_step(rec)
            except Exception as e:                              # a live-view problem must never disturb the recording
                logger.error(f"live estimate failed: {e}", exc_info=True)

    def _live_step(self, rec: _Recording):
        with self._lock:
            if not rec.blocks or rec.live is None:
                return
            blocks, events = list(rec.blocks), list(rec.events)
            a0 = rec.audio_start_ts or rec.start_ts
        if not self._analysis_lock.acquire(blocking=False):      # the previous song's final analysis is using the model
            return
        try:
            rec.live.update(np.concatenate(blocks))
        finally:
            self._analysis_lock.release()
        if self._live_stop.is_set():
            return
        timings = rec.live.timings(rec.presentation.slides, events, rec.start_ts, a0,
                                   offset_s=self.offset_ms / 1000.0,
                                   min_confidence=self.timing_cfg.get("min_match_confidence", 0.25),
                                   max_window_s=self.timing_cfg.get("max_match_window_s", 20.0),
                                   late_margin_s=self.margins[1])
        with self._lock:
            if self._rec is not rec:
                return
            self._live_timings = timings
            fresh = [t for t in timings if t.delay is not None and not t.initial and t.ordinal not in self._live_announced]
            self._live_announced.update(t.ordinal for t in fresh)
        if self.console:
            for t in fresh:
                print("\r" + " " * 90 + f"\r  ≈ slide {t.slide_index + 1:<3} {t.delay:+.1f}s  (live estimate)", flush=True)
        self._push_live()

    def live_graph_svg(self, late_s: float, early_s: float) -> Optional[str]:
        """The step graph for the song being recorded right now (provisional), or None."""
        from report.graph import render_song_graph
        with self._lock:
            rec = self._rec
            timings = list(self._live_timings)
            if rec is None:
                return None
            elapsed, name, operator = time.time() - rec.start_ts, rec.name, self.operator
        return render_song_graph(f"{name} — live estimate", max(elapsed, 10.0), timings, [], [], late_s, early_s,
                                 operator=operator)

    def toggle_song(self):
        with self._lock:
            recording = self._rec is not None
        return self.stop_song() if recording else self.start_song()

    def handle_command(self, cmd: str) -> bool:
        """Entry point for the dashboard buttons."""
        if cmd == "start":
            return self.start_song()
        if cmd == "stop":
            return self.stop_song()
        return False

    # ------------------------------------------------------------------
    # Analysis (background)
    # ------------------------------------------------------------------

    def _analyze(self, song: SongData, number: int):
        try:
            with self._analysis_lock:
                whisper = self.whisper_provider()
                analyze_songdata(song, whisper, self.timing_cfg, self.offset_ms)
            early, late = self.margins
            json_path = save_song(song, self.out_dir, number, self.offset_ms, late, early)
            self.last_song_id = None
            try:
                self.last_song_id = json_path.resolve().relative_to(Path(self.cfg["report"]["output_dir"]).resolve()).as_posix()
            except ValueError:
                pass
            report = format_song_report(song.name, song.song_start_ts, song.duration, song.timings,
                                        self.offset_ms, late, unshown=song.unshown,
                                        heard_other=song.heard_other, early_s=early)
            self._finished_reports.append(report)
            if self.console:
                print("\r" + " " * 90 + "\r" + report)
            s = summarize(song.timings, late, early)
            self._last_song_name = song.name
            tail = (f"{s['pct_on_time']:.0f}% on time, avg {s['mean_delay']:+.1f}s" if s["measured"]
                    else "no slide changes could be measured")
            self.log(f"✓ '{song.name}' analyzed — {tail}", kind="done")
        except Exception as e:
            logger.error(f"Analysis failed for '{song.name}': {e}", exc_info=True)
            self.log(f"✗ Analysis failed for '{song.name}': {e} (raw audio + slide data were saved)", kind="error")
        finally:
            self._refresh_status()
            self._push_live()

    def _refresh_status(self):
        with self._lock:
            if self._rec is not None:
                self.status = "recording"
            elif any(t.is_alive() for t in self._analysis_threads):
                self.status = "analyzing"
            else:
                self.status = "idle"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start capturing audio (does not start a song)."""
        self.source.start()
        self.log("Monitoring started — start a song when the music begins.", kind="info")
        self._push_live()

    def finish(self):
        """Stop the current song (if any), the audio source, wait for analyses, write the session report."""
        self._stop_event.set()
        with self._lock:
            recording = self._rec is not None
        if recording:
            self.stop_song()
        if self.source:
            self.source.stop()
        pending = [t for t in self._analysis_threads if t.is_alive()]
        if pending:
            self.log(f"Waiting for {len(pending)} song analysis job(s) to finish …", kind="info")
            for t in pending:
                t.join()
        if self._finished_reports:
            path = self.out_dir / "session_report.txt"
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(self._finished_reports), encoding="utf-8")
            logger.info(f"Session report: {path}")
        self.status = "idle"
        self._push_live()

    # --- terminal runner (CLI) ----------------------------------------

    def run(self, key_poller):
        """Blocking terminal loop: S = start/stop song, Q = quit. Call start() first;
        the caller finishes the session (finish()) after this returns."""
        self.log("Ready. Press S to start/stop a song, Q to quit.", kind="info")
        last_tick = 0.0
        try:
            while not self._stop_event.is_set():
                key = key_poller.poll()
                if key == "s":
                    self.toggle_song()
                elif key == "q":
                    break
                now = time.time()
                if now - last_tick >= 0.5:
                    last_tick = now
                    self._refresh_status()
                    self._status_line()
                    self._push_live()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    # ------------------------------------------------------------------
    # Messages / display
    # ------------------------------------------------------------------

    def log(self, text: str, kind: str = "info"):
        with self._lock:
            self._messages.append({"t": time.time(), "text": text, "kind": kind})
        if self.console:
            print("\r" + " " * 90 + "\r" + text, flush=True)
        else:
            logger.info(text)

    def _status_line(self):
        from colorama import Fore, Style
        with self._lock:
            rec = self._rec
            idx = self.pro7.current_slide_index if self.pro7 else None
        level = self.source.get_level_db()
        if rec is not None:
            line = (f"{Fore.RED}● REC{Style.RESET_ALL} {rec.name}  {fmt_clock(time.time() - rec.start_ts)}  "
                    f"slide {idx + 1 if idx is not None else '-'}/{len(rec.presentation.slides)}  "
                    f"audio {level:+.0f} dB")
        else:
            busy = "  (analyzing…)" if self.status == "analyzing" else ""
            line = f"idle — press S to start a song{busy}   audio {level:+.0f} dB"
        print("\r" + line.ljust(90)[:110], end="", flush=True)

    def snapshot(self) -> dict:
        """State for the GUI (live part only; completed songs come from the library)."""
        with self._lock:
            rec = self._rec
            idx = self.pro7.current_slide_index if self.pro7 else None
            state = {
                "status": self.status,
                "level_db": self.source.get_level_db(),
                "last_song_id": self.last_song_id,
                "last_song_name": self._last_song_name,
                "operator": self.operator,
                "messages": list(self._messages),
                "session_dir": str(self.out_dir),
            }
            if rec is not None:
                slide = self._slide(rec, idx) if idx is not None else None
                state.update(
                    song_name=rec.name,
                    elapsed=time.time() - rec.start_ts,
                    slide_index=idx,
                    slide_count=len(rec.presentation.slides),
                    slide_text=slide.text if slide else "",
                    slide_group=slide.group if slide else "",
                    changes=len(rec.events),
                    live_events=[
                        {"at": e.ts - rec.start_ts, "slide": e.index + 1,
                         "group": e.group if e.text is not None else (self._slide(rec, e.index).group if self._slide(rec, e.index) else ""),
                         "text": one_line(e.text, 60) if e.text is not None else (one_line(self._slide(rec, e.index).text, 60) if self._slide(rec, e.index) else ""),
                         "initial": e.initial}
                        for e in rec.events[-12:]
                    ],
                    live={
                        "enabled": rec.live is not None,
                        "covered": rec.live.covered_s if rec.live else 0.0,
                        "rows": [
                            {"ordinal": t.ordinal, "slide": t.slide_index + 1, "group": t.group,
                             "text": one_line(t.text, 50), "at": t.change_time, "delay": t.delay,
                             "pending": LiveAnalyzer.is_pending(t), "initial": t.initial,
                             "blank": not t.text.strip()}
                            for t in self._live_timings
                        ],
                    },
                )
            else:
                state.update(song_name=None, elapsed=0.0, slide_index=idx, slide_text="", slide_group="",
                             changes=0, slide_count=0, live_events=[], live={"enabled": False, "covered": 0.0, "rows": []})
        return state

    def _push_live(self):
        if self.dashboard:
            self.dashboard.broadcast(self.snapshot())
