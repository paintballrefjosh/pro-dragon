"""
Controller
===========
The application core, shared by the terminal runner and the GUI. It owns:

  * settings               (SettingsStore: config.yaml + GUI overrides)
  * the ProPresenter client
  * the speech-recognition model (loaded lazily, reloaded if its settings change)
  * the current monitoring Session (audio source + songs)
  * the song library (saved songs, re-analysis)

Every public action returns (ok, message) or a dict so the web API can hand the
result straight back to the page. Live state is pushed to browsers through
`dashboard.broadcast()`.
"""

import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

from library import LibraryError, SongLibrary
from session import Session
from settings import SettingsStore
from utils.logger import get_logger

logger = get_logger("app")


def _default_whisper_factory(wcfg: dict):
    from transcription.whisper_asr import WhisperASR
    asr = WhisperASR(model_size=wcfg.get("model", "small"), language=wcfg.get("language", "en"),
                     device=wcfg.get("device", "cpu"), compute_type=wcfg.get("compute_type", "int8"))
    asr.load_model()
    return asr


def _default_pro7_factory(pcfg: dict):
    from propresenter.client import ProPresenterClient
    return ProPresenterClient(pcfg["host"], pcfg["port"], pcfg.get("password", ""), pcfg.get("poll_interval", 0.1))


class Controller:
    def __init__(self, settings: SettingsStore, dashboard=None, *,
                 whisper_factory: Callable = _default_whisper_factory,
                 pro7_factory: Callable = _default_pro7_factory,
                 cli_offset_ms: Optional[float] = None,
                 console: bool = False,
                 auto: bool = False):
        """auto=True (the GUI): connect to ProPresenter by itself and keep trying, and keep audio monitoring
        running with the last audio source used — nothing to press. Terminal mode starts both explicitly."""
        self.settings = settings
        self.dashboard = dashboard
        self._whisper_factory = whisper_factory
        self._pro7_factory = pro7_factory
        self.cli_offset_ms = cli_offset_ms
        self.console = console
        self.auto = auto

        self.pro7 = None
        self._pro7_lock = threading.RLock()
        self.session: Optional[Session] = None
        self._stopping = False
        self._switching = False                   # changing the audio source (stop old session, start new)
        self._auto_retry_at = 0.0                 # next time auto-monitoring may try again after a failure
        self._auto_failures = 0
        self._started_at = time.time()
        self.operator = ""            # current operator name (see set_operator)
        self._known_operators: Optional[List[str]] = None
        self._ops_for_song: Optional[str] = None

        self._whisper = None
        self._whisper_key: Optional[tuple] = None
        self._whisper_lock = threading.Lock()
        self._whisper_state = {"status": "not loaded", "detail": ""}

        self._notices: Deque[dict] = deque(maxlen=30)
        self._job: Dict[str, str] = {}
        self._ticker: Optional[threading.Thread] = None
        self._shutdown = threading.Event()

        self.library = SongLibrary(lambda: self.settings.cfg["report"]["output_dir"])
        self.settings.on_change(self._on_settings_changed)
        if dashboard is not None:
            dashboard.controller = self

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_ticker(self):
        """Background thread that refreshes the live state (levels, clocks) for browsers."""
        if self._ticker and self._ticker.is_alive():
            return
        self._ticker = threading.Thread(target=self._tick_loop, daemon=True, name="ControllerTick")
        self._ticker.start()
        threading.Thread(target=self._spectrum_loop, daemon=True, name="ControllerSpectrum").start()
        if self.auto:
            threading.Thread(target=self._auto_loop, daemon=True, name="ControllerAuto").start()

    def _auto_loop(self):
        """Keep ProPresenter connected and audio monitoring running without anyone pressing anything."""
        while True:
            try:
                self._auto_tick()
            except Exception as e:
                logger.error(f"auto-connect/monitor error: {e}", exc_info=True)
            if self._shutdown.wait(2.0):
                return

    def _auto_tick(self):
        if self.pro7 is None:
            self._ensure_pro7()                       # its poll loop then reconnects by itself, forever
        if self.session is not None or self._stopping or self._switching or time.time() < self._auto_retry_at:
            return
        spec = self.settings.cfg["audio"].get("last")
        if not (isinstance(spec, dict) and spec.get("kind")):
            return                                    # first run: waiting for the user to choose an audio source
        if time.time() - self._started_at < 3.0:
            return                                    # give Windows a moment to finish listing audio devices at launch
        ok, _msg = self.start_session(spec)
        if ok:
            self._auto_failures = 0
            self.notice(f"Monitoring started — {self.session.source.describe() if self.session else 'audio'}", "done")
        else:
            self._auto_failures += 1                  # back off (5 s, 10 s, 20 s … 60 s) instead of retrying noisily
            self._auto_retry_at = time.time() + min(60.0, 5.0 * 2 ** (self._auto_failures - 1))

    def _spectrum_loop(self):
        """~12 times a second, send the equalizer bands to browsers (only while monitoring and watched)."""
        from audio.capture import spectrum_edges
        edges = [round(e, 1) for e in spectrum_edges()]
        while not self._shutdown.wait(0.08):
            try:
                sess, dash = self.session, self.dashboard
                if sess is None or dash is None or not dash.has_clients():
                    continue
                dash.send_transient({"_spectrum": {"db": [round(v, 1) for v in sess.source.get_bands_db()], "edges": edges}})
            except Exception as e:
                logger.error(f"spectrum error: {e}", exc_info=True)

    def _tick_loop(self):
        while not self._shutdown.wait(0.5):
            try:
                if self.session:
                    self.session._refresh_status()
                self._push()
            except Exception as e:
                logger.error(f"ticker error: {e}", exc_info=True)

    def shutdown(self):
        self._shutdown.set()
        if self.session:
            try:
                self.session.finish()
            except Exception as e:
                logger.error(f"error finishing session: {e}", exc_info=True)
            self.session = None
        if self.pro7:
            self.pro7.disconnect()
            self.pro7 = None

    def notice(self, text: str, kind: str = "info"):
        self._notices.append({"t": time.time(), "text": text, "kind": kind})
        logger.info(text) if kind != "error" else logger.error(text)
        self._push()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_settings(self) -> dict:
        return self.settings.view()

    def update_settings(self, patch: dict) -> dict:
        errors = self.settings.update(patch)
        return {"ok": not errors, "errors": errors, "settings": self.settings.view()}

    def reset_settings(self) -> dict:
        self.settings.reset()
        return {"ok": True, "settings": self.settings.view()}

    def _on_settings_changed(self, cfg: dict):
        if self.session:
            self.session.update_config(cfg)
        pro7, pcfg = self.pro7, cfg["propresenter"]
        if pro7 is not None and (pro7.host, pro7.port, pro7.password) != (pcfg["host"], int(pcfg["port"]), pcfg.get("password", "")):
            pro7.reconfigure(pcfg["host"], pcfg["port"], pcfg.get("password", ""), pcfg.get("poll_interval"))   # follow the new address
        key = self._whisper_cfg_key(cfg)
        if self._whisper is not None and key != self._whisper_key:
            self._whisper = None                     # reload with the new model/device next time
            self._whisper_state = {"status": "not loaded", "detail": "settings changed"}
        self._push()

    @staticmethod
    def _whisper_cfg_key(cfg: dict) -> tuple:
        w = cfg["whisper"]
        return (w.get("model"), w.get("language"), w.get("device"), w.get("compute_type"))

    # ------------------------------------------------------------------
    # ProPresenter
    # ------------------------------------------------------------------

    def connect_propresenter(self, host: Optional[str] = None, port=None) -> Tuple[bool, str]:
        if host is not None or port is not None:
            patch = {}
            if host is not None:
                patch["propresenter.host"] = host
            if port is not None:
                patch["propresenter.port"] = port
            errors = self.settings.update(patch)
            if errors:
                return False, "; ".join(errors.values())
        pcfg = self.settings.cfg["propresenter"]
        client = self._ensure_pro7()                  # one long-lived client; it keeps retrying in the background
        if (client.host, client.port, client.password) != (pcfg["host"], int(pcfg["port"]), pcfg.get("password", "")):
            client.reconfigure(pcfg["host"], pcfg["port"], pcfg.get("password", ""), pcfg.get("poll_interval"))
        if not client.check_connection():
            msg = (f"Can't connect to ProPresenter at {pcfg['host']}:{pcfg['port']} — check the address and that "
                   "Network is enabled in ProPresenter's preferences. Still trying in the background.")
            self.notice(msg, "error")
            return False, msg
        return True, "connected"

    def _ensure_pro7(self):
        """The ProPresenter client, created (and polling) on first use. It polls whether or not ProPresenter is
        reachable, so it connects as soon as ProPresenter is available and reconnects after a drop."""
        with self._pro7_lock:
            if self.pro7 is None:
                pcfg = self.settings.cfg["propresenter"]
                client = self._pro7_factory(pcfg)
                client.on_slide_changed = self._on_slide
                client.on_presentation_changed = self._on_presentation
                client.on_slides_updated = self._on_slides_updated
                client.on_slides_mismatch = self._on_slides_mismatch
                client.on_connection_changed = self._on_pro7_connection
                client.start_polling()
                self.pro7 = client
            return self.pro7

    def _on_pro7_connection(self, up: bool):
        p = self.settings.cfg["propresenter"]
        if up:
            self.notice(f"Connected to ProPresenter at {p['host']}:{p['port']}", "done")
        else:
            self.notice("Lost the connection to ProPresenter — reconnecting automatically", "warn")
        self._push()

    def _on_slide(self, index: int, ts: float):
        if self.session:
            self.session._on_slide_changed(index, ts)
        self._push()

    def _on_presentation(self, pres):
        if self.session:
            self.session._on_presentation_changed(pres)
        self._push()

    def _on_slides_updated(self, old_pres, new_pres, since_ts: float):
        if self.session:
            self.session._on_slides_updated(old_pres, new_pres, since_ts)
        self._push()

    def _on_slides_mismatch(self, cues: int, loaded: int):
        if self.session:
            self.session._on_slides_mismatch(cues, loaded)
        self._push()

    # ------------------------------------------------------------------
    # Speech recognition (lazy)
    # ------------------------------------------------------------------

    def get_whisper(self):
        """Return a loaded model, loading it first if needed (blocks)."""
        with self._whisper_lock:
            cfg = self.settings.cfg
            key = self._whisper_cfg_key(cfg)
            if self._whisper is not None and key == self._whisper_key:
                return self._whisper
            self._whisper_state = {"status": "loading",
                                   "detail": f"loading '{key[0]}' model (the first time this downloads it)…"}
            self._push()
            try:
                self._whisper = self._whisper_factory(cfg["whisper"])
                self._whisper_key = key
                self._whisper_state = {"status": "ready", "detail": f"'{key[0]}' model ready"}
            except Exception as e:
                self._whisper = None
                self._whisper_state = {"status": "error", "detail": str(e)}
                self.notice(f"Speech recognition failed to load: {e}", "error")
                raise
            finally:
                self._push()
            return self._whisper

    def preload_whisper(self):
        def _go():
            try:
                self.get_whisper()
            except Exception:
                pass
        threading.Thread(target=_go, daemon=True, name="WhisperPreload").start()

    # ------------------------------------------------------------------
    # Audio sources + session
    # ------------------------------------------------------------------

    def list_audio_sources(self, include_ndi: bool = False) -> List[dict]:
        from audio.capture import list_devices
        return list_devices(include_loopback=True, include_ndi=include_ndi,
                            ndi_addresses=self.settings.cfg["audio"].get("ndi_address") or "")

    def ndi_available(self) -> bool:
        from audio.capture import _ndi_module
        return _ndi_module() is not None

    @staticmethod
    def _parse_channels(value) -> List[int]:
        if isinstance(value, str):
            value = [c for c in value.replace(" ", "").split(",") if c]
        chans = [int(c) for c in (value or [1])]
        if not chans or any(c < 1 for c in chans):
            raise ValueError("channels must be positive numbers like 1 or 1,2")
        return chans

    def _source_factory(self, spec: dict) -> Callable:
        from audio.capture import DeviceSource, FileSource, LoopbackSource, NdiSource
        kind = spec.get("kind")
        if kind == "ndi":
            name = str(spec.get("name") or "").strip()
            if not name:
                raise ValueError("choose an NDI source")
            chans = self._parse_channels(spec["channels"]) if str(spec.get("channels") or "").strip() else None
            addresses = self.settings.cfg["audio"].get("ndi_address") or ""      # read now: the factory may run later
            return lambda cb: NdiSource(name, chans, on_frame=cb, addresses=addresses)
        if kind == "loopback":
            idx = spec.get("index")
            return lambda cb: LoopbackSource(None if idx in (None, "") else int(idx), on_frame=cb)
        if kind == "input":
            if spec.get("index") in (None, ""):
                raise ValueError("choose an audio device")
            idx, chans = int(spec["index"]), self._parse_channels(spec.get("channels"))
            return lambda cb: DeviceSource(idx, chans, on_frame=cb)
        if kind == "file":
            from pathlib import Path
            path = str(spec.get("file") or "").strip().strip('"')
            if not path or not Path(path).is_file():
                raise ValueError(f"audio file not found: {path or '(empty)'}")
            return lambda cb: FileSource(path, on_frame=cb)
        raise ValueError("choose an audio source")

    def start_session(self, spec: dict, source_factory: Optional[Callable] = None,
                      operator: Optional[str] = None) -> Tuple[bool, str]:
        if self.session is not None:
            return False, "Monitoring is already running."
        if self._stopping:
            return False, "Still finishing the previous session — one moment."
        if operator is not None:                      # whatever is in the operator box right now
            ok, msg = self.set_operator(operator)
            if not ok:
                return False, msg
        self._ensure_pro7()           # audio monitoring doesn't wait for ProPresenter: it connects whenever it can
        try:
            factory = source_factory or self._source_factory(spec)
            cfg = self.settings.cfg
            # The controller does all the pushing to browsers, so the session gets no dashboard of its own.
            session = Session(cfg, self.pro7, factory, self.get_whisper, None,
                              offset_ms=self.cli_offset_ms, console=self.console, operator=self.operator)
            session.start()
        except Exception as e:
            logger.error(f"could not start monitoring: {e}")        # one line: this repeats on retries; --debug shows the trace
            logger.debug("monitoring start failure", exc_info=True)
            msg = f"Couldn't start audio: {e}"
            self.notice(msg, "error")
            return False, msg
        self.session = session
        if spec:
            self.settings.set_runtime("audio.last", spec)
        self.preload_whisper()
        self._push()
        return True, "monitoring"

    def set_audio_source(self, spec: dict) -> Tuple[bool, str]:
        """Use this audio source from now on: (re)starts monitoring with it, in the background (stopping the old
        session can take a while if a song is still being analyzed). Refused while a song is recording."""
        try:
            self._source_factory(spec)                # validate now so mistakes are reported to the page at once
        except (ValueError, TypeError) as e:
            return False, str(e)
        sess = self.session
        if sess is not None and sess.status == "recording":
            return False, "Stop the song first — the audio source can't change while it is being recorded."
        if self._switching:
            return False, "Already switching audio sources — one moment."
        self._switching = True
        self._push()

        def _go():
            try:
                if self.session is not None:
                    self.stop_session(wait=True)
                ok, msg = self.start_session(spec)
                if ok:
                    self._auto_failures, self._auto_retry_at = 0, 0.0
                    self.notice(f"Audio source: {self.session.source.describe()}", "done")
                else:
                    self._auto_retry_at = time.time() + 5.0
            except Exception as e:
                logger.error(f"switching audio source failed: {e}", exc_info=True)
                self.notice(f"Couldn't switch the audio source: {e}", "error")
            finally:
                self._switching = False
                self._push()

        threading.Thread(target=_go, daemon=True, name="SwitchAudio").start()
        return True, "switching"

    def stop_session(self, wait: bool = False) -> Tuple[bool, str]:
        session = self.session
        if session is None:
            return False, "Monitoring isn't running."

        def _finish():
            try:
                session.finish()
            finally:
                if self.session is session:
                    self.session = None
                self._stopping = False
                self._push()

        self._stopping = True
        if wait:
            _finish()
        else:
            threading.Thread(target=_finish, daemon=True, name="SessionFinish").start()
        return True, "stopping"

    def live_graph_svg(self, late_s: Optional[float] = None, early_s: Optional[float] = None) -> Optional[str]:
        from analysis.song_timing import margins_from_cfg
        if self.session is None:
            return None
        d_early, d_late = margins_from_cfg(self.settings.cfg["timing"])
        return self.session.live_graph_svg(d_late if late_s is None else late_s, d_early if early_s is None else early_s)

    def set_operator(self, name) -> Tuple[bool, str]:
        """Who is running the slides now. Stamped on each song when it stops; remembered for next time."""
        try:
            name = self.library.clean_operator(name)
        except LibraryError as e:
            return False, str(e)
        self.operator = name
        if self.session:
            self.session.operator = name
        self.settings.set_runtime("operator.last", name)
        self._known_operators = None
        self._push()
        return True, name

    def _operators(self) -> List[str]:
        """Operator names seen in saved songs (cached: reading every song is too heavy to do on each push)."""
        if self._known_operators is None:
            try:
                self._known_operators = self.library.operator_names()
            except Exception as e:
                logger.error(f"could not read operator names: {e}", exc_info=True)
                self._known_operators = []
        names = list(self._known_operators)
        if self.operator and self.operator not in names:
            names.append(self.operator)
        return sorted(names, key=str.lower)

    def set_song_operator(self, song_id: str, name: str) -> dict:
        try:
            detail = self.library.set_operator(song_id, name)
        except LibraryError as e:
            return {"ok": False, "error": str(e)}
        self._known_operators = None
        return {"ok": True, "detail": detail}

    def remap_song_slides(self, song_id: str) -> dict:
        """Fix a song recorded while ProPresenter's slide list changed: re-read its slide changes through the
        presentation's CURRENT slide list (needs ProPresenter connected with that song live)."""
        if self.pro7 is None:
            return {"ok": False, "error": "Connect to ProPresenter first (Live tab) and put this song live."}
        try:
            detail = self.library.detail(song_id)
        except LibraryError as e:
            return {"ok": False, "error": str(e)}
        pres = self.pro7.fetch_active_presentation()
        if pres is None or not pres.slides:
            return {"ok": False, "error": "ProPresenter has no live presentation — put this song live first."}
        if detail.get("presentation_uuid") and pres.uuid != detail["presentation_uuid"]:
            return {"ok": False, "error": f"The live presentation in ProPresenter is '{pres.name}', not this song — put this song live first."}
        cues = self.pro7.live_total_cues
        if cues is not None and cues != len(pres.slides):
            return {"ok": False, "error": f"ProPresenter reports {cues} slides but {len(pres.slides)} could be loaded, so the list can't be trusted."}
        new_slides = [{"index": s.index, "text": s.text, "group": s.group, "label": s.label} for s in pres.slides]
        try:
            res = self.library.remap_slides(song_id, new_slides, self.settings.cfg["timing"],
                                            float((detail.get("settings") or {}).get("offset_ms", 0)))
        except LibraryError as e:
            return {"ok": False, "error": str(e)}
        self._known_operators = None
        return {"ok": True, "detail": res}

    def delete_songs(self, song_ids: List[str]) -> dict:
        """Permanently delete saved songs (files included). Refused for a song that is being re-analyzed."""
        ids = [str(i) for i in (song_ids or []) if str(i).strip()]
        if not ids:
            return {"ok": False, "error": "no songs selected"}
        if self._job.get("state") == "running" and self._job.get("id") in ids:
            return {"ok": False, "error": "that song is being re-analyzed right now — wait for it to finish"}
        res = self.library.delete(ids)
        self._known_operators = None                      # a deleted song may have been someone's only one
        sess = self.session
        if sess is not None and sess.last_song_id in res["deleted"]:
            sess.last_song_id = None                      # the "view last song" button must not point at a deleted file
        self._push()
        return {"ok": bool(res["deleted"]), "error": "; ".join(f"{k}: {v}" for k, v in res["failed"].items()), **res}

    def operator_stats(self, late_s: Optional[float] = None, early_s: Optional[float] = None) -> List[dict]:
        from analysis.song_timing import margins_from_cfg
        d_early, d_late = margins_from_cfg(self.settings.cfg["timing"])
        return self.library.operators(d_late if late_s is None else late_s, d_early if early_s is None else early_s)

    def song_command(self, cmd: str) -> Tuple[bool, str]:
        if self.session is None:
            return False, "Start monitoring first."
        ok = self.session.handle_command(cmd)
        self._push()
        return ok, "" if ok else "see messages"

    # ------------------------------------------------------------------
    # Library
    # ------------------------------------------------------------------

    def reanalyze_song(self, song_id: str, offset_ms=None, retranscribe: bool = False) -> dict:
        cfg = self.settings.cfg
        offset = float(cfg["timing"].get("offset_ms", 0)) if offset_ms is None else float(offset_ms)

        def _run():
            return self.library.reanalyze(song_id, cfg["timing"], offset, self.get_whisper, retranscribe)

        if not retranscribe:
            try:
                detail = self.library.detail(song_id)
                if detail["has_words"]:
                    return {"ok": True, "detail": _run()}
            except LibraryError as e:
                return {"ok": False, "error": str(e)}

        # needs Whisper: run in the background and report progress through the live state
        self._job = {"id": song_id, "state": "running", "message": "re-transcribing (this can take a minute or two)…"}
        self._push()

        def _bg():
            try:
                _run()
                self._job = {"id": song_id, "state": "done", "message": "re-analysis finished"}
            except Exception as e:
                logger.error(f"re-analysis failed: {e}", exc_info=True)
                self._job = {"id": song_id, "state": "error", "message": str(e)}
            self._push()

        threading.Thread(target=_bg, daemon=True, name="Reanalyze").start()
        return {"ok": True, "job": True}

    # ------------------------------------------------------------------
    # State for browsers
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        cfg = self.settings.cfg
        pro7 = self.pro7
        pres = pro7.current_presentation if pro7 else None
        state = {
            "pro7": {
                "connected": bool(pro7 and pro7.connected),
                "host": cfg["propresenter"]["host"], "port": cfg["propresenter"]["port"],
                "slide_index": pro7.current_slide_index if pro7 else None,
                "presentation": pres.name if pres else None,
            },
            "whisper": dict(self._whisper_state, model=cfg["whisper"]["model"]),
            "monitoring": self.session is not None,
            "stopping": self._stopping,
            "switching": self._switching,
            "audio_desc": self.session.source.describe() if self.session else "",
            "job": self._job,
            "offset_ms": (self.session.offset_ms if self.session else
                          (self.cli_offset_ms if self.cli_offset_ms is not None
                           else float(cfg["timing"].get("offset_ms", 0)))),
            "operator": self.operator,
            "operators": self._operators(),
            "live": {"enabled": False, "covered": 0.0, "rows": []},
            "status": "idle", "song_name": None, "elapsed": 0.0, "slide_text": "", "slide_group": "",
            "changes": 0, "slide_count": 0, "live_events": [], "level_db": -96.0,
            "last_song_id": None, "last_song_name": "", "messages": [], "session_dir": "",
        }
        msgs = list(self._notices)
        if self.session:
            snap = self.session.snapshot()
            msgs += snap.pop("messages")
            state.update(snap)
            if snap.get("last_song_id") != self._ops_for_song:     # a song finished: its operator may be new
                self._ops_for_song = snap.get("last_song_id")
                self._known_operators = None
                state["operators"] = self._operators()
        state["messages"] = sorted(msgs, key=lambda m: m["t"])[-30:]
        return state

    def _push(self):
        if self.dashboard is not None:
            try:
                self.dashboard.broadcast(self.snapshot())
            except Exception as e:
                logger.error(f"broadcast failed: {e}", exc_info=True)
