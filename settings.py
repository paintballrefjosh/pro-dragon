"""
Settings
=========
Effective configuration = built-in defaults  <-  config.yaml  <-  user_settings.json

`config.yaml` stays hand-edited (and keeps its comments). Anything changed in the
GUI is saved to `user_settings.json` (next to config.yaml) and layered on top, so
the GUI never rewrites your YAML. "Reset" deletes the overrides.

The FIELDS table drives both validation and the Settings page in the GUI.
"""

import copy
import json
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from analysis.song_timing import margins_from_cfg
from utils.logger import get_logger

logger = get_logger("settings")

DEFAULT_CONFIG: Dict[str, Any] = {
    "propresenter": {"host": "127.0.0.1", "port": 1025, "password": "", "poll_interval": 0.1},
    "audio": {"device": None, "channels": [1], "loopback": False, "file": None, "ndi": None, "ndi_address": None, "last": None},
    "timing": {"offset_ms": 0, "on_time_window_s": 0.75, "min_match_confidence": 0.25,
               "max_match_window_s": 20.0, "other_min_words": 4, "other_min_confidence": 0.5,
               "use_lyrics_prompt": True, "vad_filter": False},
    "whisper": {"model": "small", "language": "en", "device": "cpu", "compute_type": "int8"},
    "live": {"enabled": True, "interval_s": 6.0, "max_window_s": 40.0},
    "report": {"output_dir": "./reports"},
    "gui": {"enabled": True, "host": "127.0.0.1", "port": 8765, "open_browser": True},
}

# key, type, label, group, help, extra (choices / min / max), restart note
FIELDS: List[dict] = [
    {"key": "propresenter.host", "type": "str", "group": "ProPresenter", "label": "Host",
     "help": "IP address of the ProPresenter computer (127.0.0.1 = this computer)."},
    {"key": "propresenter.port", "type": "int", "group": "ProPresenter", "label": "Port", "min": 1, "max": 65535,
     "help": "ProPresenter → Preferences → Network (default 1025)."},
    {"key": "propresenter.password", "type": "password", "group": "ProPresenter", "label": "Password",
     "help": "Only if you set one in ProPresenter."},
    {"key": "propresenter.poll_interval", "type": "float", "group": "ProPresenter", "label": "Slide check interval (s)",
     "min": 0.02, "max": 2.0, "help": "How often slide changes are checked. Slide times are accurate to about half this."},

    {"key": "timing.offset_ms", "type": "float", "group": "Timing", "label": "Audio offset (ms)", "min": -5000, "max": 5000,
     "help": "Positive = the captured audio arrives LATE compared with the room (e.g. NDI adds 100–300 ms). "
             "Negative = it arrives early. Applied when a song is analyzed."},
    {"key": "timing.on_time_late_s", "type": "float", "group": "Timing", "label": "Margin of error — late (s)", "min": 0, "max": 30,
     "help": "A slide change up to this many seconds AFTER the singing starts still counts as on time."},
    {"key": "timing.on_time_early_s", "type": "float", "group": "Timing", "label": "Margin of error — early (s)", "min": 0, "max": 30,
     "help": "A slide change up to this many seconds BEFORE the singing starts still counts as on time."},
    {"key": "timing.max_match_window_s", "type": "float", "group": "Matching", "label": "Match window (s)", "min": 1, "max": 120,
     "help": "A slide change is only paired with sung words of its own text within this many seconds."},
    {"key": "timing.min_match_confidence", "type": "float", "group": "Matching", "label": "Min. words heard (fraction)", "min": 0.05, "max": 1,
     "help": "A slide needs at least this fraction of its words heard to be measured."},
    {"key": "timing.other_min_words", "type": "int", "group": "Matching", "label": "'No slide' min. words", "min": 1, "max": 50,
     "help": "Words heard that match no slide are listed only if there are at least this many in a row."},
    {"key": "timing.other_min_confidence", "type": "float", "group": "Matching", "label": "'No slide' min. confidence", "min": 0, "max": 1,
     "help": "…and the recognizer was at least this confident (filters phantom text over music)."},
    {"key": "timing.use_lyrics_prompt", "type": "bool", "group": "Matching", "label": "Give Whisper the lyrics as a hint",
     "help": "Strongly recommended for sung vocals. Used when a song is transcribed."},
    {"key": "timing.vad_filter", "type": "bool", "group": "Matching", "label": "Whisper speech filter (VAD)",
     "help": "Usually off for singing — it tends to cut out sung words."},

    {"key": "whisper.model", "type": "choice", "group": "Speech recognition", "label": "Model",
     "choices": ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
     "help": "Bigger = hears sung words better but slower (and a bigger first download). Takes effect next time the model loads."},
    {"key": "whisper.language", "type": "optstr", "group": "Speech recognition", "label": "Language",
     "help": "Language code (en, es, …). Leave empty to auto-detect."},
    {"key": "whisper.device", "type": "choice", "group": "Speech recognition", "label": "Device", "choices": ["cpu", "cuda"],
     "help": "cuda needs an NVIDIA GPU with CUDA/cuDNN installed."},
    {"key": "whisper.compute_type", "type": "choice", "group": "Speech recognition", "label": "Compute type",
     "choices": ["int8", "float16", "float32"], "help": "int8 for CPU, float16 for GPU."},

    {"key": "audio.ndi_address", "type": "addrlist", "group": "NDI audio", "label": "Static NDI address(es)",
     "help": "Optional. IP address or host name of the NDI sender (mixer, camera, PC), e.g. 192.168.20.15 or "
             "192.168.20.15:5959; several separated by commas. Needed when the source is on a different subnet/VLAN, where "
             "NDI's automatic discovery can't see it. Leave empty for normal same-network discovery. "
             "(A multicast sender needs nothing special: once found, the stream joins the sender's multicast group by itself.) "
             "Press ↻ on the Live tab to look again."},

    {"key": "live.enabled", "type": "bool", "group": "Live view", "label": "Live delay estimates while a song plays",
     "help": "Shows provisional delays a few seconds after each slide change (the final report still replaces them). "
             "Uses the speech model repeatedly, so turn it off on a slow computer. Applies to the next song."},
    {"key": "live.interval_s", "type": "float", "group": "Live view", "label": "Update every (s)", "min": 2, "max": 60,
     "help": "Pause between live updates (on top of the time each one takes). Shorter = quicker estimates, more CPU."},

    {"key": "report.output_dir", "type": "str", "group": "Output", "label": "Reports folder",
     "help": "Each run creates a session_<date> folder here. Applies to the next monitoring session."},
]
FIELD_BY_KEY = {f["key"]: f for f in FIELDS}


def _deep_merge(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str) -> dict:
    """defaults <- config.yaml (no GUI overrides)."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    p = Path(path)
    if p.exists():
        user_cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        _deep_merge(cfg, user_cfg)
        logger.info(f"Config loaded from {p}")
    else:
        logger.warning(f"{path} not found — using defaults")
    return cfg


def _get(cfg: dict, dotted: str):
    cur: Any = cfg
    for part in dotted.split("."):
        cur = cur.get(part) if isinstance(cur, dict) else None
    return cur


def _set(cfg: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def coerce(field: dict, value: Any) -> Any:
    """Validate/convert a value from the GUI. Raises ValueError with a readable message."""
    t, label = field["type"], field["label"]
    if t in ("str", "password"):
        s = "" if value is None else str(value).strip() if t == "str" else str(value)
        if t == "str" and not s:
            raise ValueError(f"{label} can't be empty")
        return s
    if t == "optstr":
        s = "" if value is None else str(value).strip()
        return s or None
    if t == "addrlist":
        from audio.capture import parse_ndi_addresses
        return parse_ndi_addresses(value) or None
    if t == "bool":
        if isinstance(value, bool):
            return value
        if str(value).lower() in ("true", "1", "yes", "on"):
            return True
        if str(value).lower() in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{label} must be on or off")
    if t == "choice":
        if value not in field["choices"]:
            raise ValueError(f"{label} must be one of {', '.join(field['choices'])}")
        return value
    if t in ("int", "float"):
        try:
            num = int(float(value)) if t == "int" else float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a number")
        if num != num:  # NaN
            raise ValueError(f"{label} must be a number")
        lo, hi = field.get("min"), field.get("max")
        if lo is not None and num < lo or hi is not None and num > hi:
            raise ValueError(f"{label} must be between {lo} and {hi}")
        return num
    raise ValueError(f"unknown field type {t}")


class SettingsStore:
    def __init__(self, config_path: str = "config.yaml", overrides_path: Optional[str] = None):
        self.config_path = Path(config_path)
        self.overrides_path = Path(overrides_path) if overrides_path else self.config_path.with_name("user_settings.json")
        self._lock = threading.RLock()
        self._listeners: List[Callable[[dict], None]] = []
        self._overrides: Dict[str, Any] = {}
        self._cfg: dict = {}
        self.reload()

    # --- loading ---------------------------------------------------------

    def reload(self):
        with self._lock:
            cfg = load_config(str(self.config_path))
            self._overrides = {}
            if self.overrides_path.exists():
                try:
                    data = json.loads(self.overrides_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        self._overrides = data
                except Exception as e:
                    logger.warning(f"Ignoring unreadable {self.overrides_path.name}: {e}")
            for key, val in self._overrides.items():
                _set(cfg, key, val)
            self._cfg = cfg

    @property
    def cfg(self) -> dict:
        with self._lock:
            return copy.deepcopy(self._cfg)

    def on_change(self, fn: Callable[[dict], None]):
        self._listeners.append(fn)

    # --- editing ---------------------------------------------------------

    def update(self, patch: Dict[str, Any]) -> Dict[str, str]:
        """Apply {dotted.key: value}. Returns {key: error} (nothing is saved if any error)."""
        errors: Dict[str, str] = {}
        clean: Dict[str, Any] = {}
        for key, value in patch.items():
            field = FIELD_BY_KEY.get(key)
            if field is None:
                errors[key] = "unknown setting"
                continue
            try:
                clean[key] = coerce(field, value)
            except ValueError as e:
                errors[key] = str(e)
        if errors:
            return errors
        with self._lock:
            self._overrides.update(clean)
            self._save()
            self.reload()
        self._notify()
        return {}

    def set_runtime(self, key: str, value: Any):
        """Remember a non-schema value (e.g. the last audio source chosen)."""
        with self._lock:
            self._overrides[key] = value
            self._save()
            self.reload()

    def reset(self):
        with self._lock:
            self._overrides = {}
            try:
                self.overrides_path.unlink()
            except FileNotFoundError:
                pass
            self.reload()
        self._notify()

    def _save(self):
        self.overrides_path.write_text(json.dumps(self._overrides, indent=2), encoding="utf-8")

    def _notify(self):
        cfg = self.cfg
        for fn in self._listeners:
            try:
                fn(cfg)
            except Exception as e:
                logger.error(f"settings listener failed: {e}", exc_info=True)

    # --- for the GUI -----------------------------------------------------

    def view(self) -> dict:
        """Values + schema + which keys are overridden, for the Settings page."""
        cfg = self.cfg
        early, late = margins_from_cfg(cfg["timing"])
        values = {f["key"]: _get(cfg, f["key"]) for f in FIELDS}
        values["timing.on_time_early_s"], values["timing.on_time_late_s"] = early, late
        return {
            "values": values,
            "fields": FIELDS,
            "overridden": sorted(k for k in self._overrides if k in FIELD_BY_KEY),
            "audio_last": cfg["audio"].get("last"),
            "operator_last": (cfg.get("operator") or {}).get("last", ""),
            "config_file": str(self.config_path),
            "overrides_file": str(self.overrides_path),
            "gui": {"host": cfg["gui"].get("host"), "port": cfg["gui"].get("port")},
        }
