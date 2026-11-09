"""
ProPresenter 7 Network API Client
==================================
Talks to the Pro7 REST API (port 1025 by default) and detects slide changes
by polling `GET /v1/presentation/slide_index` every `poll_interval` seconds.

Why polling instead of a WebSocket/status-stream subscription:
  Pro Dragon's whole job is measuring *when* a slide changed, so the slide
  timestamp matters more than anything else. Polling a tiny local endpoint at
  ~10 Hz needs no knowledge of the (version-dependent) subscription message
  formats, survives reconnects trivially, and gives a bounded timestamp error
  (<= poll_interval / 2, since we stamp each change at the midpoint between the
  last poll that saw the old slide and the poll that saw the new one).

REST endpoints used:
  GET /version                       - connectivity / version check (root, no /v1)
  GET /v1/presentation/active        - live presentation incl. all slide text
                                       (older builds: /v1/presentation/current)
  GET /v1/presentation/slide_index   - which slide of which presentation is live

Slide indexes: Pro7 numbers slides 0..N-1 across ALL groups in order, including
slides that contain no text (blank/title slides). We keep every slide — even
empty ones — so the index Pro7 reports always lines up with `Presentation.slides`.

All responses are logged at DEBUG level (run with --debug) so field names can be
checked against your Pro7 version if something doesn't parse.
"""

import json
import threading
import time
from typing import Callable, List, Optional, Tuple

import requests

from utils.logger import get_logger

logger = get_logger("propresenter")


class Slide:
    def __init__(self, index: int, text: str, label: str = "", group: str = "", enabled: bool = True):
        self.index = index
        self.text = (text or "").strip()
        self.label = label or ""
        self.group = group or ""
        self.enabled = enabled

    def __repr__(self):
        preview = self.text[:50].replace("\n", " ")
        return f"Slide({self.index}: '{preview}')"


class Presentation:
    def __init__(self, name: str, uuid: str, slides: List[Slide]):
        self.name = name
        self.uuid = uuid
        self.slides = slides

    def __repr__(self):
        return f"Presentation('{self.name}', {len(self.slides)} slides)"


class ProPresenterClient:
    def __init__(self, host: str, port: int, password: str = "", poll_interval: float = 0.1):
        self.host = host
        self.port = port
        self.password = password
        self.poll_interval = max(0.02, float(poll_interval))
        self.base_url = f"http://{host}:{port}"

        self._http = requests.Session()  # keep-alive: keeps per-poll latency low
        self._poll_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._connected = threading.Event()

        # Callbacks — set by the session manager
        self.on_slide_changed: Optional[Callable[[int, float], None]] = None   # (index, timestamp)
        self.on_presentation_changed: Optional[Callable[[Presentation], None]] = None
        # The live presentation's slide count changed (someone edited it): (old, new, since_ts) / (cues, loaded)
        self.on_slides_updated: Optional[Callable[[Presentation, Presentation, float], None]] = None
        self.on_slides_mismatch: Optional[Callable[[int, int], None]] = None
        self._mismatch_since: Optional[float] = None
        self._mismatch_gave_up: Optional[tuple] = None
        self._last_reload_try = 0.0

        # Connection health: the poll loop keeps running whether or not ProPresenter is reachable, so the
        # link heals by itself (ProPresenter started later, restarted, network blip …).
        self._last_ok = 0.0                       # wall-clock time of the last successful HTTP reply
        self._was_connected = False
        self.on_connection_changed: Optional[Callable[[bool], None]] = None

        # Latest known state (updated by the poll loop)
        self.current_presentation: Optional[Presentation] = None
        self.current_uuid: Optional[str] = None
        self.current_slide_index: Optional[int] = None
        self.live_total_cues: Optional[int] = None

    # ------------------------------------------------------------------
    # REST helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if self.password:
            h["Authorization"] = f"Bearer {self.password}"
        return h

    def _get(self, path: str, quiet: bool = False) -> Optional[dict]:
        url = f"{self.base_url}{path}"
        try:
            resp = self._http.get(url, headers=self._headers(), timeout=3)
            resp.raise_for_status()
            self._last_ok = time.time()
            if not resp.content:
                return {}
            data = resp.json()
            logger.debug(f"GET {path} → {json.dumps(data)[:2000]}")
            return data
        except requests.exceptions.ConnectionError:
            if not quiet:
                logger.error(f"Cannot reach ProPresenter at {self.base_url} — is it running with Network enabled?")
        except requests.exceptions.Timeout:
            if not quiet:
                logger.error(f"Timeout on GET {path}")
        except requests.exceptions.HTTPError as e:
            if not quiet:
                logger.error(f"HTTP {e.response.status_code} on GET {path}: {e.response.text[:200]}")
        except Exception as e:
            if not quiet:
                logger.error(f"Unexpected error on GET {path}: {e}")
        return None

    # ------------------------------------------------------------------
    # Connection health / settings
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        """True while ProPresenter answered recently (the poll loop asks several times a second)."""
        return time.time() - self._last_ok < max(3.0, self.poll_interval * 10)

    def reconfigure(self, host: str, port, password: str = "", poll_interval: Optional[float] = None):
        """Point this client at another ProPresenter (settings changed). The poll loop carries on with the new
        address; everything learned about the old one is forgotten."""
        self.host, self.port, self.password = host, int(port), password or ""
        if poll_interval is not None:
            self.poll_interval = max(0.02, float(poll_interval))
        self.base_url = f"http://{self.host}:{self.port}"
        self._last_ok = 0.0
        self._connected.clear()
        self.current_uuid = self.current_presentation = self.current_slide_index = self.live_total_cues = None
        self._mismatch_since = self._mismatch_gave_up = None

    # ------------------------------------------------------------------
    # Connection check
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        logger.info(f"Checking ProPresenter connection at {self.base_url} ...")
        # /version lives at the server root (no /v1 prefix); try /v1/version as a fallback.
        data = self._get("/version", quiet=True)
        if data is None:
            data = self._get("/v1/version")
        if data is None:
            return False
        name = data.get("name", "ProPresenter")
        version = data.get("host_description") or data.get("version") or data.get("api_version") or "unknown"
        platform = data.get("platform", "")
        logger.info(f"Connected to {name} — {version} {platform}".strip())
        self._connected.set()
        return True

    # ------------------------------------------------------------------
    # Presentation / slide fetching
    # ------------------------------------------------------------------

    def fetch_active_presentation(self) -> Optional[Presentation]:
        """Fetch the currently LIVE presentation and all of its slides."""
        data = self._get("/v1/presentation/active", quiet=True)
        if data is None:
            data = self._get("/v1/presentation/current")  # older API name
        if not data:
            return None
        return self._parse_presentation(data)

    def fetch_slide_index(self) -> Tuple[Optional[int], Optional[str]]:
        """Return (slide_index, presentation_uuid) of the live slide, or (None, None)."""
        data = self._get("/v1/presentation/slide_index", quiet=True)
        if not data:
            return None, None
        pi = data.get("presentation_index", data)
        if not isinstance(pi, dict):
            return None, None
        idx = pi.get("index")
        self.live_total_cues = pi.get("total_cues")      # for a sanity check against our slide list
        pid = pi.get("presentation_id") or {}
        uuid = pid.get("uuid") if isinstance(pid, dict) else None
        if idx is None:
            return None, uuid
        try:
            return int(idx), uuid
        except (TypeError, ValueError):
            return None, uuid

    def _parse_presentation(self, data: dict) -> Optional[Presentation]:
        try:
            pres = data.get("presentation") or data
            pid = pres.get("id", {})
            if isinstance(pid, dict):
                name = pid.get("name") or pres.get("name") or "Unknown"
                uuid = pid.get("uuid", "")
            else:
                name = pres.get("name") or "Unknown"
                uuid = str(pid)

            slides: List[Slide] = []
            groups = self._apply_arrangement(pres, name)
            if groups:
                for group in groups:
                    gname = group.get("name", "")
                    for raw in group.get("slides", []):
                        slides.append(Slide(
                            len(slides), self._extract_slide_text(raw),
                            label=raw.get("label", ""), group=gname,
                            enabled=raw.get("enabled", True),
                        ))
            else:
                for raw in pres.get("slides", []):
                    slides.append(Slide(
                        len(slides), self._extract_slide_text(raw),
                        label=raw.get("label", ""), enabled=raw.get("enabled", True),
                    ))

            if not any(s.text for s in slides):
                logger.warning(f"Presentation '{name}' loaded but no slide text found — run with --debug to inspect the JSON")

            pres_obj = Presentation(name, str(uuid), slides)
            logger.info(f"Loaded: {pres_obj}")
            for s in slides:
                logger.debug(f"  {s}")
            return pres_obj
        except Exception as e:
            logger.error(f"Failed to parse presentation: {e}")
            logger.debug(f"Raw data: {json.dumps(data)[:2000]}")
            return None

    @staticmethod
    def _apply_arrangement(pres: dict, name: str = "") -> list:
        """
        The slide index Pro7 reports while a song is live counts cues in the
        song's *current arrangement*, where a group (e.g. Chorus 1) can appear
        several times. The `groups` list in the API response holds each group
        once, so expand it by the arrangement's group-uuid order to get one
        entry per cue — otherwise every slide after the first repeat is mislabelled.
        """
        groups = pres.get("groups") or []
        current = pres.get("current_arrangement")
        if not groups or not current:
            return groups
        arrangement = None
        for a in pres.get("arrangements") or []:
            aid = a.get("id")
            if (aid.get("uuid") if isinstance(aid, dict) else aid) == current:
                arrangement = a
                break
        order = (arrangement or {}).get("groups") or []
        if not order:
            return groups
        by_uuid = {g.get("uuid"): g for g in groups}
        expanded = [by_uuid[u] for u in order if u in by_uuid]
        if len(expanded) != len(order):
            logger.warning(f"'{name}': arrangement references groups that weren't found — using plain group order")
            return groups
        return expanded

    @staticmethod
    def _extract_slide_text(raw_slide: dict) -> str:
        """Plain text of a slide, tolerating the different shapes Pro7 versions use."""
        val = raw_slide.get("text")
        if isinstance(val, str):
            return val.strip()
        if isinstance(val, dict):
            return (val.get("text") or val.get("plainText") or "").strip()

        parts = []
        for el in raw_slide.get("elements", []):
            el_text = el.get("element", {}) if "element" in el else el
            text_val = el_text.get("text", {})
            if isinstance(text_val, dict):
                t = text_val.get("text") or text_val.get("plainText") or ""
            elif isinstance(text_val, str):
                t = text_val
            else:
                t = ""
            if t.strip():
                parts.append(t.strip())
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Slide-change polling
    # ------------------------------------------------------------------

    def start_polling(self):
        """Start the background poll loop that detects slide / presentation changes."""
        self._stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="ProPresenter-Poll")
        self._poll_thread.start()
        logger.info(f"Watching ProPresenter slide index every {self.poll_interval * 1000:.0f} ms")

    def _poll_loop(self):
        last_ok_ts: Optional[float] = None
        warned = False
        while not self._stop.is_set():
            started = time.time()
            idx, uuid = self.fetch_slide_index()
            now = time.time()

            if idx is None and uuid is None:
                # Nothing live (or a transient failure) — don't treat as a slide change.
                if not warned and last_ok_ts is not None and now - last_ok_ts > 5:
                    logger.warning("No slide index from ProPresenter (nothing live, or connection lost)")
                    warned = True
            else:
                warned = False
                # Best estimate of when the change happened: halfway between the last poll
                # that still saw the old state and the poll that saw the new one.
                change_ts = (last_ok_ts + now) / 2 if last_ok_ts is not None else now
                last_ok_ts = now

                if uuid and uuid != self.current_uuid:
                    self.current_uuid = uuid
                    pres = self.fetch_active_presentation()
                    if pres:
                        self.current_presentation = pres
                        if self.on_presentation_changed:
                            self._safe(self.on_presentation_changed, pres)
                    # the index belongs to the new presentation: report it below as a change
                    self.current_slide_index = None
                    self._mismatch_since = self._mismatch_gave_up = None

                self._check_slide_list(uuid, now)       # before reporting a change: the index may mean a different slide now

                if idx is not None and idx != self.current_slide_index:
                    first_sight = self.current_slide_index is None
                    self.current_slide_index = idx
                    logger.info(f"Slide → {idx + 1}")
                    if self.on_slide_changed:
                        # On first sight we don't know when it actually changed.
                        self._safe(self.on_slide_changed, idx, now if first_sight else change_ts)

            is_up = self.connected
            if is_up != self._was_connected:
                self._was_connected = is_up
                if self.on_connection_changed:
                    self._safe(self.on_connection_changed, is_up)

            elapsed = time.time() - started
            self._stop.wait(max(0.0, self.poll_interval - elapsed))

    def _check_slide_list(self, uuid: Optional[str], now: float):
        """
        ProPresenter reports how many cues the live presentation has. If that no longer equals the
        number of slides we loaded, the presentation was edited (a slide or group added/removed) and
        every slide index after the edit now means a different slide — so reload the list.
        """
        pres, cues = self.current_presentation, self.live_total_cues
        if pres is None or not isinstance(cues, int) or uuid != self.current_uuid or cues == len(pres.slides):
            self._mismatch_since = None
            return
        if self._mismatch_since is None:
            self._mismatch_since = now
        if self._mismatch_gave_up == (cues, len(pres.slides)) or now - self._last_reload_try < 2.0:
            return
        self._last_reload_try = now
        fresh = self.fetch_active_presentation()
        if fresh is None or fresh.uuid != pres.uuid:
            return
        since = self._mismatch_since - self.poll_interval
        if len(fresh.slides) == cues:
            logger.warning(f"The slide list of '{pres.name}' changed ({len(pres.slides)} → {len(fresh.slides)} slides) — reloaded")
            self.current_presentation = fresh
            self._mismatch_since = self._mismatch_gave_up = None
            if self.on_slides_updated:
                self._safe(self.on_slides_updated, pres, fresh, since)
        else:                                           # still disagrees after a reload: warn once, don't keep hammering
            self._mismatch_gave_up = (cues, len(pres.slides))
            logger.warning(f"ProPresenter reports {cues} slides live but {len(fresh.slides)} could be loaded")
            if self.on_slides_mismatch:
                self._safe(self.on_slides_mismatch, cues, len(fresh.slides))

    @staticmethod
    def _safe(fn, *args):
        try:
            fn(*args)
        except Exception as e:  # never let a callback bug kill the poll thread
            logger.error(f"Error in ProPresenter callback: {e}", exc_info=True)

    def disconnect(self):
        self._stop.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=2)
        self._http.close()
