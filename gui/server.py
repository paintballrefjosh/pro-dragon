"""
Web GUI server
===============
Serves the single-page GUI (gui/static/index.html), pushes live state to it over a
WebSocket, and exposes a small JSON API that drives the Controller (app.py):

  GET  /api/settings            POST /api/settings {patch}      POST /api/settings/reset
  GET  /api/devices
  POST /api/propresenter/connect {host?, port?}
  POST /api/monitoring/start {source:{kind,index,channels,file}}   POST /api/monitoring/stop
  POST /api/audio/source {source:{...}}      switch what the always-on monitoring listens to
  POST /api/song/start | /api/song/stop
  GET  /api/songs               GET  /api/song?id=              GET /api/graph.svg?id=&late=&early=
  GET  /api/song/file?id=&kind=json|txt|svg|wav
  POST /api/song/reanalyze {id, offset_ms?, retranscribe?}
  POST /api/operator {name}                     POST /api/song/operator {id, operator}
  GET  /api/operators?late=&early=              GET  /api/operator/graph.svg?name=&late=&early=
  GET  /api/live/graph.svg?late=&early=         (provisional graph while a song is being recorded)

Runs its own aiohttp app + asyncio loop in a background thread (the rest of the app
is thread-based). `broadcast()` is the thread-safe way to push state.

Security: there is no login — it's meant for the booth computer or a trusted LAN.
By default it listens on 127.0.0.1 only. Because a web page you visit could try to
call a local service, every POST must carry an `X-ProDragon: 1` header (which other
sites can't send without a CORS preflight we never grant) and the WebSocket checks
its Origin.
"""

import asyncio
import json
import threading
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from utils.logger import get_logger

logger = get_logger("dashboard")

STATIC_DIR = Path(__file__).parent / "static"


class DashboardServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.controller = None            # set by Controller
        self.on_command = None            # legacy hook (unused when a controller is attached)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._runner = None
        self._clients: set = set()
        self._state: dict = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        ready = threading.Event()
        self._start_error: Optional[Exception] = None
        self._thread = threading.Thread(target=self._run_loop, args=(ready,), daemon=True, name="Dashboard")
        self._thread.start()
        ready.wait(timeout=10)
        if self._start_error:
            raise self._start_error
        logger.info(f"GUI running at http://{self.host}:{self.port}")

    def stop(self):
        if not self._loop:
            return
        fut = asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop)
        try:
            fut.result(timeout=5)
        except Exception as e:
            logger.warning(f"Dashboard shutdown error: {e}")
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self, ready: threading.Event):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_start())
        except Exception as e:                 # e.g. port already in use
            self._start_error = e
            ready.set()
            return
        ready.set()
        self._loop.run_forever()

    # ------------------------------------------------------------------
    # aiohttp app
    # ------------------------------------------------------------------

    async def _async_start(self):
        from aiohttp import web

        @web.middleware
        async def guard(request, handler):
            if request.method == "POST" and request.headers.get("X-ProDragon") != "1":
                return web.json_response({"ok": False, "error": "missing X-ProDragon header"}, status=403)
            return await handler(request)

        app = web.Application(middlewares=[guard])
        r = app.router
        r.add_get("/", self._handle_index)
        r.add_get("/ws", self._handle_ws)
        r.add_get("/api/settings", self._api_settings_get)
        r.add_post("/api/settings", self._api_settings_post)
        r.add_post("/api/settings/reset", self._api_settings_reset)
        r.add_get("/api/devices", self._api_devices)
        r.add_post("/api/propresenter/connect", self._api_connect)
        r.add_post("/api/monitoring/start", self._api_monitor_start)
        r.add_post("/api/monitoring/stop", self._api_monitor_stop)
        r.add_post("/api/audio/source", self._api_audio_source)
        r.add_post("/api/song/reanalyze", self._api_reanalyze)
        r.add_post("/api/song/operator", self._api_song_operator)      # must come before /api/song/{cmd}
        r.add_post("/api/song/{cmd}", self._api_song_cmd)
        r.add_post("/api/operator", self._api_operator_set)
        r.add_get("/api/live/graph.svg", self._api_live_graph)
        r.add_get("/api/operators", self._api_operators)
        r.add_get("/api/operator/graph.svg", self._api_operator_graph)
        r.add_get("/api/songs", self._api_songs)
        r.add_post("/api/songs/delete", self._api_songs_delete)
        r.add_post("/api/songs/remap", self._api_song_remap)
        r.add_get("/api/song", self._api_song)
        r.add_get("/api/graph.svg", self._api_graph)
        r.add_get("/api/song/file", self._api_song_file)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

    async def _async_stop(self):
        for ws in list(self._clients):
            await ws.close()
        if self._runner:
            await self._runner.cleanup()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    async def _run(self, fn, *args):
        """Run blocking controller code off the event loop."""
        return await asyncio.get_running_loop().run_in_executor(None, fn, *args)

    @staticmethod
    async def _body(request) -> dict:
        try:
            data = await request.json()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _need_controller(self):
        from aiohttp import web
        if self.controller is None:
            raise web.HTTPServiceUnavailable(text=json.dumps({"ok": False, "error": "no controller"}),
                                             content_type="application/json")
        return self.controller

    @staticmethod
    def _err(msg: str, status: int = 400):
        from aiohttp import web
        return web.json_response({"ok": False, "error": msg}, status=status)

    # ------------------------------------------------------------------
    # pages + websocket
    # ------------------------------------------------------------------

    async def _handle_index(self, request):
        from aiohttp import web
        return web.FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    def _current_state(self) -> dict:
        state = dict(self._state)
        if self.controller is not None:
            state.update(self.controller.snapshot())
        return state

    async def _handle_ws(self, request):
        from aiohttp import web

        origin = request.headers.get("Origin")
        if origin and urlparse(origin).netloc != request.host:
            return web.Response(status=403, text="bad origin")

        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._clients.add(ws)
        logger.debug(f"GUI client connected ({len(self._clients)} total)")
        try:
            await ws.send_str(json.dumps(self._current_state(), default=str))
            async for _ in ws:
                pass
        finally:
            self._clients.discard(ws)
            logger.debug(f"GUI client disconnected ({len(self._clients)} total)")
        return ws

    # ------------------------------------------------------------------
    # API: settings / devices / connection / monitoring
    # ------------------------------------------------------------------

    async def _api_settings_get(self, request):
        from aiohttp import web
        return web.json_response(self._need_controller().get_settings())

    async def _api_settings_post(self, request):
        from aiohttp import web
        body = await self._body(request)
        patch = body.get("patch") if isinstance(body.get("patch"), dict) else {}
        res = await self._run(self._need_controller().update_settings, patch)
        return web.json_response(res, status=200 if res["ok"] else 422)

    async def _api_settings_reset(self, request):
        from aiohttp import web
        return web.json_response(await self._run(self._need_controller().reset_settings))

    async def _api_devices(self, request):
        from aiohttp import web
        try:
            ctrl = self._need_controller()
            want_ndi = request.query.get("ndi") == "1"
            devices = await self._run(ctrl.list_audio_sources, want_ndi)
            ndi_ok = await self._run(ctrl.ndi_available)
        except Exception as e:
            logger.error(f"listing audio devices failed: {e}", exc_info=True)
            return self._err(f"couldn't list audio devices: {e}", 500)
        return web.json_response({"ok": True, "devices": devices, "ndi_available": ndi_ok})

    async def _api_connect(self, request):
        from aiohttp import web
        body = await self._body(request)
        ok, msg = await self._run(self._need_controller().connect_propresenter, body.get("host"), body.get("port"))
        return web.json_response({"ok": ok, "message": msg}, status=200 if ok else 502)

    async def _api_monitor_start(self, request):
        from aiohttp import web
        body = await self._body(request)
        spec = body.get("source") if isinstance(body.get("source"), dict) else {}
        operator = body.get("operator")
        ok, msg = await self._run(self._need_controller().start_session, spec, None,
                                  None if operator is None else str(operator))
        return web.json_response({"ok": ok, "message": msg}, status=200 if ok else 400)

    async def _api_audio_source(self, request):
        """Use this audio source from now on (monitoring is always on; this just switches what it listens to)."""
        from aiohttp import web
        body = await self._body(request)
        spec = body.get("source") if isinstance(body.get("source"), dict) else {}
        ok, msg = await self._run(self._need_controller().set_audio_source, spec)
        return web.json_response({"ok": ok, "message": msg}, status=200 if ok else 400)

    async def _api_monitor_stop(self, request):
        from aiohttp import web
        ok, msg = await self._run(self._need_controller().stop_session, False)
        return web.json_response({"ok": ok, "message": msg}, status=200 if ok else 400)

    async def _api_song_cmd(self, request):
        from aiohttp import web
        cmd = request.match_info["cmd"]
        if cmd not in ("start", "stop"):
            return self._err("unknown command")
        ok, msg = await self._run(self._need_controller().song_command, cmd)
        return web.json_response({"ok": ok, "message": msg}, status=200 if ok else 400)

    # ------------------------------------------------------------------
    # API: songs
    # ------------------------------------------------------------------

    async def _api_songs(self, request):
        from aiohttp import web
        songs = await self._run(self._need_controller().library.list)
        return web.json_response({"ok": True, "songs": songs})

    async def _api_song(self, request):
        from aiohttp import web
        from library import LibraryError
        try:
            detail = await self._run(self._need_controller().library.detail, request.query.get("id", ""))
        except LibraryError as e:
            return self._err(str(e), 404)
        return web.json_response({"ok": True, "song": detail})

    async def _api_graph(self, request):
        from aiohttp import web
        from library import LibraryError
        try:
            late = float(request.query.get("late", "0.75"))
            early = float(request.query.get("early", str(late)))
            if not (0 <= late <= 60 and 0 <= early <= 60):
                raise ValueError
        except ValueError:
            return self._err("bad margin")
        try:
            svg = await self._run(self._need_controller().library.graph_svg, request.query.get("id", ""), late, early)
        except LibraryError as e:
            return self._err(str(e), 404)
        return web.Response(text=svg, content_type="image/svg+xml", headers={"Cache-Control": "no-store"})

    async def _api_song_file(self, request):
        from aiohttp import web
        from library import LibraryError
        try:
            path = await self._run(self._need_controller().library.file_path,
                                   request.query.get("id", ""), request.query.get("kind", ""))
        except LibraryError as e:
            return self._err(str(e), 404)
        return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="{path.name}"'})

    # ------------------------------------------------------------------
    # API: operators
    # ------------------------------------------------------------------

    @staticmethod
    def _margins(request):
        """(late, early) from ?late=&early=, either may be omitted (None = use the saved default)."""
        def one(key):
            raw = request.query.get(key)
            if raw in (None, ""):
                return None
            v = float(raw)
            if not (0 <= v <= 60):
                raise ValueError
            return v
        late = one("late")
        early = one("early")
        return late, (late if early is None and late is not None else early)

    async def _api_operators(self, request):
        from aiohttp import web
        try:
            late, early = self._margins(request)
        except ValueError:
            return self._err("bad margin")
        entries = await self._run(self._need_controller().operator_stats, late, early)
        return web.json_response({"ok": True, "operators": entries})

    async def _api_operator_graph(self, request):
        from aiohttp import web
        from library import LibraryError
        try:
            late, early = self._margins(request)
        except ValueError:
            return self._err("bad margin")
        ctrl = self._need_controller()

        def build():
            from analysis.song_timing import margins_from_cfg
            d_early, d_late = margins_from_cfg(ctrl.settings.cfg["timing"])
            lt = d_late if late is None else late
            return ctrl.library.operator_graph_svg(request.query.get("name", ""), lt, d_early if early is None else early)
        try:
            svg = await self._run(build)
        except LibraryError as e:
            return self._err(str(e), 404)
        return web.Response(text=svg, content_type="image/svg+xml", headers={"Cache-Control": "no-store"})

    async def _api_live_graph(self, request):
        """Provisional graph of the song being recorded right now (404 when no song is being recorded)."""
        from aiohttp import web
        try:
            late, early = self._margins(request)
        except ValueError:
            return self._err("bad margin")
        svg = await self._run(self._need_controller().live_graph_svg, late, early)
        if svg is None:
            return self._err("no song is being recorded", 404)
        return web.Response(text=svg, content_type="image/svg+xml", headers={"Cache-Control": "no-store"})

    async def _api_operator_set(self, request):
        from aiohttp import web
        body = await self._body(request)
        ok, msg = await self._run(self._need_controller().set_operator, body.get("name", ""))
        return web.json_response({"ok": ok, "message": msg}, status=200 if ok else 400)

    async def _api_song_operator(self, request):
        from aiohttp import web
        body = await self._body(request)
        res = await self._run(self._need_controller().set_song_operator, str(body.get("id", "")), body.get("operator", ""))
        return web.json_response(res, status=200 if res.get("ok") else 400)

    async def _api_song_remap(self, request):
        from aiohttp import web
        body = await self._body(request)
        res = await self._run(self._need_controller().remap_song_slides, str(body.get("id", "")))
        return web.json_response(res, status=200 if res.get("ok") else 400)

    async def _api_songs_delete(self, request):
        from aiohttp import web
        body = await self._body(request)
        ids = body.get("ids")
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            return self._err("ids must be a list of song ids")
        res = await self._run(self._need_controller().delete_songs, ids)
        return web.json_response(res, status=200 if res.get("ok") else 400)

    async def _api_reanalyze(self, request):
        from aiohttp import web
        body = await self._body(request)
        offset = body.get("offset_ms")
        try:
            offset = None if offset in (None, "") else float(offset)
        except (TypeError, ValueError):
            return self._err("offset must be a number")
        res = await self._run(self._need_controller().reanalyze_song, str(body.get("id", "")), offset,
                              bool(body.get("retranscribe")))
        return web.json_response(res, status=200 if res.get("ok") else 400)

    # ------------------------------------------------------------------
    # Broadcast (thread-safe)
    # ------------------------------------------------------------------

    def broadcast(self, updates: dict):
        self._state.update(updates)
        if not self._loop:
            return
        payload = json.dumps(self._state, default=str)
        asyncio.run_coroutine_threadsafe(self._broadcast_async(payload), self._loop)

    def has_clients(self) -> bool:
        return bool(self._clients) and self._loop is not None

    def send_transient(self, message: dict):
        """Send a small message to every browser WITHOUT merging it into the stored state
        (used for the fast-updating equalizer; new browsers don't need to replay it)."""
        if not self._loop or not self._clients:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(json.dumps(message)), self._loop)

    async def _broadcast_async(self, payload: str):
        dead = []
        for ws in list(self._clients):
            try:
                await ws.send_str(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)
