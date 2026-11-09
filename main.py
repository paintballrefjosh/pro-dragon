"""
Pro Dragon — Main Entry Point
===============================
Measures how early/late a ProPresenter operator changes each lyric slide
relative to when the vocalist starts singing it.

Usage:
  python main.py --gui                    # browser GUI (setup, live view, song graphs, settings)
  python main.py                          # terminal mode (prompts for audio source; S = start/stop song)
  python main.py --loopback               # terminal mode, capture the computer's speakers (testing, Windows)
  python main.py --audio-file test.wav    # terminal mode, play a wav file as the audio source
  python main.py --ndi "MIXER (Audio)"    # terminal mode, audio from an NDI source (+ --offset-ms for its delay)
  python main.py --offset-ms 200          # audio arrives 200 ms late (e.g. NDI)
  python main.py --operator "Sam"         # name the person running slides (stored with each song)
  python main.py --operators              # per-operator report over all saved songs
  python main.py --list-devices           # show audio sources and exit
  python main.py --analyze reports/session_x/song_01_x.json [--offset-ms N] [--retranscribe]
                                          # re-run the analysis on a saved song
"""

import argparse
import logging
import signal
import socket
import sys
import time
from pathlib import Path

from settings import SettingsStore, load_config  # noqa: F401  (load_config re-exported for convenience)
from utils.logger import get_logger, set_level

logger = get_logger("main")


def make_whisper(cfg: dict):
    from transcription.whisper_asr import WhisperASR
    w = cfg["whisper"]
    asr = WhisperASR(model_size=w.get("model", "small"), language=w.get("language", "en"),
                     device=w.get("device", "cpu"), compute_type=w.get("compute_type", "int8"))
    asr.load_model()
    return asr


def _local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "localhost"
    finally:
        s.close()


# ──────────────────────────────────────────────────────────────────────
# Audio source selection (terminal mode)
# ──────────────────────────────────────────────────────────────────────

def build_audio_source_factory(cfg: dict, args):
    """Resolve which audio source to use (CLI > config > interactive menu) and
    return a factory `f(on_frame) -> AudioSource`."""
    from audio.capture import (DeviceSource, FileSource, LoopbackSource, NdiSource, list_devices,
                               print_devices, select_channels_interactive, select_device_interactive)
    a = cfg["audio"]
    audio_file = args.audio_file or a.get("file")
    ndi_name = args.ndi or a.get("ndi")
    ndi_addr = args.ndi_address or a.get("ndi_address") or ""
    loopback = args.loopback or a.get("loopback", False)
    device_setting = a.get("device")
    channels = a.get("channels") or [1]

    if audio_file:
        return lambda cb: FileSource(audio_file, on_frame=cb)

    if ndi_name:                                   # all NDI channels are mixed to mono
        return lambda cb: NdiSource(str(ndi_name), None, on_frame=cb, addresses=ndi_addr)

    if loopback:
        dev_index = None
        if isinstance(device_setting, str) and device_setting:
            matches = [d for d in list_devices() if d["kind"] == "loopback" and device_setting.lower() in d["name"].lower()]
            if not matches:
                logger.error(f"No speaker device matching '{device_setting}'. Run --list-devices.")
                sys.exit(1)
            dev_index = matches[0]["index"]
        return lambda cb: LoopbackSource(dev_index, on_frame=cb)

    if device_setting is None:
        print_devices(ndi_addresses=ndi_addr)
        dev = select_device_interactive(ndi_addresses=ndi_addr)
        if dev["kind"] == "ndi":
            return lambda cb: NdiSource(dev["name"], None, on_frame=cb, addresses=ndi_addr)
        if dev["kind"] == "loopback":
            return lambda cb: LoopbackSource(dev["index"], on_frame=cb)
        chans = select_channels_interactive(dev)
        return lambda cb: DeviceSource(dev["index"], chans, on_frame=cb)

    devices = [d for d in list_devices(include_loopback=False)]
    if isinstance(device_setting, int):
        idx = device_setting
    else:
        matches = [d for d in devices if str(device_setting).lower() in d["name"].lower()]
        if not matches:
            logger.error(f"No input device matching '{device_setting}'. Run --list-devices.")
            sys.exit(1)
        if len(matches) > 1:
            logger.warning(f"Multiple devices match '{device_setting}': {[d['name'] for d in matches]} — using the first")
        idx = matches[0]["index"]
    return lambda cb: DeviceSource(idx, channels, on_frame=cb)


# ──────────────────────────────────────────────────────────────────────
# Modes
# ──────────────────────────────────────────────────────────────────────

def run_analyze(args, settings: SettingsStore):
    from analysis.song_runner import analyze_songdata, load_song, save_song
    from analysis.song_timing import margins_from_cfg
    from report.timing_report import format_song_report

    cfg = settings.cfg
    json_path = Path(args.analyze)
    song = load_song(json_path)
    offset_ms = args.offset_ms if args.offset_ms is not None else cfg["timing"].get("offset_ms", 0)
    early, late = margins_from_cfg(cfg["timing"])

    need_whisper = song.words is None or args.retranscribe
    whisper = make_whisper(cfg) if need_whisper else None
    analyze_songdata(song, whisper, cfg["timing"], offset_ms, force_transcribe=args.retranscribe)

    print(format_song_report(song.name, song.song_start_ts, song.duration, song.timings, offset_ms, late,
                             unshown=song.unshown, heard_other=song.heard_other, early_s=early))
    out = save_song(song, json_path.parent, 0, offset_ms, late, early, stem=json_path.stem)
    logger.info(f"Updated {out.name} and its .txt report and .svg graph")


def run_operators(args, settings: SettingsStore):
    """Print the per-operator report across every saved song (no GUI, no ProPresenter needed)."""
    from analysis.operators import aggregate, format_operator_report
    from analysis.song_timing import margins_from_cfg
    from library import SongLibrary

    cfg = settings.cfg
    early, late = margins_from_cfg(cfg["timing"])
    lib = SongLibrary(lambda: cfg["report"]["output_dir"])
    entries = aggregate(lib._all_details(), late, early)
    text = format_operator_report(entries, late, early)
    print(text)
    out = Path(cfg["report"]["output_dir"]) / "operator_report.txt"
    try:
        out.write_text(text, encoding="utf-8")
        logger.info(f"Also saved to {out}")
    except OSError as e:
        logger.warning(f"Could not save {out}: {e}")


def _make_dashboard(cfg: dict):
    from gui.server import DashboardServer
    gui = cfg["gui"]
    return DashboardServer(host=gui.get("host", "127.0.0.1"), port=gui.get("port", 8765))


def run_live(args, settings: SettingsStore):
    """Terminal mode: the same Controller as the GUI, driven by the keyboard."""
    from app import Controller
    from utils.keys import KeyPoller

    cfg = settings.cfg
    factory = build_audio_source_factory(cfg, args)

    dashboard = None
    if cfg["gui"].get("enabled", True):
        dashboard = _make_dashboard(cfg)
    controller = Controller(settings, dashboard, cli_offset_ms=args.offset_ms, console=True)

    ok, msg = controller.connect_propresenter()
    if not ok:
        logger.error("Cannot connect to ProPresenter. Check propresenter.host/port in config.yaml "
                     "and that Network is enabled in ProPresenter's preferences.")
        sys.exit(1)

    if dashboard:
        try:
            dashboard.start()
        except OSError as e:
            # Don't let a busy port take down a live session: carry on in the terminal.
            logger.warning(f"The GUI page isn't available ({dashboard.host}:{dashboard.port}: {e}). "
                           "Is another Pro Dragon already running? Continuing in the terminal only.")
            dashboard = None
            controller.dashboard = None
        else:
            controller.start_ticker()
            shown = _local_ip() if dashboard.host == "0.0.0.0" else dashboard.host
            logger.info(f"GUI: http://{shown}:{dashboard.port}")

    logger.info("Loading Whisper model (first run downloads it) ...")
    controller.get_whisper()

    ok, msg = controller.start_session({}, source_factory=factory, operator=args.operator)
    if not ok:
        logger.error(msg)
        controller.shutdown()
        if dashboard:
            dashboard.stop()
        sys.exit(1)
    session = controller.session
    logger.info(f"Audio source: {session.source.describe()}   |   offset: {session.offset_ms:+.0f} ms")
    logger.info(f"Output folder: {session.out_dir}")

    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        with KeyPoller() as keys:
            session.run(keys)
    finally:
        print()
        controller.stop_session(wait=True)
        controller.shutdown()
        if dashboard:
            dashboard.stop()


def run_gui(args, settings: SettingsStore):
    """Browser GUI: everything (connect, choose audio, start/stop songs, graphs, settings) happens in the page."""
    from app import Controller

    cfg = settings.cfg
    dashboard = _make_dashboard(cfg)
    # auto: connects to ProPresenter by itself (and keeps retrying) and keeps monitoring the last audio source
    controller = Controller(settings, dashboard, cli_offset_ms=args.offset_ms, console=False, auto=True)
    try:
        dashboard.start()
    except OSError as e:
        logger.error(f"Couldn't start the GUI on {dashboard.host}:{dashboard.port}: {e}. "
                     "Is Pro Dragon already running? Change gui.port in config.yaml to use another port.")
        sys.exit(1)
    controller.start_ticker()

    shown = "localhost" if dashboard.host in ("127.0.0.1", "0.0.0.0") else dashboard.host
    url = f"http://{shown}:{dashboard.port}"
    print(f"\n  Pro Dragon GUI:  {url}\n  (leave this window open; press Ctrl+C to quit)\n")
    if dashboard.host == "0.0.0.0":
        logger.warning(f"The GUI is reachable from other computers on the network (http://{_local_ip()}:{dashboard.port}) "
                       "and has no login. Set gui.host to 127.0.0.1 to limit it to this computer.")

    if cfg["gui"].get("open_browser", True) and not args.no_browser:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print()
        logger.info("Shutting down ...")
    finally:
        controller.shutdown()
        dashboard.stop()


def main():
    parser = argparse.ArgumentParser(description="Pro Dragon — per-slide timing audit for ProPresenter operators")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--gui", action="store_true", help="run the browser GUI")
    parser.add_argument("--no-browser", action="store_true", help="with --gui: don't open the browser automatically")
    parser.add_argument("--debug", action="store_true", help="verbose logging (shows raw Pro7 JSON)")
    parser.add_argument("--list-devices", action="store_true", help="list audio sources and exit")
    parser.add_argument("--loopback", action="store_true", help="capture the computer's speakers (Windows, testing)")
    parser.add_argument("--audio-file", help="use a 16-bit .wav file as the audio source")
    parser.add_argument("--ndi", metavar="NAME", help="use the audio of an NDI source (name or part of it); "
                        "network/mixing delay: set --offset-ms (typically +100…+300)")
    parser.add_argument("--ndi-address", metavar="IP[:PORT]", help="IP address / host name of the NDI sender (several: "
                        "comma-separated) — for sources on another subnet/VLAN that automatic discovery can't see")
    parser.add_argument("--offset-ms", type=float, default=None,
                        help="fixed audio offset in ms (+ = audio arrives late, e.g. NDI; - = early); overrides the GUI/config value")
    parser.add_argument("--operator", metavar="NAME", help="who is running the slides (recorded with each song; "
                        "used by the per-operator reports)")
    parser.add_argument("--operators", action="store_true",
                        help="print the per-operator report over all saved songs and exit")
    parser.add_argument("--analyze", metavar="SONG_JSON", help="re-analyze a saved song instead of running live")
    parser.add_argument("--retranscribe", action="store_true", help="with --analyze: run Whisper again")
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    set_level(level)
    logging.getLogger().setLevel(level)

    settings = SettingsStore(args.config)
    if args.ndi_address is not None:
        from audio.capture import parse_ndi_addresses
        try:
            args.ndi_address = parse_ndi_addresses(args.ndi_address)
        except ValueError as e:
            parser.error(str(e))

    if args.list_devices:
        from audio.capture import print_devices
        print_devices(ndi_addresses=args.ndi_address or settings.cfg["audio"].get("ndi_address") or "")
        return

    if args.operators:
        run_operators(args, settings)
    elif args.analyze:
        run_analyze(args, settings)
    elif args.gui:
        run_gui(args, settings)
    else:
        run_live(args, settings)


if __name__ == "__main__":
    main()
