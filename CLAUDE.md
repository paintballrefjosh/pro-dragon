# Pro Dragon — Claude Context

Read automatically by Claude Code. Summarizes goals, architecture and key
decisions so work can continue without the original chat history.

## What This Project Does

Audits ProPresenter 7 operators during live worship songs. For **every slide
change** it reports how many seconds before/after the vocalist started singing
that slide the operator changed it:

    delay = slide_change_time - vocal_onset_time
    + = LATE (singing started before the slide appeared)   - = EARLY (ideal)

The report lists each transition with the delay and the words on the slide so
it's easy to see where the operator was slow. It never controls ProPresenter.
Today it is a **report after the fact** (per song, shortly after Stop); a
near-real-time view is a possible future step.

## How It Works

0. **Two front-ends, one core:** `python main.py --gui` (browser GUI: connect, choose
   audio, monitor, start/stop songs, graphs, settings) or `python main.py` (terminal;
   S = start/stop song, Q = quit). Both drive `app.py::Controller`.
1. **Operator starts/stops each song** (terminal key `S`, or Start/Stop buttons
   in the GUI). Only audio + slide changes inside that window are kept.
2. **Slide events:** `propresenter/client.py` polls `GET /v1/presentation/slide_index`
   every ~100 ms. Each change is stamped at the midpoint between the last poll
   that saw the old slide and the one that saw the new (error <= poll_interval/2).
   Slide indexes are Pro7's own (0..N-1 across all groups, blank slides included).
3. **Audio:** `audio/capture.py` — `AudioSource` interface; sources: `DeviceSource`
   (mic / line-in / Dante, via sounddevice), `LoopbackSource` (Windows speakers via
   PyAudioWPatch — for testing against YouTube), `NdiSource` (NDI audio over the network via
   ndi-python, optional: `requirements-ndi.txt`), `FileSource` (wav playback).
   All deliver mono float32 @16 kHz blocks stamped with the capture wall-clock time.
   NDI: discovered by name (`--ndi`, `audio.ndi`, GUI "NDI" group); frames are planar float32 @ sender rate;
   NDI's own timestamps use the SENDER's clock, so blocks are stamped arrival-time minus duration and the
   network/mix delay (often 100-300 ms) is handled by `timing.offset_ms`. Verified against a real local NDI
   sender (not against a real mixer/camera). `audio.ndi_address` / `--ndi-address` (GUI: Live tab box + Settings)
   = static unicast IP(s) fed to the finder's `extra_ips` so sources on another subnet/VLAN are found
   (`parse_ndi_addresses` validates; verified only against a local sender with 127.0.0.1). No multicast-address setting
   exists on purpose: NDI receivers can't pick one, the runtime joins the sender's group itself. To add another source: subclass `AudioSource` and call
   `self._emit(block, rate, ts)`.
4. **Analysis** (background thread after Stop, `session.py` -> `analysis/`):
   Whisper (`transcription/whisper_asr.py::transcribe_words`, word timestamps,
   lyrics given as `initial_prompt`) -> heard words. `analysis/song_timing.py`
   matches **by words, never by slide position** (the band may sing a different
   arrangement than Pro7's, skip or repeat sections, and the operator may jump
   around): (a) for each distinct slide text, local alignment (Smith-Waterman,
   fuzzy words) of the heard words finds every occurrence where it was sung;
   (b) an order-preserving DP pairs each slide change with an occurrence of ITS OWN
   text within `max_match_window_s`, maximizing pairs then minimizing |delay|;
   (c) onset = median of the occurrence's first 3 matched words' implied start;
   offset applied; delay computed. Unpaired slide changes are reported as "not heard"
   / "words sung at this point didn't match this slide" (the note names the nearest time
   those words ARE sung and how far off — not "sung near this change"); well-heard occurrences with no slide change are reported as
   "sung but never shown" (near-duplicate overlaps suppressed). Finally, heard words
   not inside ANY slide's occurrence are grouped into stretches (split on >2.5 s pauses)
   and reported as "words heard, no slide matched" (talking, ad-libs, off-slide lyrics)
   if they have >= `other_min_words` words, >= 3 distinct words, and mean Whisper
   word-probability >= `other_min_confidence` (guards against Whisper's phantom text over
   music/silence). Probabilities are saved with the words (4th tuple element; older
   saved songs lack it and default to 1.0 — `--retranscribe` adds them). Speech vs singing
   can't be distinguished by text; the label is deliberately neutral. Thresholds are
   untuned against real interlude talking (none in the Abide test recording).
4b. **Live estimates while the song plays** (`analysis/live.py::LiveAnalyzer`, driven by
   `Session._live_loop` every `live.interval_s`): only the newest audio is transcribed (a few
   seconds of overlap for context, never the whole song again, windows capped at
   `live.max_window_s`); words ending >2 s before "now" are committed (words at the edge may still
   change); the same word-matching analysis re-runs on the committed words, giving each slide change
   a provisional delay. Changes whose singing isn't transcribed yet show as pending ("…"), not
   "not heard" (`PENDING_NOTE`, horizon 8 s). Measured on the user's CPU with Whisper `small` int8
   against the real Abide recording: ~3.7 s compute per ~13 s window (0.28x real time, worst 9.1 s),
   estimate appears ~7 s after the slide change (median), median |live - final| = 0.12 s but only
   72% within 0.5 s — it is an estimate. Live yields to a running final analysis (non-blocking lock)
   and is discarded at Stop. GUI: "Live delays" card + `/api/live/graph.svg`.
4c. **Operators:** the operator's name is typed on the Live tab (sent with Start monitoring; editable while
   monitoring) or `--operator`; it is stamped on each song when it stops (`SongData.operator`) and can be
   reassigned later on the song's page (regenerates its json/txt/svg without re-analysis).
   `analysis/operators.py` pools measured slide changes (excl. not-heard and already-live-at-start) per operator
   at READ time with the current margins; Operators tab + `/api/operators` + `--operators` (also saves
   `reports/operator_report.txt`) + trend graph (`render_operator_trend`, one box per song).
5. **Output:** `reports/session_<date>/song_NN_<name>.{wav,json,txt,svg}` (svg = the
   step graph) + `session_report.txt`. The json contains slides, events, heard words, timings, so
   `python main.py --analyze <song.json> [--offset-ms N] [--retranscribe]`
   re-runs the analysis offline (instant if words are cached).

**Offset (`timing.offset_ms` / `--offset-ms`):** positive = captured audio arrives
LATE vs. the room (NDI etc.) -> vocal times shift earlier -> delays increase by
that amount. Negative = the opposite.

## Module Map

```
main.py                    Entry: --gui, terminal live session, --analyze, --list-devices
app.py                     Controller: owns settings, Pro7 client, lazy Whisper, current Session, library;
                           pushes live state to browsers (shared by terminal + GUI)
settings.py                SettingsStore (defaults <- config.yaml <- user_settings.json), FIELDS schema + validation
session.py                 Session: song start/stop, audio window, slide events, analysis jobs (no Pro7/web ownership)
library.py                 SongLibrary: list/load saved songs, re-analyze, path-safe ids
propresenter/client.py     Pro7 REST client + slide-index polling + arrangement expansion
audio/capture.py           AudioSource, DeviceSource, LoopbackSource, FileSource, resampler
analysis/song_timing.py    Word matching + per-slide delay + summary + margins (pure logic, no I/O)
analysis/song_runner.py    SongData, wav/json/txt/svg save+load, runs Whisper + analysis
analysis/live.py           LiveAnalyzer: windowed transcription + provisional per-slide delays during a song
analysis/operators.py      Per-operator pooled statistics + text report (pure functions)
transcription/whisper_asr.py   faster-whisper wrapper (transcribe_words used)
report/timing_report.py    Text report formatting
report/graph.py            Boxy step-graph SVG (green on time / yellow early / red late + problems lane)
gui/server.py              aiohttp: page + WebSocket state + JSON API (see its docstring)
gui/static/index.html      The GUI: Live / Songs / Settings tabs (vanilla JS, no CDN, safe DOM building)
utils/keys.py, utils/logger.py
tests/                     pytest: timing logic, graph, parsing, session e2e, GUI API over real HTTP (fakes at the edges)
```

**Legacy / dormant (not used by the main flow):** `alignment/engine.py`,
`transcription/vosk_asr.py`, `notification/notifier.py`, `report/generator.py`.
They belong to the earlier "live lag alert" design (vocalist N slides ahead for
N seconds) that was dropped because a threshold alert isn't a per-slide delay
measurement. Kept for reference; safe to delete.

## Key Decisions

- **Polling, not WebSocket, for slide changes** — Pro7's status-subscription format
  is version-dependent and unverified here; polling a tiny endpoint is robust and
  has bounded timestamp error.
- **Match slides to audio by words, not position** — user requirement: YouTube/band
  arrangements differ from Pro7's, and bands skip around live. (Replaced an earlier
  "expected lyrics = slides in order shown" alignment that assumed the operator tracked
  the band.) Known limit: when the operator skips a slide whose exact line is sung
  elsewhere in the song (e.g. "I depend on You" in every verse), the change can pair
  with the wrong occurrence (~2-5% in simulation with heavy ASR noise).
- **Pro7 arrangements:** the live slide index counts cues in the song's *current
  arrangement* (groups repeat), so `client.py` expands `groups` by
  `current_arrangement` (verified against real Pro7 21.4.2: 38 cues vs 26 unique slides).
- **Whisper with lyrics prompt, VAD off** — sung vocals; VAD tends to drop singing.
- **Record + re-analyze** — wav+json per song so tuning never needs a re-run of a service.
- **GUI has no auth** — see the security posture above; localhost by default.
- **One port:** terminal mode and `--gui` both serve on `gui.port` (8765), so only one runs at a
  time; terminal mode continues without the page if the port is busy, `--gui` exits with a message.
- Platform: Windows for testing, macOS for production (loopback is Windows-only;
  on macOS use a BlackHole input device).

## GUI (user-specified requirements — first version built 2026-11-07)

- The GUI exposes **every CLI feature plus the user-configurable parameters**: connect to
  ProPresenter, pick audio (speakers / input+channels / wav file), offset, start/stop
  monitoring and songs, browse saved songs, re-analyze (new offset / re-transcribe), download
  report/graph/data/audio, and a Settings page generated from `settings.FIELDS`.
- **Margin of error** is user-configurable and **asymmetric** (late and early separately;
  `timing.on_time_window_s` sets both, `on_time_late_s` / `on_time_early_s` override). In the Songs
  tab, changing it re-labels rows, summary tiles and the graph live (saved delays don't depend on
  it); "Save as default" persists it.
- **Song graph** (also saved as `song_NN_*.svg`): delay vs song time, **green = on time,
  yellow = early, red = late**, boxy step style (each slide holds its delay until the next change,
  vertical risers with a colour gradient), up = early / down = late (user asked for red below the 0s line, green/yellow above). A "details" lane under the time axis prints
  every slide's delay value (coloured like the line, staggered over rows so they never overlap, thin guide
  line to the exact time) so values and gaps read straight off the timeline; it also marks slides whose
  words weren't heard (✕ in place of a value), lyrics sung but never shown (◆) and words with no slide (▬). In the table,
  NOT HEARD rows are red-highlighted with their reason; chips jump to each problem list.
- **Settings layering:** GUI changes go to `user_settings.json` (next to config.yaml), never
  rewriting the YAML (keeps its comments); "Reset" deletes it. Precedence: CLI flag > GUI/user_settings >
  config.yaml > defaults. The last audio source is remembered there too (`audio.last`).
- **Security posture:** no login. Binds 127.0.0.1 by default (`gui.host`); every POST needs an
  `X-ProDragon: 1` header and the WebSocket checks Origin (guards against hostile web pages
  calling a local service); song ids are path-checked inside the reports folder. If `gui.host` is
  0.0.0.0, anyone on the LAN can control it.
- **Always-on GUI + header pop-outs** (2026-11-09): `Controller(auto=True)` (GUI mode only; terminal mode keeps its explicit
  connect/start) runs `_auto_loop`: it creates the ONE long-lived `ProPresenterClient` (polls whether or not Pro7 is reachable,
  `connected` = answered within ~3 s, `on_connection_changed` → notices, `reconfigure()` when host/port settings change) and starts
  `start_session(audio.last)` by itself (backoff 5→60 s on failure; nothing happens until an `audio.last` exists).
  `set_audio_source(spec)` / `POST /api/audio/source` switches the source in a background thread (refused while recording a song);
  there are no Start/Stop-monitoring buttons any more (`/api/monitoring/*` still exist for tests/terminal). The operator name is NOT
  remembered across launches on purpose (no Start press to confirm it any more). GUI: Setup and Messages moved out of the page into
  header buttons with hover/click pop-out panels (`makePanel` in index.html); header status items (Speech, Audio, GUI, and
  ProPresenter on the Setup button) turn red with ✗ when not ready (`setPill(id, ready, …)`); Live tab is a fixed two-column
  grid (Song | Live delays) with fixed-height lists so nothing re-flows when a song starts.
- **Three accuracy fixes from real recordings** (2026-11-09, Another In The Fire 02:40 take):
  (1) Whisper `no_speech_threshold=None` in `transcribe_words`: Whisper discards a whole 30 s window it calls "no speech" —
  Verse 3 (3:47–4:18) vanished, slides reported "not heard"; with it off the same audio gives the whole verse (403→443 words, no
  junk stretches). Existing songs only benefit after "Re-analyze → also re-transcribe". (2) `WEAK_MATCH_COVERAGE/_MAX_DELAY_S` in
  `analyze_song`: a pairing with <40% of the slide's words matched AND >8 s from the change is a coincidence ("−18.5 s early" from
  shared small words with the next chorus line) → reported "not clearly heard"; on all saved songs it changed only that one slide.
  (3) `LiveAnalyzer.timings(late_margin_s=…)`: while recording, a LATE estimate for a slide change whose transcript hasn't yet run
  `pending_horizon_s` past it is shown as pending: the slide's own words weren't transcribed yet, so it latched onto the previous
  line (+7…12 s false late), which "sorted itself out" later. Replay harness idea: truncate the cached words at T−2 s and
  events at T, step T — reproduces the live view offline (7 wrong newest-slide estimates → 0 on that song).
- **Loopback "default speakers" can be unresolvable at launch** (2026-11-09): twice on the user's PC Windows/PortAudio reported the
  default output as an INPUT endpoint ('Microphone (2- USB Audio CODEC )') with the USB codec's speakers and a monitor missing from
  the loopback list (a USB device appearing/disappearing), while a copy launched a minute later resolved fine. `LoopbackSource` now
  matches by the bracketed device name (`_pick_loopback`), re-enumerates up to 3× (0.7 s apart), then falls back to the best
  available loopback (`_guess_loopback`: a "Speakers…" device, else the first) and says so in `describe()` and a log warning,
  instead of failing auto-start. Not verified against the real flapping state — only simulated.
- **Slide list edited mid-song** (2026-11-09; found when the user deleted a duplicate blank "Intro" while a song was recording, which
  shifted every later index by one and made 8 on-time slides look 7 s late). Fix: each `SlideEvent` now carries a snapshot
  (`text`, `group`) of the slide as it was when it went live (older songs have None → looked up by index in `slides`);
  `ProPresenterClient._check_slide_list` compares Pro7's `total_cues` (from /v1/presentation/slide_index) with the loaded slide
  count each poll and, on a change, re-fetches the presentation once (`on_slides_updated`; `on_slides_mismatch` if it still disagrees,
  warned once); `Session._on_slides_updated` swaps `rec.presentation`, re-snapshots events since the edit and records a note
  (`SongData.notes`, shown in the GUI + .txt report). Known limits: the slide live at the instant of the edit keeps its index so no event
  is created for its new meaning; count-preserving edits are invisible to the cue-count check. Repair for old songs:
  `SongLibrary.remap_slides` / `Controller.remap_song_slides` / `POST /api/songs/remap` ("Fix slide list…" button) re-reads all events
  through ProPresenter's current list (needs it connected, same presentation uuid, matching cue count); keeps the old list in
  `slides_before_remap`.
- **Sections strip on the graph** (2026-11-09): `report/graph.py` draws a strip under the time axis with one block per run of
  consecutive slides sharing a Pro7 group name (`SlideTiming.group`), tinted by that run's mean delay (measured, non-initial slides),
  labelled "name  ±x.xs" when it fits, else the name, else truncated; dashed boundary lines run up through the chart; tooltip has the
  average. Only drawn when any slide has a group (adds 26 px of height; the chart itself keeps its size). The GUI renders graphs
  on demand from the song json, so old songs get it immediately; saved `.svg` files update on the next re-analysis.
- **Deleting songs** (2026-11-09, for clearing test recordings): `SongLibrary.delete(ids)` (permanent; removes .json/.txt/.svg/.wav,
  path-checked inside the reports folder; a session folder with no songs left is removed with its session_report.txt),
  `Controller.delete_songs` (refuses a song being re-analyzed; clears a stale `last_song_id`), `POST /api/songs/delete {ids}`.
  GUI: Songs tab "Select…" mode (+ "Select short / empty": <30 s or nothing measured) and a "Delete song" button on the detail card.
- **Equalizer instead of a level bar** (2026-11-09): `audio/capture.py::band_levels_db` (24 log bands 60 Hz–8 kHz of the newest
  2048 samples of the 16 kHz mono signal, 0 dB = full-scale sine; computed lazily by `AudioSource.get_bands_db`, silent after 0.6 s
  without audio). `Controller._spectrum_loop` sends `{"_spectrum": {db, edges}}` ~12×/s via `DashboardServer.send_transient`
  (not merged into the stored state; the client draws it on a canvas with its own decay/peak caps). Overall `level_db` still
  rides in the normal state push.
- Added after the first GUI version (2026-11-07): live in-song delay view, per-operator reports, NDI audio source
  (see How It Works 3, 4b, 4c). The user explicitly decided against a GUI login for now.
- Not done yet: GUI auth (declined for now), editing `gui.host/port` from the page (needs restart), audio device
  hot-plug detection, NDI tested only against a local test sender.
- Process hygiene (lesson): when starting helper servers for testing, record the exact PID/command line and stop
  only that — never "whatever is on the port" (that once killed a user's own `main.py --loopback` session).

## Known Limits / Unverified

- Automated tests use fakes (fake Pro7 server, scripted Whisper). A real run against
  Pro7 21.4.2 + Whisper `small` + speaker loopback has been done by the user (2026-11-07)
  and worked; `/version` is at the server root (no /v1).
- Whisper word timestamps on sung music are approximate (expect roughly +-0.3-0.5 s);
  fine for seconds-scale slowness, not for 100 ms precision.
- Pro7 slide indexing of *disabled* slides is assumed to count them.
- Audio timestamps use the driver's capture time; clock drift over a song is ignored.
- Only 16-bit PCM wav for `FileSource`.

## Not Built Yet

- Per-operator tracking, auto song detection, near-real-time delays, NDI source,
  Planning Center lyrics, dashboard auth.
