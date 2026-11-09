# Pro Dragon — Configuration Reference

All settings live in `config.yaml` (see comments there). Settings changed in the GUI are saved to
`user_settings.json` next to it and layered on top (precedence: command-line flag > GUI settings >
`config.yaml` > built-in defaults); delete that file, or press "Reset" in Settings, to go back to `config.yaml`.
Summary:

## propresenter
| Key | Meaning |
|-----|---------|
| `host`, `port` | ProPresenter machine and Network API port (default 1025) |
| `password` | only if you set one |
| `poll_interval` | seconds between slide checks (default 0.1). Slide timestamps are accurate to about half this. |

## audio
Choose one source (CLI flags `--loopback`, `--audio-file` override these):
| Key | Meaning |
|-----|---------|
| `device` | input device name substring or number (mic / line-in / Dante). `null` = pick from a menu at startup. With `loopback: true` it can name which output to capture. |
| `channels` | 1-based channel(s) of an input device; several are averaged to mono |
| `loopback` | capture the computer's speakers (Windows only, needs PyAudioWPatch) |
| `file` | 16-bit PCM `.wav` played as if live |
| `ndi` | name (or part of it) of an NDI source to take audio from; needs `requirements-ndi.txt`. All channels are mixed to mono. NDI adds delay (often 100–300 ms): set `timing.offset_ms` to match. |
| `ndi_address` | optional static (unicast) address of the NDI sender: IP or host name, optional `:port`, several separated by commas (`--ndi-address`, GUI Live tab "NDI address", Settings → NDI audio). Use it when the source is on another subnet/VLAN where automatic (mDNS) discovery can't see it. Not a multicast address: NDI receivers don't choose one — a multicast sender is joined automatically once found (if it's on another subnet, give its IP here). |

Sources run at the device's native sample rate and are resampled to 16 kHz internally.

## timing
| Key | Meaning |
|-----|---------|
| `offset_ms` | fixed audio-vs-slide offset, either direction. **Positive = captured audio arrives late** (NDI/network), so vocal times are moved earlier (delays increase). **Negative** = audio arrives early. Override with `--offset-ms`. |
| `on_time_window_s` | margin of error: a slide change within ± this of the vocal start = ON TIME (default 0.75) |
| `on_time_late_s` / `on_time_early_s` | optional: set the late and early margins separately (they override `on_time_window_s`). Early is the ideal for congregational singing, so you might allow 0.5 s late but 2 s early. The GUI's Settings page and Songs tab edit these. |
| `min_match_confidence` | fraction of a slide's words that must be heard for it to be measured (default 0.25) |
| `max_match_window_s` | a slide change is paired only with sung words of *its own text* within this many seconds (default 20). Matching is by words, never by slide position, so a different arrangement or skipped sections are fine. |
| `other_min_words` / `other_min_confidence` | a stretch of heard words that no slide explains is listed as "words heard, no slide matched" only if it has at least this many words (default 4, with 3+ distinct) and at least this mean recognizer confidence (default 0.5). Raise them if Whisper's invented words show up; lower them to catch short remarks. |
| `use_lyrics_prompt` | give Whisper the lyrics as a hint (recommended) |
| `vad_filter` | Whisper's speech filter; keep off for singing |

## whisper
`model` (tiny/base/small/medium/large-v2/large-v3), `language`, `device` (cpu/cuda), `compute_type` (int8 CPU, float16 GPU).
Transcription runs once per song after you press Stop. Bigger models are slower but hear sung words better.

## live
| Key | Meaning |
|-----|---------|
| `enabled` | show provisional delay estimates while a song is recording (default on). They use the speech model repeatedly — turn off on a slow computer. |
| `interval_s` | pause between live updates, on top of the time each takes (default 6) |
| `max_window_s` | longest stretch of audio transcribed in one go (default 40) |

## operators
No settings: the operator's name is entered on the Live tab (or `--operator`) and stored with each song. The last name used is
remembered in `user_settings.json` and pre-filled next time (it is only used when you press Start monitoring).

## report
`output_dir` — each run creates `session_<date>/` containing, per song, `song_NN_<name>.txt|json|wav`, plus `session_report.txt` at exit.

## gui
`enabled`, `host` (default `127.0.0.1` = this computer only; `0.0.0.0` = also the LAN), `port` (default 8765),
`open_browser` (open the page automatically with `--gui`).
No login: with `host: 0.0.0.0` anyone on the network can start/stop songs and change settings, so only do that
on a trusted network. The default changed from 0.0.0.0 to 127.0.0.1 when the GUI gained settings control.
