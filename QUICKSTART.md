# Pro Dragon — Quick Start

Pro Dragon measures how early or late each ProPresenter slide change is compared
with when the vocalist starts singing that slide, and prints a per-slide report.
It only observes — it never controls ProPresenter.

## Setup

1. Python 3.10+ . Run `install.bat` (Windows) or `./install.sh` (macOS) to create `venv/` and install packages.
   If `install.bat` fails because `python` isn't found, use: `py -3.11 -m venv venv` then
   `venv\Scripts\pip install -r requirements.txt`.
2. ProPresenter 7: Preferences → Network → **Enable Network**, note the port (default 1025).
3. Edit `config.yaml` → `propresenter.host` (`127.0.0.1` if ProPresenter is on this computer).
4. The first run downloads the Whisper model (`small`, ~250 MB).

## The GUI (recommended)

```
venv\Scripts\python main.py --gui
```
Opens `http://localhost:8765` (it only listens on this computer unless you change `gui.host`).

- **Always on.** Pro Dragon connects to ProPresenter by itself, keeps retrying if it isn't running yet or the link drops,
  and starts listening to the last audio source you used — there is no Connect or Start monitoring button. The top bar shows
  what's ready: **Speech**, **Audio**, **GUI** and the **ProPresenter** status on the **Setup** button; anything not ready is
  red with an ✗. (First run: choose an audio source under Setup once.)
- **Setup** and **Messages** (buttons at the top right) pop out when you hover them; click to keep one open (click again,
  click elsewhere or press Esc to close). Setup holds the ProPresenter address, the audio source (changing it switches
  right away, but not while a song is recording), NDI address, audio offset and the operator. Messages shows what's been
  happening (a number on the button counts new ones).
- **Live** — press **Start song / Stop song** for each song. The slide on screen, the changes so far, the equalizer and the live
  delays sit in fixed places, so nothing jumps around when a song starts.
  The card under the slide list is a live **equalizer** (24 bands, 60 Hz – 8 kHz, with peak caps) plus the overall level,
  so you can see at a glance that audio is arriving and whether the vocals are there. It shows what Pro Dragon actually
  hears — the 16 kHz mono signal fed to speech recognition — not the full-range mix.
- **Songs** — every saved song with its **graph** (green on time · yellow early · red late, boxy step style), summary,
  per-slide table, and clearly marked problems: ✕ slides whose words weren't heard, ◆ lyrics sung but slide never shown,
  ▬ words heard that match no slide (e.g. talking). Change the **margin of error** (late/early separately) and the
  labels, percentages and graph update instantly; **Re-analyze** with a different audio offset; download the report,
  graph, data or audio.
  A **sections** strip under the graph's time axis names each part of the song (Verse 1, Chorus, Bridge… taken from
  ProPresenter's slide groups) and tints it by that section's average delay — green on time, yellow early, red late — so you can
  see at a glance which parts are the problem (hover a section for its average). Dashed lines mark where sections start.
  **If you edit a presentation while a song is recording** (add/delete a slide or group), every later slide number in ProPresenter
  points at a different slide. Pro Dragon notices (the live slide count changes), shows a warning, reloads the list, and
  remembers what each slide was when it went live, so the report stays correct; a ⚠ note on the song says when it happened.
  (The slide that was on screen at the moment of the edit may not be measured, and edits that keep the slide count the same
  can't be detected.) For a song recorded before this existed, open it and press **Fix slide list…**: with ProPresenter
  connected and that song live, it re-reads every slide change using the current list (the original list is kept in the file).
  **Clean up test recordings:** press **Select…** above the list, tick songs (or **Select short / empty** to tick everything
  under 30 s or with no measured slide changes), then **Delete selected**; or open a song and press **Delete song**. Deleting is
  permanent (it asks first) and removes the song's audio, report, graph and data. A session folder left with no songs is removed too.
- **Settings** — every configurable parameter, with explanations. Saved to `user_settings.json` (your `config.yaml` is
  never rewritten; "Reset" goes back to it).

- **Operator** — type the name of the person running the slides under **Setup** (or `--operator Sam` in terminal mode). It
  applies right away and is saved with each song; it can be corrected later on the song's page. It isn't remembered between
  launches (so a new operator is never credited by mistake) — pick or type the name each time.
- **Live delays** — while a song is recording, the Live delays card shows each slide change's *estimated* delay a few
  seconds after it happens, with a live graph. They're estimates (they can still shift a little); the final report
  after Stop replaces them. If your computer is slow, turn this off in Settings → Live view.
- **Operators tab** — compares everyone: songs, slide changes, average delay, on time / late / early %, worst cases,
  and a per-person trend graph (one box per song). Click a person for details; the margin of error can be changed there.
  Terminal: `python main.py --operators` prints the same and saves `reports/operator_report.txt`.

The terminal mode below still works and uses the same engine. Only one of them can run at a time (they share port 8765).

## Test with a YouTube video (speakers)

1. In ProPresenter, put the song's slides **live** (click the first slide) — lyrics should match the video.
2. Play the YouTube video through the computer's speakers.
3. Run:
   ```
   venv\Scripts\python main.py --loopback
   ```
4. When the video's song starts, press **S** (or click **Start song** at the dashboard URL printed at startup).
5. Click through the slides in ProPresenter as the song plays — be early, on time and late on purpose.
6. When the song ends press **S** again. After a short analysis the report prints and is saved.

Press **Q** (or Ctrl+C) to quit.

## Audio sources

| Option | Use |
|--------|-----|
| `--loopback` | what's playing out of the computer's speakers (Windows, testing) |
| (no flag) | menu at startup: pick a microphone / line-in / Dante Virtual Soundcard and channel(s) |
| `--audio-file x.wav` | play a 16-bit .wav as if live (repeatable tests) |
| `--list-devices` | show everything available |

## NDI audio

Install once: `venv\Scripts\pip install -r requirements-ndi.txt` (the NDI runtime is included). Then NDI sources on your
network appear in the GUI's audio list under **NDI** (↻ looks again), or use `--ndi "MIXER (Audio)"` in terminal mode
(`python main.py --list-devices` lists them). NDI adds delay between the room and what you capture — set the audio offset
(e.g. `+150`) to match; if you're not sure, record a song and compare results at a few offsets with **Re-analyze**.

**Source on another subnet / VLAN?** NDI's automatic discovery only sees the local network. Enter the sender's IP address
in the **NDI address** box on the Live tab (then **Save & look**), or use `--ndi-address 192.168.20.15` (add `:port` if
it isn't the default; several addresses can be comma-separated). Multicast senders need nothing extra — the stream joins
the multicast group by itself once the source is found — but across subnets you still need the address, and the network
must pass multicast (IGMP snooping needs a querier).

## Audio offset (NDI etc.)

If the captured audio is delayed relative to what the room sees/hears, tell the tool:

```
python main.py --offset-ms 200      # audio arrives 200 ms LATE  (positive)
python main.py --offset-ms -100     # audio arrives 100 ms EARLY (negative)
```
Or set `timing.offset_ms` in `config.yaml`. You can also change it afterwards when re-analyzing a saved song.

## Reading the report

```
    #      At  Section      Delay  Verdict  Words on slide
    2  0:02.5  Verse 1      +1.0s  LATE     Holy holy holy Lord God almighty
    3  0:03.9  Verse 1      -0.7s  ON TIME  Early in the morning our song shall rise to thee
```
- **At** — when into the song the slide changed. **Delay** — `+` = slide changed after singing started (late), `−` = before (early).
- **ON TIME** = within ±`timing.on_time_window_s` (default 0.75 s).
- Each slide is matched to where **its words** are sung — not to its position — so a band that sings a different
  arrangement than ProPresenter, skips sections, or repeats the chorus is fine.
- "not heard in the audio" = those words weren't sung/recognized (band skipped the section, too quiet, wrong song).
- "the words sung at this point didn't match this slide" = something else was being sung when this slide appeared (the band
  moved on or skipped around). The note says where in the song this slide's words *are* sung (nearest time, and how far
  before/after the slide change), which helps tell "band skipped it" from "operator was far too late".
- **NOT SHOWN** (list at the end) = lyrics from one of your slides that the band sang but that never had a matching slide change.
- **NO SLIDE** (list at the end) = words heard that aren't on any slide — talking during an interlude, an ad-lib, or lyrics
  that weren't put on a slide. Only stretches of several confidently-recognized words are listed (so Whisper's phantom text over
  music isn't). Tune with `timing.other_min_words` / `timing.other_min_confidence`.
- A slide already live when you pressed Start shows `≤` (it was on screen at least that early).

Files go to `reports/session_<date>/`: per song a `.txt` report, `.json` data and `.wav` audio.

## Re-analyze a saved song

```
python main.py --analyze reports/session_2026-.../song_01_xxx.json --offset-ms 150
python main.py --analyze ... --retranscribe      # run Whisper again (e.g. after changing whisper.model)
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Cannot connect to ProPresenter" | Check host/port, that Network is enabled, and the firewall. |
| "No live presentation found" | Click a slide of the song in ProPresenter so it's live, then press S. |
| Few/no slides measured | Lyrics in ProPresenter must match what's sung; try `whisper.model: medium`; check audio level in the status line. |
| Slide numbers look off / no text | Run with `--debug` and compare the raw JSON with `propresenter/client.py`. |
| NDI source listed but the level says "silent" | The sender is connected but sending zeros. ProPresenter: Preferences → Audio → **SDI & NDI → Enable** (off by default) and route channels to it. Remember ProPresenter's NDI only carries audio ProPresenter plays, not the room vocals. |
| Loopback error | `pip install PyAudioWPatch` (Windows). On macOS use a BlackHole input device. |

Run tests: `venv\Scripts\python -m pytest tests`
