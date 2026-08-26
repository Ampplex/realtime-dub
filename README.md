---
title: Realtime Dub
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8500
short_description: Korean video to live Hindi or English dub
---

# realtime-dub

Korean video → live Hindi or English dub, in the browser.

Plays the original for the first few seconds while the pipeline builds a buffer;
switching language restarts playback with the dubbed track and keeps rendering
chunks ahead of the playhead.

## How it performs

Measured on an M1 / 8GB, 40s of real dialogue-dense video, both languages at once:

| | value |
|---|---|
| time to first dubbed audio | **5.5s** (warm) |
| sustained throughput | **1.20x realtime** (sustains, does not stall) |
| model warm-up | 3.7s, once at boot |

Time-to-first-audio is bounded by the **window size, not the video length** - a
3-minute video starts just as fast as a 40-second one. Because throughput exceeds
realtime, the buffer grows while you watch instead of draining, so the default
buffer is only 4s.

## Why this is tractable

The whole file is on disk, so this is **look-ahead dubbing**, not live dubbing. The
pipeline runs *ahead* of the playhead and the buffer is permanent headroom that
workers keep refilling. Measured end to end on an M1/8GB: **1.34× realtime for both
languages at once**.

## Stack (measured, not guessed)

| Stage | Choice | Throughput |
|---|---|---|
| STT | faster-whisper `base`, int8 (no torch) | 7.6× realtime |
| Translate | Bedrock `amazon.nova-lite-v1:0` | 0.6 s/line, ~14× with fan-out |
| TTS | Piper ONNX (`hi_IN-pratham`, `en_US-lessac`) | 10–29× realtime |

Whisper `base` beat `small`: faster **and** better segmented. `small` merged three
sentences into one 5.5 s blob, which makes sync coarse. Sentence-level segments let
each line land on its own timestamp.

ASR runs **once** and fans out to both languages, so Hindi alongside English costs a
translate+TTS pass, not a second transcription.

## Setup

```sh
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./fetch-voices                 # Piper voices, ~124 MB
cp .env.example .env           # add AWS creds
```

## Run

```sh
./run                          # http://127.0.0.1:8500
./run --model small            # more accurate ASR, coarser segments
./run --buffer 5               # seconds to build before playback (2-10)
./run --translator ollama      # offline fallback (degraded, see below)
./run --tts say                # macOS dev fallback
```

Paste a path or drag a video onto the page. Pick **Korean / हिन्दी / English**.

## How sync works

The video clock is the source of truth; audio bends to it. Each segment is scheduled at

```
anchor.ctxTime + (segment.start - anchor.videoTime)
```

so it is pinned to its position on the **source** timeline. Drift cannot accumulate,
because segments are scheduled against the anchor independently rather than relative
to each other. Pause, seek, rate change and language switch invalidate the anchor.

Duration fitting is two-stage: Piper's `length_scale` (drives the duration predictor,
so formants stay intact) with an ffmpeg `atempo` trim after. Stretch is clamped to
0.70–1.45 — past that speech stops sounding human, so overrun is absorbed by the
silence that follows instead.

## Known limits

- **Translation is the bottleneck** at ~0.6 s/line. A long video with dense dialogue
  will build buffer more slowly than a sparse one.
- **Lines may end early, by design.** A line is sped up to fit its slot but never
  slowed to fill one. Stretching short translations to pad the slot is what made the
  voice drawl; a short line simply finishes early and the bed carries the gap.
- **No speaker diarisation.** One voice for everyone, regardless of who is talking.
- **Ducking is the fallback, not the default.** With `--no-separate` The source is no longer muted: it plays as a bed under
  the dub and dips while each line speaks, so music and effects survive. The original
  Korean dialogue stays faintly audible underneath (voice-over / lektor style).
  Removing it entirely needs stem separation - see below.
- **Local fallback is degraded.** `llama3.2:1b` failed badly on ko→hi in testing —
  emitted Thai script, leaked Korean, hallucinated refusals. Bedrock is the real path.
- **`nova-micro` is not enabled** on this account; `mistral-large-2407` throttles hard
  on Hindi (8.3 s/line, half the calls failed). `nova-lite` was the smallest that worked.

## Script-leak guard

Untranslated Korean must never reach a Hindi or English voice. Every translation is
checked for Hangul/CJK/kana (and Devanagari in English output); on a hit it retries
once with a stricter prompt, then sanitises, and if it still cannot produce clean
text it returns **empty and the line is skipped** - the bed keeps playing. Critically
it never falls back to the source line, which was a real leak path in an earlier
version. Verified end to end: whisper hears 0 Hangul characters in either render.

`mistral-large-3` also obeys the write-numbers-as-words rule that nova-lite ignored,
which removes digit mispronunciation ("3:30" -> "तीन बजकर तीस मिनट").

## TTS backends

`--tts polly` (default) is the most natural and the only combination that both sounds
good and sustains above realtime. Polly is network-bound, so it parallelises and
leaves the CPU free for demucs - which is what makes the stage overlap pay off.

| backend | throughput | notes |
|---|---|---|
| `polly` | 1.23x | natural; needs AWS creds; **no Hindi male voice** |
| `kokoro` | 0.73x | natural, fully local, but stalls on long video |
| `piper` | 1.38x | fastest, fully local, robotic |
| `say` | - | macOS dev only |

Polly has no Hindi male voice (Aditi/Kajal/Raveena are all female, and are filed
under `en-IN`, not `hi-IN`), so `(hi, male)` falls back to Kokoro's `hm_omega`
automatically and gender casting still works.

## Pipeline shape

Everything is windowed. Each window is separated, transcribed, translated and voiced
before the next is touched, so audio becomes playable in window-time:

```
window N: demucs ─┬─> bed  (music + effects, full level)
                  └─> vocals ──> whisper ──> gender (F0) ──> translate ──> TTS
```

Separation runs on its own thread, producing windows into a bounded queue while the
main loop consumes them. Running the stages in sequence made throughput the SUM of
every stage cost (0.89x); overlapping makes it the slowest single stage (1.23x).

An earlier version ran separation as a blocking pre-pass over the whole file. That
worked but destroyed streaming: nothing was playable until the last frame was
separated (~190s for a 40s clip). Windowing restored it.

**Voice casting.** Median F0 over the voiced frames of each segment picks a male or
female voice (boundary 158 Hz, which sits in the empty gap between the two clusters
on real material). This must run on the **separated vocals**, not the mix - music
energy sits in the same F0 band and turns the result into a coin flip.

**Duration fitting is two-pass.** Guess a `length_scale` from the backend's measured
speaking rate, synthesise, then re-synthesise only if the result is outside
0.82-1.22x of its slot. Below that the time stretcher is transparent and a second
synthesis is wasted compute. Measured rates (chars/sec at scale 1.0):

| | Hindi | English |
|---|---|---|
| Kokoro | 8.93 | 14.06 |
| Piper | 11.47 | 18.56 |

Using one prior for both backends is what previously crushed 7 of 16 Hindi lines
against the compressor limit; per-backend rates plus the corrective pass took that
to 0-1.

## Removing the original dialogue (optional, not installed)

Ducking keeps music and effects but leaves the source voice audible underneath. To
strip dialogue properly you need stem separation:

```sh
pip install demucs          # ~2 GB with torch
```

Run it as a **pre-pass, not in the realtime loop**: `htdemucs` runs ~0.3-1x realtime on
an M1 CPU, so it is slower than playback and would break the look-ahead. Separate once,
cache the `no_vocals` stem, then use that as the bed. Costs an upfront wait per video;
worth it on a GPU box.

## Deploying

Deployed from the `Dockerfile`; the image carries ffmpeg and bakes the Whisper,
demucs and Piper weights so the first request does not wait on a download.

**Not deployable to Vercel** (or any serverless platform), for four independent
reasons: torch alone is ~500MB against a 250MB function limit; ffmpeg is shelled
out to 9 times and is not in the runtime; sessions live in process memory with
background threads that keep working for minutes after the response is sent; and
request bodies are capped at 4.5MB, well under a real video upload. It needs a
container with a persistent process.

```sh
docker build -t realtime-dub .
docker run -p 8500:8500 --env-file .env realtime-dub
```

On Render, `render.yaml` is a blueprint — point **New > Blueprint** at this repo
and set `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in the dashboard. Any other
container host (Railway, Fly.io, Cloud Run) works the same way from the Dockerfile.

**Memory, measured on the 40s clip** (peak RSS, lean build):

| config | peak | verdict |
|---|---|---|
| whisper base, unbounded voice cache | 538MB | OOMs a 512MB instance |
| whisper base, `PIPER_VOICE_CACHE=1` | 847MB | worse — base's decode arena, not its weights |
| whisper tiny, `PIPER_VOICE_CACHE=1` | 293MB | fits 512MB |

The Piper voice cache is the lever that matters: casting both genders in both
languages touches four ONNX voices and an unbounded cache kept every one resident.
`tiny` costs real Korean accuracy, so raise it to `base` the moment you have 2GB.

**On the full pipeline,** not the lean one: demucs, Whisper and Kokoro are all
resident at once, so a 512MB instance cannot start. 2GB is the floor, 4GB is
comfortable — which rules out the usual free tiers.

Run exactly **one** worker (`wsgi.py` explains why): sessions are a module-level
dict, so a second worker would answer polls for sessions it has never seen.
Concurrency comes from threads instead.

### Uploads and the local-path route

A hosted instance accepts **uploads only**. The `path=` route that the CLI-ish
local flow uses reads an arbitrary server path and hands the bytes straight back
via `/api/session/<id>/video`, so on a public host it is arbitrary file
disclosure — `path=/proc/self/environ` would surrender the AWS keys. It is off
unless `ALLOW_LOCAL_PATH=1`, which belongs on your laptop and nowhere else.

## Notes on the original deployment sketch

Piper and faster-whisper are both CPU-only and Linux-friendly, so a container works.
Three things to change:

1. `.env` — inject AWS creds as real secrets, never a file in the image.
2. Bake voices + the Whisper model into the image so the first request isn't a download.
3. Flask's dev server is single-process; put gunicorn in front and give sessions a
   real store (they are in-memory here and die with the process).
