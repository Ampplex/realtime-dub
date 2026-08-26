# realtime-dub

Korean video → live English dub, in the browser. Upload a clip, hit play, and switch
between the original audio and a dubbed track while it is still being generated.

The interesting problem here is not "call an ASR API and a TTS API". It is keeping a
generated audio track **in sync with a video that is already playing**, on a pipeline
that is still producing that audio a few seconds ahead of the playhead.

```
video  ──────────────────────────────────────────────▶ playhead
dub    ████████████████░░░░░░                          ▲
       already voiced   being voiced now          ~4s buffer
```

## How it performs

Measured on an M1/8GB against 40s of dialogue-dense video:

| | value |
|---|---|
| time to first dubbed audio | **~5s** (warm) |
| sustained throughput | **1.1–1.25× realtime** |
| segments recovered from 40s | 20 |
| Korean leaking into the English track | 0 |

Time-to-first-audio is bounded by the **window size, not the video length** — a
3-minute video starts as fast as a 40-second one. Because throughput exceeds
realtime, the buffer grows while you watch rather than draining.

## Why look-ahead, not "realtime"

The whole file is on disk, so this is **look-ahead dubbing**. The pipeline runs *ahead*
of the playhead and the buffer is permanent headroom that workers keep refilling. That
is what makes the sync problem tractable at all.

## Stack

| Stage | Choice | Why |
|---|---|---|
| Separation | demucs `htdemucs` | removes the original dialogue so it is a dub, not a voice-over |
| ASR | faster-whisper `small`, int8 | CTranslate2, no torch; streams segments as it decodes |
| Translate | Amazon Bedrock (`mistral-large-3`) | ~0.6s/line, parallelised |
| TTS | Amazon Polly (generative) | natural, network-bound, so it leaves CPU for demucs |

ASR runs **once** and fans out to every target language, so a second language costs a
translate+TTS pass, not a second transcription.

## How sync works

The video clock is the source of truth; audio bends to it. Each segment is scheduled at

```
anchor.ctxTime + (segment.start − anchor.videoTime)
```

so it is pinned to its position on the **source** timeline. Drift cannot accumulate,
because segments are scheduled against the anchor independently rather than relative to
each other. Pause, seek, rate change and language switch all invalidate the anchor.

Duration fitting is two-stage: the backend's speaking-rate prior picks an initial
length, then an ffmpeg `atempo` pass trims the result to its slot. Stretch is clamped —
past the clamp speech stops sounding human, so overrun is absorbed by the silence that
follows instead.

**Lines may end early, by design.** A line is sped up to fit its slot but never slowed to
fill one. Stretching a short translation to pad the slot is what made the voice drawl.

## Script-leak guard

Untranslated Korean must never reach an English voice. Every translation is checked for
Hangul/CJK/kana; on a hit it retries once with a stricter prompt, then sanitises, and if
it still cannot produce clean text it returns **empty and the line is skipped** — the
background keeps playing. Critically it never falls back to the source line, which was a
real leak path in an earlier version.

Verified end to end by transcribing the *rendered output* back: 0 Hangul characters in
the English track, and 0 Korean speech in the separated background.

## Architecture

[**ARCHITECTURE.md**](ARCHITECTURE.md) covers the full technical design: the sync model
and why drift cannot accumulate, the concurrency layout, every pipeline stage, and the
measurements behind each decision.

## Setup

```sh
python3 -m venv .venv
./.venv/bin/pip install -r requirements-full.txt   # includes demucs/torch
./fetch-voices                                     # Piper voices (local TTS fallback)
cp .env.example .env                               # add AWS credentials
brew install ffmpeg
```

`.env` needs `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` with access to Bedrock
(translation) and Polly (voices).

## Run

```sh
./run                                  # http://127.0.0.1:8500
./run --model base                     # faster ASR, less accurate on proper nouns
./run --no-separate                    # skip demucs: much lighter, but the Korean
                                       # stays audible under the dub
./run --tts piper                      # fully local voices, no AWS
```

Then drop a Korean clip on the page.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TTS_BACKEND` | `polly` | `polly` / `piper` / `kokoro` / `say` |
| `POLLY_ENGINE` | `generative` | `neural` restores SSML duration control |
| `POLLY_FALLBACK` | `kokoro` | `none` avoids pulling torch in for one voice |
| `ASR_BACKEND` | `whisper` | `transcribe` uses Amazon Transcribe (no local model) |
| `WHISPER_MODEL` | `small` | `tiny` fits small instances but segments badly |
| `SEPARATE` | `1` | `0` ducks the original instead of removing it |
| `ALLOW_LOCAL_PATH` | `0` | dev only — see below |

## What the measurements taught us

Most of the design here came from being wrong first:

- **Whisper `base` beat `small` on segmentation but loses proper nouns.** `small` reads
  `서울대 로스쿨 졸업생 강휴민입니다` correctly; `base` garbles the name. `tiny` collapses
  40s into 2 segments, which destroys sync entirely.
- **An unbounded TTS voice cache cost ~250MB.** Casting both genders in two languages
  touches four ONNX voices and every one stayed resident. Bounding it took peak RSS from
  538MB to 293MB.
- **Polly's Kokoro fallback cost ~800MB** — torch, loaded at startup to cover a single
  missing voice, whether or not that voice was ever used.
- **Running stages in sequence made throughput the SUM of every stage** (0.89×);
  overlapping them makes it the slowest single stage (1.23×).
- **Memory, not CPU, is the binding constraint.** demucs + Whisper + voices is ~1.6GB;
  below that the machine swaps and everything looks "slow" for reasons no profiler
  attributes to the code.

## Security note

The `path=` parameter lets the server open an arbitrary local file and serves it back
via `/api/session/<id>/video` — arbitrary file disclosure on a public host. It is off
unless `ALLOW_LOCAL_PATH=1`, which belongs on a development machine and nowhere else.
Hosted instances accept uploads only.

## Deploying

Runs from the `Dockerfile`. **Not deployable to serverless** (Vercel and friends): torch
alone exceeds a 250MB function limit, ffmpeg is shelled out to, and sessions are
in-process state with background threads that outlive the response.

Sizing is about memory:

| config | peak RSS |
|---|---|
| whisper `small` + demucs + Polly | ~1.6GB |
| whisper `tiny` + Piper, no separation | 293MB |
| `ASR_BACKEND=transcribe` + Polly | no resident models |

Run exactly **one** worker (see `wsgi.py`): sessions live in a module-level dict, so a
second worker would answer polls for sessions it has never seen. Concurrency comes from
threads.

## Known limits

- **No speaker diarisation** — one voice per gender, regardless of who is talking.
- **Gender casting is F0-based** (boundary 158 Hz) and must run on separated vocals;
  on a mix, music energy sits in the same band and makes it a coin flip.
- **Translation occasionally emits meta-text** (`"Sure, please provide the subtitle
  lines."`) for very short inputs like `네`. The script-leak guard does not catch this,
  because the output is valid English.
- **Hindi is disabled.** The code still maps Hindi voices, but Polly exposes no `hi-IN`
  voice, so Hindi male fell through to a torch-backed local model for one voice.
