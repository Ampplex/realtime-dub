# realtime-dub — Technical Architecture

How a Korean video becomes an English dub that stays in sync with a video already
playing, while the audio for it is still being generated.

---

## 1. The core problem

Batch dubbing is easy: transcribe everything, translate everything, synthesise
everything, mux, done. The user waits for the whole file.

This system plays the video *immediately* and generates the dub **ahead of the
playhead**. That creates two problems batch dubbing never has:

1. **Sync** — generated speech must land at the same instant as the original line,
   with no cumulative drift, on a clock the browser controls, not the server.
2. **Throughput** — the pipeline must beat realtime, or the buffer drains and playback
   stalls.

Everything below is downstream of those two constraints.

```
source video ─▶ [separate] ─▶ [ASR] ─▶ [gender] ─▶ [translate] ─▶ [TTS] ─▶ [fit] ─▶ chunk
                    │                                                                │
                    └────────────▶ background bed ──────────────────────────────────┐│
                                                                                    ▼▼
                                                            browser: Web Audio scheduler
```

---

## 2. Why "look-ahead", not "realtime"

The whole file is on disk before we start. That means this is not live dubbing — it is
**look-ahead** dubbing. The pipeline runs ahead of the playhead and the buffer is
permanent headroom that workers keep refilling.

This distinction is what makes the sync problem solvable. Every segment knows its exact
position on the source timeline *before* it is scheduled, so audio can be pinned to
absolute positions instead of being chased in real time.

Default buffer is **4 seconds**. That is small because throughput exceeds realtime — the
buffer grows during playback rather than draining.

---

## 3. Stage-by-stage

### 3.1 Separation — `dubber/separate.py`

**demucs `htdemucs`** splits the audio into stems; we keep everything *except* vocals.

```python
bed = sources.sum(dim=0) - sources[vocal_idx]
```

This is the difference between a **dub** and a **voice-over**. Ducking the original
leaves the Korean audible underneath (lektor style). Removing the vocal stem means the
music and effects bed plays at full level with nothing underneath.

**Windowed, not whole-file.** An earlier version separated the entire file first — which
worked but destroyed streaming: nothing was playable until the last frame was separated
(~190 s for a 40 s clip). Now it emits 20 s windows as they finish.

**The first window is 5 s, not 20 s.** Time-to-first-audio is what a viewer experiences
as "buffering", and a 20 s first window means waiting 20 s before anything can play.

**Seam handling:** windows are cut with 50 ms of padding on each side, then trimmed back
after separation, so boundaries butt together without artefacts.

**CPU beats MPS here** — measured 8.4 s vs 55.7 s on the same input. At this window size
MPS spends nearly all wall time in GPU compile and transfer.

**Vocals are kept too**, not discarded — gender detection needs them (§3.3).

### 3.2 ASR — `dubber/asr.py`

**faster-whisper** (CTranslate2), not openai-whisper. No torch dependency, int8
quantised.

The critical property is that `transcribe()` returns a **generator**:

```python
for i, s in enumerate(segments):      # yields as it decodes
    yield Segment(idx=i, start=s.start, end=s.end, source_text=text)
```

Segment 1 can be translated while Whisper is still decoding segment 8. Nothing waits for
the full file.

Settings that matter:
- `beam_size=1` — greedy, ~2× faster, fine for dubbing
- `vad_filter=True` — drops silence so dead air is never dubbed
- `condition_on_previous_text=False` — stops runaway repetition loops

**Model size is a real trade-off, measured:**

| model | behaviour |
|---|---|
| `tiny` | collapses 40 s into **2 segments** — destroys sync entirely |
| `base` | good segmentation, garbles proper nouns (`서울대 로스쿨 저러…`) |
| `small` | **20 segments**, correct names (`졸업생 강휴민입니다` → "I'm Kang Hyumin") |
| `medium` | will not fit in 8 GB — thrashes swap, 0 segments in 10 min |

An alternative cloud backend exists (`dubber/asr_cloud.py`, Amazon Transcribe Streaming)
selected by `ASR_BACKEND`. Streaming, not batch — batch needs an S3 bucket, an upload and
a poll loop, while streaming needs no bucket and still yields incrementally. It holds
**no model at all**, which matters on memory-constrained hosts.

### 3.3 Gender casting — `dubber/gender.py`

Which voice to synthesise with is a binary choice, so full diarisation (pyannote — gated
model, heavy) is overkill. Instead: **median F0 by autocorrelation** over the voiced
frames of each segment.

```
40 ms frames, 20 ms hop
periodicity < 0.30 → unvoiced, skipped
< 6 voiced frames  → None, keep previous speaker rather than guess
boundary: 158 Hz   → below = male, above = female
```

158 Hz sits in the empty gap between the two clusters on real material (male ~85–180,
female ~165–255).

**This must run on the separated vocals, not the mix.** Music energy sits in the same F0
band as speech and turns the result into a coin flip.

### 3.4 Translation — `dubber/translate.py`

**Amazon Bedrock** (`mistral-large-3`) via the Converse API, `temperature=0`.

Calls are latency-bound, so they fan out across a thread pool — parallelism is what keeps
translation from being the bottleneck.

**The script-leak guard** is the most safety-critical code in the project. Untranslated
Korean must never reach an English voice:

```
translate → contains Hangul/CJK/kana?
          → retry once with a stricter prompt
          → still dirty? strip foreign characters
          → still nothing usable? return "" and SKIP the line
```

It **never falls back to the source line** — that was a real leak path in an earlier
version. A skipped line is silent and the bed carries the gap, which is strictly better
than speaking Korean in an English voice.

**Known gap:** the guard checks *script*, not *meaning*. A meta-response like
`"Sure, please provide the subtitle lines."` (which `네` provokes) is valid English and
passes straight through.

### 3.5 TTS — `dubber/tts.py`

Four interchangeable backends behind one interface:

| backend | nature | notes |
|---|---|---|
| **Polly** | cloud | most natural; network-bound so it parallelises and leaves CPU for demucs |
| **Piper** | local ONNX | fastest, robotic, ~61 MB per voice |
| **Kokoro** | local torch | natural, but ~800 MB resident |
| **say** | macOS | development only |

**Polly has no `hi-IN` voice at all** on a standard account — Kajal is filed under
`en-IN`. That gap is why a local fallback existed, and why loading it cost 800 MB of torch
to cover a single voice. `POLLY_FALLBACK=none` disables that.

`POLLY_ENGINE=generative` is the most natural engine, but it **rejects SSML**, so the
prosody-rate trick is refused and duration control falls entirely to the ffmpeg pass.
`neural` restores SSML timing.

**The voice cache is bounded (LRU).** Casting both genders in two languages touches four
ONNX voices, and an unbounded cache kept every one resident — ~250 MB on top of Whisper.
Bounding it took peak RSS from 538 MB to 293 MB.

### 3.6 Duration fitting — `dubber/audio.py`

A translated line rarely matches the length of the original. Two-stage fit:

**Stage 1 — speaking-rate prior.** Each backend has a *measured* chars/sec figure
(Piper 18.56 en, Polly 18.01 en, Kokoro 14.06 en). Guess a length scale, synthesise, and
re-synthesise only if the result lands outside 0.82–1.22× of its slot. Inside that band
the time stretcher is transparent and a second synthesis is wasted compute.

Using one prior for all backends is what previously crushed 7 of 16 lines against the
compressor limit.

**Stage 2 — ffmpeg `atempo`,** pitch-preserving. `atempo` only accepts 0.5–2.0, so wider
ratios chain filters:

```python
while ratio > 2.0: parts.append("atempo=2.0"); ratio /= 2.0
```

**Compression only, never expansion** (`min_tempo=1.0`). A line shorter than its slot is
left exactly as it is. Stretching short translations to pad the slot is what made the
voice drawl — a short line simply finishes early and the bed carries the gap.

Overrun past the clamp is accepted rather than compressed further; past that point speech
stops sounding human.

---

## 4. Concurrency model — `dubber/pipeline.py`

```
separation thread ──▶ Queue(maxsize=2) ──▶ consumer loop ──▶ ThreadPoolExecutor
   (producer)                               ASR + gender      (translate → TTS → fit)
```

**Why overlapped, not sequential.** Running stages in sequence makes total throughput the
*sum* of every stage's cost (measured 0.89× realtime). Overlapping makes it the **slowest
single stage** (1.23×). That is the difference between stalling and sustaining.

The queue is bounded at 2 so separation cannot run arbitrarily far ahead and exhaust
memory.

ASR runs **once** and fans out to every target language — a second language costs a
translate+TTS pass, not a second transcription.

`ready_to[lang]` is the point up to which that language is *continuously* dubbed, which is
what the UI gates playback on.

---

## 5. Browser scheduling — `static/player.js`

This is where sync actually happens. The server produces chunks; the browser decides when
they play.

### 5.1 The anchor

The video clock is the source of truth; audio bends to it. On play, capture one pairing:

```js
anchor = { ctx: audioContext.currentTime, video: video.currentTime }
```

Every segment is then scheduled at:

```js
when = anchor.ctx + (segment.start - anchor.video)
```

**Why drift cannot accumulate:** each segment is scheduled against the *anchor*, not
relative to the previous segment. An error in one line cannot propagate into the next.

Pause, seek, rate change and language switch all invalidate the anchor and re-capture it.

### 5.2 Epochs

`fetch` + `decodeAudioData` are async. A chunk requested before a language switch can
resolve *after* it, and play the old language over the new one.

Every invalidating action bumps `epoch`. Work in flight across a bump refuses to start.
Belt and braces: `newDubBus()` also *disconnects the entire gain node*, so any source
still attached to it — including buffers that were never registered in `scheduled` and so
could not be stopped individually — is orphaned and silent.

### 5.3 Late arrival

If a fetch overruns and `when` is already in the past, the chunk is not dropped and not
played from its start — it starts **mid-buffer**:

```js
const offset = ctx.currentTime - when;
src.start(ctx.currentTime, offset);
```

The line stays pinned to its timeline position instead of sliding everything later.

### 5.4 Bed mixing

Two paths, depending on whether separation ran:

- **Separated** — the bed is a genuinely voice-free track. Play it at full level, keep the
  `<video>` element muted. No ducking, no bleed-through.
- **Not separated** — route the video element through a `GainNode` and duck it to 28% while
  each dubbed line speaks, with a 120 ms ramp so the dip does not click. Music and effects
  survive; the Korean stays faintly audible.

The element is routed through Web Audio rather than muted precisely so ducking is possible
— muting would kill music and effects along with dialogue.

---

## 6. Server — `server.py`

Flask, deliberately thin. Session state is a module-level dict; the pipeline runs on
background threads inside the worker process.

| Endpoint | Purpose |
|---|---|
| `POST /api/session` | upload a video, start the pipeline |
| `GET  /api/session/<id>/manifest?lang=` | segment list + progress (polled ~700 ms) |
| `GET  /api/session/<id>/chunk/<lang>/<idx>` | one dubbed line as WAV |
| `GET  /api/session/<id>/bed/<idx>` | one background window |
| `GET  /api/session/<id>/video` | the source video |

**Exactly one worker.** Sessions are in-process state, so a second worker would answer
polls for sessions it has never seen. Concurrency comes from threads (`wsgi.py`).

**Security:** the `path=` parameter opens an arbitrary server file and serves it back via
`/video` — arbitrary file disclosure. Gated behind `ALLOW_LOCAL_PATH`, default off.
Hosted instances accept uploads only.

---

## 7. What the numbers taught us

Most of the design came from being wrong first.

| Finding | Consequence |
|---|---|
| Sequential stages = 0.89× realtime | overlap them → 1.23× |
| Whole-file separation blocked playback ~190 s | window it → ~5 s to first audio |
| Unbounded voice cache held 4 ONNX models | bound it → 538 MB → 293 MB |
| Kokoro fallback loaded torch for one voice | ~800 MB for a voice often never used |
| `tiny` collapsed 40 s into 2 segments | quality floor is `base`; `small` for names |
| One chars/sec prior for all backends | crushed 7/16 lines; per-backend priors → 0–1 |
| MPS slower than CPU for 20 s windows | 55.7 s vs 8.4 s |
| **Memory, not CPU, is the binding constraint** | ~1.6 GB working set; below that the machine swaps and everything looks "slow" for reasons no profiler blames on the code |

---

## 8. Verification

Correctness here is not "it produced audio" — it is "no Korean survived". Both are checked
by transcribing the *rendered output back*:

```
separated bed  → transcribe as Korean → 0 characters  (dialogue truly removed)
English chunks → transcribe as English → exact match to manifest text
```

One caveat worth recording: force-decoding English audio *as Korean* makes Whisper
hallucinate Hangul (`"전ely rub white on the clothing stains"` — a Hangul character bolted
onto correct English mid-word). That is a decoder artefact, not a leak. Decoded normally,
the same chunk matches its manifest text exactly.

`dubber/render.py` renders the same mix offline — the ground truth for confirming the
browser scheduler produced what the pipeline intended.

---

## 9. Known limits

- **No speaker diarisation** — one voice per gender, regardless of who is talking.
- **Translation can emit meta-text** for very short inputs; the script guard cannot catch
  it because the output is valid English.
- **Hindi is disabled** — Polly exposes no `hi-IN` voice, so Hindi male fell through to a
  torch-backed local model for a single voice. Mappings remain in `tts.py`.
- **Not deployable to serverless** — torch exceeds function size limits, ffmpeg is shelled
  out to, and sessions are in-process state with threads that outlive the response.
