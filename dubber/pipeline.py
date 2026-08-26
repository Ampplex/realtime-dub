"""
Look-ahead dubbing pipeline.

Because the whole file is on disk, we are not doing true live dubbing — we run the
stages *ahead* of the playhead. That is what makes the sync problem tractable: the
buffer is permanent headroom that workers keep refilling, not just startup latency.

Stage layout (all concurrent, connected by queues):

    ASR ──> segment queue ──> per-language workers ──> manifest
    (1 thread, streams        (translate -> TTS -> fit duration)
     segments as decoded)

ASR runs ONCE and fans out to every target language, so adding Hindi alongside
English costs a translate+TTS pass, not a second transcription.

Measured on an M1/8GB: ASR ~7.6x realtime, translation ~0.6s/line (parallel),
Piper TTS 10-29x realtime. The binding constraint is translation, and it still
clears realtime comfortably.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .asr import WhisperASR, get_asr
from .audio import extract_audio, probe_duration, wav_duration, fit_duration, OUT_SR, run
from .models import Segment, Progress
from .separate import Separator
from .gender import SegmentGender
from .translate import get_translator
from .tts import get_backend, scale_for, MIN_SCALE, MAX_SCALE

# Speaking-rate priors used to guess a length_scale before synthesising, so most
# lines land near their slot on the first attempt instead of needing a re-render.
# Re-synthesis band: outside this the time stretch becomes audible.
RESYNTH_LO, RESYNTH_HI = 0.82, 1.22


class DubSession:
    def __init__(self, video: Path, work: Path, langs: tuple[str, ...] = ("hi", "en"),
                 source_lang: str = "ko", model_size: str = "base",
                 buffer_seconds: float = 4.0, tts_backend: str = "auto",
                 translator: str = "auto", separate: bool = True,
                 sep_device: str = "cpu"):
        self.video = Path(video)
        self.work = Path(work)
        self.langs = tuple(langs)
        self.source_lang = source_lang
        self.buffer_seconds = buffer_seconds
        self.work.mkdir(parents=True, exist_ok=True)
        for l in self.langs:
            (self.work / l).mkdir(exist_ok=True)

        self.asr = get_asr(model_size=model_size)
        self.tts = get_backend(tts_backend)
        self.translator = get_translator(translator)

        self.separate = separate
        self.separator = Separator(device=sep_device) if separate else None
        self.beds: list = []            # BedChunk, background-only audio
        self.bed_ready_to = 0.0

        self._next_idx = 0                      # segments are numbered across windows
        self.gender_of: dict[int, str] = {}     # segment idx -> "male"/"female"
        self._sg: SegmentGender | None = None

        self.segments: dict[int, Segment] = {}
        self.progress = Progress(ready_to={l: 0.0 for l in self.langs})
        self.lock = threading.Lock()
        self._started = False
        self._done = threading.Event()

    # ------------------------------------------------------------------ api

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._run, daemon=True).start()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def manifest(self, lang: str) -> dict:
        with self.lock:
            segs = [s.public(lang) for s in sorted(self.segments.values(), key=lambda x: x.idx)
                    if lang in s.audio]
            p = self.progress.public()
        return {
            "lang": lang,
            "segments": segs,
            "beds": [b.public() for b in list(self.beds)],
            "bed_ready_to": round(self.bed_ready_to, 3),
            "separated": bool(self.separate),
            "ready_to": p["ready_to"].get(lang, 0.0),
            "ready_all": p["ready_to"],          # per-language, so the UI never locks
            "asr_done_to": p["asr_done_to"],
            "total": p["total"],
            "stage": p["stage"],
            "error": p["error"],
            "buffer_seconds": self.buffer_seconds,
            "complete": self._done.is_set(),
        }

    def bed_path(self, idx: int) -> Path | None:
        for b in list(self.beds):
            if b.idx == idx:
                return self.work / "bed" / b.filename
        return None

    def chunk_path(self, lang: str, idx: int) -> Path | None:
        with self.lock:
            seg = self.segments.get(idx)
        if not seg or lang not in seg.audio:
            return None
        return self.work / lang / seg.audio[lang]

    # -------------------------------------------------------------- internals

    def _run(self) -> None:
        try:
            self.progress.stage = "extracting audio"
            self.progress.total = probe_duration(self.video)

            if not self.separate:
                # No separation: one pass over the whole file, gender off mixed audio.
                wav = extract_audio(self.video, self.work / "source16k.wav")
                try:
                    self._sg = SegmentGender(wav)
                except Exception:
                    self._sg = None
                self.progress.stage = "dubbing"
                with ThreadPoolExecutor(max_workers=len(self.langs) * 2) as pool:
                    futures = []
                    for seg in self.asr.stream(wav, language=self.source_lang):
                        self._admit(seg, futures, pool)
                    for f in futures:
                        f.result()
                self.progress.stage = "complete"
                return

            # Windowed streaming. Each window is separated, transcribed, translated
            # and voiced before the next one is touched, so the first few seconds
            # become playable in window-time rather than whole-file time. This is the
            # difference between waiting ~10s and waiting for the entire video.
            self.progress.stage = "separating voice"
            src44 = extract_audio(self.video, self.work / "source44.wav",
                                  sr=44100, channels=2)

            # Separation PRODUCES windows on its own thread while the main loop
            # CONSUMES them. Running them in sequence made total throughput the sum
            # of every stage's cost (0.89x); overlapping makes it the slowest single
            # stage instead. This only became worthwhile once TTS moved to Polly,
            # which is network-bound and leaves the CPU free for demucs.
            queue_: "queue.Queue" = queue.Queue(maxsize=2)

            def produce() -> None:
                try:
                    for bed in self.separator.stream(src44, self.work / "bed"):
                        with self.lock:
                            self.beds.append(bed)
                            self.bed_ready_to = bed.end
                        queue_.put(bed)
                except Exception as e:
                    self.progress.error = f"separation failed: {type(e).__name__}: {e}"
                finally:
                    queue_.put(None)

            producer = threading.Thread(target=produce, daemon=True)
            producer.start()

            with ThreadPoolExecutor(max_workers=len(self.langs) * 3) as pool:
                futures: list = []
                while True:
                    bed = queue_.get()
                    if bed is None:
                        break
                    self.progress.stage = "dubbing"
                    voc = self.work / "bed" / f"voc_{bed.idx:05d}.wav"
                    if voc.exists():
                        self._window(voc, bed.start, futures, pool)
                for f in futures:
                    f.result()
            producer.join(timeout=5)

            self.progress.stage = "complete"
        except Exception as e:
            self.progress.stage = "error"
            self.progress.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            self._done.set()

    def _window(self, voc: Path, offset: float, futures: list, pool) -> None:
        """Transcribe + dub one separated window, shifting times onto the real timeline."""
        w16 = voc.with_name(voc.stem + "_16k.wav")
        try:
            run(["ffmpeg", "-y", "-v", "error", "-i", str(voc), "-ac", "1",
                 "-ar", "16000", "-c:a", "pcm_s16le", str(w16)])
        except Exception as e:
            self.progress.error = f"window resample failed: {e}"
            return
        try:
            sg = SegmentGender(w16)
        except Exception:
            sg = None

        for seg in self.asr.stream(w16, language=self.source_lang):
            local_start, local_end = seg.start, seg.end
            seg.start += offset                    # window-local -> source timeline
            seg.end += offset
            if sg is not None:
                g, _f0 = sg.for_segment(local_start, local_end)
                seg.gender = g
            self._admit(seg, futures, pool)

    def _admit(self, seg: Segment, futures: list, pool) -> None:
        """Register a segment and queue its per-language rendering."""
        with self.lock:
            seg.idx = self._next_idx
            self._next_idx += 1
            self.segments[seg.idx] = seg
            self.gender_of[seg.idx] = seg.gender
        self.progress.asr_done_to = max(self.progress.asr_done_to, seg.end)
        for lang in self.langs:
            futures.append(pool.submit(self._render, seg, lang))

    def _concat_vocals(self) -> Path | None:
        """Join the per-window vocal stems into one 16k mono file for ASR/gender."""
        import glob
        parts = sorted(glob.glob(str(self.work / "bed" / "voc_*.wav")))
        if not parts:
            return None
        listing = self.work / "voc_list.txt"
        # Absolute paths: ffmpeg's concat demuxer resolves relative entries against
        # the LIST FILE's directory, not the cwd, which silently finds nothing.
        listing.write_text("".join(f"file '{Path(p).resolve()}'\n" for p in parts))
        joined = self.work / "vocals16k.wav"
        try:
            run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                 "-i", str(listing), "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", str(joined)])
            return joined
        except Exception as e:
            # Do not swallow this: falling back to mixed audio silently degrades
            # gender detection, which is exactly the bug this once caused.
            self.progress.error = f"vocals concat failed: {e}"
            return None

    def _separate(self, src44: Path) -> None:
        """Background-only stem, emitted window by window so it streams ahead."""
        try:
            for bed in self.separator.stream(src44, self.work / "bed"):
                with self.lock:
                    self.beds.append(bed)
                    self.bed_ready_to = bed.end
        except Exception as e:
            with self.lock:
                self.progress.error = f"separation failed: {type(e).__name__}: {e}"
                self.separate = False      # fall back to ducking in the player

    def _render(self, seg: Segment, lang: str) -> None:
        """translate -> synthesise -> fit to the source slot."""
        try:
            text = self.translator.translate(seg.source_text, lang, self.source_lang)
            with self.lock:
                seg.text[lang] = text
            if not text.strip():
                return      # nothing safe to say here; leave the bed playing alone

            slot = seg.source_duration
            gender = self.gender_of.get(seg.idx, "female")
            raw = self.work / lang / f"{seg.idx:05d}.raw.wav"
            out = self.work / lang / f"{seg.idx:05d}.wav"

            # Pass 1: guess a length_scale from the backend's measured speaking rate.
            est = len(text) / self.tts.chars_per_sec(lang)
            scale = scale_for(est, slot)
            self.tts.synth(text, lang, raw, length_scale=scale, gender=gender)
            dur = wav_duration(raw)

            # Pass 2: the prior is only a guess, so correct it against what the voice
            # ACTUALLY produced. Duration is ~linear in length_scale for both backends.
            # Re-synthesising is far kinder to intelligibility than letting the time
            # stretcher compress a long line into a short slot.
            # Only re-synthesise when the stretch needed would actually be audible.
            # Between ~0.85x and ~1.20x the time stretcher is transparent, so paying
            # for a second synthesis there is wasted compute.
            # Only correct OVERRUNS. An underrun just ends early, which is correct
            # dubbing behaviour - re-synthesising it slower would drawl.
            if dur > 0 and dur / slot > RESYNTH_HI:
                corrected = max(MIN_SCALE, min(MAX_SCALE, scale * (slot / dur)))
                if abs(corrected - scale) > 0.02:
                    self.tts.synth(text, lang, raw, length_scale=corrected, gender=gender)
                    scale = corrected
                    dur = wav_duration(raw)

            dur, tempo = fit_duration(raw, out, slot)
            raw.unlink(missing_ok=True)

            with self.lock:
                seg.audio[lang] = out.name
                seg.duration[lang] = dur
                seg.tempo[lang] = tempo
                # ready_to = the point up to which this language is continuously dubbed
                done = [s for s in self.segments.values() if lang in s.audio]
                self.progress.ready_to[lang] = max(s.end for s in done) if done else 0.0
        except Exception as e:
            # A segment that fails to render has no audio, so manifest() filters it
            # out entirely and the run looks like it simply produced nothing. Put the
            # reason where it can actually be seen: the log, and the manifest.
            traceback.print_exc()
            with self.lock:
                seg.text.setdefault(lang, f"[{type(e).__name__}]")
                if not self.progress.error:
                    self.progress.error = f"render failed ({lang}): {type(e).__name__}: {e}"
