"""
Speech recognition via Amazon Transcribe Streaming.

Why streaming and not batch: batch transcription requires the audio to be in S3
first, which means a bucket, an upload and a poll loop. Streaming takes raw PCM over
a websocket, needs no bucket, and — more importantly — yields results *as it goes*,
which is the same property the look-ahead pipeline relies on from Whisper.

The point of this backend is memory: it holds no model at all. faster-whisper's
tiny model plus its decode arena is the single largest resident cost in the lean
build, and on a 512MB instance that is the difference between running and being
OOM-killed.
"""
from __future__ import annotations

import asyncio
import queue
import threading
import wave
from pathlib import Path
from typing import Iterator

from .models import Segment

# Transcribe wants a language code, not the two-letter code the rest of the app uses.
LANG_CODES = {"ko": "ko-KR", "en": "en-US", "hi": "hi-IN", "ja": "ja-JP", "zh": "zh-CN"}

CHUNK = 8192          # bytes of PCM per frame sent upstream
_SENTINEL = object()


class TranscribeASR:
    """Drop-in replacement for WhisperASR: same stream() -> Iterator[Segment]."""

    def __init__(self, region: str | None = None, **_ignored):
        import os
        self.region = region or os.getenv("AWS_REGION", "us-west-2")

    # WhisperASR exposes `.model` and server.py's warm-up touches it. There is no
    # model to warm here, so make that a cheap no-op rather than an AttributeError.
    @property
    def model(self):
        return None

    def stream(self, audio: Path, language: str = "ko",
               task: str = "transcribe") -> Iterator[Segment]:
        results: "queue.Queue" = queue.Queue()
        t = threading.Thread(target=self._run, args=(audio, language, results), daemon=True)
        t.start()

        idx = 0
        while True:
            item = results.get()
            if item is _SENTINEL:
                break
            if isinstance(item, Exception):
                raise item
            start, end, text = item
            text = (text or "").strip()
            if not text:
                continue
            yield Segment(idx=idx, start=float(start), end=float(end), source_text=text)
            idx += 1

    # ------------------------------------------------------------------ internals

    def _run(self, audio: Path, language: str, out: "queue.Queue") -> None:
        try:
            asyncio.run(self._transcribe(audio, language, out))
        except Exception as e:                      # surface, don't hang the consumer
            out.put(e)
        finally:
            out.put(_SENTINEL)

    async def _transcribe(self, audio: Path, language: str, out: "queue.Queue") -> None:
        from amazon_transcribe.client import TranscribeStreamingClient
        from amazon_transcribe.handlers import TranscriptResultStreamHandler

        with wave.open(str(audio), "rb") as w:
            rate = w.getframerate()
            channels = w.getnchannels()
            frames = w.readframes(w.getnframes())

        if channels != 1:
            raise ValueError(f"Transcribe streaming needs mono PCM, got {channels} channels")

        client = TranscribeStreamingClient(region=self.region)
        stream = await client.start_stream_transcription(
            language_code=LANG_CODES.get(language, language),
            media_sample_rate_hz=rate,
            media_encoding="pcm",
        )

        class Handler(TranscriptResultStreamHandler):
            async def handle_transcript_event(self, event):
                for r in event.transcript.results:
                    # Partials rewrite themselves as more audio arrives; only a
                    # finalised result has a stable transcript and timing.
                    if r.is_partial or not r.alternatives:
                        continue
                    out.put((r.start_time, r.end_time, r.alternatives[0].transcript))

        async def pump():
            for i in range(0, len(frames), CHUNK):
                await stream.input_stream.send_audio_event(audio_chunk=frames[i:i + CHUNK])
            await stream.input_stream.end_stream()

        await asyncio.gather(pump(), Handler(stream.output_stream).handle_events())

    def detect_language(self, audio: Path) -> tuple[str, float]:
        raise NotImplementedError("Transcribe streaming requires an explicit language")
