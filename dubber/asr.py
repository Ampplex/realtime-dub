"""
Speech recognition via faster-whisper (CTranslate2 — no torch dependency).

The key property for look-ahead dubbing: `transcribe()` returns a *generator* that
yields segments as they are decoded, so the pipeline can start translating segment 1
while Whisper is still working on segment 8. Nothing waits for the full file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .models import Segment

# int8 keeps a 1.5GB model inside an 8GB machine's budget with room for Piper + Ollama.
_MODEL_CACHE: dict = {}
_CACHE_LOCK = None

DEFAULT_MODEL = "small"
DEFAULT_COMPUTE = "int8"


class WhisperASR:
    def __init__(self, model_size: str = DEFAULT_MODEL, compute_type: str = DEFAULT_COMPUTE,
                 device: str = "cpu", cpu_threads: int = 0):
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.cpu_threads = cpu_threads
        self._model = None

    @property
    def model(self):
        # Process-wide cache: loading Whisper costs seconds, and paying that per
        # session shows up directly as time-to-first-audio.
        if self._model is None:
            import threading
            global _CACHE_LOCK
            if _CACHE_LOCK is None:
                _CACHE_LOCK = threading.Lock()
            key = (self.model_size, self.device, self.compute_type)
            with _CACHE_LOCK:
                if key not in _MODEL_CACHE:
                    from faster_whisper import WhisperModel
                    _MODEL_CACHE[key] = WhisperModel(
                        self.model_size, device=self.device,
                        compute_type=self.compute_type, cpu_threads=self.cpu_threads)
                self._model = _MODEL_CACHE[key]
        return self._model

    def stream(self, audio: Path, language: str = "ko",
               task: str = "transcribe") -> Iterator[Segment]:
        """Yield Segments as Whisper decodes them."""
        segments, _info = self.model.transcribe(
            str(audio),
            language=language,
            task=task,                      # "translate" would force English output
            beam_size=1,                    # greedy: ~2x faster, fine for dubbing
            vad_filter=True,                # drop silence so we don't dub dead air
            vad_parameters={
                "min_silence_duration_ms": 400,
                "speech_pad_ms": 150,
            },
            condition_on_previous_text=False,   # stops runaway repetition loops
        )
        for i, s in enumerate(segments):
            text = (s.text or "").strip()
            if not text:
                continue
            yield Segment(idx=i, start=float(s.start), end=float(s.end), source_text=text)

    def detect_language(self, audio: Path) -> tuple[str, float]:
        _segments, info = self.model.transcribe(str(audio), beam_size=1, vad_filter=True)
        return info.language, float(info.language_probability)


def get_asr(model_size: str = DEFAULT_MODEL, prefer: str | None = None):
    """
    Pick an ASR backend.

    `transcribe` holds no model at all, which is the whole point on a small
    instance: faster-whisper is the last resident model in an otherwise
    cloud-only pipeline (Bedrock for translation, Polly for voices), and even
    `tiny` costs enough to put a 512MB box at risk.

    Falls back to Whisper when Transcribe is not usable — an AWS account without
    the service enabled answers with SubscriptionRequiredException — so a missing
    subscription degrades instead of breaking the app.
    """
    import os
    prefer = (prefer or os.getenv("ASR_BACKEND", "whisper")).lower()
    if prefer in ("transcribe", "aws", "cloud"):
        try:
            from .asr_cloud import TranscribeASR
            return TranscribeASR()
        except Exception:
            if prefer != "cloud":
                raise
    return WhisperASR(model_size=model_size)
