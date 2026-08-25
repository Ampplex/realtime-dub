"""
Offline render: bed + dubbed speech -> a finished audio track, muxed back to video.

This is the ground truth for "did it actually work". The browser schedules the same
chunks live; this produces the same mix as a file you can play or measure.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def _read(path: Path, sr: int = SR) -> np.ndarray:
    data, in_sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono_or_st = data.T
    if in_sr != sr:                       # resample via ffmpeg for correctness
        tmp = path.with_suffix(".rs.wav")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path),
                        "-ar", str(sr), str(tmp)], check=True)
        data, _ = sf.read(str(tmp), dtype="float32", always_2d=True)
        tmp.unlink(missing_ok=True)
        mono_or_st = data.T
    if mono_or_st.shape[0] == 1:
        mono_or_st = np.repeat(mono_or_st, 2, axis=0)
    return mono_or_st[:2]


def render(session, lang: str, out_video: Path, bed_gain: float = 1.0,
           dub_gain: float = 1.0) -> Path:
    """Mix the voice-free bed with the dubbed lines and mux onto the original video."""
    total = session.progress.total or 0.0
    n = int(total * SR) + SR
    track = np.zeros((2, n), dtype=np.float32)

    # 1. lay down the background bed at full level - music and effects untouched
    for b in session.beds:
        p = session.work / "bed" / b.filename
        if not p.exists():
            continue
        a = _read(p)
        i = int(b.start * SR)
        j = min(n, i + a.shape[1])
        track[:, i:j] += a[:, : j - i] * bed_gain

    # 2. drop each dubbed line in at its source timestamp
    for seg in sorted(session.segments.values(), key=lambda s: s.idx):
        fn = seg.audio.get(lang)
        if not fn:
            continue
        p = session.work / lang / fn
        if not p.exists():
            continue
        a = _read(p)
        i = int(seg.start * SR)
        j = min(n, i + a.shape[1])
        track[:, i:j] += a[:, : j - i] * dub_gain

    peak = float(np.max(np.abs(track))) or 1.0
    if peak > 0.99:
        track *= 0.99 / peak

    wav = session.work / f"mix_{lang}.wav"
    sf.write(str(wav), track.T, SR, subtype="PCM_16")

    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(session.video),
                    "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", str(out_video)], check=True)
    return out_video
