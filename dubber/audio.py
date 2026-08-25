"""ffmpeg helpers. Everything shells out — no heavyweight audio deps."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

SR = 16000            # what Whisper wants
OUT_SR = 22050        # what we render dubbed audio at


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def probe_duration(path: Path) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "json", str(path)]).stdout
    return float(json.loads(out)["format"]["duration"])


def extract_audio(video: Path, dest: Path, sr: int = SR, channels: int = 1) -> Path:
    """PCM at `sr`. Mono 16k is what Whisper decodes fastest; demucs wants 44.1k stereo."""
    run(["ffmpeg", "-y", "-v", "error", "-i", str(video),
         "-vn", "-ac", str(channels), "-ar", str(sr), "-c:a", "pcm_s16le", str(dest)])
    return dest


def wav_duration(path: Path) -> float:
    return probe_duration(path)


def atempo_chain(ratio: float) -> str:
    """ffmpeg's atempo only accepts 0.5-2.0, so chain filters for wider ratios."""
    ratio = max(0.25, min(4.0, ratio))
    parts = []
    while ratio > 2.0:
        parts.append("atempo=2.0")
        ratio /= 2.0
    while ratio < 0.5:
        parts.append("atempo=0.5")
        ratio /= 0.5
    parts.append(f"atempo={ratio:.4f}")
    return ",".join(parts)


def fit_duration(src: Path, dest: Path, target: float,
                 min_tempo: float = 1.0, max_tempo: float = 1.6) -> tuple[float, float]:
    """
    Time-stretch `src` toward `target` seconds without changing pitch.

    Returns (final_duration, tempo_applied). We clamp the stretch: forcing a long
    translation into a short slot destroys intelligibility, so past the clamp we
    accept overrun and let the scheduler absorb it in the following silence.
    """
    dur = wav_duration(src)
    if dur <= 0 or target <= 0:
        return dur, 1.0

    tempo = dur / target
    # min_tempo defaults to 1.0: audio shorter than its slot is left exactly as it is.
    # Slowing it to fill the slot is what made short translations sound like 0.5x.
    tempo = max(min_tempo, min(max_tempo, tempo))

    if abs(tempo - 1.0) < 0.03:
        run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c", "copy", str(dest)])
        return dur, 1.0

    run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-filter:a", atempo_chain(tempo), "-ar", str(OUT_SR), "-ac", "1", str(dest)])
    return wav_duration(dest), tempo
