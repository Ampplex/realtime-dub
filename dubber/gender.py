"""
Per-segment speaker gender estimation from fundamental frequency.

Full diarisation (pyannote) needs a gated model and a lot of compute for what is,
here, a binary choice: which voice to synthesise with. Median F0 over the voiced
frames of a segment separates adult male from adult female reliably enough for
voice casting, and costs microseconds.

Typical adult ranges: male ~85-180 Hz (median ~120), female ~165-255 (median ~210).
Anything with too few voiced frames to judge returns None so the caller can fall
back to the previous speaker rather than guessing.
"""
from __future__ import annotations

import numpy as np

F0_MIN, F0_MAX = 70.0, 300.0
BOUNDARY = 158.0          # Hz; below -> male, above -> female
MIN_VOICED_FRAMES = 6


def _f0_autocorr(frame: np.ndarray, sr: int) -> float | None:
    """Single-frame F0 by autocorrelation peak."""
    frame = frame - frame.mean()
    if np.sqrt(np.mean(frame ** 2)) < 1e-3:      # silence
        return None
    corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    lo, hi = int(sr / F0_MAX), int(sr / F0_MIN)
    if hi >= len(corr) or lo >= hi:
        return None
    seg = corr[lo:hi]
    peak = int(np.argmax(seg)) + lo
    if corr[peak] <= 0 or corr[0] <= 0:
        return None
    if corr[peak] / corr[0] < 0.30:              # weak periodicity -> unvoiced
        return None
    return sr / peak


def median_f0(samples: np.ndarray, sr: int) -> float | None:
    win = int(0.040 * sr)                        # 40 ms frames, 20 ms hop
    hop = win // 2
    vals = []
    for i in range(0, max(1, len(samples) - win), hop):
        f = _f0_autocorr(samples[i:i + win], sr)
        if f is not None:
            vals.append(f)
    if len(vals) < MIN_VOICED_FRAMES:
        return None
    return float(np.median(vals))


def classify(samples: np.ndarray, sr: int) -> tuple[str | None, float | None]:
    """Return ("male"|"female"|None, median_f0)."""
    f0 = median_f0(samples, sr)
    if f0 is None:
        return None, None
    return ("male" if f0 < BOUNDARY else "female"), f0


class SegmentGender:
    """Reads segment ranges out of a mono wav and classifies each."""

    def __init__(self, wav_path, sr_hint: int | None = None):
        import wave
        with wave.open(str(wav_path)) as w:
            self.sr = w.getframerate()
            raw = w.readframes(w.getnframes())
            a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if w.getnchannels() == 2:
                a = a.reshape(-1, 2).mean(axis=1)
        self.samples = a
        self._last = "female"

    def for_segment(self, start: float, end: float) -> tuple[str, float | None]:
        a = int(max(0, start) * self.sr)
        b = int(min(len(self.samples) / self.sr, end) * self.sr)
        if b <= a:
            return self._last, None
        g, f0 = classify(self.samples[a:b], self.sr)
        if g is None:
            return self._last, f0        # keep the previous speaker rather than guess
        self._last = g
        return g, f0
