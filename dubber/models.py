"""Shared data types for the dubbing pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Segment:
    """One unit of speech on the source timeline, plus its dubbed renderings."""
    idx: int
    start: float                 # seconds into the source video
    end: float
    source_text: str
    # per-language state, keyed by lang code
    text: dict[str, str] = field(default_factory=dict)
    audio: dict[str, str] = field(default_factory=dict)      # lang -> wav filename
    duration: dict[str, float] = field(default_factory=dict)  # lang -> rendered seconds
    tempo: dict[str, float] = field(default_factory=dict)     # lang -> stretch applied
    gender: str = "female"                                    # detected speaker gender

    @property
    def source_duration(self) -> float:
        return max(0.05, self.end - self.start)

    def public(self, lang: str) -> dict:
        return {
            "idx": self.idx,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "source_text": self.source_text,
            "text": self.text.get(lang, ""),
            "dur": round(self.duration.get(lang, 0.0), 3),
            "tempo": round(self.tempo.get(lang, 1.0), 3),
            "gender": self.gender,
            "url": f"/api/chunk/{lang}/{self.idx}",
        }


@dataclass
class Progress:
    asr_done_to: float = 0.0     # source seconds transcribed so far
    ready_to: dict = field(default_factory=dict)   # lang -> seconds fully dubbed
    total: float = 0.0
    stage: str = "idle"
    error: str = ""

    def public(self) -> dict:
        return asdict(self)
