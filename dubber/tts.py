"""
Pluggable TTS backends.

Piper is the deployment default: MIT-licensed, ONNX, CPU-only, ~10-30x realtime,
no per-request cost and no network hop in the latency budget. The abstraction
exists so a hosted voice (Azure/Google/ElevenLabs) can be dropped in without the
pipeline noticing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import wave
from abc import ABC, abstractmethod
from pathlib import Path

VOICES_DIR = Path(__file__).resolve().parent.parent / "voices"

# Piper: one model per (language, gender). pratham is male, priyamvada female.
PIPER_VOICES = {
    ("hi", "male"):   "hi_IN-pratham-medium",
    ("hi", "female"): "hi_IN-priyamvada-medium",
    ("en", "male"):   "en_US-ryan-medium",
    ("en", "female"): "en_US-lessac-medium",
}

# Kokoro: 82M params, Apache-2.0, ~5x realtime warm. Noticeably more natural than
# Piper, which is why it is the default despite being ~6x slower.
KOKORO_LANG = {"hi": "h", "en": "a"}
KOKORO_VOICES = {
    ("hi", "female"): "hf_alpha",
    ("hi", "male"):   "hm_omega",
    ("en", "female"): "af_heart",
    ("en", "male"):   "am_michael",
}

# Piper's length_scale drives the duration predictor, so stretching this way keeps
# formants intact. Outside this band speech stops sounding human, so we clamp and
# let the scheduler absorb any residual overrun.
# A line may be SPED UP to fit its slot, but must never be slowed down to pad one
# out. A short translation finishing early is a natural pause; stretching it to fill
# the gap makes the voice drawl, which is far more noticeable than the silence.
MIN_SCALE, MAX_SCALE = 0.70, 1.04


class TTSBackend(ABC):
    name = "base"

    # Measured chars/sec at length_scale=1.0, per language. Used to guess an initial
    # length_scale so most lines land in their slot on the first synthesis. These are
    # empirical, not assumed: a wrong prior forces the duration fitter to compress
    # hard, which is what made lines sound rushed.
    CHARS_PER_SEC: dict[str, float] = {"hi": 11.0, "en": 15.0}

    def chars_per_sec(self, lang: str) -> float:
        return self.CHARS_PER_SEC.get(lang, 13.0)

    @abstractmethod
    def synth(self, text: str, lang: str, out: Path, length_scale: float = 1.0,
              gender: str = "female") -> Path:
        ...

    def available_langs(self) -> list[str]:
        return ["hi", "en"]


class PiperBackend(TTSBackend):
    """
    Self-hosted ONNX voices, cached because loading costs ~0.5s each.

    The cache is BOUNDED. Casting both genders in both languages touches four
    voices, and an unbounded cache kept every one resident — roughly 250MB of ONNX
    on top of Whisper, which is what pushed a 512MB instance over the edge. Holding
    two covers the common case (one language, both genders) with no reloads;
    PIPER_VOICE_CACHE=1 trades a ~0.5s reload on each gender change for ~60MB.
    """
    name = "piper"
    CHARS_PER_SEC = {"hi": 11.47, "en": 18.56}     # measured

    def __init__(self, voices: dict | None = None):
        from piper import PiperVoice  # noqa: F401  (import cost paid once)
        import collections
        self.voices = voices or PIPER_VOICES
        self._cache: "collections.OrderedDict" = collections.OrderedDict()
        self._max_cache = max(1, int(os.getenv("PIPER_VOICE_CACHE", "2")))

    def _resolve(self, lang: str, gender: str) -> str:
        name = self.voices.get((lang, gender))
        if name and (VOICES_DIR / f"{name}.onnx").exists():
            return name
        for g in ("female", "male"):               # fall back to whichever exists
            n = self.voices.get((lang, g))
            if n and (VOICES_DIR / f"{n}.onnx").exists():
                return n
        raise FileNotFoundError(f"No Piper voice for {lang}. Run ./fetch-voices")

    def _voice(self, lang: str, gender: str):
        name = self._resolve(lang, gender)
        cached = self._cache.get(name)
        if cached is not None:
            self._cache.move_to_end(name)          # mark as most recently used
            return cached

        from piper import PiperVoice
        voice = PiperVoice.load(str(VOICES_DIR / f"{name}.onnx"))
        self._cache[name] = voice
        while len(self._cache) > self._max_cache:
            self._cache.popitem(last=False)        # drop least recently used
        return voice

    def synth(self, text: str, lang: str, out: Path, length_scale: float = 1.0,
              gender: str = "female") -> Path:
        from piper import SynthesisConfig
        voice = self._voice(lang, gender)
        cfg = SynthesisConfig(
            length_scale=max(MIN_SCALE, min(MAX_SCALE, length_scale)),
            normalize_audio=True,
        )
        with wave.open(str(out), "wb") as wf:
            voice.synthesize_wav(text, wf, syn_config=cfg)
        return out

    def available_langs(self) -> list[str]:
        return sorted({l for (l, _g), v in self.voices.items()
                       if (VOICES_DIR / f"{v}.onnx").exists()})


class KokoroBackend(TTSBackend):
    """Higher-quality neural voices with a male/female option per language."""
    name = "kokoro"
    SR = 24000
    CHARS_PER_SEC = {"hi": 8.93, "en": 14.06}      # measured

    def __init__(self):
        from kokoro import KPipeline  # noqa: F401
        self._pipes: dict[str, object] = {}

    def _pipe(self, lang: str):
        code = KOKORO_LANG.get(lang, "a")
        if code not in self._pipes:
            import warnings
            from kokoro import KPipeline
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._pipes[code] = KPipeline(lang_code=code)
        return self._pipes[code]

    def synth(self, text: str, lang: str, out: Path, length_scale: float = 1.0,
              gender: str = "female") -> Path:
        import warnings
        import numpy as np
        import soundfile as sf

        voice = KOKORO_VOICES.get((lang, gender)) or KOKORO_VOICES[(lang, "female")]
        # Kokoro speaks faster as `speed` rises; length_scale is the inverse notion.
        speed = 1.0 / max(0.5, min(2.0, length_scale))

        chunks = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _gs, _ps, audio in self._pipe(lang)(text, voice=voice, speed=speed):
                if audio is not None:
                    chunks.append(np.asarray(audio, dtype=np.float32))
        if not chunks:
            raise RuntimeError(f"kokoro produced no audio for {lang}/{voice}")
        sf.write(str(out), np.concatenate(chunks), self.SR, subtype="PCM_16")
        return out

    def available_langs(self) -> list[str]:
        return list(KOKORO_LANG)


class PollyBackend(TTSBackend):
    """
    Amazon Polly neural voices. The most natural option available here, and calls are
    network-bound so they parallelise properly (Kokoro serialises internally, which is
    why extra threads never helped it).

    One real gap: Polly has NO Hindi male voice - Aditi, Kajal and Raveena are all
    female, and they are filed under en-IN rather than hi-IN. A local backend covers
    (hi, male) so gender casting still works.
    """
    name = "polly"
    VOICES = {
        ("hi", "female"): ("Kajal", "hi-IN"),
        ("en", "female"): ("Joanna", "en-US"),
        ("en", "male"):   ("Matthew", "en-US"),
    }
    CHARS_PER_SEC = {"hi": 13.27, "en": 18.01}    # measured

    def __init__(self, engine: str = "neural", region: str | None = None,
                 fallback: TTSBackend | None = None):
        import boto3
        self.engine = engine
        self.region = region or os.getenv("AWS_REGION", "us-west-2")
        self._polly = boto3.client("polly", region_name=self.region)
        self._fallback = fallback          # covers (hi, male)

    def available(self) -> bool:
        try:
            self._polly.describe_voices(LanguageCode="en-US")
            return True
        except Exception:
            return False

    def synth(self, text: str, lang: str, out: Path, length_scale: float = 1.0,
              gender: str = "female") -> Path:
        key = (lang, gender)
        if key not in self.VOICES:
            if self._fallback is None:
                key = (lang, "female")     # last resort: any voice for that language
            else:
                return self._fallback.synth(text, lang, out,
                                            length_scale=length_scale, gender=gender)

        voice, lang_code = self.VOICES[key]
        # SSML prosody gives precise duration control, which matters far more for
        # lip-sync than the small quality edge of the generative engine.
        rate = int(round(100 / max(0.5, min(2.0, length_scale))))
        ssml = f'<speak><prosody rate="{rate}%">{_xml_escape(text)}</prosody></speak>'

        try:
            resp = self._polly.synthesize_speech(
                Text=ssml, TextType="ssml", VoiceId=voice, Engine=self.engine,
                LanguageCode=lang_code, OutputFormat="mp3", SampleRate="24000")
        except Exception:
            resp = self._polly.synthesize_speech(          # engine may reject SSML
                Text=text, VoiceId=voice, Engine=self.engine,
                LanguageCode=lang_code, OutputFormat="mp3", SampleRate="24000")

        mp3 = out.with_suffix(".mp3")
        mp3.write_bytes(resp["AudioStream"].read())
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(mp3),
                        "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(out)],
                       check=True)
        mp3.unlink(missing_ok=True)
        return out

    def available_langs(self) -> list[str]:
        return ["hi", "en"]


def _xml_escape(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class SayBackend(TTSBackend):
    """macOS `say`. Dev convenience only — will not exist on a Linux host."""
    name = "say"
    VOICES = {"hi": "Lekha", "en": "Samantha", "ko": "Yuna"}

    def synth(self, text: str, lang: str, out: Path, length_scale: float = 1.0,
              gender: str = "female") -> Path:
        rate = int(180 / max(0.5, length_scale))
        subprocess.run(
            ["say", "-v", self.VOICES.get(lang, "Samantha"), "-r", str(rate),
             "-o", str(out), "--file-format=WAVE", "--data-format=LEI16@22050", text],
            check=True, capture_output=True)
        return out

    def available_langs(self) -> list[str]:
        return list(self.VOICES)


_BACKEND_CACHE: dict = {}


def get_backend(name: str = "auto") -> TTSBackend:
    """Cached: voice models are expensive to load and safe to share across sessions."""
    if name in _BACKEND_CACHE:
        return _BACKEND_CACHE[name]
    b = _build_backend(name)
    _BACKEND_CACHE[name] = b
    return b


def _build_backend(name: str = "auto") -> TTSBackend:
    if name in ("auto", "polly"):
        try:
            local = None
            # Kokoro exists only to cover (hi, male), which Polly has no voice for —
            # but it is torch-backed, and building it costs ~800MB resident whether
            # or not a single Hindi male line is ever spoken. Measured: Polly with
            # this fallback peaked at 1102MB on the 40s clip; without it, Polly is
            # pure network. On a small instance set POLLY_FALLBACK=none and accept
            # that Hindi male lines are voiced by the female Hindi voice.
            if os.getenv("POLLY_FALLBACK", "kokoro").lower() not in ("none", "0", ""):
                try:
                    local = KokoroBackend()
                except Exception:
                    pass
            b = PollyBackend(fallback=local)
            if b.available():
                return b
        except Exception:
            if name == "polly":
                raise
    if name in ("auto", "kokoro"):
        try:
            return KokoroBackend()
        except Exception:
            if name == "kokoro":
                raise
    if name in ("auto", "piper"):
        try:
            backend = PiperBackend()
            if backend.available_langs():
                return backend
        except Exception:
            if name == "piper":
                raise
    if name in ("auto", "say") and shutil.which("say"):
        return SayBackend()
    raise RuntimeError("No usable TTS backend. Install piper voices via ./fetch-voices")


def scale_for(text_seconds_estimate: float, slot: float) -> float:
    """
    Initial length_scale guess. Capped at ~1.0 on purpose: we compress a long line to
    make it fit, but we do not stretch a short one to fill dead air.
    """
    if slot <= 0 or text_seconds_estimate <= 0:
        return 1.0
    return max(MIN_SCALE, min(MAX_SCALE, slot / text_seconds_estimate))
