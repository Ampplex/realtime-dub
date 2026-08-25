"""
Translation stage.

Two backends behind one interface:
  * BedrockTranslator - amazon.nova-lite by default. Benchmarked correct on ko->hi
    and ko->en at ~0.6-0.8s/line. (mistral-large was throttled hard on Hindi in
    us-west-2 and is not viable for realtime; claude-3-haiku needs a model-access grant.)
  * OllamaTranslator  - local fallback so the pipeline still runs offline. Small
    models are unreliable at ko->hi, so this is a degraded mode, not a peer.
"""
from __future__ import annotations

import os
import re
import unicodedata
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

LANG_NAMES = {"hi": "Hindi", "en": "English", "ko": "Korean"}

SYSTEM = (
    "You translate subtitle lines for video dubbing into {target}. "
    "Output ONLY the translation: no quotes, no notes, no romanization, no source "
    "script, no commentary. Keep the translation close to the original length so it "
    "fits the same speaking time. Preserve proper names. Never refuse; if a line is "
    "untranslatable, output it unchanged."
)


# Scripts that must never reach a Hindi or English voice.
HANGUL = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
CJK    = re.compile(r"[\u4e00-\u9fff]")
KANA   = re.compile(r"[\u3040-\u30ff]")
DEVA   = re.compile(r"[\u0900-\u097f]")

EXPECTED = {"hi": DEVA, "en": re.compile(r"[A-Za-z]")}


def leaked(text: str, target: str) -> bool:
    """True if `text` contains script that does not belong in the target language."""
    if HANGUL.search(text) or CJK.search(text) or KANA.search(text):
        return True
    if target == "en" and DEVA.search(text):
        return True
    return False


def strip_foreign(text: str, target: str) -> str:
    """Last resort: drop characters that would be mispronounced, then tidy spacing."""
    out = HANGUL.sub("", text)
    out = CJK.sub("", out)
    out = KANA.sub("", out)
    if target == "en":
        out = DEVA.sub("", out)
    return re.sub(r"\s{2,}", " ", out).strip(" ,.-—·").strip()


def _clean(out: str, fallback: str) -> str:
    out = (out or "").strip()
    out = re.sub(r"^(translation|hindi|english|output)\s*[:\-]\s*", "", out, flags=re.I).strip()
    out = re.sub(r'^["“”\'](.*)["“”\']$', r"\1", out, flags=re.S).strip()
    if "\n" in out:
        out = out.split("\n")[0].strip()
    # Deliberately NOT falling back to the source line: feeding untranslated Korean
    # to a Hindi or English voice is worse than skipping the line entirely.
    return out


class Translator(ABC):
    name = "base"

    @abstractmethod
    def translate(self, text: str, target: str, source: str = "ko") -> str: ...

    def translate_many(self, items: list[tuple[str, str]], workers: int = 4) -> list[str]:
        """items = [(text, target_lang)]. Parallel — the API call is latency-bound,
        so fanning out is what keeps the pipeline ahead of the playhead."""
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda it: self.translate(it[0], it[1]), items))


class BedrockTranslator(Translator):
    name = "bedrock"

    def __init__(self, model_id: str | None = None, region: str | None = None):
        import boto3
        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID", "mistral.mistral-large-3-675b-instruct")
        self.region = region or os.getenv("AWS_REGION", "us-west-2")
        self._rt = boto3.client("bedrock-runtime", region_name=self.region)
        self.leaks = 0          # lines dropped because they could not be cleaned
        self.last_error: str | None = None   # why the most recent call failed

    def available(self) -> bool:
        """
        translate() deliberately never raises — it returns "" so one bad line cannot
        kill a run. That makes "it did not throw" worthless as a health check: with
        no credentials every call failed, this still returned True, and the session
        reported translator="bedrock" while silently producing an empty dub. Require
        actual output.
        """
        try:
            out = self.translate("테스트", "en")
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return False
        if not (out or "").strip():
            self.last_error = self.last_error or "Bedrock returned an empty translation"
            return False
        return True

    def translate(self, text: str, target: str, source: str = "ko") -> str:
        text = (text or "").strip()
        if not text or target == source:
            return text
        out = self._call(text, target)
        if out and not leaked(out, target):
            return out

        # One stricter retry, then sanitise. Never return the Korean source.
        retry = self._call(text, target, strict=True)
        if retry and not leaked(retry, target):
            return retry

        best = retry or out
        cleaned = strip_foreign(best, target) if best else ""
        if cleaned:
            return cleaned
        self.leaks += 1
        return ""        # caller skips synthesis rather than voicing the wrong script

    def _call(self, text: str, target: str, strict: bool = False) -> str:
        sys_prompt = SYSTEM.format(target=LANG_NAMES.get(target, target))
        if strict:
            sys_prompt += (" CRITICAL: your reply must contain ONLY " +
                           LANG_NAMES.get(target, target) +
                           " script. Absolutely no Korean, Chinese or Japanese "
                           "characters, and no source text of any kind.")
        try:
            resp = self._rt.converse(
                modelId=self.model_id,
                system=[{"text": sys_prompt}],
                messages=[{"role": "user", "content": [{"text": text}]}],
                inferenceConfig={"maxTokens": 220, "temperature": 0.0},
            )
            return _clean(resp["output"]["message"]["content"][0]["text"], text)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return ""


class OllamaTranslator(Translator):
    name = "ollama"

    def __init__(self, model: str = "llama3.2:1b", host: str = "http://127.0.0.1:11434"):
        import requests
        self.model, self.host = model, host
        self._s = requests.Session()

    def available(self) -> bool:
        try:
            tags = self._s.get(f"{self.host}/api/tags", timeout=3).json()
            return any(m["name"] == self.model for m in tags.get("models", []))
        except Exception:
            return False

    def translate(self, text: str, target: str, source: str = "ko") -> str:
        text = (text or "").strip()
        if not text or target == source:
            return text
        try:
            r = self._s.post(f"{self.host}/api/generate", json={
                "model": self.model,
                "system": SYSTEM.format(target=LANG_NAMES.get(target, target)),
                "prompt": text, "stream": False,
                "options": {"temperature": 0.1, "num_predict": 220},
            }, timeout=60)
            r.raise_for_status()
            out = _clean(r.json().get("response", ""), text)
            if out and leaked(out, target):
                out = strip_foreign(out, target)
            return out
        except Exception:
            return ""


def get_translator(prefer: str = "auto") -> Translator:
    why = None
    if prefer in ("auto", "bedrock"):
        try:
            t = BedrockTranslator()
            if t.available():
                return t
            why = t.last_error
        except Exception as e:
            why = f"{type(e).__name__}: {e}"
            if prefer == "bedrock":
                raise
    t = OllamaTranslator()
    if t.available():
        return t
    raise RuntimeError(
        (f"Bedrock unusable: {why}\n" if why else "") +
        "No translation backend available.\n"
        "  Bedrock: check AWS creds in .env (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION).\n"
        "  Ollama:  no local model present. `ollama pull qwen2.5:3b` for a usable offline\n"
        "           fallback - note llama3.2:1b was tested and is NOT good enough for ko->hi."
    )
