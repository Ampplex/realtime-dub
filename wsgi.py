#!/usr/bin/env python3
"""
Gunicorn entrypoint.

`server.main()` is what normally populates CFG and warms the models, and gunicorn
never calls it — so without this shim the app would serve with the module-level
defaults and pay the full model load inside the first user request.

Run with ONE worker. Sessions live in a module-level dict and the pipeline runs on
background threads inside the worker process; a second worker would answer polls
for sessions it has never heard of. Concurrency comes from threads, not workers.
"""
from __future__ import annotations

import os

from server import CFG, app, warm_models


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


CFG.update(
    model_size=os.environ.get("WHISPER_MODEL", "base"),
    buffer=max(2.0, min(10.0, _f("BUFFER_SECONDS", 4.0))),
    tts=os.environ.get("TTS_BACKEND", "polly"),
    translator=os.environ.get("TRANSLATOR", "auto"),
    separate=os.environ.get("SEPARATE", "1") == "1",
    sep_device=os.environ.get("SEP_DEVICE", "cpu"),
)

# Warm-up loads Whisper AND synthesises in every language/gender pair, which pulls
# all four Piper voices into memory at once — measured 538MB peak, over a 512MB
# instance, so the worker was OOM-killed at boot and gunicorn crash-looped. On a
# small box, skip it: the models then load lazily inside the first request, which
# is slower once but survives.
if os.environ.get("WARM_MODELS", "1") == "1":
    warm_models()
else:
    print("  warm-up skipped (WARM_MODELS=0): first request will load models.",
          flush=True)

__all__ = ["app"]
