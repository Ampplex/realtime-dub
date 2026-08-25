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

warm_models()

__all__ = ["app"]
