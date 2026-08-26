#!/usr/bin/env python3
"""
Hugging Face Spaces entrypoint.

Spaces' free tier only offers the Gradio SDK — the Docker SDK is paid — so the
app cannot be run from its Dockerfile here. This runs the same Flask app instead:
the code is fetched from the public GitHub repo at boot and served through ASGI,
so the player, the API and the pipeline are byte-for-byte what the Dockerfile
would have run.

`fetch-voices` and the Whisper weights are pulled on first boot rather than baked,
because there is no build step to bake them into.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = "https://codeload.github.com/Ampplex/realtime-dub/tar.gz/refs/heads/main"
HERE = Path(__file__).resolve().parent
SRC = HERE / "src"


def fetch_source() -> Path:
    """Download the repo tarball once and strip its top-level directory."""
    marker = SRC / "server.py"
    if marker.exists():
        return SRC
    print("fetching source from GitHub…", flush=True)
    SRC.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(REPO, timeout=120) as r:
        blob = r.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        root = tf.getnames()[0].split("/")[0]
        for m in tf.getmembers():
            rel = m.name[len(root):].lstrip("/")
            if not rel:
                continue
            m.name = rel
            tf.extract(m, SRC)          # nosec - our own repo, fixed URL
    print("source ready.", flush=True)
    return SRC


def fetch_voices(src: Path) -> None:
    """Piper voices are ~250MB and there is no build step here to bake them."""
    if list((src / "voices").glob("*.onnx")):
        return
    script = src / "fetch-voices"
    script.chmod(0o755)
    print("downloading Piper voices…", flush=True)
    subprocess.run([str(script)], cwd=str(src), check=False)


src = fetch_source()
fetch_voices(src)
sys.path.insert(0, str(src))

# Work inside the writable Space filesystem, and keep sessions off the repo copy.
os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("ALLOW_LOCAL_PATH", "0")
os.environ.setdefault("WARM_MODELS", "0")      # lazy: boot must not hit the healthcheck timeout
os.chdir(src)

from server import CFG, app as flask_app  # noqa: E402

CFG.update(
    model_size=os.environ.get("WHISPER_MODEL", "base"),
    buffer=float(os.environ.get("BUFFER_SECONDS", "4")),
    tts=os.environ.get("TTS_BACKEND", "piper"),
    translator=os.environ.get("TRANSLATOR", "auto"),
    separate=os.environ.get("SEPARATE", "0") == "1",
    sep_device="cpu",
)

# Flask is WSGI; Spaces serves ASGI. Wrap rather than run Flask's dev server, which
# is single-process and would serialise uploads behind the dubbing threads.
from fastapi import FastAPI                      # noqa: E402
from fastapi.middleware.wsgi import WSGIMiddleware  # noqa: E402

api = FastAPI()
api.mount("/", WSGIMiddleware(flask_app))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
