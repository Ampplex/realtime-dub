#!/usr/bin/env python3
"""Realtime dubbing web app. Local by default: ./run"""
from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from dubber.pipeline import DubSession

BASE = Path(__file__).resolve().parent
WORK = BASE / "work"
UPLOADS = WORK / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


@app.after_request
def _no_cache(resp):
    """Never let the browser hold on to an old player.js: a cached bundle makes a
    deployed fix look like it did nothing."""
    if request.path.startswith("/static/") or request.path == "/":
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
    return resp
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 ** 3       # 2 GB uploads

# Local-path loading is a dev affordance; hosted deployments accept uploads only.
ALLOW_LOCAL_PATH = os.environ.get("ALLOW_LOCAL_PATH", "0") == "1"

SESSIONS: dict[str, DubSession] = {}
CFG = {"model_size": "base", "buffer": 8.0, "tts": "auto", "translator": "auto",
       "separate": True, "sep_device": "cpu"}


@app.get("/")
def index():
    return render_template("index.html", buffer_seconds=CFG["buffer"],
                           allow_local_path=ALLOW_LOCAL_PATH)


@app.post("/api/session")
def create_session():
    sid = uuid.uuid4().hex[:12]
    sdir = WORK / f"session_{sid}"
    sdir.mkdir(parents=True, exist_ok=True)

    if "file" in request.files:
        f = request.files["file"]
        video = UPLOADS / f"{sid}_{Path(f.filename or 'video.mp4').name}"
        f.save(video)
    elif ALLOW_LOCAL_PATH:
        # Reading an arbitrary server path is a local-dev convenience ONLY. Left on in
        # a hosted deployment it is arbitrary file disclosure: the file is handed
        # straight back by /api/session/<sid>/video, so `path=/proc/self/environ`
        # would surrender the AWS credentials. Off unless explicitly enabled.
        raw = (request.form.get("path") or (request.json or {}).get("path", "")).strip()
        video = Path(raw).expanduser()
        if not video.exists():
            return jsonify({"error": f"No such file: {video}"}), 400
    else:
        return jsonify({"error": "Upload a video file — this server does not read "
                                 "local paths."}), 400

    source_lang = request.form.get("source_lang") or "ko"
    try:
        session = DubSession(
            video=video, work=sdir, langs=("hi", "en"), source_lang=source_lang,
            model_size=CFG["model_size"], buffer_seconds=CFG["buffer"],
            tts_backend=CFG["tts"], translator=CFG["translator"],
            separate=CFG["separate"], sep_device=CFG["sep_device"])
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    SESSIONS[sid] = session
    session.start()
    return jsonify({
        "id": sid,
        "video_url": f"/api/session/{sid}/video",
        "buffer_seconds": session.buffer_seconds,
        "tts": session.tts.name,
        "translator": session.translator.name,
    })


def _session(sid: str) -> DubSession:
    s = SESSIONS.get(sid)
    if not s:
        abort(404)
    return s


@app.get("/api/session/<sid>/manifest")
def manifest(sid: str):
    lang = request.args.get("lang", "hi")
    return jsonify(_session(sid).manifest(lang))


@app.get("/api/session/<sid>/chunk/<lang>/<int:idx>")
def chunk(sid: str, lang: str, idx: int):
    path = _session(sid).chunk_path(lang, idx)
    if not path or not path.exists():
        abort(404)
    return send_file(path, mimetype="audio/wav", conditional=True)


@app.get("/api/session/<sid>/bed/<int:idx>")
def bed(sid: str, idx: int):
    path = _session(sid).bed_path(idx)
    if not path or not path.exists():
        abort(404)
    return send_file(path, mimetype="audio/wav", conditional=True)


@app.get("/api/session/<sid>/video")
def video(sid: str):
    return send_file(_session(sid).video, conditional=True)


def warm_models() -> None:
    """Load the heavy models once at boot so the first session does not pay for it."""
    import threading

    def _warm():
        try:
            from pathlib import Path
            from dubber.asr import WhisperASR
            from dubber.tts import get_backend
            from dubber.separate import _get_model
            import tempfile, time
            t0 = time.perf_counter()
            WhisperASR(model_size=CFG["model_size"]).model
            if CFG["separate"]:
                _get_model()

            # Actually SYNTHESISE in every language. Constructing the backend only
            # imports it; the per-language pipeline (spaCy for en, misaki G2P for hi)
            # is built lazily on first use, and that cost otherwise lands inside the
            # first user session as ~45s of silence.
            b = get_backend(CFG["tts"])
            for lang, probe in (("hi", "नमस्ते"), ("en", "Hello")):
                for gender in ("female", "male"):
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                            b.synth(probe, lang, Path(f.name), gender=gender)
                    except Exception:
                        pass
            print(f"  models warm in {time.perf_counter()-t0:.1f}s "
                  f"(whisper + demucs + {b.name} voices).", flush=True)
        except Exception as e:
            print(f"  warm-up skipped: {type(e).__name__}: {e}", flush=True)

    threading.Thread(target=_warm, daemon=True).start()


def main() -> None:
    p = argparse.ArgumentParser(description="Realtime Korean->Hindi/English dubbing")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8500)))
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--model", default="base",
                   help="whisper size: tiny/base/small (base is fastest AND best-segmented)")
    p.add_argument("--buffer", type=float, default=4.0,
                   help="seconds of dub to build before playback may start (2-10). "
                        "4 is safe because throughput exceeds realtime, so the "
                        "buffer grows during playback rather than draining.")
    p.add_argument("--tts", default="polly",
                   choices=["auto", "polly", "kokoro", "piper", "say"],
                   help="kokoro = most natural (auto picks it); piper = ~6x faster")
    p.add_argument("--translator", default="auto", choices=["auto", "bedrock", "ollama"])
    p.add_argument("--no-separate", action="store_true",
                   help="skip demucs; duck the original instead of removing the voice")
    p.add_argument("--sep-device", default="cpu", choices=["cpu", "mps"],
                   help="cpu is faster than mps for this workload")
    a = p.parse_args()

    CFG.update(model_size=a.model, buffer=max(2.0, min(10.0, a.buffer)),
               tts=a.tts, translator=a.translator,
               separate=not a.no_separate, sep_device=a.sep_device)
    warm_models()
    print(f"\n  realtime-dub  ->  http://{a.host}:{a.port}")
    print(f"  whisper={a.model}  buffer={CFG['buffer']}s  tts={a.tts}  mt={a.translator}"
          f"  separate={CFG['separate']}\n")
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()
