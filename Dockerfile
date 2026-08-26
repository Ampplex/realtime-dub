# CPU-only image. Every model here runs on CPU, so there is no CUDA base to carry.
FROM python:3.10-slim

# ffmpeg is not optional: the pipeline shells out to ffmpeg/ffprobe for extraction,
# resampling and the atempo duration fit.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Separation (demucs) needs torch, which is ~500MB installed and cannot fit on a
# small instance. Default build is the lean one; pass --build-arg WITH_SEPARATION=1
# for the full pipeline on a box with 2GB+.
ARG WITH_SEPARATION=0

COPY requirements.txt requirements-full.txt ./
RUN if [ "$WITH_SEPARATION" = "1" ]; then \
        pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
            -r requirements-full.txt ; \
    else \
        pip install --no-cache-dir -r requirements.txt ; \
    fi

COPY . .

# Bake the weights in. Downloading them on first request instead would make the
# first user wait minutes, and a restarted container would pay it again.
RUN ./fetch-voices \
    && python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8'); WhisperModel('tiny', device='cpu', compute_type='int8')" \
    && if [ "$WITH_SEPARATION" = "1" ]; then \
         python -c "from demucs.pretrained import get_model; get_model('htdemucs')" ; \
       fi

# Defaults sized for a 512MB instance. Measured peak RSS on the 40s clip:
#   base  + unbounded voice cache = 538MB  (OOM on 512MB)
#   base  + PIPER_VOICE_CACHE=1   = 847MB  (worse — base's decode arena is the cost,
#                                           not its weights)
#   tiny  + PIPER_VOICE_CACHE=1   = 293MB  (fits)
# On a 2GB+ host raise these: WHISPER_MODEL=base, PIPER_VOICE_CACHE=4, and for the
# full pipeline SEPARATE=1 + TTS_BACKEND=polly with --build-arg WITH_SEPARATION=1.
ENV PORT=8500 \
    HOST=0.0.0.0 \
    TTS_BACKEND=piper \
    SEPARATE=0 \
    WHISPER_MODEL=tiny \
    PIPER_VOICE_CACHE=1 \
    BUFFER_SECONDS=4 \
    ALLOW_LOCAL_PATH=0 \
    WARM_MODELS=0 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2

EXPOSE 8500

# ONE worker on purpose — see wsgi.py. Threads carry the concurrency; the long
# timeout covers large uploads and video range requests.
CMD gunicorn --workers 1 --threads 4 --timeout 600 \
    --bind "0.0.0.0:${PORT}" --access-logfile - wsgi:app
