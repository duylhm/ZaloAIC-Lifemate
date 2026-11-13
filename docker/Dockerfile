FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Build-time offline weight prefetch (override BASE_MODEL / ASR_MODEL via --build-arg)
ARG BASE_MODEL="Qwen/Qwen3-4B"
ARG ASR_MODEL="vinai/PhoWhisper-large"
ENV LIFEMATE_BASE_MODEL="${BASE_MODEL}" \
    LIFEMATE_ASR_MODEL="${ASR_MODEL}"

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${BASE_MODEL}'); snapshot_download('${ASR_MODEL}')"

COPY src /app/src
COPY run_inference.py evaluate.py /app/
COPY docker/run.sh /app/run.sh
# Place fine-tuned LoRA adapter files in ./adapter/ before docker build
COPY adapter /app/adapter

RUN sed -i 's/\r$//' /app/run.sh && chmod +x /app/run.sh

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["/app/run.sh"]
