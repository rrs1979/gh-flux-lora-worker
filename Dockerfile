# Flux-schnell + per-request LoRA serverless worker (GroupHunter Phase 2).
# Thin image: НЕ запекаем базу Flux внутрь — качаем на cold-start с кэшем на
# network volume (/runpod-volume/hf). Образ остаётся ~маленьким (torch/diffusers).
FROM runpod/base:0.6.3-cuda12.1.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

# Python 3.11 — стандарт для runpod/base 0.6.3
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 || true

COPY requirements.txt /requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r /requirements.txt

ADD handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
