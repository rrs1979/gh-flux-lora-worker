"""RunPod serverless handler — Flux-schnell + per-request LoRA image generation.

Phase 2 production-LoRA worker for GroupHunter persona faces.

Cold start:
    - Загружает базу FLUX.1-schnell (diffusers FluxPipeline, bf16) один раз
      в global PIPE. База качается из HF (gated → нужен HF_TOKEN env) ИЛИ
      из ungated-зеркала, см. BASE_MODEL.
    - HF cache указывает на /runpod-volume/hf (network volume) если он
      примонтирован, иначе /tmp/hf — чтобы повторные cold-start'ы были быстрее.

Per request input:
    {
        "prompt":     str   (required) — напр. "ohwx woman, portrait, photorealistic"
        "lora_url":   str   (optional, https only) — .safetensors LoRA. Если нет —
                            генерим чистым schnell (для сравнения с/без LoRA).
        "lora_scale": float (default 1.0)
        "width":      int   (default 1024)
        "height":     int   (default 1024)
        "steps":      int   (default 4)   — schnell быстрый, 4 шага достаточно
        "seed":       int   (optional)
    }

Output:
    {"image_base64": str, "content_type": "image/png", "cost": float,
     "lora_applied": bool, "duration_ms": int, "model": "flux-lora"}
    или {"error": str}

Безопасность:
    - lora_url принимается ТОЛЬКО https (no file://, no http://).
    - LoRA кэшируется по sha256(url) в LORA_CACHE_DIR; повторные запросы с тем
      же url не качают заново.
"""

import base64
import hashlib
import io
import os
import time
import urllib.request

import torch

import runpod

# ── Конфигурация путей кэша (network volume если есть) ───────────────────────
VOLUME = "/runpod-volume" if os.path.isdir("/runpod-volume") else "/tmp"
HF_HOME = os.path.join(VOLUME, "hf")
LORA_CACHE_DIR = os.path.join(VOLUME, "loras")
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(LORA_CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# База: schnell. Gated на black-forest-labs → нужен HF_TOKEN.
# Если HF_TOKEN не задан, падаем на ungated-зеркало диффьюзеров.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
BASE_MODEL = os.environ.get("BASE_MODEL", "black-forest-labs/FLUX.1-schnell")
BASE_MODEL_FALLBACK = "YuCollection/FLUX.1-schnell-Diffusers"

# Грубая оценка стоимости генерации (для логов SH). Уточняется по факту GPU-часов.
COST_PER_IMAGE = float(os.environ.get("COST_PER_IMAGE", "0.012"))

PIPE = None  # глобальный пайплайн, инициализируется на первом запросе (cold start)
_LOADED_LORA_KEY = None  # какой LoRA сейчас прикручен к PIPE


def _load_pipeline():
    """Загрузить FluxPipeline один раз. bf16, model_cpu_offload для 24GB GPU."""
    global PIPE
    if PIPE is not None:
        return PIPE

    from diffusers import FluxPipeline

    model_id = BASE_MODEL
    kwargs = {"torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    try:
        pipe = FluxPipeline.from_pretrained(model_id, **kwargs)
    except Exception as exc:  # gated/нет токена → зеркало
        print(f"[init] primary base {model_id} failed: {exc}; trying fallback {BASE_MODEL_FALLBACK}")
        pipe = FluxPipeline.from_pretrained(BASE_MODEL_FALLBACK, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

    # 24GB GPU: cpu offload экономит VRAM при загрузке LoRA. Чуть медленнее, но
    # надёжно влезает в L4/4090/A5000.
    # sequential offload: lowest peak VRAM, reliably fits 24GB (model offload OOM'd на 24GB)
    pipe.enable_sequential_cpu_offload()
    PIPE = pipe
    print(f"[init] pipeline ready: {model_id} (cpu_offload)")
    return PIPE


def _download_lora(lora_url: str) -> str:
    """Скачать LoRA по https с кэшем по sha256(url). Вернуть путь к файлу."""
    if not lora_url.lower().startswith("https://"):
        raise ValueError("lora_url must be https://")
    key = hashlib.sha256(lora_url.encode()).hexdigest()[:24]
    dest = os.path.join(LORA_CACHE_DIR, f"{key}.safetensors")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    print(f"[lora] downloading {lora_url} → {dest}")
    req = urllib.request.Request(lora_url, headers={"User-Agent": "gh-flux-lora-worker"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    if not data:
        raise ValueError("empty LoRA download")
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dest)
    print(f"[lora] cached {len(data)} bytes")
    return dest


def _apply_lora(pipe, lora_path: str, cache_key: str):
    """Прикрутить LoRA к пайплайну (с unload предыдущей, если другая)."""
    global _LOADED_LORA_KEY
    if _LOADED_LORA_KEY == cache_key:
        return  # уже эта LoRA загружена
    if _LOADED_LORA_KEY is not None:
        try:
            pipe.unload_lora_weights()
        except Exception as exc:
            print(f"[lora] unload prev failed (ignored): {exc}")
    directory, weight_name = os.path.split(lora_path)
    pipe.load_lora_weights(directory, weight_name=weight_name)
    _LOADED_LORA_KEY = cache_key
    print(f"[lora] applied {weight_name}")


def handler(job):
    start = time.monotonic()
    inp = job.get("input") or {}

    prompt = inp.get("prompt")
    if not prompt or not isinstance(prompt, str):
        return {"error": "prompt (str) is required"}

    lora_url = inp.get("lora_url")
    lora_scale = float(inp.get("lora_scale", 1.0))
    width = int(inp.get("width", 1024))
    height = int(inp.get("height", 1024))
    steps = int(inp.get("steps", 4))
    seed = inp.get("seed")

    try:
        pipe = _load_pipeline()
    except Exception as exc:
        return {"error": f"pipeline load failed: {type(exc).__name__}: {exc}"}

    lora_applied = False
    if lora_url:
        try:
            lora_path = _download_lora(lora_url)
            cache_key = hashlib.sha256(lora_url.encode()).hexdigest()[:24]
            _apply_lora(pipe, lora_path, cache_key)
            lora_applied = True
        except Exception as exc:
            return {"error": f"lora apply failed: {type(exc).__name__}: {exc}"}

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    # Flux LoRA scale прокидывается через joint_attention_kwargs.
    joint_kwargs = {"scale": lora_scale} if lora_applied else None

    try:
        image = pipe(
            prompt=prompt,
            guidance_scale=0.0,          # schnell timestep-distilled
            num_inference_steps=steps,
            max_sequence_length=256,
            width=width,
            height=height,
            generator=generator,
            joint_attention_kwargs=joint_kwargs,
        ).images[0]
    except Exception as exc:
        return {"error": f"generation failed: {type(exc).__name__}: {exc}"}

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "image_base64": b64,
        "content_type": "image/png",
        "cost": COST_PER_IMAGE,
        "lora_applied": lora_applied,
        "lora_scale": lora_scale if lora_applied else None,
        "model": "flux-lora",
        "duration_ms": int((time.monotonic() - start) * 1000),
    }


runpod.serverless.start({"handler": handler})
