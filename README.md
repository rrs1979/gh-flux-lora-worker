# gh-flux-lora-worker

RunPod **serverless** worker для GroupHunter Phase 2: генерация изображений
**FLUX.1-schnell + per-request LoRA** (лица персон).

## Что делает

- Cold start: грузит базу `FLUX.1-schnell` (bf16, diffusers) один раз.
- Каждый запрос: качает LoRA по `lora_url` (https, кэш по hash на network volume),
  прикручивает её к пайплайну со `scale`, генерит картинку, возвращает base64 PNG.

## Input

```json
{
  "input": {
    "prompt": "ohwx woman, portrait, photorealistic",
    "lora_url": "https://huggingface.co/rrs1979/gh-persona-loras/resolve/main/ohwx_face_lora.safetensors",
    "lora_scale": 1.0,
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "seed": 12345
  }
}
```

`lora_url` опционален (без него — чистый schnell, для сравнения). Только `https://`.

## Output

```json
{
  "image_base64": "<...>",
  "content_type": "image/png",
  "cost": 0.012,
  "lora_applied": true,
  "model": "flux-lora",
  "duration_ms": 9000
}
```

## Деплой

- **RunPod GitHub integration**: New Endpoint → GitHub Repo → этот репо →
  Dockerfile path `Dockerfile`, GPU 24GB (L4/4090/A5000), min workers 0
  (scale-to-zero), max 1-2.
- Env vars: `HF_TOKEN` (FLUX gated), опц. `BASE_MODEL`, `COST_PER_IMAGE`.
- Рекомендуется **network volume** (`/runpod-volume`) для кэша базы/LoRA →
  быстрые повторные cold-start'ы.

## GPU / VRAM

`enable_model_cpu_offload()` → влезает в 24GB. Flux-schnell, 4 шага → быстро.
