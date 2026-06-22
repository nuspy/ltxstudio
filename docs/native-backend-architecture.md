# Backend nativo (sostituzione ComfyUI) — Architettura Fase 0

Branch: `feature/native-backend`. Obiettivo: sostituire ComfyUI con un backend nostro,
incrementale (strangler) dietro un'astrazione, partendo dalle immagini.

## Principi
1. **Stesso pattern di ComfyUI**: un **InferenceServer** persistente gira in background (processo
   separato, proprio venv torch+diffusers) e riceve job via HTTP; l'app PyQt resta solo UI/builder.
2. **Astrazione `Backend`**: per ogni modulo si sceglie `comfy` o `native` via feature-flag
   (`data/backend_config.json`). Default: tutto `comfy` → zero regressioni finché un modulo non è pronto.
3. **Stessa interfaccia driver**: `native_driver` espone `submit/poll/interrupt/system_stats` con la
   stessa forma di `comfy_driver` → le pipeline/tab cambiano il minimo.
4. **Gestione VRAM nostra**: `ModelManager` con budget, lazy-load, unload LRU, offload diffusers,
   LoRA hot-swap (vedi `docs/comfyui-migration-analysis.md` §F).

## Componenti (questo scaffold)
| File | Ruolo |
|---|---|
| `inference_server/model_manager.py` | Registry modelli + budget VRAM + lazy/LRU + offload + LoRA |
| `inference_server/backends/image_flux.py` | PoC: FLUX/FLUX.2 text-to-image via diffusers sul ModelManager |
| `inference_server/server.py` | HTTP: /submit /history /interrupt /system_stats + coda seriale |
| `core/native_driver.py` | Client app → InferenceServer (API stile comfy_driver) |
| `core/backend.py` | Sceglie comfy/native per-modulo (feature-flag) |
| `scripts/install_inference_server.py` | venv isolato torch cu128 + diffusers |
| `data/backend_config.json` | flag per-modulo (default: comfy) |

## Flusso di un job (nativo)
app → `backend.get("image")` → se native: `native_driver.submit(job)` → InferenceServer mette in coda →
worker chiama l'handler (`image_flux.run`) → `ModelManager.ensure(...)` (carica/scarica) → genera →
salva output → status `success` + path; l'app fa polling come con ComfyUI.

## Roadmap
- **F0 (questo scaffold)**: infrastruttura + PoC immagini FLUX dietro flag `image=native`.
- **F1**: parità immagini (FLUX.1/2, Z-Image `ZImagePipeline`, Ernie `ErnieImagePipeline`, Kontext,
  inpaint/integrate/upscale/restore) + LoRA + encoder per-variante (riuso `core/flux2_encoder.py`).
- **F2**: enhance (SeedVR2/RIFE) + volti (CodeFormer/GFPGAN) + segmentazione (SAM/GroundingDINO).
- **F3**: video LTX-2.3 via pacchetto ufficiale `Lightricks/LTX-2` (t2v/i2v + audio) → elimina lo shim.
- **F4**: extend autoregressivo + ReTake/Talk.
- **F5**: dismissione ComfyUI (3D/TTS restano su ambienti isolati).

Test di parità per ogni modulo prima di spostare il flag su `native`.
