"""Router immagini: sceglie il MOTORE (comfy-native vs diffusers) secondo user_prefs.get_engine('image').

- 'comfy'  → workers/img_comfy_native.py (model_management ComfyUI, VRAM dinamica, scala 8→48GB; FLUX.2
             ~11x più veloce del block-swap diffusers). Solo t2i per ora; con FALLBACK automatico a
             diffusers per op non coperte (inpaint/kontext/integra) o in caso di errore → niente rotture.
- 'diffusers' → core.native_driver (inference_server :8189, image_flux.py). Path storico, intatto.

Default 'comfy'. Mantiene ENTRAMBI i percorsi (scelta in GUI/settings).
"""
from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

from core import native_driver, user_prefs

COMFY_PY = r"E:\AI\Comfy_Nexus.bak\ComfyUI\venv\Scripts\python.exe"
REPO = str(Path(__file__).resolve().parent.parent)
OUTPUT_DIR = r"E:\AI\ltxstudio\output"
_DM_ROOT = r"E:\AI\Models\diffusion_models"

# componenti di default per arch (ComfyUI folder-relative): clip(+clip2), type, vae
_ARCH_COMPONENTS = {
    "flux2": {"clip1": "Flux/mistral_3_small_flux2.safetensors", "clip2": "", "clip_type": "flux2",
              "vae": "Flux/flux2-vae.safetensors"},
    "flux":  {"clip1": "clip_l.safetensors", "clip2": "t5xxl_fp16.safetensors", "clip_type": "flux",
              "vae": "ae.safetensors"},
}


def _dm_relative(model: str) -> str:
    if not model:
        return model
    m = model.replace("/", "\\")
    root = _DM_ROOT.rstrip("\\") + "\\"
    if m.lower().startswith(root.lower()):
        m = m[len(root):]
    return m.replace("\\", "/")


def _detect_arch(model_path: str) -> str:
    """Famiglia → arch comfy worker (flux2/flux). Default flux."""
    try:
        from core import model_detect
        fam = (model_detect.get(model_path) or {}).get("family", "")
        return "flux2" if fam == "flux2" else "flux"
    except Exception:                                          # noqa: BLE001
        m = (model_path or "").lower()
        return "flux2" if any(t in m for t in ("flux2", "flux_2", "flux 2")) else "flux"


def _comfy_t2i(params: dict, timeout: int) -> dict:
    """Genera t2i col worker comfy-native (one-shot subprocess). Ritorna {status, output:[png]}."""
    model_abs = params.get("model", "")
    arch = _detect_arch(model_abs)
    comp = _ARCH_COMPONENTS.get(arch, _ARCH_COMPONENTS["flux"])
    p = {
        "arch": arch, "model": _dm_relative(model_abs),
        "clip1": params.get("encoder") or comp["clip1"], "clip2": comp["clip2"],
        "clip_type": comp["clip_type"], "vae": params.get("vae") or comp["vae"],
        "prompt": params.get("prompt", ""), "negative": params.get("negative", ""),
        "width": int(params.get("width", 1024)), "height": int(params.get("height", 1024)),
        "steps": int(params.get("steps", 20)), "guidance": float(params.get("guidance", 3.5)),
        "sampler": params.get("sampler") or "euler", "scheduler": params.get("scheduler") or "simple",
        "seed": int(params.get("seed", 0)),
        "output": params.get("output") or os.path.join(OUTPUT_DIR, f"comfy_{uuid.uuid4().hex[:8]}.png"),
    }
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pj = os.path.join(OUTPUT_DIR, f"_iparams_{uuid.uuid4().hex[:8]}.json")
    Path(pj).write_text(json.dumps(p), encoding="utf-8")
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    # libera la VRAM del server diffusers :8189 (single-GPU: un modello pesante alla volta)
    try:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8189/free", data=b"{}",
                               headers={"Content-Type": "application/json"}), timeout=20).read()
    except Exception:                                          # noqa: BLE001
        pass
    try:
        r = subprocess.run([COMFY_PY, os.path.join(REPO, "workers", "img_comfy_native.py"), pj],
                           cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "timeout comfy t2i"}
    finally:
        try:
            os.remove(pj)
        except OSError:
            pass
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        if line.startswith("OK:"):
            return {"status": "success", "output": [line[3:].strip()]}
    return {"status": "error", "error": out[-1500:]}


def generate(params: dict, timeout: int = 1800) -> dict:
    """Genera un'immagine instradando al motore scelto. t2i → comfy (se selezionato) con FALLBACK a
    diffusers; inpaint/kontext/integra/altri → diffusers (il worker comfy copre per ora solo t2i)."""
    arch = (params.get("task") or params.get("arch") or "flux").lower()
    is_t2i = not params.get("image_path") and arch in ("flux", "flux2", "zimage", "ernie")
    if user_prefs.get_engine("image") == "comfy" and is_t2i:
        res = _comfy_t2i(params, timeout)
        if res.get("status") == "success":
            return res
        print(f"[img_router] comfy fallito → fallback diffusers: {str(res.get('error',''))[:200]}", flush=True)
    return native_driver.run_and_wait("image", params, timeout=timeout)
