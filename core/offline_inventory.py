"""Inventario modelli OFFLINE (scan filesystem) — sostituisce ComfyUI object_info/models_in.

Per il no-ComfyUI: quando ComfyUI è assente, la GUI popola i dropdown leggendo direttamente
`E:\\AI\\Models\\*`. Espone la STESSA interfaccia che la GUI usa dal driver ComfyUI:
`models_in(category)`, `object_info()` (formato v1: {cls:{input:{required:{inp:[[opzioni]]}}}}),
`gpu_vram_gb()`, `is_up()`. Path relativi alla cartella categoria (come ComfyUI), separatore '\\'.
"""
from __future__ import annotations

import os
from pathlib import Path

MODELS_ROOT = Path(os.environ.get("AI_MODELS_DIR", r"E:\AI\Models"))

# categoria → (sottocartelle, estensioni)
_CATS = {
    "diffusion_models": (["diffusion_models"], (".safetensors", ".gguf", ".sft")),
    "checkpoints":      (["Checkpoints"], (".safetensors", ".ckpt", ".gguf")),
    "loras":            (["LoRA", "loras"], (".safetensors",)),
    "vae":              (["VAE"], (".safetensors", ".pt")),
    "text_encoders":    (["TextEncoders", "text_encoders", "clip"], (".safetensors",)),
    "upscale_models":   (["Upscale", "upscale_models"], (".pth", ".safetensors")),
    "sams":             (["Segmentation", "sams"], (".pth", ".safetensors", ".pt")),
    "facerestore":      (["facerestore_models", "FaceRestore"], (".pth", ".safetensors", ".onnx")),
}
# clip-type accettati dai loader (per i menu encoder)
_CLIP_TYPES = ["flux", "flux2", "ltxv", "sd3", "stable_diffusion", "sdxl", "qwen_image", "lumina2"]


# sottocartelle che NON contengono modelli del tipo della categoria (es. Upscale\LTX = upscaler LATENTI di LTX,
# non caricabili da spandrel: non devono finire nel picker dell'upscaler immagini)
_SKIP_SUBDIRS = {"upscale_models": {"ltx"}}


# Copie di CACHE generate dall'app accanto al modello originale: non sono modelli e non vanno in nessun menu.
# `<file>.safetensors.normalized.safetensors` = FLUX.1 con prefisso/fp8-E5M2 riscritto da
# inference_server/backends/image_flux.py::_normalize_flux_prefix (si rigenera da sola dall'originale).
CACHE_SUFFIXES = (".safetensors.normalized.safetensors",)


def is_cache_copy(name: str) -> bool:
    return str(name).lower().endswith(CACHE_SUFFIXES)


def list_models(category: str) -> list[str]:
    subs, exts = _CATS.get(category, ([], ()))
    skip = _SKIP_SUBDIRS.get(category, set())
    out: list[str] = []
    for sub in subs:
        base = MODELS_ROOT / sub
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if f.is_file() and f.suffix.lower() in exts and not is_cache_copy(f.name):
                if skip and any(part.lower() in skip for part in f.relative_to(base).parts[:-1]):
                    continue
                out.append(str(f.relative_to(base)).replace("/", "\\"))
    # dedup mantenendo l'ordine
    seen: set[str] = set()
    return [x for x in sorted(out) if not (x in seen or seen.add(x))]


def models_in(category: str) -> list[str]:
    """Alias compatibile con comfy_driver.models_in()."""
    return list_models(category)


def _combo(options: list[str]) -> dict:
    return {"input": {"required": {}}}  # placeholder, riempito sotto


def object_info(_node: str | None = None) -> dict:
    """object_info COMPATIBILE (formato v1: spec = [[opzioni]]) per i nodi che la GUI legge."""
    diff = list_models("diffusion_models")
    gguf = [m for m in diff if m.lower().endswith(".gguf")]
    tes = list_models("text_encoders")
    vaes = list_models("vae")
    sams = list_models("sams")
    ups = list_models("upscale_models")
    face = list_models("facerestore")

    def node(inp_map: dict) -> dict:
        return {"input": {"required": {k: [v] for k, v in inp_map.items()}}}

    oi = {
        "UNETLoader": node({"unet_name": diff}),
        "UnetLoaderGGUF": node({"unet_name": gguf}),
        "SAMLoader": node({"model_name": sams}),
        "UpscaleModelLoader": node({"model_name": ups}),
        "FaceRestoreModelLoader": node({"model_name": face}),
        "CLIPLoader": node({"clip_name": tes, "type": _CLIP_TYPES}),
        "DualCLIPLoader": node({"clip_name1": tes, "clip_name2": tes}),
        "VAELoader": node({"vae_name": vaes}),
    }
    if _node:                                                   # comfy_driver.object_info(node) → singolo
        return {_node: oi.get(_node, {})}
    return oi


def gpu_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    except Exception:                                          # noqa: BLE001
        pass
    return 0.0


def is_up() -> bool:
    return MODELS_ROOT.exists()
