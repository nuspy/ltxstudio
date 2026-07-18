"""Worker inference Krea 2 — architettura IMMAGINI alternativa (accanto a FLUX/Z-Image/Ernie).

Eseguito nel venv `E:\\AI\\Tools\\lingbot_venv311` (py3.11 + torch 2.11 cu130 + diffusers 0.39 + Qwen3-VL +
FlashAttention-2 sm120 — stesse deps di LingBot) via subprocess da `core/img_gen_job.py`.

Krea 2 è diffusers (`Krea2Pipeline`): text-encoder Qwen3-VL, VAE Qwen-Image, transformer ~13B → su 32GB
serve **fp8** (layerwise casting, come FLUX). Due varianti: Turbo (8 step, cfg 0, mu 1.15) e Raw (52 step,
cfg 3.5). Prompt in TESTO semplice (niente JSON). Legge params JSON (argv[1]); scrive un PNG; stampa OK:<path>.

⚠️ LIVE-TEST PENDENTE: firma esatta di `Krea2Pipeline.__call__` (mu/guidance) e fp8 da validare al 1° run.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_grad_enabled(False)

_VARIANT_DEFAULTS = {                                        # step/cfg/mu per variante (dai docs Krea 2)
    "turbo": {"steps": 8, "guidance": 0.0, "mu": 1.15},
    "raw":   {"steps": 52, "guidance": 3.5, "mu": None},
}


def _resolve_model_dir(p: dict) -> tuple[str, str]:
    """(model_dir, variant). variant ∈ {turbo, raw}."""
    variant = str(p.get("krea2_variant", "turbo")).lower()
    m = p.get("model") or ""
    if m and os.path.isdir(m):
        return m, ("raw" if "raw" in m.lower() else "turbo")
    try:
        from core import weights_catalog as wc
        entries = {e["id"]: e for e in wc.load()}
        eid = "krea2_raw" if variant == "raw" else "krea2_turbo"
        e = entries.get(eid)
        if e and wc.is_present(e):
            return str(wc.dest_for(e)), variant
        if e:
            return str(wc.dest_for(e)), variant
    except Exception as ex:                                   # noqa: BLE001
        print(f"[krea2] risoluzione catalogo fallita: {ex}", flush=True)
    raise RuntimeError("Modello Krea 2 non trovato. Scaricalo dal wizard 'Pesi' (modulo Immagini).")


def _free_vram_mb() -> int:
    try:
        from core import gpu_mem
        return int(gpu_mem.free_vram_mb())
    except Exception:                                        # noqa: BLE001
        return 0


_PIPE_CACHE: dict = {}


def _load_pipeline(model_dir: str, variant: str):
    key = (model_dir, variant)
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]
    from diffusers import DiffusionPipeline
    print(f"[krea2] carico Krea2Pipeline: {model_dir} (variant={variant})", flush=True)
    try:
        from krea2 import Krea2Pipeline as _Pipe                # se il pacchetto è installato
        pipe = _Pipe.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    except Exception:                                          # noqa: BLE001 — fallback custom-code HF
        pipe = DiffusionPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True)

    # fp8: transformer ~13B bf16 (~26GB) non entra residente sui 32GB → layerwise casting fp8_e4m3 (~13GB),
    # come FLUX in inference_server/backends/image_flux. Compute in bf16.
    free = _free_vram_mb()
    try:
        if free < 40000 and hasattr(pipe, "transformer"):
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
            print(f"[krea2] transformer fp8 layerwise (VRAM libera {free}MB)", flush=True)
    except Exception as ex:                                    # noqa: BLE001
        print(f"[krea2] fp8 non applicato ({ex})", flush=True)

    try:
        if free < 30000:
            pipe.enable_model_cpu_offload()
            print("[krea2] model_cpu_offload (VRAM stretta)", flush=True)
        else:
            pipe = pipe.to("cuda")
    except Exception:                                          # noqa: BLE001
        pipe = pipe.to("cuda")
    try:
        if hasattr(pipe, "vae"):
            pipe.vae.enable_tiling()
    except Exception:                                          # noqa: BLE001
        pass
    _PIPE_CACHE[key] = pipe
    return pipe


def _guidance_of(p: dict, default: float) -> float:
    """La GUI emette `cfg`; il worker t2i storicamente leggeva `guidance` → default sempre. Accetta entrambe."""
    for k in ("guidance", "cfg"):
        if p.get(k) is not None:
            return float(p[k])
    return float(default)


def _resolve_lora_paths(p: dict, edit: bool) -> list[tuple[str, float]]:
    """(path, weight) delle LoRA da applicare. In edit include SEMPRE `krea2_identity_edit_v1` @1.0
    (obbligatoria per l'edit d'identità), più le LoRA scelte dall'utente in `p['loras']`."""
    out: list[tuple[str, float]] = []
    try:
        from core import weights_catalog as wc
        entries = {e["id"]: e for e in wc.load()}
        if edit:
            e = entries.get("krea2_identity_edit_v1")
            if e and wc.is_present(e):
                out.append((str(wc.dest_for(e)), 1.0))
    except Exception as ex:                                       # noqa: BLE001
        print(f"[krea2] LoRA identity_edit non risolta ({ex})", flush=True)
    for lo in (p.get("loras") or []):
        path = lo.get("path") or lo.get("name")
        if path and os.path.isfile(path):
            out.append((path, float(lo.get("strength", 1.0))))
    return out


def _apply_loras(pipe, loras: list[tuple[str, float]]) -> None:
    if not loras:
        return
    names, weights = [], []
    for i, (path, w) in enumerate(loras):
        try:
            an = f"krea2_lora_{i}"
            pipe.load_lora_weights(path, adapter_name=an)
            names.append(an); weights.append(w)
            print(f"[krea2] LoRA caricata: {os.path.basename(path)} @{w}", flush=True)
        except Exception as ex:                                   # noqa: BLE001
            print(f"[krea2] LoRA fallita {os.path.basename(path)}: {ex}", flush=True)
    if names:
        try:
            pipe.set_adapters(names, adapter_weights=weights)
        except Exception:                                         # noqa: BLE001
            pass


def _load_images(p: dict):
    """Immagini sorgente per l'edit: `images[]` (multi) o `image_path` (single). PIL RGB."""
    from PIL import Image
    paths = p.get("images") or ([p["image_path"]] if p.get("image_path") else [])
    imgs = []
    for pth in paths[:2]:                                        # il modello gestisce fino a 2 riferimenti
        if pth and os.path.isfile(pth):
            imgs.append(Image.open(pth).convert("RGB"))
    return imgs


def run_edit(p: dict) -> str:
    """Editing per istruzione (single/multi) via Krea2EditPipeline (port nativo di comfyui-krea2edit)."""
    out = p["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    imgs = _load_images(p)
    if not imgs:
        raise RuntimeError("Krea 2 Edit: nessuna immagine sorgente (image_path/images).")
    model_dir, variant = _resolve_model_dir(p)
    pipe = _load_pipeline(model_dir, variant)
    # trasforma la pipeline t2i caricata in quella di edit (stessi componenti, aggiunge edit())
    from workers.krea2_edit_pipeline import Krea2EditPipeline
    from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline as _Base
    if isinstance(pipe, _Base) and not isinstance(pipe, Krea2EditPipeline):
        pipe.__class__ = Krea2EditPipeline
    _apply_loras(pipe, _resolve_lora_paths(p, edit=True))
    d = _VARIANT_DEFAULTS.get(variant, _VARIANT_DEFAULTS["turbo"])

    result = pipe.edit(
        images=imgs,
        prompt=p.get("prompt", ""),
        negative_prompt=(p.get("negative") or ""),
        grounding_px=int(p.get("grounding_px", 768)),
        num_inference_steps=int(p.get("steps") or d["steps"]),
        guidance_scale=_guidance_of(p, d["guidance"]),
        generator=torch.Generator(device=("cuda" if torch.cuda.is_available() else "cpu")).manual_seed(int(p.get("seed", 0))),
    )
    img = result.images[0] if hasattr(result, "images") else result[0]
    img.save(out)
    _sign_and_watermark(out)
    return out


def _sign_and_watermark(out: str) -> str:
    try:                                                          # #107 stego + #106 watermark
        from core import stego
        if stego.enabled():
            stego.sign_file(out)
    except Exception as _se:                                      # noqa: BLE001
        print(f"[stego] {_se}", flush=True)
    try:
        from core import watermark
        out = watermark.apply_to_file(out)
    except Exception as _we:                                      # noqa: BLE001
        print(f"[watermark] {_we}", flush=True)
    return out


def run_generation(p: dict) -> str:
    out = p["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    model_dir, variant = _resolve_model_dir(p)
    pipe = _load_pipeline(model_dir, variant)
    _apply_loras(pipe, _resolve_lora_paths(p, edit=False))        # colma il gap: il t2i ignorava le LoRA
    d = _VARIANT_DEFAULTS.get(variant, _VARIANT_DEFAULTS["turbo"])

    def _m16(v, default):
        v = int(p.get(v, default))
        return max(512, (v // 16) * 16)
    kwargs = dict(
        prompt=p.get("prompt", ""),
        num_inference_steps=int(p.get("steps") or d["steps"]),
        guidance_scale=_guidance_of(p, d["guidance"]),
        width=_m16("width", 1024), height=_m16("height", 1024),
        generator=torch.Generator(device=("cuda" if torch.cuda.is_available() else "cpu")).manual_seed(int(p.get("seed", 0))),
    )
    neg = p.get("negative") or ""
    if neg:
        kwargs["negative_prompt"] = neg
    # NB (validato 17/07): Krea2Pipeline.__call__ NON accetta `mu` — la firma espone `sigmas`.
    # Lo shift-timestep del Turbo (mu 1.15) è già codificato nel FlowMatchEulerDiscreteScheduler del
    # model_index (is_distilled=true). Quindi non passiamo mu: lo scheduler applica lo shift corretto.

    result = pipe(**kwargs)
    img = result.images[0] if hasattr(result, "images") else (result[0] if isinstance(result, (list, tuple)) else result)
    # NaN/immagine nera-degenere? affidiamo la validità all'utente; salviamo il PNG.
    img.save(out)
    out = _sign_and_watermark(out)
    return out


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    is_edit = str(p.get("arch", "")).lower() == "krea2_edit" or str(p.get("mode", "")).lower() == "edit"
    print("OK:" + (run_edit(p) if is_edit else run_generation(p)), flush=True)


if __name__ == "__main__":
    main()
