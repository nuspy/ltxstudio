"""Worker inference Krea Realtime 14B — architettura VIDEO alternativa (autoregressiva, accanto a LTX/LingBot).

Eseguito nel venv `E:\\AI\\Tools\\krea_rt_venv` (py3.11 + torch 2.11 cu130 + diffusers 0.36 MODULAR + Wan deps +
FlashAttention-2 sm120) via subprocess da `core/ltx_video_job.py` (arch="krea_rt").

Krea Realtime è diffusers **modular** (`WanModularPipeline` + `WanRTBlocks`): transformer krea (~14B) +
componenti Wan 2.1 14B (VAE AutoencoderKLWan, TE UMT5, scheduler UniPC). **Distillato/autoregressivo
(Self-Forcing)**: genera a blocchi (`num_frames_per_block=3`) con KV-cache rolling (`kv_cache_num_frames`),
4 step. → **long-video ed extend NATIVI** (num_blocks) senza chunking. Su 32GB serve **fp8 + kv-cache ridotta**.

⚠️ Licenza: usare SOLO la pipeline diffusers Apache-2.0 di HF; MAI il server GitHub CC-BY-NC-SA.
⚠️ LIVE-TEST PENDENTE: l'API modular esatta (caricamento WanModularPipeline con componenti Wan, chiamata
autoregressiva, nomi arg) va confermata al 1° run — punti marcati # VALIDATE (prassi come LingBot).
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_grad_enabled(False)

NEG_DEFAULT = "worst quality, blurry, jittery, distorted, low quality, deformed, static"


def _free_vram_mb() -> int:
    try:
        from core import gpu_mem
        return int(gpu_mem.free_vram_mb())
    except Exception:                                        # noqa: BLE001
        return 0


def _resolve_dirs(p: dict) -> tuple[str, str]:
    """(krea_rt_dir, wan_base_dir) dal catalogo pesi."""
    from core import weights_catalog as wc
    ent = {e["id"]: e for e in wc.load()}
    def d(eid):
        e = ent.get(eid)
        return str(wc.dest_for(e)) if e else ""
    krea = p.get("model") if (p.get("model") and os.path.isdir(str(p.get("model")))) else d("krea_rt")
    wan = d("wan21_14b_diffusers")
    if not krea or not os.path.isdir(krea):
        raise RuntimeError("Modello Krea Realtime non trovato. Scaricalo dal wizard 'Pesi' (modulo Video) "
                           "insieme alla base Wan 2.1 14B Diffusers.")
    return krea, wan


_PIPE = {}


def _load_pipeline(krea_dir: str, wan_dir: str, kv_frames: int):
    key = (krea_dir, kv_frames)
    if key in _PIPE:
        return _PIPE[key]
    # VALIDATE: caricamento della pipeline modular Wan-RT. La strada canonica diffusers 0.36:
    #   from diffusers import ModularPipeline ; pipe = ModularPipeline.from_pretrained(krea_dir, ...)
    # Il modular_model_index punta ai componenti Wan (repo separato) → si passa il base Wan locale.
    from diffusers import ModularPipeline
    print(f"[krea_rt] carico WanModularPipeline: {krea_dir} (base Wan={wan_dir}, kv={kv_frames})", flush=True)
    pipe = ModularPipeline.from_pretrained(krea_dir, trust_remote_code=True)
    # componenti Wan dal base locale (VAE/TE/tokenizer/scheduler) se il modular li richiede da repo
    try:
        if wan_dir and os.path.isdir(wan_dir) and hasattr(pipe, "load_components"):
            pipe.load_components(wan_dir)                    # VALIDATE: nome metodo modular per i componenti
    except Exception as ex:                                  # noqa: BLE001
        print(f"[krea_rt] componenti Wan: {ex}", flush=True)

    free = _free_vram_mb()
    try:                                                    # fp8 sul transformer 14B (~28GB bf16 → ~14GB)
        tr = getattr(pipe, "transformer", None)
        if tr is not None and free < 44000:
            tr.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
            print(f"[krea_rt] transformer fp8 (VRAM libera {free}MB)", flush=True)
    except Exception as ex:                                 # noqa: BLE001
        print(f"[krea_rt] fp8 non applicato ({ex})", flush=True)
    try:
        pipe = pipe.to("cuda")
    except Exception:                                       # noqa: BLE001
        try:
            pipe.enable_model_cpu_offload()
        except Exception:                                   # noqa: BLE001
            pass
    _PIPE[key] = pipe
    return pipe


def run_generation(p: dict) -> str:
    out = p["output"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fps = int(p.get("fps", 24))
    krea_dir, wan_dir = _resolve_dirs(p)

    # KV-cache ridotta sui 32GB (regola VRAM): meno frame di contesto = meno memoria (contesto più corto)
    free = _free_vram_mb()
    kv_frames = int(p.get("kv_cache_num_frames", 3 if free >= 40000 else 2))

    # long-video/extend NATIVO: num_blocks dalla durata (num_frames_per_block=3 di default)
    fpb = int(p.get("num_frames_per_block", 3))
    total = int(p.get("num_frames", 121))
    num_blocks = max(1, round(total / (fpb * 4)))            # VALIDATE: relazione blocchi↔frame del VAE Wan

    pipe = _load_pipeline(krea_dir, wan_dir, kv_frames)

    def _m16(k, dflt):
        return max(256, (int(p.get(k, dflt)) // 16) * 16)
    kwargs = dict(
        prompt=p.get("prompt", ""),
        negative_prompt=p.get("negative", NEG_DEFAULT),
        height=_m16("height", 480), width=_m16("width", 832),
        num_inference_steps=int(p.get("steps", 4)),         # distillato: ~4 step
        guidance_scale=float(p.get("guidance", 1.0)),
        num_blocks=num_blocks, num_frames_per_block=fpb, kv_cache_num_frames=kv_frames,
        generator=torch.Generator(device="cpu").manual_seed(int(p.get("seed", 7))),
    )
    if p.get("image_path") and os.path.exists(str(p["image_path"])):
        from PIL import Image
        kwargs["image"] = Image.open(p["image_path"]).convert("RGB")   # i2v/v2v se supportato — VALIDATE

    print(f"[krea_rt] t2v autoregressivo: {kwargs['width']}x{kwargs['height']} ~{total}f "
          f"({num_blocks} blocchi × {fpb}, kv={kv_frames}, {kwargs['num_inference_steps']} step)", flush=True)
    result = pipe(**kwargs)
    frames = getattr(result, "frames", result)
    if isinstance(frames, (list, tuple)) and frames and not hasattr(frames[0], "save"):
        frames = frames[0]

    # NaN-guard + encode
    import numpy as np, imageio
    outs = []
    for f in frames:
        a = np.asarray(f)
        if a.dtype.kind == "f":
            if np.isnan(a).any() or np.isinf(a).any():
                raise RuntimeError("Frame NaN/inf: pesi Krea Realtime corrotti/incompleti.")
            a = (a * 255).clip(0, 255).astype("uint8") if a.max() <= 1.01 else a.clip(0, 255).astype("uint8")
        if a.ndim == 3 and a.shape[0] in (1, 3):
            a = a.transpose(1, 2, 0)
        outs.append(a)
    imageio.mimsave(out, outs, fps=fps, codec="libx264", quality=8, macro_block_size=16)

    try:
        from core import stego
        if stego.enabled():
            stego.sign_file(out)
    except Exception as _se:                                # noqa: BLE001
        print(f"[stego] {_se}", flush=True)
    try:
        from core import watermark
        out = watermark.apply_to_file(out)
    except Exception as _we:                                # noqa: BLE001
        print(f"[watermark] {_we}", flush=True)
    return out


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)
    print("OK:" + run_generation(p), flush=True)


if __name__ == "__main__":
    main()
