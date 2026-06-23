"""Worker inference LTX-2.3 — pacchetto UFFICIALE Lightricks (github.com/Lightricks/LTX-2).

Eseguito nel venv dedicato `E:\\AI\\Tools\\ltx2_venv` (torch ~2.7) via subprocess dal backend.
ZERO ComfyUI, ZERO diffusers: usa ltx-core + ltx-pipelines col checkpoint 2.3 LOCALE.
Legge i parametri da un file JSON (argv[1]), genera video+audio, salva mp4.

Pesi (tutti già locali):
- checkpoint  ltx-2.3-22b-dev-fp8.safetensors
- upsampler   ltx-2.3-spatial-upscaler-x2-1.1.safetensors
- gemma       cartella HF gemma3
"""
import json
import os
import sys

import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps, LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.media_io import encode_video

# Fix meta-tensor: il meta-device loading di ltx-core + l'ottimizzazione SDPA-mask di transformers
# fa `.item()` su un tensore 'meta' (crash su qualsiasi torch recente). Disabilitando lo skip,
# la causal mask viene costruita normalmente (un filo più lavoro, nessun crash).
try:
    import transformers.masking_utils as _mu
    _mu._ignore_causal_mask_sdpa = lambda *a, **k: False
except Exception:  # noqa: BLE001
    pass


def _materialize_meta(model, device):
    """ltx-core costruisce su 'meta' e carica i pesi con assign=True, ma i BUFFER calcolati
    (rotary inv_freq, embed_scale) non sono nel checkpoint Gemma di ComfyUI → restano su meta.
    Qui li ricreiamo sul device reale (inv_freq via rope_init_fn)."""
    import torch
    cfg = getattr(model, "config", None)
    txt = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    hidden = getattr(txt, "hidden_size", None)
    for mod in model.modules():
        for bn, buf in list(getattr(mod, "_buffers", {}).items()):
            if buf is None or not getattr(buf, "is_meta", False):
                continue
            if bn in ("inv_freq", "original_inv_freq") and hasattr(mod, "rope_init_fn") and hasattr(mod, "config"):
                inv, scaling = mod.rope_init_fn(mod.config, device)
                mod.register_buffer("inv_freq", inv.to(device=device), persistent=False)
                if hasattr(mod, "attention_scaling"):
                    mod.attention_scaling = scaling
                if hasattr(mod, "original_inv_freq"):
                    mod.original_inv_freq = mod.inv_freq
            elif bn == "embed_scale" and hidden:
                mod.register_buffer("embed_scale", torch.tensor(float(hidden) ** 0.5, device=device,
                                                                dtype=torch.float32), persistent=False)
            else:
                mod.register_buffer(bn, torch.zeros(buf.shape, dtype=buf.dtype, device=device), persistent=False)


try:
    import ltx_core.loader.single_gpu_model_builder as _sgmb
    _orig_build = _sgmb.SingleGPUModelBuilder.build

    def _patched_build(self, device=None, dtype=None, **kw):
        import torch
        dev = torch.device("cuda") if device is None else device
        m = _orig_build(self, device=device, dtype=dtype, **kw)
        metas = [n for n, b in m.named_buffers() if getattr(b, "is_meta", False)]
        metap = [n for n, p in m.named_parameters() if getattr(p, "is_meta", False)]
        print(f"[meta-fix] build {type(m).__name__}: {len(metas)} buffer meta, {len(metap)} param meta", flush=True)
        print("[meta-fix] ATTESI(primi):", metap[:6], flush=True)
        print("[meta-fix] ATTESI(text):", [n for n in metap if 'language_model' in n][:5], flush=True)
        print("[meta-fix] ATTESI(altro):", [n for n in metap if 'vision' not in n and 'language_model' not in n][:6], flush=True)
        try:
            _materialize_meta(m, dev)
            left = [n for n, b in m.named_buffers() if getattr(b, "is_meta", False)]
            print(f"[meta-fix] dopo fix: {len(left)} buffer meta rimasti {left[:3]}", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[meta-fix] warn: {_e}", flush=True)
        return m

    _sgmb.SingleGPUModelBuilder.build = _patched_build
except Exception:  # noqa: BLE001
    pass

CHECKPOINT = r"E:\AI\Models\diffusion_models\LTX\ltx-2.3-22b-dev-fp8.safetensors"
UPSAMPLER = r"E:\AI\Models\latent_upscale_models\ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA = r"E:\AI\Models\TextEncoders\gemma-3-ltx"              # Gemma-3 12B UFFICIALE Lightricks (non-gated, formato ltx-core)
LORA_DIR = r"E:\AI\Models\LoRA"


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as fh:
        p = json.load(fh)

    # LoRA (formato ComfyUI → renaming map ufficiale). path relativo risolto su LORA_DIR.
    loras = []
    for lo in p.get("loras", []):
        path = lo["name"]
        if not os.path.isabs(path):
            path = os.path.join(LORA_DIR, path)
        loras.append(LoraPathStrengthAndSDOps(path, float(lo.get("strength", 1.0)),
                                              LTXV_LORA_COMFY_RENAMING_MAP))

    # MODELLO scelto dalla GUI (checkpoint): la PRECISIONE la determina il modello stesso.
    import time
    ckpt = p.get("model") or CHECKPOINT
    if not os.path.isabs(ckpt):
        for base in (r"E:\AI\Models\diffusion_models\LTX", r"E:\AI\Models\Checkpoints\LTX"):
            cand = os.path.join(base, ckpt)
            if os.path.exists(cand):
                ckpt = cand
                break
    # quant: dedotta dal nome del modello (fp8 → scaled_mm, bf16 → cast); override opz. via param
    qmode = p.get("quant") or ("fp8_scaled_mm" if "fp8" in os.path.basename(ckpt).lower() else "fp8_cast")
    quant = None
    if qmode == "fp8_scaled_mm":
        from ltx_core.quantization import fp8_scaled_mm
        quant = fp8_scaled_mm.build_policy(ckpt)
    elif qmode == "fp8_cast":
        from ltx_core.quantization import fp8_cast
        quant = fp8_cast.build_policy(ckpt)
    t0 = time.time()
    pipe = TI2VidTwoStagesPipeline(
        checkpoint_path=ckpt, distilled_lora=[], spatial_upsampler_path=UPSAMPLER,
        gemma_root=GEMMA, loras=loras, quantization=quant)
    print(f"[load] {time.time() - t0:.0f}s (model={os.path.basename(ckpt)} quant={qmode})", flush=True)

    vg = MultiModalGuiderParams(cfg_scale=float(p.get("guidance", 3.0)), stg_scale=1.0,
                                rescale_scale=0.7, modality_scale=3.0, stg_blocks=[29])
    ag = MultiModalGuiderParams(cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
                                modality_scale=3.0, stg_blocks=[29])

    images = []
    if p.get("image_path"):                                  # i2v: condiziona sul primo frame
        images = [ImageConditioningInput(p["image_path"], 0, 1.0, 33)]

    nf = int(p.get("num_frames", 121))
    fps = float(p.get("fps", 25.0))
    tg = time.time()
    video, audio = pipe(
        prompt=p.get("prompt", ""),
        negative_prompt=p.get("negative", "worst quality, blurry"),
        seed=int(p.get("seed", 42)),
        height=int(p.get("height", 512)),
        width=int(p.get("width", 768)),
        num_frames=nf,
        frame_rate=fps,
        num_inference_steps=int(p.get("steps", 40)),
        video_guider_params=vg,
        audio_guider_params=ag,
        images=images,
        tiling_config=TilingConfig.default(),
        enhance_prompt=bool(p.get("enhance", False)),
    )

    print(f"[gen] {time.time() - tg:.0f}s ({nf} frames @ {fps}fps = {nf / fps:.1f}s video)", flush=True)
    out = p["output"]
    encode_video(video=video, fps=fps, audio=audio, output_path=out,
                 video_chunks_number=get_video_chunks_number(nf, TilingConfig.default()))
    print("OK:" + out, flush=True)


if __name__ == "__main__":
    main()
