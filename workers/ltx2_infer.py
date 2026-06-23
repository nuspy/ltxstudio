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

CHECKPOINT = r"E:\AI\Models\diffusion_models\LTX\ltx-2.3-22b-dev-fp8.safetensors"
UPSAMPLER = r"E:\AI\Models\latent_upscale_models\ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
GEMMA = r"E:\AI\Models\TextEncoders\gemma3"
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

    # quantizzazione fp8 (equivalente di --fast fp8_matrix_mult di ComfyUI). dev-fp8 27GB → 32GB ok.
    import time
    quant = None
    qmode = p.get("quant", "fp8_scaled_mm")
    if qmode == "fp8_scaled_mm":
        from ltx_core.quantization import fp8_scaled_mm
        quant = fp8_scaled_mm.build_policy(CHECKPOINT)
    elif qmode == "fp8_cast":
        from ltx_core.quantization import fp8_cast
        quant = fp8_cast.build_policy(CHECKPOINT)
    t0 = time.time()
    pipe = TI2VidTwoStagesPipeline(
        checkpoint_path=CHECKPOINT, distilled_lora=[], spatial_upsampler_path=UPSAMPLER,
        gemma_root=GEMMA, loras=loras, quantization=quant)
    print(f"[load] {time.time() - t0:.0f}s (quant={qmode})", flush=True)

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
