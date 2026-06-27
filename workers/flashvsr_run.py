"""Runner FlashVSR v1.1 (Tiny) parametrizzato — super-risoluzione video x4 streaming.

Riusa le helper del repo OpenImagingLab/FlashVSR (examples/WanVSR/infer_flashvsr_v1.1_tiny.py:
prepare_input_tensor / tensor2video / save_video) ma con pesi/input/output PARAMETRICI (lo script
originale ha i path hardcoded). Va eseguito col venv FlashVSR e cwd=examples/WanVSR (per gli import
`from utils...` e `from diffsynth...`). Output mp4; l'audio viene rimuxato a valle dal service.

Uso: python flashvsr_run.py <input> <output> <model_dir> [scale] [seed] [sparse_ratio] [local_range]
"""
import importlib.util
import os
import sys

import torch


def _load_infer_module(wanvsr_dir: str):
    """Carica infer_flashvsr_v1.1_tiny.py come modulo per riusarne le helper."""
    path = os.path.join(wanvsr_dir, "infer_flashvsr_v1.1_tiny.py")
    spec = importlib.util.spec_from_file_location("flashvsr_infer_tiny", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def init_pipeline(model_dir: str, M):
    """init_pipeline parametrico (path pesi assoluti invece di ./FlashVSR-v1.1/)."""
    from diffsynth import ModelManager, FlashVSRTinyPipeline
    from utils.utils import Causal_LQ4x_Proj
    from utils.TCDecoder import build_tcdecoder
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([os.path.join(model_dir, "diffusion_pytorch_model_streaming_dmd.safetensors")])
    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device="cuda")
    pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    lq = os.path.join(model_dir, "LQ_proj_in.ckpt")
    if os.path.exists(lq):
        pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq, map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in.to("cuda")
    pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768)
    pipe.TCDecoder.load_state_dict(torch.load(os.path.join(model_dir, "TCDecoder.ckpt")), strict=False)
    pipe.to("cuda"); pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(); pipe.load_models_to_device(["dit", "vae"])
    return pipe


def main():
    if len(sys.argv) < 4:
        print("uso: flashvsr_run.py <input> <output> <model_dir> [scale] [seed] [sparse] [local_range]")
        sys.exit(2)
    inp, out, model_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    sparse = float(sys.argv[6]) if len(sys.argv) > 6 else 2.0
    local_range = int(sys.argv[7]) if len(sys.argv) > 7 else 11
    wanvsr_dir = os.getcwd()                                  # cwd = examples/WanVSR
    M = _load_infer_module(wanvsr_dir)
    pipe = init_pipeline(model_dir, M)
    torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    LQ, th, tw, F, fps = M.prepare_input_tensor(inp, scale=scale, dtype=torch.bfloat16, device="cuda")
    video = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=seed,
                 LQ_video=LQ, num_frames=F, height=th, width=tw, is_full_block=False, if_buffer=True,
                 topk_ratio=sparse * 768 * 1280 / (th * tw), kv_ratio=3.0,
                 local_range=local_range, color_fix=True)
    M.save_video(M.tensor2video(video), out, fps=fps, quality=6)
    print(f"[flashvsr] OK → {out}")


if __name__ == "__main__":
    main()
