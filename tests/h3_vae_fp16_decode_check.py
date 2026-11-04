r"""Verifica NUMERICA PESANTE (CPU, ~15 GB di RAM, ~1 minuto): decode del VAE H3 con pesi fp16 contro fp32 su latenti veri. NON fa parte dei test leggeri: si lancia nel
turno di misura, quando il PC e nostro. Esito del 04/11: PSNR 63,5 dB, errore massimo 2,6/255, niente NaN.
Si lancia:  E:\AI\Tools\diffusers_next_venv\Scripts\python.exe tests\h3_vae_fp16_decode_check.py"""
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, r"E:\ai\ltxstudio\.claude\worktrees\openjutsu\workers")
sys.path.insert(0, r"E:\ai\ltxstudio\.claude\worktrees\openjutsu")
import minimax_h3_infer as worker  # noqa: E402

SETTINGS = {"vae": r"E:\AI\Models\VAE\minimax_h3_video_vae_fp16-native.safetensors", "configs": r"E:\AI\Models\configs\minimax-h3"}
VIDEO = r"C:\Users\atain\.claude\projects\E--ai-ltxstudio\orch\app_live\fixtures\persona_intera_12s.mp4"
SIZE = (192, 128)                                        # larghezza, altezza: multipli di 16

capture = cv2.VideoCapture(VIDEO)
frames = []
while len(frames) < 22:
    ok, frame = capture.read()
    if not ok:
        break
    frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), SIZE, interpolation=cv2.INTER_AREA))
pixels = torch.from_numpy(np.stack(frames[:22])).float().div(255).permute(3, 0, 1, 2)[None]
mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
x = (pixels - mean) / std

t0 = time.time()
vae = worker.load_vae(SETTINGS, device="cpu")
with torch.inference_mode():
    posterior = vae.encode(x, return_dict=False)[0]
    latents = posterior.mode()
    print("latenti", tuple(latents.shape), f"{time.time() - t0:.0f}s", flush=True)
    t1 = time.time()
    out32 = vae.decode(latents, return_dict=False)[0].float()
    print(f"decode fp32 {time.time() - t1:.0f}s", tuple(out32.shape), flush=True)
    vae16 = vae.half()
    t2 = time.time()
    out16 = vae16.decode(latents.half(), return_dict=False)[0].float()
    print(f"decode fp16 {time.time() - t2:.0f}s", flush=True)
assert torch.isfinite(out16).all(), "NaN/inf nel decode fp16"


def to_pixels(t):
    return ((t.clamp(-1, 1) * 0.5 + 0.5) if t.min() < -0.01 else t.clamp(0, 1)) * 255.0


p32, p16 = to_pixels(out32), to_pixels(out16)
mse = float(((p32 - p16) ** 2).mean())
psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
ref_frames = pixels[:, :, : out32.shape[2]] * 255.0
mse_ref = float(((p32 - ref_frames) ** 2).mean())
print(f"fp16 contro fp32: PSNR {psnr:.1f} dB, errore massimo {float((p32 - p16).abs().max()):.1f}/255, medio {float((p32 - p16).abs().mean()):.2f}/255")
print(f"(per confronto: fp32 contro il fotogramma originale: PSNR {10 * np.log10(255.0 ** 2 / mse_ref):.1f} dB)")
print("min/max uscita fp32", float(out32.min()), float(out32.max()), "fp16", float(out16.min()), float(out16.max()))
