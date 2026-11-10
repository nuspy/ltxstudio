"""Verifica su CPU (venv diffusers, VAE vero) del VAE H3 «solo encoder» del worker: stessa codifica del VAE intero (bit a bit), 0,7 GB invece di 9,7, decoder sostituito.
Si lancia:  E:\\AI\\Tools\\diffusers_next_venv\\Scripts\\python.exe tests\\h3_vae_encoder_only_check.py   (stampa «VAE ENCODER-ONLY OK»)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT))
import minimax_h3_infer as worker  # noqa: E402

VAE_FILE = r"E:\AI\Models\VAE\minimax_h3_video_vae_fp16-native.safetensors"
CONFIGS = r"E:\AI\Models\configs\minimax-h3"
SETTINGS = {"vae": VAE_FILE, "configs": CONFIGS}


def parameter_gigabytes(module) -> float:
    return sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers())) / 2**30


def encode_tile(vae, tile: torch.Tensor):
    with torch.inference_mode():
        return vae.encode(tile, return_dict=False)[0].parameters


def main() -> int:
    torch.manual_seed(7)
    tile = torch.randn(1, 3, 17, 96, 128)                         # un clip da 17 frame, un solo riquadro spaziale
    encoder_only = worker.load_vae(SETTINGS, encoder_only=True, device="cpu")
    small = parameter_gigabytes(encoder_only)
    assert small < 1.2, f"encoder-only troppo grande: {small:.2f} GB"
    assert isinstance(encoder_only.decoder, torch.nn.Identity) and isinstance(encoder_only.post_quant_conv, torch.nn.Identity)
    moments_small = encode_tile(encoder_only, tile)
    del encoder_only
    full = worker.load_vae(SETTINGS, device="cpu")
    big = parameter_gigabytes(full)
    moments_full = encode_tile(full, tile)
    print(f"parametri: encoder-only {small:.2f} GB, intero {big:.2f} GB; momenti {tuple(moments_small.shape)}")
    assert moments_small.shape == moments_full.shape and torch.isfinite(moments_small).all()
    assert torch.equal(moments_small, moments_full), "il VAE solo encoder codifica diversamente da quello intero"
    assert big > 9.0 and small < big / 8
    print("VAE ENCODER-ONLY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
