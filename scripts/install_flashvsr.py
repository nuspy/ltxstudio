"""Installer FlashVSR v1.1 (super-risoluzione video x4) — venv isolato + diffsynth.

Repo: OpenImagingLab/FlashVSR (clonato in E:\\AI\\Tools\\FlashVSR). Crea il venv e installa torch CUDA +
requirements (diffsynth, einops, imageio-ffmpeg, ...). block_sparse_attn/flash_attn sono opzionali
(fallback ad attention standard se non compilano su Windows). I pesi sono già in E:\\AI\\Models\\FlashVSR.

Uso: python scripts/install_flashvsr.py
"""
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(r"E:\AI\Tools\FlashVSR")
VENV = REPO_DIR / "venv"
REPO = "https://github.com/OpenImagingLab/FlashVSR"


def main():
    if not REPO_DIR.exists():
        print(f"[flashvsr] clono {REPO}")
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(REPO_DIR)], check=True)
    if not VENV.exists():
        print(f"[flashvsr] creo venv {VENV}")
        subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
    py = VENV / "Scripts" / "python.exe"
    subprocess.run([str(py), "-m", "pip", "install", "-U", "pip"], check=True)
    print("[flashvsr] installo torch CUDA 12.8 (Blackwell sm_120, ultimo) + requirements…")
    # NB: niente pin di versione — su RTX 5090 (sm_120) serve torch >=2.7 cu128; il 2.6 non ha i kernel.
    subprocess.run([str(py), "-m", "pip", "install", "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu128"], check=False)
    subprocess.run([str(py), "-m", "pip", "install", "setuptools<81", "wheel"], check=False)
    # diffsynth: --no-deps (il suo pin torch==2.6+cu124 romperebbe il cu128 Blackwell) + transformers 4.46
    # (diffsynth importa PretrainedConfig, rimosso in transformers 5). Venv ISOLATO → nessun impatto altrove.
    subprocess.run([str(py), "-m", "pip", "install", "-e", str(REPO_DIR),
                    "--no-deps", "--no-build-isolation"], check=False)
    subprocess.run([str(py), "-m", "pip", "install", "transformers==4.46.2", "einops", "imageio",
                    "imageio-ffmpeg", "modelscope", "sentencepiece", "huggingface_hub", "safetensors",
                    "opencv-python-headless", "ftfy", "tqdm", "numpy<2.3"], check=False)
    # PATCH Windows: block_sparse_attn non compila → import opzionale + fallback SDPA in wan_video_dit.py
    dit = REPO_DIR / "diffsynth" / "models" / "wan_video_dit.py"
    if dit.exists():
        t = dit.read_text(encoding="utf-8")
        if "block_sparse_attn_func = None" not in t:
            t = t.replace("from block_sparse_attn import block_sparse_attn_func",
                          "try:\n    from block_sparse_attn import block_sparse_attn_func\n"
                          "except Exception:\n    block_sparse_attn_func = None")
            t = t.replace(
                "    if attention_mask is not None:\n        seqlen = q.shape[1]",
                "    if attention_mask is not None and block_sparse_attn_func is None:\n"
                "        import torch.nn.functional as _F\n"
                "        q = rearrange(q, 'b s (n d) -> b n s d', n=num_heads)\n"
                "        k = rearrange(k, 'b s (n d) -> b n s d', n=num_heads)\n"
                "        v = rearrange(v, 'b s (n d) -> b n s d', n=num_heads)\n"
                "        x = _F.scaled_dot_product_attention(q, k, v)\n"
                "        x = rearrange(x, 'b n s d -> b s (n d)', n=num_heads)\n"
                "    elif attention_mask is not None:\n        seqlen = q.shape[1]")
            dit.write_text(t, encoding="utf-8")
            print("[flashvsr] patch block_sparse_attn (fallback SDPA) applicata")
    print("[flashvsr] fatto. Validato: 512x320→2048x1280 in ~22s.")


if __name__ == "__main__":
    main()
