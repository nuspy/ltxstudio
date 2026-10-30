"""Installer FlashVSR+ (super-risoluzione video Turbo/Base) — solo CODICE (i pesi stanno nel registro pesi).

Clona OpenImagingLab/FlashVSR_Plus (Apache-2.0) in E:\AI\Tools\FlashVSR_Plus al commit del 29/10/2026.
Nessun venv nuovo: il runner usa il venv consolidato (inference_server_venv: torch cu128, einops, imageio,
opencv, safetensors, psutil gia' presenti) con attenzione PyTorch SDPA. I pesi (Turbo ~6,8 GB, Base ~6,8 GB)
si scaricano dal tab «Pesi modelli» (id flashvsr-plus-turbo / flashvsr-plus-base) o al primo uso, con consenso.

Uso: python scripts/install_flashvsrplus.py [--dir PATH]
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/OpenImagingLab/FlashVSR_Plus"
PINNED = "b548a2cc2c929d3d4676f2973abe27876ec4dee7"      # commit del 29/10/2026 (import e geometria provati su CPU; prova GPU: vedi report task 20)
DEFAULT_DIR = Path(r"E:\AI\Tools\FlashVSR_Plus")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    d = Path(ap.parse_args().dir)
    if not (d / "inference" / "run_stream.py").exists():
        print(f"[flashvsr+] clono {REPO} in {d}")
        d.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", REPO, str(d)], check=True)
        r = subprocess.run(["git", "-C", str(d), "checkout", PINNED])
        if r.returncode != 0:
            print("[flashvsr+] commit pinnato non trovato: resto sull'ultimo (verifica a mano)")
    else:
        print(f"[flashvsr+] gia' presente: {d}")
    ok = (d / "prompt_tensor" / "posi_prompt.pth").exists()
    print("[flashvsr+] codice OK" if ok else "[flashvsr+] ATTENZIONE: manca prompt_tensor/posi_prompt.pth")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
