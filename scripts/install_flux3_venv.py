"""Installer del venv «flux3_venv» per FLUX.3 Image Klein (Flux3Transformer2DModel, uscita il 15/10/2026 solo su
diffusers `main`: nessuna release la contiene).

Perché un venv a parte e non l'aggiornamento di `diffusers_next_venv`: quello serve anche Qwen-Image 2.1 e MiniMax H3
(diffusers fissato al commit del 26/09) e non si ripinna sul posto senza averli riprovati dal vivo. Questo venv ha lo
stesso torch 2.9.1+cu130 (wheel già in cache pip), le stesse dipendenze e lo stesso transformers (≥ 5.17: Qwen3-VL e
Qwen3.5), ma diffusers a un commit di main ≥ 15/10 (default b800f828f, 16/10).

Uso:  python scripts/install_flux3_venv.py
Env:  AI_TOOLS_DIR (default E:\\AI\\Tools), SKIP_TORCH=1, DIFFUSERS_REF=<sha|branch>

Le funzioni di appoggio (run, _py311, checkout sparse) sono quelle di install_diffusers_next.py: stesso modo di
installare, nessuna logica nuova da mantenere.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import install_diffusers_next as base  # noqa: E402

TOOLS = base.TOOLS
VENV = TOOLS / "flux3_venv"
VENV_PY = VENV / "Scripts" / "python.exe"
DIFFUSERS_SRC = TOOLS / "flux3_diffusers_src"            # checkout PROPRIO: non tocca quello di diffusers_next
# main del 16/10/2026: contiene Flux3Transformer2DModel / Flux3ModularPipeline (PR #15046, 15/10) e il blocco
# prompt_upsampler (PR #15058). Verificato dalla ricerca del task 13 (HEAD main 16/10).
DIFFUSERS_REF = os.environ.get("DIFFUSERS_REF", "b800f828f")
DEPS = list(base.DEPS)

CHECK = ("import json, diffusers, transformers, torch;"
         "f=all(hasattr(diffusers,n) for n in ('Flux3Transformer2DModel','AutoencoderKLFlux2'));"
         "p=hasattr(diffusers,'Flux3ModularPipeline');"
         "import importlib.util as u;"
         "print(json.dumps({'flux3_transformer':f,'flux3_pipeline':p,'qwen3_5':hasattr(transformers,'Qwen3_5ForConditionalGeneration'),"
         "'qwen3_vl':hasattr(transformers,'Qwen3VLForConditionalGeneration'),'comfy_kitchen':u.find_spec('comfy_kitchen') is not None,"
         "'diffusers':diffusers.__version__,'transformers':transformers.__version__,'torch':torch.__version__}))")


def fetch_diffusers() -> bool:
    d = DIFFUSERS_SRC
    if not (d / ".git").is_dir():
        d.parent.mkdir(parents=True, exist_ok=True)
        if base.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                     "https://github.com/huggingface/diffusers", d]).returncode:
            return False
        base.run(["git", "-C", d, "sparse-checkout", "init", "--no-cone"])
        patterns = ["/src/", "/setup.py", "/pyproject.toml", "/setup.cfg", "/README.md", "/LICENSE"]
        (d / ".git" / "info" / "sparse-checkout").write_text("\n".join(patterns) + "\n", encoding="utf-8")
    else:
        base.run(["git", "-C", d, "fetch", "origin"])
    return base.run(["git", "-C", d, "checkout", "--force", DIFFUSERS_REF]).returncode == 0


def check_classes() -> dict:
    r = base.run([VENV_PY, "-c", CHECK], capture_output=True, text=True)
    try:
        return json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception:                                                # noqa: BLE001
        print("[!] verifica classi fallita:\n", (r.stderr or "")[-2000:])
        return {}


def main() -> int:
    if not VENV_PY.exists():
        VENV.parent.mkdir(parents=True, exist_ok=True)
        if base.run([base._py311(), "-m", "venv", VENV]).returncode:
            print("[!] creazione venv fallita"); return 1
    base.run([VENV_PY, "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"])
    if os.environ.get("SKIP_TORCH") != "1":
        if base.run([VENV_PY, "-m", "pip", "install", "torch==2.9.1", "torchvision",
                     "--index-url", base.TORCH_INDEX]).returncode:
            print("[!] installazione torch fallita"); return 1
    for w in base.WHEELS:
        base.run([VENV_PY, "-m", "pip", "install", "--no-deps", f"{base.HF_WHEELS}/{w}"])
    if not fetch_diffusers():
        print("[!] checkout di diffusers fallito"); return 1
    if base.run([VENV_PY, "-m", "pip", "install", str(DIFFUSERS_SRC), *DEPS]).returncode:
        print("[!] installazione dipendenze fallita"); return 1
    ok = check_classes()
    print("[=] supporto:", ok)
    if not (ok.get("flux3_transformer") and ok.get("qwen3_vl")):
        print("[!] diffusers non espone Flux3Transformer2DModel: prova DIFFUSERS_REF=main"); return 1
    (VENV / ".ltx_ready").write_text("ok", encoding="utf-8")
    print(f"\n[OK] venv flux3 pronto: {VENV_PY}\n     Motore: FLUX.3 Image Klein")
    return 0


if __name__ == "__main__":
    sys.exit(main())
