"""Installer del venv CONDIVISO «diffusers_next» per i motori che richiedono diffusers più recente dell'ultima
release: Qwen-Image 2.1 (QwenImage21Pipeline, arrivata dopo la 0.40) e MiniMax H3 (classi MiniMaxH3*).

Un venv invece di due (coerente col consolidamento dei venv, #61): stesso torch 2.9.1+cu130 del venv LongCat/
TRELLIS (wheel già in cache pip), diffusers fissato a un commit di main (riproducibile; override con
DIFFUSERS_REF), transformers ≥ 5.17 (Qwen3-VL/Qwen3.5-VL), comfy-kitchen per il dequant NVFP4.

Uso:  python scripts/install_diffusers_next.py
Env:  AI_TOOLS_DIR (default E:\\AI\\Tools), SKIP_TORCH=1, DIFFUSERS_REF=<sha|branch>
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TOOLS = Path(os.environ.get("AI_TOOLS_DIR", r"E:\AI\Tools"))
VENV = TOOLS / "diffusers_next_venv"
VENV_PY = VENV / "Scripts" / "python.exe"
TORCH_INDEX = os.environ.get("TORCH_INDEX", "https://download.pytorch.org/whl/cu130")
# main del 26/09/2026: contiene QwenImage21Pipeline e le classi MiniMax H3 (verificato via API GitHub)
DIFFUSERS_REF = os.environ.get("DIFFUSERS_REF", "e0abab83b5df05de9e7abd788643c1a7c1e42e28")
HF_WHEELS = "https://huggingface.co/MonsterMMORPG/Wan_GGUF/resolve/main"
WHEELS = ["flash_attn-2.8.3+torch2.9.1.cuda13.1-cp311-cp311-win_amd64.whl"]
DIFFUSERS_SRC = TOOLS / "diffusers_src"
# NB: lo zip GitHub di diffusers non si estrae su Windows (pip: OSError 22 su un file di examples/): si
# clona SOLO src/ + i file di build (sparse-checkout) e si installa dalla cartella locale.
DEPS = [
    "transformers>=5.17", "accelerate", "safetensors>=0.4.5", "sentencepiece", "protobuf", "pillow",
    "huggingface_hub", "einops", "ftfy", "imageio", "imageio-ffmpeg", "soundfile", "av", "numpy<2.3",
    "comfy-kitchen", "peft",
    "opencv-python-headless",            # preprocessori ControlNet di Qwen-Image 2.1 (canny, contorni)
]


def _env() -> dict:
    """pip del venv appena creato non ha truststore: bundle certifi + store di Windows (core/net_tls)."""
    env = dict(os.environ)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from core.net_tls import child_env
        env.update(child_env())
    except Exception:                                                # noqa: BLE001
        pass
    return env


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    kw.setdefault("env", _env())
    return subprocess.run([str(c) for c in cmd], **kw)


def _py311() -> str:
    for c in (os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python311\python.exe"),
              r"C:\Python311\python.exe", r"C:\Program Files\Python311\python.exe"):
        if Path(c).is_file():
            return c
    raise SystemExit("Python 3.11 non trovato (i wheel precompilati sono cp311). Installa Python 3.11.")


def fetch_diffusers() -> bool:
    """Checkout sparse di diffusers al commit DIFFUSERS_REF (solo src/ e file di build)."""
    d = DIFFUSERS_SRC
    if not (d / ".git").is_dir():
        if run(["git", "clone", "--filter=blob:none", "--no-checkout",
                "https://github.com/huggingface/diffusers", d]).returncode:
            return False
        run(["git", "-C", d, "sparse-checkout", "init", "--no-cone"])
        patterns = ["/src/", "/setup.py", "/pyproject.toml", "/setup.cfg", "/README.md", "/LICENSE"]
        (d / ".git" / "info" / "sparse-checkout").write_text("\n".join(patterns) + "\n", encoding="utf-8")
    else:
        run(["git", "-C", d, "fetch", "origin"])
    return run(["git", "-C", d, "checkout", "--force", DIFFUSERS_REF]).returncode == 0


def check_classes() -> dict:
    """Quali motori il venv supporta davvero (import delle classi, non solo la versione)."""
    code = ("import json, diffusers, transformers;"
            "q=all(hasattr(diffusers,n) for n in ('QwenImage21Pipeline','QwenImage21Transformer2DModel',"
            "'AutoencoderKLQwenImage21'));"
            "h=all(hasattr(diffusers,n) for n in ('MiniMaxH3Transformer3DModel','AutoencoderKLMiniMaxH3',"
            "'AutoencoderKLMiniMaxH3Audio','ModularPipeline'));"
            "import importlib.util as u;"
            "print(json.dumps({'qwen_image21':q,'minimax_h3':h,'nvfp4':u.find_spec('comfy_kitchen') is not None,"
            "'diffusers':diffusers.__version__,'transformers':transformers.__version__}))")
    r = run([VENV_PY, "-c", code], capture_output=True, text=True)
    import json
    try:
        return json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception:                                                # noqa: BLE001
        print("[!] verifica classi fallita:\n", (r.stderr or "")[-2000:])
        return {}


def main() -> int:
    if not VENV_PY.exists():
        VENV.parent.mkdir(parents=True, exist_ok=True)
        if run([_py311(), "-m", "venv", VENV]).returncode:
            print("[!] creazione venv fallita"); return 1
    run([VENV_PY, "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"])
    if os.environ.get("SKIP_TORCH") != "1":
        if run([VENV_PY, "-m", "pip", "install", "torch==2.9.1", "torchvision", "--index-url", TORCH_INDEX]).returncode:
            print("[!] installazione torch fallita"); return 1
    for w in WHEELS:
        run([VENV_PY, "-m", "pip", "install", "--no-deps", f"{HF_WHEELS}/{w}"])
    if not fetch_diffusers():
        print("[!] checkout di diffusers fallito"); return 1
    if run([VENV_PY, "-m", "pip", "install", str(DIFFUSERS_SRC), *DEPS]).returncode:
        print("[!] installazione dipendenze fallita"); return 1
    ok = check_classes()
    print("[=] supporto:", ok)
    if not ok.get("qwen_image21"):
        print("[!] diffusers non espone QwenImage21Pipeline: prova DIFFUSERS_REF=main"); return 1
    (VENV / ".ltx_ready").write_text("ok", encoding="utf-8")
    print(f"\n[OK] venv diffusers_next pronto: {VENV_PY}\n"
          "     Motori: Qwen-Image 2.1" + (", MiniMax H3" if ok.get("minimax_h3") else "") +
          ("" if ok.get("nvfp4") else "  (NVFP4 non disponibile: comfy-kitchen mancante)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
