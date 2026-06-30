"""Installer ACCELERATORI velocità/memoria — completo e DISTRIBUTION-AWARE.

Acceleratori gestiti (#73): Triton · FlashAttention · SageAttention · xFormers · DeepSpeed · ReActor(insightface).

Strategia per ciascuno, scelta in base a (OS, Python, torch, CUDA compute-capability sm_XX):
  - "pip"        : pip install dal PyPI / index pytorch (cross-platform, quando esiste un wheel pubblico per
                   la combo torch+cuda del venv) — es. xformers, insightface, onnxruntime-gpu, triton(linux).
  - "pip_win"    : nome pacchetto diverso su Windows (es. triton → triton-windows).
  - "wheel"      : SOLO da wheel pre-compilato in WHEELS_DIR (no compilazione in distribuzione), match per
                   torch-ver + arch sm_XX/cpXY nel filename — es. flash_attn (Win), sageattention.
  - "pip_linux"  : pip ma SOLO su Linux (su Windows si flagga "non supportato / build complessa") — es. deepspeed.
  - "optional"   : utile ma non bloccante.

Niente compilazione lato utente: i wheel arch-specific si scaricano una volta (vedi docs/precompiled.pdf →
"All_Compiled_Libraries_Online_Install_Links.txt") e si mettono in WHEELS_DIR. Mappa sm_XX→GPU in
[[precompiled-libs-distribution]].

Uso:
  python scripts/install_accelerators.py [--venv <python.exe>] [--wheels <dir>] [--only a,b] [--dry-run]
"""
from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

DEFAULT_VENV = r"E:\AI\Comfy_Nexus.bak\ComfyUI\venv\Scripts\python.exe"   # runtime principale LTX Studio
DEFAULT_WHEELS = r"E:\AI\Tools\precompiled_wheels"

# Alcuni acceleratori vanno in un venv DIVERSO da quello principale:
#   insightface (ReActor/face-swap) → `inference_server_venv` (nel comfy venv dà KeyError numpy.uint32
#   per conflitto onnxruntime/numpy2). Il face-swap nativo gira già lì. Usa --venv per puntarlo.
PREFERRED_VENV = {
    "insightface": r"E:\AI\Tools\inference_server_venv\Scripts\python.exe",
    "onnxruntime_gpu": r"E:\AI\Tools\inference_server_venv\Scripts\python.exe",
}

# nome_pacchetto → (import_name, strategia, extra)
#   extra: per "pip" = eventuale index-url; per "wheel" = pattern-hint; per "pip_win" = nome win
ACCELERATORS = {
    "triton":        ("triton",        "pip_win",  "triton-windows"),
    "xformers":      ("xformers",      "pip",      "torch_index"),
    "sageattention": ("sageattention", "wheel",    None),
    "flash_attn":    ("flash_attn",    "wheel",    None),          # Win: solo wheel; Linux: può pip-buildare
    "deepspeed":     ("deepspeed",     "pip_linux", None),
    "insightface":   ("insightface",   "pip",      None),          # ReActor/face-swap (+ onnxruntime-gpu)
    "onnxruntime_gpu": ("onnxruntime", "pip",      "onnxruntime-gpu"),
}


def _probe(py: str):
    code = (
        "import importlib.util as u\n"
        "try:\n import torch; tv=torch.__version__; "
        "cc=torch.cuda.get_device_capability() if torch.cuda.is_available() else (0,0)\n"
        "except Exception: tv='?'; cc=(0,0)\n"
        "import platform as P\n"
        "mods={m:(u.find_spec(m) is not None) for m in "
        "('triton','xformers','sageattention','flash_attn','deepspeed','insightface','onnxruntime')}\n"
        "print(tv+'|sm'+str(cc[0])+str(cc[1])+'|cp'+''.join(P.python_version_tuple()[:2])+'|'"
        "+','.join(k for k,v in mods.items() if v))")
    out = subprocess.run([py, "-c", code], capture_output=True, text=True)
    line = (out.stdout or "").strip().splitlines()[-1] if out.stdout else "?|sm00|cp00|"
    tv, sm, cp, present = (line.split("|") + ["", "", "", ""])[:4]
    return tv, sm, cp, set(p for p in present.split(",") if p)


def _cuda_tag(tv: str) -> str:
    # 'cu128' da '2.8.0+cu128'
    return tv.split("+")[1] if "+" in tv else "cu121"


def _pip(py, *args, dry=False):
    cmd = [py, "-m", "pip", "install", *args]
    if dry:
        print("    [dry-run]", " ".join(str(c) for c in cmd)); return True
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and ("Successfully installed" in r.stdout or "already satisfied" in r.stdout)
    if not ok:
        print("    ERR:", (r.stdout + r.stderr).strip().splitlines()[-1][:160] if (r.stdout + r.stderr).strip() else "")
    return ok


def _install_one(name, py, is_win, tv, sm, cp, wdir, dry):
    imp, strat, extra = ACCELERATORS[name]
    cu = _cuda_tag(tv)
    tmm = "torch" + "".join(tv.split("+")[0].split(".")[:2]) if tv and tv != "?" else ""

    if strat == "pip_win":
        pkg = extra if is_win else name
        return _pip(py, "-U", pkg, dry=dry)

    if strat == "pip":
        pkg = extra if (extra and extra != "torch_index") else name
        if extra == "torch_index":
            # xformers: prova prima il wheel PyPI che matcha il torch installato
            return _pip(py, "-U", name, dry=dry)
        return _pip(py, "-U", pkg, dry=dry)

    if strat == "pip_linux":
        if is_win:
            print(f"    [{name}] Windows: build complessa → SALTATO (usare su Linux/WSL, oppure wheel community).")
            return False
        return _pip(py, "-U", name, dry=dry)

    if strat == "wheel":
        if not wdir.is_dir():
            print(f"    [{name}] serve wheel pre-compilato per {tmm}/{sm}/{cp} in {wdir} (docs/precompiled.pdf)."); return False
        base = imp.replace("_", "-").split("-")[0]
        cands = [w for w in wdir.glob("*.whl")
                 if base in w.name.lower().replace("_", "-")
                 and (not tmm or tmm in w.name.lower())
                 and (sm in w.name.lower() or cp in w.name.lower() or cu in w.name.lower())]
        if not cands:
            print(f"    [{name}] nessun wheel compatibile in {wdir} (serve {tmm}+{sm}/{cp}/{cu}) → fornisci il file.")
            return False
        whl = sorted(cands, key=lambda w: len(w.name))[0]
        print(f"    [{name}] wheel: {whl.name}")
        return _pip(py, "--no-deps", str(whl), dry=dry)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default=DEFAULT_VENV)
    ap.add_argument("--wheels", default=DEFAULT_WHEELS)
    ap.add_argument("--only", default="", help="lista CSV di acceleratori da installare (default: tutti i mancanti)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    py = a.venv
    if not Path(py).is_file():
        sys.exit(f"venv non trovato: {py}")
    is_win = platform.system() == "Windows"
    tv, sm, cp, present = _probe(py)
    wdir = Path(a.wheels)
    print(f"[accel] OS={platform.system()} · torch={tv} · arch={sm} · py={cp} · cuda={_cuda_tag(tv)}")
    print(f"[accel] già presenti: {sorted(present) or 'nessuno'}")

    want = [x.strip() for x in a.only.split(",") if x.strip()] or list(ACCELERATORS)
    todo = [n for n in want if ACCELERATORS[n][0] not in present]
    if not todo:
        print("[accel] tutti gli acceleratori richiesti sono già presenti.")
        return
    print(f"[accel] da installare: {todo}\n")
    done, failed = [], []
    for name in todo:
        print(f"  → {name} ({ACCELERATORS[name][1]})")
        ok = _install_one(name, py, is_win, tv, sm, cp, wdir, a.dry_run)
        (done if ok else failed).append(name)
    print(f"\n[accel] INSTALLATI: {done or 'nessuno'}")
    if failed:
        print(f"[accel] da completare (wheel/Linux): {failed}")
        print("        → flash_attn/sageattention: wheel arch-specific in WHEELS_DIR; "
              "deepspeed: Linux/WSL; vedi docs/precompiled.pdf.")


if __name__ == "__main__":
    main()
