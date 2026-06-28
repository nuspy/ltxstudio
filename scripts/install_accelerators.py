"""Installer acceleratori (flash-attn / xformers / sageattention) da WHEEL PRE-COMPILATI per arch GPU.

Niente compilazione in distribuzione: rileva (torch, compute-capability) del venv target e installa il wheel
giusto da una cartella locale WHEELS_DIR (dove l'utente mette i wheel scaricati da docs/precompiled.pdf —
"All_Compiled_Libraries_Online_Install_Links.txt"). Mappa sm_XX→GPU in [[precompiled-libs-distribution]].

Uso:  python scripts/install_accelerators.py [--venv <python.exe>] [--wheels <dir>]
Default: venv inference_server (torch 2.11 cu128 sm_120) + E:\AI\Tools\precompiled_wheels
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_VENV = r"E:\AI\Tools\inference_server_venv\Scripts\python.exe"
DEFAULT_WHEELS = r"E:\AI\Tools\precompiled_wheels"
PKGS = ("flash_attn", "xformers", "sageattention", "triton")


def _probe(py: str):
    """Ritorna (torch_ver, cc_str es. 'sm120', mancanti[])."""
    code = (
        "import torch,importlib.util as u;"
        "cc=torch.cuda.get_device_capability() if torch.cuda.is_available() else (0,0);"
        "miss=[m for m in ('flash_attn','xformers','sageattention','triton') if u.find_spec(m) is None];"
        "print(torch.__version__+'|sm'+str(cc[0])+str(cc[1])+'|'+','.join(miss))")
    out = subprocess.run([py, "-c", code], capture_output=True, text=True)
    line = (out.stdout or "").strip().splitlines()[-1] if out.stdout else "?|?|"
    tv, cc, miss = (line.split("|") + ["", "", ""])[:3]
    return tv, cc, [m for m in miss.split(",") if m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default=DEFAULT_VENV)
    ap.add_argument("--wheels", default=DEFAULT_WHEELS)
    a = ap.parse_args()
    py = a.venv
    if not Path(py).is_file():
        sys.exit(f"venv non trovato: {py}")
    tv, cc, miss = _probe(py)
    print(f"[accel] venv: torch {tv}, arch {cc}; mancanti: {miss or 'nessuno'}")
    if not miss:
        print("[accel] tutti gli acceleratori già presenti.")
        return
    wdir = Path(a.wheels)
    if not wdir.is_dir():
        print(f"[accel] cartella wheel assente: {wdir}\n"
              f"        → scarica i wheel per {tv}/{cc} (vedi docs/precompiled.pdf) e mettili lì.")
        wdir.mkdir(parents=True, exist_ok=True)
    # token di match: versione torch major.minor (es. 'torch211') + arch (es. 'sm120' / 'cp311')
    tmm = "torch" + "".join(tv.split("+")[0].split(".")[:2]) if tv and tv != "?" else ""
    installed = []
    for pkg in miss:
        # cerca un wheel che contenga il nome pkg + (torch-ver o arch) nel filename
        cands = [w for w in wdir.glob("*.whl")
                 if pkg.replace("_", "-").split("-")[0] in w.name.lower().replace("_", "-")
                 and (not tmm or tmm in w.name.lower() or cc in w.name.lower())]
        if not cands:
            print(f"  [{pkg}] nessun wheel compatibile in {wdir} (serve {tmm}/{cc}) → fornisci il file.")
            continue
        whl = sorted(cands, key=lambda w: len(w.name))[0]
        r = subprocess.run([py, "-m", "pip", "install", "--no-deps", str(whl)],
                           capture_output=True, text=True)
        ok = "Successfully installed" in (r.stdout or "")
        print(f"  [{pkg}] {'OK' if ok else 'FALLITO'} ← {whl.name}")
        if ok:
            installed.append(pkg)
    print(f"[accel] installati: {installed or 'nessuno'}. "
          f"Mancanti residui: {[m for m in miss if m not in installed] or 'nessuno'}")


if __name__ == "__main__":
    main()
