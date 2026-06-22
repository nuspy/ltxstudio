"""Installer delle feature 3D di LTX Studio: TRELLIS.2 (immagine→3D) + SkinTokens (rig).

Idempotente: può essere rieseguito senza danni (clona se manca, altrimenti aggiorna).

Esegue:
  1. Clona/aggiorna  custom_nodes/ComfyUI-TRELLIS2  → pip install -r requirements + python install.py
  2. Clona/aggiorna  custom_nodes/ComfyUI-SkinToken → pip install -r requirements
  3. (best-effort)   Installa Blender via winget + imposta SKINTOKEN_BLENDER_BIN  (serve a SkinTokens)

Uso (con il python del venv di ComfyUI):
  E:\\AI\\Comfy_Nexus\\ComfyUI\\venv\\Scripts\\python.exe scripts\\install_3d_features.py

Variabili d'ambiente opzionali:
  COMFYUI_DIR   override del percorso radice di ComfyUI (default: E:\\AI\\Comfy_Nexus\\ComfyUI)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", r"E:\AI\Comfy_Nexus\ComfyUI"))
CUSTOM_NODES = COMFYUI_DIR / "custom_nodes"
PY = sys.executable   # atteso: il python del venv di ComfyUI

NODES = [
    # (cartella, url, ha_install_py)
    ("ComfyUI-TRELLIS2", "https://github.com/PozzettiAndrea/ComfyUI-TRELLIS2.git", True),
    ("ComfyUI-SkinToken", "https://github.com/Rizzlord/ComfyUI-SkinToken.git", False),
]


def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return r.returncode


def clone_or_update(folder: str, url: str) -> Path:
    dest = CUSTOM_NODES / folder
    if dest.exists():
        print(f"[=] {folder}: già presente, aggiorno")
        run(["git", "-C", str(dest), "pull", "--ff-only"])
    else:
        print(f"[+] {folder}: clono")
        run(["git", "clone", "--depth", "1", url, str(dest)])
    return dest


def install_node(folder: str, url: str, has_install_py: bool):
    print(f"\n=== {folder} ===")
    dest = clone_or_update(folder, url)
    req = dest / "requirements.txt"
    if req.exists():
        run([PY, "-m", "pip", "install", "-r", str(req)])
    if has_install_py and (dest / "install.py").exists():
        print(f"[i] {folder}: install.py (setup ambiente isolato comfy-env/pixi)")
        run([PY, str(dest / "install.py")], cwd=dest)


def install_blender() -> None:
    print("\n=== Blender (per SkinTokens) ===")
    if shutil.which("blender") or os.environ.get("SKINTOKEN_BLENDER_BIN"):
        print("[=] Blender già disponibile.")
        return
    if not shutil.which("winget"):
        print("[!] winget non disponibile: installa Blender 4.0+ manualmente e imposta SKINTOKEN_BLENDER_BIN.")
        return
    print("[+] Installo Blender via winget…")
    rc = run([
        "winget", "install", "-e", "--id", "BlenderFoundation.Blender",
        "--accept-source-agreements", "--accept-package-agreements", "--disable-interactivity",
    ])
    # individua il binario e persiste la variabile
    bin_path = None
    pf = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Blender Foundation"
    if pf.exists():
        cands = sorted(pf.glob("**/blender.exe"), reverse=True)
        bin_path = str(cands[0]) if cands else None
    if bin_path:
        subprocess.run(["setx", "SKINTOKEN_BLENDER_BIN", bin_path], capture_output=True, text=True)
        print(f"[✓] Blender: {bin_path} (SKINTOKEN_BLENDER_BIN impostata)")
    elif rc == 0:
        print("[✓] winget ok; riavvia il terminale per il PATH di Blender.")
    else:
        print("[!] Installazione Blender non confermata: verifica manualmente.")


def predownload_trellis2():
    """Pre-scarica i pesi TRELLIS.2 (~25GB) PRIMA del primo run.

    Necessario perché i nodi TRELLIS2 girano in ambiente isolato comfy-env con
    timeout IPC di 600s hardcoded: se i pesi venissero scaricati al primo run
    (HuggingFace lento), il job andrebbe in timeout. Pre-scaricandoli qui il
    problema è eliminato. Saltabile con SKIP_TRELLIS2_DOWNLOAD=1.
    """
    print("\n=== Pre-download pesi TRELLIS.2 (~25GB) ===")
    if os.environ.get("SKIP_TRELLIS2_DOWNLOAD"):
        print("[=] saltato (SKIP_TRELLIS2_DOWNLOAD impostata). "
              "NB: il primo run di TRELLIS andrà in timeout finché i pesi non sono presenti.")
        return
    script = Path(__file__).with_name("download_trellis2_weights.py")
    env = dict(os.environ, TRELLIS2_MODELS_DIR=str(COMFYUI_DIR / "models" / "trellis2"))
    print("[+] download pesi (ripartibile; può richiedere tempo a banda lenta)…")
    subprocess.run([PY, str(script)], env=env)


def main() -> int:
    print(f"ComfyUI:      {COMFYUI_DIR}")
    print(f"custom_nodes: {CUSTOM_NODES}")
    print(f"python:       {PY}")
    if not CUSTOM_NODES.exists():
        print(f"[ERRORE] custom_nodes non trovata: {CUSTOM_NODES}\n"
              f"Imposta COMFYUI_DIR alla radice corretta di ComfyUI.")
        return 1
    for folder, url, has_install in NODES:
        install_node(folder, url, has_install)
    install_blender()
    predownload_trellis2()
    print("\n✔ Feature 3D installate. Riavvia ComfyUI per registrare i nodi.")
    print("  - TRELLIS.2: pesi pre-scaricati (evita il timeout IPC al primo run).")
    print("  - SkinTokens: pesi (~4GB) scaricati al primo run del rig.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
