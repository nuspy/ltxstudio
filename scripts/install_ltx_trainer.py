"""Installa il trainer ufficiale LTX-2 (packages/ltx-trainer) nel venv ltx2 (editable).
Non scarica modelli: usa il checkpoint LTX e il text-encoder Gemma già presenti.
Uso:  ltx2_venv\\Scripts\\python.exe scripts\\install_ltx_trainer.py
"""
import subprocess
import sys
from pathlib import Path

TRAINER = Path(r"E:\AI\Tools\LTX-2\packages\ltx-trainer")
LTX2_PY = Path(r"E:\AI\Tools\ltx2_venv\Scripts\python.exe")


def main():
    if not TRAINER.exists():
        print(f"[err] trainer non trovato: {TRAINER}")
        sys.exit(1)
    py = str(LTX2_PY if LTX2_PY.exists() else sys.executable)
    print(f"[install] pip install -e {TRAINER}  (venv: {py})")
    r = subprocess.run([py, "-m", "pip", "install", "-e", str(TRAINER)], cwd=str(TRAINER))
    if r.returncode != 0:
        print("[install] fallito. Riprova con la rete attiva.")
        sys.exit(r.returncode)
    chk = subprocess.run([py, "-c", "import ltx_trainer; print('ltx_trainer OK')"])
    sys.exit(chk.returncode)


if __name__ == "__main__":
    main()
