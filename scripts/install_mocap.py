"""Setup dell'ambiente mocap (venv dedicato, isolato dal venv ComfyUI).

FreeMoCap e MediaPipe hanno dipendenze pesanti/conflittuali e FreeMoCap è AGPL:
vanno tenuti FUORI dal processo principale. Questo script crea `mocap_env/` e vi
installa il necessario per: (a) cattura webcam in-app (MediaPipe→BVH), (b) import
di sessioni FreeMoCap (opzionale, pacchetto grande).

Uso:
    python scripts/install_mocap.py            # mediapipe + opencv + numpy
    python scripts/install_mocap.py --freemocap # aggiunge anche freemocap (grande)
Idempotente: se il venv esiste, installa/aggiorna soltanto.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / "mocap_env"
PY = VENV / "Scripts" / "python.exe"        # Windows
# mediapipe>=0.10.21 ha RIMOSSO l'API legacy `mp.solutions` (ora tasks-only), che il worker usa
# (mp.solutions.pose.Pose) → pin a una versione che la include ancora.
BASE_PKGS = ["mediapipe==0.10.18", "opencv-python", "numpy"]


def run(cmd: list) -> None:
    print("[mocap-setup] $", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    want_freemocap = "--freemocap" in sys.argv
    if not PY.exists():
        print(f"[mocap-setup] creo il venv {VENV}")
        run([sys.executable, "-m", "venv", str(VENV)])
    run([str(PY), "-m", "pip", "install", "--upgrade", "pip", "wheel"])
    run([str(PY), "-m", "pip", "install", *BASE_PKGS])
    if want_freemocap:
        print("[mocap-setup] installo freemocap (AGPL, pacchetto grande)…")
        try:
            run([str(PY), "-m", "pip", "install", "freemocap"])
        except subprocess.CalledProcessError as e:
            print(f"[mocap-setup] freemocap non installato ({e}). "
                  "La cattura MediaPipe e l'import di sessioni esistenti restano disponibili.")
    # smoke test
    run([str(PY), "-c", "import mediapipe, cv2, numpy; print('mocap env OK')"])
    print(f"[mocap-setup] fatto. Interprete: {PY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
