"""Compila l'ATF Detector in un .exe AUTONOMO (Windows), one-file.

Il bundle include SOLO ciò che serve al Detector: detector_gui.py + core/stego.py (logica + chiave) +
numpy + opencv + PySide6. NON include il resto di LTX Studio. Distribuibile come strumento indipendente.

Uso:  python tools/atf_detector/build_detector.py
Output: tools/atf_detector/dist/ATF-Detector.exe

NB: scarica PyInstaller la prima volta (connessione). Per offuscare la chiave nel binario si può passare
a Nuitka; qui usiamo PyInstaller per semplicità (la chiave resta in core/stego.py compilato — vedi nota
sicurezza nel modulo).
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
GUI = os.path.join(HERE, "detector_gui.py")
STEGO = os.path.join(ROOT, "core", "stego.py")


def main() -> int:
    py = sys.executable
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("Installo PyInstaller…")
        subprocess.run([py, "-m", "pip", "install", "pyinstaller"], check=True)
    sep = ";" if os.name == "nt" else ":"
    cmd = [
        py, "-m", "PyInstaller", "--noconfirm", "--onefile", "--windowed",
        "--name", "ATF-Detector",
        # includi core/stego.py mantenendo il package path 'core/'
        "--add-data", f"{STEGO}{sep}core",
        "--hidden-import", "cv2", "--hidden-import", "numpy",
        "--distpath", os.path.join(HERE, "dist"),
        "--workpath", os.path.join(HERE, "build"),
        "--specpath", HERE,
        GUI,
    ]
    print("Compilo:", " ".join(cmd))
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
