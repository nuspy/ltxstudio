"""Setup test GUI: Qt headless (offscreen) + root del progetto sul sys.path.
Eseguire con il venv dell'app:  E:\\AI\\Comfy_Nexus\\ComfyUI\\venv\\Scripts\\python.exe -m pytest tests -q
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # nessuna finestra reale, nessuna GPU
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")    # prima di qualunque import di cv2 (altrimenti i test EXR dipendono dall ordine)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
