"""Lato GUI del mocap: avvia il worker (venv isolato `mocap_env`) e ne relaya
gli eventi via segnali Qt. Il worker (workers/mocap_worker.py) cattura la webcam
con MediaPipe e produce un BVH. Isolamento in processo/venv anche per la licenza
AGPL di FreeMoCap.
"""
from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal

ROOT = Path(__file__).resolve().parents[1]
MOCAP_PY = ROOT / "mocap_env" / "Scripts" / "python.exe"
WORKER = ROOT / "workers" / "mocap_worker.py"
_CREATE_NO_WINDOW = 0x08000000


def mocap_env_ready() -> bool:
    return MOCAP_PY.exists() and WORKER.is_file()


class MocapService(QObject):
    """Gestisce il processo worker mocap e i suoi eventi."""

    ready = Signal()
    frame = Signal(str, int)        # path anteprima, n frame
    finished_ok = Signal(str, int)  # path bvh, n frame
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None

    def start(self, out_bvh: str, preview_path: str, camera: int = 0,
              fps: int = 30, max_seconds: float = 0) -> bool:
        if not mocap_env_ready():
            self.failed.emit("Ambiente mocap assente: esegui scripts/install_mocap.py")
            return False
        if self._proc is not None:
            self.failed.emit("Cattura già in corso")
            return False
        try:
            self._proc = subprocess.Popen(
                [str(MOCAP_PY), str(WORKER)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
                creationflags=_CREATE_NO_WINDOW)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(f"Avvio worker mocap fallito: {e}")
            return False
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._send({"cmd": "start", "camera": camera, "out": out_bvh,
                    "fps": fps, "preview": preview_path, "max_seconds": max_seconds})
        return True

    def stop(self):
        self._send({"cmd": "stop"})

    def _send(self, obj: dict):
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(json.dumps(obj) + "\n")
                self._proc.stdin.flush()
            except Exception:
                pass

    def _read_loop(self):
        proc = self._proc
        if not proc or not proc.stdout:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = ev.get("event")
            if kind == "ready":
                self.ready.emit()
            elif kind == "frame":
                self.frame.emit(ev.get("preview", ""), int(ev.get("n", 0)))
            elif kind == "done":
                self.finished_ok.emit(ev.get("bvh", ""), int(ev.get("frames", 0)))
                break
            elif kind == "error":
                self.failed.emit(ev.get("msg", "errore mocap"))
                break
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        self._proc = None
