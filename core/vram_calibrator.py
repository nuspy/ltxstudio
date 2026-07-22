"""CalibrationScheduler — misura livello-2 in background a GPU INATTIVA (F7 unified-model-manager).

Prende la lista `to_measure` dell'audit (F2) e misura UN modello alla volta lanciando il
measure-worker nel venv giusto (immagini → inference_server/measure_worker; video LTX →
workers/ltx2_measure). Subordinato ai job dell'utente:
  - parte SOLO se la GenerationQueue è vuota, nessun server è busy, la VRAM libera basta e il
    COMMIT disponibile supera size×1.2 (regola pagefile — vedi core/st_direct);
  - si CANCELLA all'istante se l'utente accoda un job (segnale queue.changed) e riprende a coda
    vuota dopo 60s di grazia;
  - prima di ogni misura passa da gpu_registry.before_job come un job normale (coordina i /free).
Modelli falliti/fuori budget → marcati `skipped` in memoria di sessione, mai ritentati qui.

Modulo GUI-side (usa QObject/QTimer del processo PySide6). Nessun torch nel processo GUI.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from PySide6.QtCore import QObject, QTimer, Signal

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKER_OF = {                                          # venv → (python, argv del measure-worker)
    "inference_server": (r"E:\AI\Tools\inference_server_venv\Scripts\python.exe",
                         ["-m", "inference_server.measure_worker"]),
    "ltx2": (r"E:\AI\Tools\ltx2_venv\Scripts\python.exe",
             [os.path.join(_REPO, "workers", "ltx2_measure.py")]),
}
_GRACE_S = 60                                           # attesa dopo l'ultimo job utente
_TICK_MS = 30_000


class CalibrationScheduler(QObject):
    progress = Signal(str)                              # riga di log per la GUI
    finished_all = Signal(int)                          # quanti misurati in questa sessione

    def __init__(self, queue, parent=None):
        super().__init__(parent)
        self._queue = queue                             # ui.job_queue.GenerationQueue
        self._targets: list[dict] = []
        self._skipped: set[str] = set()
        self._proc: subprocess.Popen | None = None
        self._cur: dict | None = None
        self._done = 0
        self._last_user_job = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._tick)
        try:
            queue.changed.connect(self._on_queue_changed)
        except Exception:                               # noqa: BLE001
            pass

    # ── API ──────────────────────────────────────────────────────────────────
    def set_targets(self, to_measure: list[dict]) -> None:
        self._targets = [dict(t) for t in (to_measure or []) if t.get("venv") in _WORKER_OF]

    def start(self) -> None:
        if self._targets:
            self._timer.start()
            self.progress.emit(f"[calibrazione] in attesa di GPU libera — {len(self._targets)} modelli in coda")

    def stop(self) -> None:
        self._timer.stop()
        self._cancel_proc("stop")

    # ── interni ──────────────────────────────────────────────────────────────
    def _on_queue_changed(self) -> None:
        """Un job utente è entrato/cambiato: la misura si toglie di mezzo ALL'ISTANTE."""
        self._last_user_job = time.monotonic()
        if self._proc is not None and self._queue.is_busy():
            self._cancel_proc("job utente in coda")

    def _cancel_proc(self, why: str) -> None:
        p, self._proc = self._proc, None
        cur, self._cur = self._cur, None
        if p is not None and p.poll() is None:
            try:
                p.kill()
            except Exception:                           # noqa: BLE001
                pass
            if cur:                                     # non consumato: torna in coda
                self._targets.insert(0, cur)
            self.progress.emit(f"[calibrazione] misura interrotta ({why}) — riprenderà a GPU libera")

    def _can_run(self, est_mb: float, size_mb: float) -> bool:
        try:
            if self._queue.is_busy():
                return False
        except Exception:                               # noqa: BLE001
            return False
        if time.monotonic() - self._last_user_job < _GRACE_S:
            return False
        try:
            from core import gpu_registry
            if any(r.busy for r in gpu_registry.residents()):
                return False
        except Exception:                               # noqa: BLE001
            pass
        try:
            from core import gpu_mem
            if gpu_mem.free_vram_mb() < est_mb + 2000:
                return False
        except Exception:                               # noqa: BLE001
            return False
        try:
            from core import st_direct
            if st_direct.commit_avail_mb() < size_mb * 1.2:
                return False                            # regola pagefile: mai sfondare il commit
        except Exception:                               # noqa: BLE001
            pass
        return True

    def _tick(self) -> None:
        if self._proc is not None:                      # misura in corso: controlla l'esito
            rc = self._proc.poll()
            if rc is None:
                return
            cur, self._cur, self._proc = self._cur, None, None
            if rc == 0:
                self._done += 1
                self.progress.emit(f"[calibrazione] misurato: {cur['name']} "
                                   f"({self._done} fatti, {len(self._targets)} rimasti)")
            else:
                self._skipped.add(cur["path"])
                self.progress.emit(f"[calibrazione] saltato (exit {rc}): {cur['name']}")
            return
        while self._targets and self._targets[0]["path"] in self._skipped:
            self._targets.pop(0)
        if not self._targets:
            self._timer.stop()
            self.progress.emit(f"[calibrazione] completata: {self._done} modelli misurati")
            self.finished_all.emit(self._done)
            return
        t = self._targets[0]
        est = float(t.get("est_mb") or t.get("size_mb") or 20000)
        if not self._can_run(est, float(t.get("size_mb") or est)):
            return
        try:                                            # coordina come un job vero (/free demote)
            from core import gpu_registry
            cat = "video" if t["venv"] == "ltx2" else "image"
            gpu_registry.before_job(cat, arch=t.get("family", ""), model=t["path"], need_mb=est)
        except Exception:                               # noqa: BLE001
            pass
        py, argv = _WORKER_OF[t["venv"]]
        if not os.path.exists(py):
            self._skipped.add(t["path"])
            return
        self._cur = self._targets.pop(0)
        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        try:
            self._proc = subprocess.Popen([py, "-u", *argv, t["path"]], cwd=_REPO, env=env,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                          creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
            self.progress.emit(f"[calibrazione] misuro {t['name']} (~{est/1024:.0f}GB, venv {t['venv']})…")
        except Exception as e:                          # noqa: BLE001
            self._skipped.add(t["path"])
            self._cur = None
            self.progress.emit(f"[calibrazione] avvio misura fallito: {e}")
