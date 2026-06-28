"""Coda di generazione SERIALE — un job alla volta, niente sovrapposizioni né cancellazioni reciproche.

Problema risolto: i worker di generazione (QThread: immagini, video, a2v, 3D, restauro, upscale, ecc.)
venivano avviati con `.start()` diretto e salvati in attributi RIUSATI (es. `self._ltx2_worker`),
quindi due generazioni da tab diversi potevano sovrapporsi (contesa GPU/OOM) o annullarsi a vicenda
(riferimento sovrascritto → QThread orfano). Qui ogni richiesta viene ACCODATA: parte un solo job per
volta; un nuovo job non cancella mai quello in corso, si mette in lista. Pannello visibile con stato +
annulla per singolo job.

Uso (in main_window):
    self.jobs = GenerationQueue(self)
    self.jobs.enqueue("Video t2v", worker, on_done=self._on_done, on_progress=self._log)
"""
from __future__ import annotations

import itertools

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QLabel,
)

try:
    from ui.i18n_manager import tr
except Exception:                                            # noqa: BLE001
    def tr(key, default=None):                               # fallback se i18n non disponibile
        return default if default is not None else key


_STATUS = {
    "queued":    ("⏳", "queue.st.queued",    "in coda"),
    "running":   ("▶",  "queue.st.running",   "in esecuzione"),
    "done":      ("✅", "queue.st.done",      "completato"),
    "error":     ("❌", "queue.st.error",     "errore"),
    "cancelled": ("⛔", "queue.st.cancelled", "annullato"),
}


class _Job:
    __slots__ = ("id", "label", "worker", "on_done", "on_progress", "status", "result")

    def __init__(self, jid: int, label: str, worker, on_done, on_progress):
        self.id = jid
        self.label = label
        self.worker = worker            # QThread (costruito, NON avviato) con segnale done(dict)
        self.on_done = on_done
        self.on_progress = on_progress
        self.status = "queued"
        self.result: dict = {}


class GenerationQueue(QObject):
    """Esegue i job di generazione uno alla volta, in ordine di arrivo."""

    changed = Signal()                  # la lista è cambiata (per il pannello)
    job_started = Signal(object)
    job_finished = Signal(object, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs: list[_Job] = []
        self._running: _Job | None = None
        self._ids = itertools.count(1)

    # ---------- API ----------
    def jobs(self) -> list[_Job]:
        return list(self._jobs)

    def is_busy(self) -> bool:
        return self._running is not None

    def enqueue(self, label: str, worker, on_done=None, on_progress=None) -> int:
        """Accoda un worker (QThread con `done(dict)` e opz. `progress(str)`), GIÀ costruito ma NON avviato."""
        job = _Job(next(self._ids), label, worker, on_done, on_progress)
        self._jobs.append(job)
        self.changed.emit()
        self._pump()
        return job.id

    def cancel(self, job_id: int) -> None:
        for j in self._jobs:
            if j.id != job_id:
                continue
            if j.status == "running":
                self._stop_worker(j.worker)
                j.status = "cancelled"
                if self._running is j:
                    self._running = None
                self.changed.emit()
                self._pump()
            elif j.status == "queued":
                j.status = "cancelled"
                self.changed.emit()
            return

    def clear_finished(self) -> None:
        self._jobs = [j for j in self._jobs if j.status in ("queued", "running")]
        self.changed.emit()

    # ---------- motore ----------
    def _pump(self) -> None:
        if self._running is not None:
            return
        nxt = next((j for j in self._jobs if j.status == "queued"), None)
        if nxt is None:
            return
        self._running = nxt
        nxt.status = "running"
        w = nxt.worker
        if nxt.on_progress is not None and hasattr(w, "progress"):
            try:
                w.progress.connect(nxt.on_progress)
            except Exception:                                # noqa: BLE001
                pass
        w.done.connect(lambda res=None, j=nxt: self._finish(j, res if isinstance(res, dict) else {}))
        self.job_started.emit(nxt)
        self.changed.emit()
        w.start()

    def _finish(self, job: _Job, result: dict) -> None:
        if job.status != "cancelled":
            job.status = "error" if result.get("status") == "error" else "done"
            job.result = result
        if self._running is job:
            self._running = None
        try:
            if job.on_done is not None and job.status != "cancelled":
                job.on_done(result)
        finally:
            self.job_finished.emit(job, result)
            self.changed.emit()
            self._pump()

    @staticmethod
    def _stop_worker(worker) -> None:
        # i worker con subprocess espongono stop() (kill del processo figlio); gli altri: terminate().
        try:
            if hasattr(worker, "stop"):
                worker.stop()
        except Exception:                                    # noqa: BLE001
            pass
        try:
            worker.requestInterruption()
        except Exception:                                    # noqa: BLE001
            pass
        try:
            worker.terminate()
        except Exception:                                    # noqa: BLE001
            pass


class JobQueuePanel(QWidget):
    """Pannello visibile della coda: lista job con stato + annulla/pulisci."""

    def __init__(self, queue: GenerationQueue, parent=None):
        super().__init__(parent)
        self.q = queue
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        self.hint = QLabel(tr("queue.empty", "Nessuna generazione in coda."))
        self.hint.setStyleSheet("color:#9aa1b0;")
        v.addWidget(self.hint)
        self.list = QListWidget()
        v.addWidget(self.list, 1)
        row = QHBoxLayout()
        self.btn_cancel = QPushButton(tr("queue.cancel", "Annulla selezionato"))
        self.btn_clear = QPushButton(tr("queue.clear", "Pulisci completati"))
        self.btn_cancel.clicked.connect(self._cancel_selected)
        self.btn_clear.clicked.connect(self.q.clear_finished)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_clear)
        v.addLayout(row)
        self.q.changed.connect(self.refresh)
        self.refresh()

    def _cancel_selected(self) -> None:
        it = self.list.currentItem()
        if it is not None:
            self.q.cancel(int(it.data(Qt.UserRole)))

    def refresh(self) -> None:
        self.list.clear()
        jobs = self.q.jobs()
        self.hint.setVisible(not jobs)
        for j in jobs:
            icon, key, default = _STATUS.get(j.status, ("•", "queue.st.unknown", j.status))
            item = QListWidgetItem(f"{icon}  {j.label}  —  {tr(key, default)}")
            item.setData(Qt.UserRole, j.id)
            self.list.addItem(item)
