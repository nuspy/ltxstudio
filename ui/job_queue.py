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
import time

from PySide6.QtCore import QObject, QTimer, Signal, Qt
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


_MEDIA = (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".webm", ".wav", ".mp3", ".glb", ".obj")


def _snap(p) -> dict:
    """Copia dei parametri del worker con soli valori JSON-safe (niente oggetti Qt/tensori)."""
    if not isinstance(p, dict):
        return {}

    def ok(v, d=0):
        if isinstance(v, (str, int, float, bool)) or v is None:
            return True
        if d < 3 and isinstance(v, (list, tuple)):
            return all(ok(x, d + 1) for x in v)
        if d < 3 and isinstance(v, dict):
            return all(isinstance(k, str) and ok(x, d + 1) for k, x in v.items())
        return False
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items() if ok(v)}


def _media_paths(obj, out=None, depth=0) -> list:
    import os
    out = [] if out is None else out
    if depth > 4:
        return out
    if isinstance(obj, str):
        if obj.lower().endswith(_MEDIA) and os.path.isfile(obj) and obj not in out:
            out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _media_paths(v, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _media_paths(v, out, depth + 1)
    return out


class _Job:
    __slots__ = ("id", "label", "worker", "on_done", "on_progress", "status", "result", "params", "outputs")

    def __init__(self, jid: int, label: str, worker, on_done, on_progress):
        self.id = jid
        self.label = label
        self.worker = worker            # QThread (costruito, NON avviato) con segnale done(dict)
        self.on_done = on_done
        self.on_progress = on_progress
        self.status = "queued"
        self.result: dict = {}
        self.params: dict = _snap(getattr(worker, "params", None))     # copia JSON-safe per «reimposta parametri»
        self.outputs: list = []


class GenerationQueue(QObject):
    """Esegue i job di generazione uno alla volta, in ordine di arrivo."""

    changed = Signal()                  # la lista è cambiata (per il pannello)
    job_started = Signal(object)
    job_finished = Signal(object, dict)
    gpu_waiting = Signal(str)           # la GPU è di un ALTRO processo (tag del proprietario)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._jobs: list[_Job] = []
        self._running: _Job | None = None
        self._ids = itertools.count(1)
        self._gpu_wait_started: float | None = None
        self._gpu_wait_logged = 0.0

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
                    self._release_lock()
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
        # LOCK GPU INTER-PROCESSO («un solo job GPU pesante alla volta», per davvero): la coda
        # seriale copre solo la GUI, ma API :8200 / training / musica sono processi/percorsi
        # propri. Acquisizione NON-bloccante (mai congelare il thread GUI: un training può tenere
        # la GPU per ore) → retry ogni 2s + avviso in statusbar con CHI la tiene.
        try:
            from core import gpu_lock
            if not gpu_lock.acquire(f"gui:{nxt.label}", timeout_s=0):
                h = gpu_lock.holder() or {}
                self.gpu_waiting.emit(str(h.get("tag", "?")))
                self._log_gpu_wait(nxt, h)
                QTimer.singleShot(2000, self._pump)
                return
            self._gpu_wait_started = None
        except Exception as exc:                             # noqa: BLE001 — senza lock il job parte lo stesso, ma si dice perche
            print(f"[gpu-lock] ERRORE nel lock GPU ({type(exc).__name__}: {exc}): il job «{nxt.label}» parte senza lock", flush=True)
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

    GPU_WAIT_LOG_EVERY_S = 60.0

    def _log_gpu_wait(self, job: _Job, holder_info: dict) -> None:
        """Una riga nel log ogni minuto con CHI tiene la GPU e da quanto il job aspetta: prima l'attesa era muta (04/11: lock a zeri dopo un riavvio, job fermo per ore senza una riga)."""
        now = time.monotonic()
        if self._gpu_wait_started is None:
            self._gpu_wait_started, self._gpu_wait_logged = now, now - self.GPU_WAIT_LOG_EVERY_S
        if now - self._gpu_wait_logged < self.GPU_WAIT_LOG_EVERY_S:
            return
        self._gpu_wait_logged = now
        owner = f"pid {holder_info.get('pid')} ({holder_info.get('tag')})" if holder_info else "illeggibile (lock vuoto o corrotto)"
        print(f"[gpu-lock] il job «{job.label}» aspetta la GPU da {now - self._gpu_wait_started:.0f} s: tenuta da {owner}", flush=True)

    @staticmethod
    def _release_lock() -> None:
        try:
            from core import gpu_lock
            gpu_lock.release()
        except Exception:                                    # noqa: BLE001
            pass

    def _finish(self, job: _Job, result: dict) -> None:
        if job.status != "cancelled":
            job.status = "error" if result.get("status") == "error" else "done"
            job.result = result
            try:
                job.outputs = _media_paths(result)
                if not job.outputs and isinstance(job.params, dict) and job.params.get("output"):
                    job.outputs = _media_paths(job.params["output"])
            except Exception:                                # noqa: BLE001
                job.outputs = []
        if self._running is job:
            self._running = None
            self._release_lock()
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

    restore_requested = Signal(dict, str)           # (parametri del lavoro, etichetta): la finestra principale li rimette nella GUI

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
        # Lista lavori precedenti (punto 5): visualizza risultato, sidecar, reimposta parametri, rimuovi dalla lista, cancella risultato.
        row2 = QHBoxLayout()
        self.btn_view = QPushButton(tr("queue.view", "Visualizza risultato"))
        self.btn_sidecar = QPushButton(tr("queue.sidecar", "Sidecar"))
        self.btn_restore = QPushButton(tr("queue.restore", "Reimposta parametri"))
        self.btn_remove = QPushButton(tr("queue.remove", "Rimuovi dalla lista"))
        self.btn_delete = QPushButton(tr("queue.delete", "Cancella risultato"))
        for b, fn in ((self.btn_view, self._view), (self.btn_sidecar, self._sidecar), (self.btn_restore, self._restore),
                      (self.btn_remove, self._remove), (self.btn_delete, self._delete_result)):
            b.clicked.connect(fn)
            row2.addWidget(b)
        v.addLayout(row2)
        self.q.changed.connect(self.refresh)
        self.refresh()

    # --- azioni sul lavoro selezionato ---
    def _sel(self):
        it = self.list.currentItem()
        if it is None:
            return None
        jid = int(it.data(Qt.UserRole))
        return next((j for j in self.q.jobs() if j.id == jid), None)

    def _view(self) -> None:
        j = self._sel()
        if not j or not j.outputs:
            return
        import os
        path = j.outputs[0]
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            from PySide6.QtGui import QPixmap
            from PySide6.QtWidgets import QDialog, QScrollArea
            dlg = QDialog(self)
            dlg.setWindowTitle(os.path.basename(path))
            dlg.setModal(False)
            lay = QVBoxLayout(dlg)
            lab = QLabel()
            pm = QPixmap(path)
            lab.setPixmap(pm.scaled(960, 960, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            sc = QScrollArea()
            sc.setWidget(lab)
            lay.addWidget(sc)
            dlg.resize(1000, 800)
            self._dlg = dlg
            dlg.show()
        elif path.lower().endswith((".mp4", ".mov", ".webm")):
            try:
                from ui.video_player_dialog import VideoPlayerDialog
                d = VideoPlayerDialog(self)
                d.play_file(path)
                d.setModal(False)
                self._dlg = d
                d.show()
            except Exception:                                # noqa: BLE001
                os.startfile(path)                           # noqa: S606 - ripiego: lettore di sistema
        else:
            os.startfile(path)                               # noqa: S606

    def _sidecar(self) -> None:
        j = self._sel()
        if not j or not j.outputs:
            return
        import json
        from core.metadata_saver import MetadataSaver
        sp = MetadataSaver.sidecar_path(j.outputs[0])
        try:
            txt = sp.read_text(encoding="utf-8") if sp.exists() else json.dumps(j.params, ensure_ascii=False, indent=2)
        except Exception as e:                               # noqa: BLE001
            txt = str(e)
        from ui.widgets.msg_dialog import show_message
        show_message(self, tr("queue.sidecar", "Sidecar"), txt, "information")

    def _restore(self) -> None:
        j = self._sel()
        if j and j.params:
            self.restore_requested.emit(dict(j.params), j.label)

    def _remove(self) -> None:
        j = self._sel()
        if j and j.status not in ("queued", "running"):
            self.q._jobs = [x for x in self.q._jobs if x.id != j.id]
            self.q.changed.emit()

    def _delete_result(self) -> None:
        j = self._sel()
        if not j or not j.outputs:
            return
        import os
        from PySide6.QtWidgets import QMessageBox
        names = "\n".join(os.path.basename(p) for p in j.outputs)
        if QMessageBox.question(self, tr("queue.delete", "Cancella risultato"),
                                tr("queue.delete.q", "Cancellare dal disco questi file?\n{n}").format(n=names)) != QMessageBox.Yes:
            return
        for pth in j.outputs:
            try:
                os.remove(pth)
            except OSError:
                pass
        j.outputs = []
        self.q.changed.emit()

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
