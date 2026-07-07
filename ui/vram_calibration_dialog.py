"""Dialog «Calibrazione VRAM» (tab Avanzate): carica/analizza tutti i modelli locali (immagini + video),
ne salva il footprint reale in cache e lo riusa per la gestione VRAM/offload. Incrementale (ricalcola solo i
nuovi) + invalidazione su richiesta. Vedi core/vram_calibration.py e core/vram_estimate.py."""
from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox,
                               QTableWidget, QTableWidgetItem, QProgressBar, QMessageBox, QHeaderView)

from core import vram_calibration as _cal
from ui.i18n_manager import tr


def _free_gpu_mb() -> int:
    # NVML/nvidia-smi: NON crea un contesto CUDA nel processo app (torch.cuda.* lo farebbe → in conflitto
    # col subprocess CUDA della generazione 3D = crash nativo su Blackwell).
    try:
        from core.gpu_mem import free_vram_mb
        return free_vram_mb()
    except Exception:                                        # noqa: BLE001
        return 0


class _CalWorker(QThread):
    prog = Signal(int, int, str, int)                       # i, n, name, mb
    done = Signal(dict)

    def __init__(self, force: bool):
        super().__init__()
        self._force = force

    def run(self):
        try:
            r = _cal.calibrate_all(force=self._force,
                                   on_progress=lambda i, n, nm, mb: self.prog.emit(i, n, nm, mb))
            self.done.emit(r)
        except Exception as e:                              # noqa: BLE001
            self.done.emit({"error": str(e)})


class VramCalibrationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("cal.title", "Calibrazione VRAM modelli"))
        self.resize(720, 520)
        self._worker = None
        v = QVBoxLayout(self)

        intro = QLabel(tr("cal.intro",
            "Analizza i modelli locali (immagini + video) e salva in cache il footprint VRAM reale, riusato "
            "per decidere cosa tenere in GPU e cosa offloadare. Incrementale: calcola solo i modelli nuovi."))
        intro.setWordWrap(True); intro.setStyleSheet("color:#aaa;padding:4px;")
        v.addWidget(intro)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([tr("cal.col.model", "Modello"),
                                              tr("cal.col.vram", "VRAM stimata"),
                                              tr("cal.col.state", "Stato (la tua GPU)")])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self.table, 1)

        self.bar = QProgressBar(); self.bar.setVisible(False)
        v.addWidget(self.bar)
        self.status = QLabel(""); self.status.setStyleSheet("color:#888;")
        v.addWidget(self.status)

        row = QHBoxLayout()
        self.force = QCheckBox(tr("cal.force", "Ricalcola tutti (ignora la cache)"))
        row.addWidget(self.force)
        row.addStretch(1)
        self.btn_inval = QPushButton(tr("cal.invalidate", "Invalida cache"))
        self.btn_inval.clicked.connect(self._invalidate)
        row.addWidget(self.btn_inval)
        self.btn_cal = QPushButton(tr("cal.run", "Calibra tutti"))
        self.btn_cal.clicked.connect(self._run)
        row.addWidget(self.btn_cal)
        v.addLayout(row)

        self._refresh_table()

    # ── tabella ──────────────────────────────────────────────────────────────
    def _refresh_table(self):
        free = _free_gpu_mb()
        models = _cal.enumerate_models()
        self.table.setRowCount(len(models))
        from core import vram_estimate as _ve
        for i, m in enumerate(models):
            mb = _ve.cache_get(m["path"]) or (m["size_mb"] if m["is_gguf"] else 0)
            self.table.setItem(i, 0, QTableWidgetItem(m["name"]))
            vi = QTableWidgetItem(f"{mb} MB" if mb else tr("cal.notyet", "— (da calibrare)"))
            self.table.setItem(i, 1, vi)
            st = QTableWidgetItem(self._state_label(mb, free))
            st.setForeground(self._state_color(mb, free))
            self.table.setItem(i, 2, st)
        self.status.setText(tr("cal.free", "VRAM libera ora: {gb:.1f} GB · {n} modelli").format(
            gb=free / 1024.0, n=len(models)))

    def _state_label(self, mb: int, free: int) -> str:
        if not mb:
            return "—"
        need = mb * 1.10 + 3200                              # +working + margine
        if need <= free:
            return tr("cal.state.gpu", "GPU residente (veloce)")
        if mb <= free:
            return tr("cal.state.partial", "Con offload encoder")
        return tr("cal.state.offload", "Offload/streaming")

    def _state_color(self, mb: int, free: int):
        from PySide6.QtGui import QColor
        if not mb:
            return QColor("#888")
        need = mb * 1.10 + 3200
        if need <= free:
            return QColor("#5cb85c")
        if mb <= free:
            return QColor("#f0ad4e")
        return QColor("#d9534f")

    # ── azioni ───────────────────────────────────────────────────────────────
    def _run(self):
        if self._worker and self._worker.isRunning():
            return
        self.btn_cal.setEnabled(False); self.btn_inval.setEnabled(False)
        self.bar.setVisible(True); self.bar.setValue(0)
        self._worker = _CalWorker(self.force.isChecked())
        self._worker.prog.connect(self._on_prog)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_prog(self, i, n, name, mb):
        self.bar.setMaximum(n); self.bar.setValue(i)
        self.status.setText(tr("cal.calibrating", "Calibro {i}/{n}: {name} = {mb} MB").format(
            i=i, n=n, name=name, mb=mb))

    def _on_done(self, r):
        self.btn_cal.setEnabled(True); self.btn_inval.setEnabled(True)
        self.bar.setVisible(False)
        if r.get("error"):
            QMessageBox.warning(self, tr("cal.title", "Calibrazione VRAM modelli"), r["error"])
        else:
            self.status.setText(tr("cal.done", "Fatto: {c} calibrati, {s} già in cache, {t} totali.").format(
                c=r.get("calibrated", 0), s=r.get("skipped", 0), t=r.get("total", 0)))
        self._refresh_table()

    def _invalidate(self):
        if QMessageBox.question(self, tr("cal.invalidate", "Invalida cache"),
                                tr("cal.invalidate.confirm", "Svuotare tutta la cache dei footprint VRAM?"),
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        n = _cal.invalidate(None)
        self.status.setText(tr("cal.invalidated", "{n} voci rimosse dalla cache.").format(n=n))
        self._refresh_table()
