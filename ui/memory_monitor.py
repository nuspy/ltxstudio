"""Finestra NON-MODALE movibile/ridimensionabile: grafici live di RAM, VRAM dedicata, VRAM shared e % GPU
(core CUDA), update 1s. Apri/chiudi dal pulsante nelle 'orecchie' del tab (accanto al pannello coda) o con la X.
Testo e grafici scalano con la dimensione della finestra.
"""
from __future__ import annotations

from collections import deque

from PySide6.QtCore import Qt, QPointF, QThread, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (QDialog, QHeaderView, QLabel, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

try:
    from ui.i18n_manager import tr                         # fix: era `ui.i18n` (inesistente) → sempre fallback EN
except Exception:                                          # noqa: BLE001
    def tr(_k, d=""):                                      # fallback
        return d


class _Spark(QWidget):
    """Grafico ad area a scorrimento (ultime N letture), si ridisegna scalando con il widget."""

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self._vals: deque[float] = deque([0.0] * 90, maxlen=90)
        self._c = QColor(color)

    def push(self, frac: float):
        # SANITIZZAZIONE CRITICA: NaN/inf NON vanno mai messi in _vals — un QPointF con coordinata NaN passato a
        # QPolygonF fa ACCESS VIOLATION in Qt6Gui.dll (crash nativo dell'app, visto nel log eventi Windows).
        # Il clamp Python NON basta: `min(1.0, nan)` restituisce nan. Durante il 3D il /free cambia la VRAM di
        # colpo e una lettura PDH transitoria può dare un valore non-finito → qui lo azzeriamo.
        try:
            import math
            f = float(frac)
            if not math.isfinite(f):
                f = 0.0
        except Exception:  # noqa: BLE001
            f = 0.0
        self._vals.append(max(0.0, min(1.0, f)))
        self.update()

    def paintEvent(self, _e):
        try:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            w, h = self.width(), self.height()
            p.fillRect(self.rect(), QColor(18, 18, 24))
            p.setPen(QPen(QColor(46, 46, 58), 1))
            for i in range(1, 4):
                p.drawLine(0, h * i // 4, w, h * i // 4)
            n = len(self._vals)
            if n < 2:
                return
            import math
            pts = [QPointF(x * w / (n - 1), h - v * h)
                   for x, v in enumerate(self._vals) if math.isfinite(v)]   # solo punti finiti (extra difesa)
            if len(pts) < 2:
                return
            grad = QLinearGradient(0, 0, 0, h)
            grad.setColorAt(0.0, QColor(self._c.red(), self._c.green(), self._c.blue(), 150))
            grad.setColorAt(1.0, QColor(self._c.red(), self._c.green(), self._c.blue(), 18))
            p.setBrush(grad)
            p.setPen(Qt.NoPen)
            p.drawPolygon(QPolygonF(pts + [QPointF(w, h), QPointF(0.0, h)]))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(self._c, 2))
            p.drawPolyline(QPolygonF(pts))
        except Exception:  # noqa: BLE001  (un paintEvent non deve MAI propagare: rischio crash nativo)
            pass


class _Sampler(QThread):
    """Campionatore in THREAD di sfondo: nvidia-smi/PDH (sysmon) + HTTP /system_stats dei server
    (residents_detailed). PRIMA girava tutto nel QTimer del thread GUI: con un server occupato dal
    denoise (GIL saturo) le 4 chiamate HTTP (timeout 2s l'una) + nvidia-smi sotto carico (1-3s)
    congelavano l'app per svariati secondi A OGNI TICK durante le generazioni (visto live 22/07).
    Qui la GUI riceve solo i DATI via signal e non blocca mai."""
    sampled = Signal(dict, object)                       # (sysmon.sample(), rows|None)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        import time as _t
        n = 0
        while not self._stop:
            try:
                from core import sysmon
                s = sysmon.sample()
            except Exception:                            # noqa: BLE001
                s = None
            rows = None
            if n % 2 == 1:                               # tabella modelli ogni 2s (metà frequenza)
                try:
                    from core import gpu_registry
                    rows = gpu_registry.residents_detailed()
                except Exception:                        # noqa: BLE001
                    rows = None
            if s is not None and not self._stop:
                self.sampled.emit(s, rows)
            n += 1
            _t.sleep(1.0)


class MemoryMonitor(QDialog):
    _GB = 1024 ** 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("mem.title", "Monitor memoria"))
        self.setWindowFlags(Qt.Window)          # movibile, ridimensionabile, con X — non modale
        self.setModal(False)
        self.setMinimumSize(280, 240)
        self.resize(380, 380)
        lay = QVBoxLayout(self)
        lay.setSpacing(6)
        self._rows: dict[str, tuple[QLabel, _Spark]] = {}
        for key, color in (("ram", "#74c0fc"), ("ded", "#63e6be"),
                           ("sha", "#ffa94d"), ("gpu", "#f783ac")):
            head = QLabel("—")
            head.setStyleSheet("font-weight:600;")
            spark = _Spark(color)
            lay.addWidget(head)
            lay.addWidget(spark, 1)
            self._rows[key] = (head, spark)
        # F4 unified-model-manager: TABELLA dei modelli in VRAM / parcheggiati in RAM (ledger
        # cross-venv + "parked" dei server). Verde = in VRAM, giallo = parcheggiato in RAM.
        self._models_head = QLabel(tr("mem.models", "Modelli caricati"))
        self._models_head.setStyleSheet("font-weight:600;")
        self._tbl = QTableWidget(0, 4)
        self._tbl.setHorizontalHeaderLabels([
            tr("mem.col.model", "Modello"), tr("mem.col.mb", "MB"),
            tr("mem.col.state", "Stato"), tr("mem.col.proc", "Processo")])
        self._tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._tbl.verticalHeader().setVisible(False)
        self._tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self._tbl.setMinimumHeight(90)
        lay.addWidget(self._models_head)
        lay.addWidget(self._tbl, 1)
        # Cache degli embedding di prompt: sta qui perché è l'altra memoria che il software occupa
        # per conto dell'utente — su disco invece che in VRAM. Solo os.scandir, nessun torch.
        from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget
        _cr = QHBoxLayout()
        _cr.setContentsMargins(0, 0, 0, 0)
        self._emb_lbl = QLabel("—")
        _cr.addWidget(self._emb_lbl, 1)
        self._emb_btn = QPushButton(tr("emb.cache.clear", "Svuota"))
        self._emb_btn.clicked.connect(self._clear_emb_cache)
        _cr.addWidget(self._emb_btn)
        _cw = QWidget(); _cw.setLayout(_cr)
        lay.addWidget(_cw)
        self._refresh_emb_cache()
        self._sampler: _Sampler | None = None   # thread di campionamento (vivo solo a finestra aperta)

    # apri/chiudi → avvia/ferma il polling (niente carico quando è chiusa)
    def showEvent(self, e):
        if self._sampler is None or not self._sampler.isRunning():
            self._sampler = _Sampler(self)
            self._sampler.sampled.connect(self._on_sample)
            self._sampler.start()
        super().showEvent(e)

    def hideEvent(self, e):
        if self._sampler is not None:
            self._sampler.stop()
            self._sampler = None                # il thread esce da solo entro ~1s (loop con flag)
        super().hideEvent(e)

    def resizeEvent(self, e):                    # scala il TESTO con l'altezza della finestra
        pt = max(8, min(16, int(self.height() / 26)))
        for head, _ in self._rows.values():
            f = head.font()
            f.setPointSize(pt)
            head.setFont(f)
        super().resizeEvent(e)

    def _refresh_emb_cache(self):
        """Stato della cache embedding. Nessun torch: emb_cache è stdlib-only all'import."""
        try:
            from core import emb_cache
            st = emb_cache.stats()
            self._emb_lbl.setText(
                tr("emb.cache.line", "Cache prompt: {n} voci · {mb} / {bud} MB")
                .format(n=st["n"], mb=f"{st['mb']:.0f}", bud=f"{st['budget_mb']:.0f}")
                if st["n"] else tr("emb.cache.empty", "Cache prompt: vuota"))
            self._emb_btn.setEnabled(bool(st["n"]))
        except Exception:                                # noqa: BLE001 — il monitor non deve mai rompersi
            self._emb_lbl.setText(tr("emb.cache.empty", "Cache prompt: vuota"))
            self._emb_btn.setEnabled(False)

    def _clear_emb_cache(self):
        try:
            from core import emb_cache
            emb_cache.clear()
        except Exception:                                # noqa: BLE001
            pass
        self._refresh_emb_cache()

    def _on_sample(self, s: dict, rows):
        """Aggiorna i widget coi dati arrivati dal thread campionatore (solo painting, mai I/O)."""
        self._refresh_emb_cache()                        # ~0,1 ms: os.scandir su una manciata di file
        g = self._GB

        def _row(key, title, used, total, pct_txt=None):
            head, spark = self._rows[key]
            frac = (used / total) if total else 0.0
            pct = pct_txt if pct_txt is not None else f"{frac * 100:.0f}%"
            if total:
                head.setText(f"{title}   {used / g:.1f} / {total / g:.0f} GB   ({pct})")
            else:
                head.setText(f"{title}   {used / g:.1f} GB")
            spark.push(frac if pct_txt is None else used / 100.0)

        _row("ram", tr("mem.ram", "RAM"), s["ram_used"], s["ram_total"])
        _row("ded", tr("mem.vram_ded", "VRAM dedicata"), s["vram_ded_used"], s["vram_ded_total"])
        _row("sha", tr("mem.vram_shared", "VRAM shared"), s["vram_shared_used"], s["vram_shared_total"])
        # GPU util: percentuale pura (grafico su 0-100), niente GB
        head, spark = self._rows["gpu"]
        head.setText(f"{tr('mem.gpu', 'GPU (CUDA)')}   {s['gpu_util']}%")
        spark.push(s["gpu_util"] / 100.0)
        if rows is not None:
            self._refresh_models(rows)

    def _refresh_models(self, rows: list):
        """F4: tabella dei modelli residenti (dati già raccolti dal thread campionatore)."""
        _vram = QColor(46, 160, 96)
        _park = QColor(196, 148, 32)
        self._tbl.setRowCount(len(rows))
        for i, r in enumerate(rows):
            name = QTableWidgetItem(str(r["key"]))
            mb = QTableWidgetItem(f"{r['mb']:.0f}")
            mb.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            state = QTableWidgetItem(tr("mem.state.vram", "VRAM") if r["state"] == "vram"
                                     else tr("mem.state.parked", "RAM (parcheggiato)"))
            state.setForeground(_vram if r["state"] == "vram" else _park)
            proc = QTableWidgetItem(f"{r['venv']} #{r['pid']}" if r["pid"] else str(r["venv"]))
            for c, it in enumerate((name, mb, state, proc)):
                self._tbl.setItem(i, c, it)
