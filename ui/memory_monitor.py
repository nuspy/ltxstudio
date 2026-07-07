"""Finestra NON-MODALE movibile/ridimensionabile: grafici live di RAM, VRAM dedicata, VRAM shared e % GPU
(core CUDA), update 1s. Apri/chiudi dal pulsante nelle 'orecchie' del tab (accanto al pannello coda) o con la X.
Testo e grafici scalano con la dimensione della finestra.
"""
from __future__ import annotations

from collections import deque

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QWidget

try:
    from ui.i18n import tr
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
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)

    # apri/chiudi → avvia/ferma il polling (niente carico quando è chiusa)
    def showEvent(self, e):
        self._tick()
        self._timer.start()
        super().showEvent(e)

    def hideEvent(self, e):
        self._timer.stop()
        super().hideEvent(e)

    def resizeEvent(self, e):                    # scala il TESTO con l'altezza della finestra
        pt = max(8, min(16, int(self.height() / 26)))
        for head, _ in self._rows.values():
            f = head.font()
            f.setPointSize(pt)
            head.setFont(f)
        super().resizeEvent(e)

    def _tick(self):
        try:
            from core import sysmon
            s = sysmon.sample()
        except Exception:                        # noqa: BLE001
            return
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
