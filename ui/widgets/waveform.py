"""Player audio con forma d'onda + selezione regione (trim) — tab Musica.

WaveformView disegna la forma d'onda di un file audio (mix mono, picchi min/max per colonna), con:
 - click = seek (posiziona la testina), trascinamento = selezione di una regione [start,end] in secondi;
 - trascinamento dei bordi della regione per aggiustarla.
WaveformPlayer aggiunge il trasporto (play/pausa, tempo) via QtMultimedia e i pulsanti regione.
La regione selezionata alimenta i task repaint/extend/audio2audio (trim visuale invece di digitare i secondi).
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal, QUrl, QRectF
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy

from ui.i18n_manager import tr

_HANDLE_PX = 6                                             # tolleranza px per afferrare un bordo regione


class WaveformView(QWidget):
    """Disegna la forma d'onda + testina + regione selezionata. Emette seek e region_changed."""
    seek_requested = Signal(float)                        # secondi
    region_changed = Signal(float, float)                # start, end (secondi)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._peaks: list[tuple[float, float]] = []       # (min,max) normalizzati [-1,1] per colonna
        self._dur = 0.0
        self._pos = 0.0                                   # posizione testina (s)
        self._sel: tuple[float, float] | None = None      # regione (s)
        self._drag = None                                 # None | 'new' | 'start' | 'end'
        self._press_x = 0

    # ── caricamento ───────────────────────────────────────────────────────────
    def load(self, path: str) -> bool:
        try:
            import soundfile as sf
            import numpy as np
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            mono = data.mean(axis=1)
            self._dur = len(mono) / float(sr) if sr else 0.0
            cols = max(200, self.width() or 600)
            step = max(1, len(mono) // cols)
            peaks = []
            for i in range(0, len(mono), step):
                chunk = mono[i:i + step]
                if len(chunk):
                    peaks.append((float(chunk.min()), float(chunk.max())))
            m = max((abs(a) for p in peaks for a in p), default=1.0) or 1.0
            self._peaks = [(mn / m, mx / m) for mn, mx in peaks]
        except Exception:                                  # noqa: BLE001 — file assente/illeggibile
            self._peaks = []; self._dur = 0.0
        self._pos = 0.0; self._sel = None
        self.update()
        return bool(self._peaks)

    def clear(self):
        self._peaks = []; self._dur = 0.0; self._pos = 0.0; self._sel = None; self.update()

    def set_position(self, seconds: float):
        self._pos = max(0.0, min(seconds, self._dur))
        self.update()

    def duration(self) -> float:
        return self._dur

    def region(self) -> tuple[float, float] | None:
        return self._sel

    def clear_region(self):
        self._sel = None; self.update(); self.region_changed.emit(0.0, 0.0)

    # ── disegno ───────────────────────────────────────────────────────────────
    def paintEvent(self, _e):
        p = QPainter(self)
        w, h = self.width(), self.height(); mid = h / 2
        p.fillRect(0, 0, w, h, QColor("#12151c"))
        if not self._peaks:
            p.setPen(QColor("#5a6272"))
            p.drawText(self.rect(), Qt.AlignCenter, tr("wave.empty", "— nessun audio —"))
            return
        # regione selezionata
        if self._sel and self._dur > 0:
            x0 = self._sel[0] / self._dur * w; x1 = self._sel[1] / self._dur * w
            p.fillRect(QRectF(x0, 0, x1 - x0, h), QColor(90, 150, 240, 60))
            p.setPen(QPen(QColor("#74c0fc"), 1))
            p.drawLine(int(x0), 0, int(x0), h); p.drawLine(int(x1), 0, int(x1), h)
        # forma d'onda
        p.setPen(QColor("#6ea8fe"))
        n = len(self._peaks)
        for x in range(w):
            idx = int(x / w * n)
            if idx >= n:
                break
            mn, mx = self._peaks[idx]
            p.drawLine(x, int(mid - mx * mid), x, int(mid - mn * mid))
        # testina
        if self._dur > 0:
            px = int(self._pos / self._dur * w)
            p.setPen(QPen(QColor("#ff6b6b"), 2)); p.drawLine(px, 0, px, h)

    # ── interazione ───────────────────────────────────────────────────────────
    def _x_to_s(self, x: float) -> float:
        w = max(1, self.width())
        return max(0.0, min(x / w * self._dur, self._dur))

    def mousePressEvent(self, e):
        if not self._peaks or self._dur <= 0:
            return
        x = e.position().x(); w = max(1, self.width())
        self._press_x = x
        if self._sel:                                      # afferra un bordo della regione?
            x0 = self._sel[0] / self._dur * w; x1 = self._sel[1] / self._dur * w
            if abs(x - x0) <= _HANDLE_PX:
                self._drag = "start"; return
            if abs(x - x1) <= _HANDLE_PX:
                self._drag = "end"; return
        self._drag = "new"

    def mouseMoveEvent(self, e):
        if not self._drag or self._dur <= 0:
            return
        s = self._x_to_s(e.position().x())
        if self._drag == "new":
            a = self._x_to_s(self._press_x)
            self._sel = (min(a, s), max(a, s))
        elif self._drag == "start" and self._sel:
            self._sel = (min(s, self._sel[1]), self._sel[1])
        elif self._drag == "end" and self._sel:
            self._sel = (self._sel[0], max(s, self._sel[0]))
        self.update()

    def mouseReleaseEvent(self, e):
        if self._drag == "new" and abs(e.position().x() - self._press_x) < 4:
            self._sel = None                               # semplice click → seek, niente regione
            self.seek_requested.emit(self._x_to_s(e.position().x()))
        elif self._sel and self._sel[1] - self._sel[0] > 0.05:
            self.region_changed.emit(self._sel[0], self._sel[1])
        else:
            self._sel = None
        self._drag = None; self.update()


class WaveformPlayer(QWidget):
    """WaveformView + trasporto (play/pausa, tempo) + pulsanti regione. use_region(start,end) al click."""
    use_region = Signal(float, float)

    def __init__(self):
        super().__init__()
        self._path = ""
        from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
        self._audio = QAudioOutput()
        self._player = QMediaPlayer(); self._player.setAudioOutput(self._audio)
        self._build()
        self._player.positionChanged.connect(lambda ms: self.wave.set_position(ms / 1000.0))
        self._player.playbackStateChanged.connect(self._on_state)
        self.wave.seek_requested.connect(lambda s: self._player.setPosition(int(s * 1000)))
        self.wave.region_changed.connect(self._on_region)

    def _build(self):
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(4)
        self.wave = WaveformView(); v.addWidget(self.wave)
        row = QHBoxLayout()
        self.btn_play = QPushButton(tr("wave.play", "▶")); self.btn_play.setFixedWidth(40)
        self.btn_play.setCursor(Qt.PointingHandCursor); self.btn_play.clicked.connect(self._toggle)
        self.lbl_time = QLabel("0.0 / 0.0 s"); self.lbl_time.setStyleSheet("color:#8a93a6;")
        self.lbl_region = QLabel(""); self.lbl_region.setStyleSheet("color:#74c0fc;")
        self.btn_use = QPushButton(tr("wave.use_region", "✂ Usa regione come sezione"))
        self.btn_use.setToolTip(tr("wave.use_region.tt",
            "Imposta l'operazione ACE-Step su 'Repaint' con l'intervallo selezionato (rigenera solo quella parte)."))
        self.btn_use.setEnabled(False); self.btn_use.setCursor(Qt.PointingHandCursor)
        self.btn_use.clicked.connect(self._emit_region)
        self.btn_clear = QPushButton(tr("wave.clear_region", "✕")); self.btn_clear.setFixedWidth(32)
        self.btn_clear.setToolTip(tr("wave.clear_region.tt", "Cancella la regione selezionata."))
        self.btn_clear.clicked.connect(self._clear_region)
        row.addWidget(self.btn_play); row.addWidget(self.lbl_time); row.addStretch(1)
        row.addWidget(self.lbl_region); row.addWidget(self.btn_use); row.addWidget(self.btn_clear)
        v.addLayout(row)
        self._player.positionChanged.connect(self._upd_time)
        self._player.durationChanged.connect(self._upd_time)

    def load(self, path: str):
        self._path = str(path)
        ok = self.wave.load(self._path)
        self._player.setSource(QUrl.fromLocalFile(self._path) if ok else QUrl())
        self.btn_use.setEnabled(False); self.lbl_region.setText("")
        self._upd_time()

    def _toggle(self):
        from PySide6.QtMultimedia import QMediaPlayer
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_state(self, st):
        from PySide6.QtMultimedia import QMediaPlayer
        self.btn_play.setText(tr("wave.pause", "⏸") if st == QMediaPlayer.PlayingState else tr("wave.play", "▶"))

    def _upd_time(self, *_):
        dur = self.wave.duration() or (self._player.duration() / 1000.0)
        self.lbl_time.setText(f"{self._player.position()/1000.0:.1f} / {dur:.1f} s")

    def _on_region(self, a: float, b: float):
        if b - a > 0.05:
            self.lbl_region.setText(f"[{a:.1f}–{b:.1f}s]"); self.btn_use.setEnabled(True)
        else:
            self.lbl_region.setText(""); self.btn_use.setEnabled(False)

    def _clear_region(self):
        self.wave.clear_region(); self.lbl_region.setText(""); self.btn_use.setEnabled(False)

    def _emit_region(self):
        r = self.wave.region()
        if r:
            self.use_region.emit(r[0], r[1])
