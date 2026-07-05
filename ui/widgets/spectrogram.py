"""Spettrogramma + confronto A/B — tab Musica.

Rende lo spettrogramma (STFT in dB) di un file audio e calcola il CENTROIDE SPETTRALE medio (Hz),
la stessa metrica usata per diagnosticare aderenza/rumore: centroide alto = suono brillante (o rumore),
basso = suono cupo. Il dialog A/B affianca due generazioni e ne confronta il centroide (es. gs_lyric on/off,
RL3B vs 3B) in modo oggettivo.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDialog, QFileDialog,
)

from ui.i18n_manager import tr


def compute_spectrogram(path: str, n_fft: int = 1024, hop: int = 512):
    """Ritorna (img_uint8 [F×T], samplerate, centroide_medio_Hz) da un file audio. Grezzo ma senza scipy."""
    import soundfile as sf
    import numpy as np
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if len(mono) < n_fft:
        mono = np.pad(mono, (0, n_fft - len(mono)))
    win = np.hanning(n_fft).astype("float32")
    idx = range(0, len(mono) - n_fft, hop)
    frames = np.stack([mono[i:i + n_fft] * win for i in idx]) if idx else np.zeros((1, n_fft), "float32")
    spec = np.abs(np.fft.rfft(frames, axis=1))                # (T, F)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # centroide per-frame → media (ignora frame silenziosi)
    denom = spec.sum(axis=1)
    cframe = (spec * freqs).sum(axis=1) / np.maximum(denom, 1e-9)
    centroid = float(cframe[denom > denom.max() * 0.02].mean()) if denom.max() > 0 else 0.0
    mag_db = 20.0 * np.log10(spec.T + 1e-6)                    # (F, T)
    lo, hi = mag_db.max() - 80.0, mag_db.max()                # finestra dinamica 80 dB
    img = np.clip((mag_db - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (img * 255).astype("uint8")[::-1]                    # bassa freq in basso
    return img, sr, centroid


class SpectrogramView(QLabel):
    """Mostra lo spettrogramma (QImage grayscale) scalato al widget."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(240, 140); self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#0e1016; border:1px solid #2a2f3a;")
        self.setText(tr("spec.empty", "— nessun audio —"))
        self._img = None

    def load(self, path: str) -> float:
        """Carica il file, disegna lo spettro e ritorna il centroide (Hz)."""
        try:
            img, _sr, centroid = compute_spectrogram(path)
        except Exception:                                     # noqa: BLE001
            self.setText(tr("spec.err", "— illeggibile —")); self._img = None; return 0.0
        h, w = img.shape
        qi = QImage(bytes(img.tobytes()), w, h, w, QImage.Format_Grayscale8)
        self._img = QPixmap.fromImage(qi)
        self._redraw()
        return centroid

    def _redraw(self):
        if self._img is not None:
            self.setPixmap(self._img.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, _e):
        self._redraw()


class _Slot(QWidget):
    """Uno slot A o B: intestazione (nome + centroide) + spettro + carica."""

    def __init__(self, title: str):
        super().__init__()
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        self.title = QLabel(f"<b>{title}</b>"); self.cent = QLabel("")
        self.cent.setStyleSheet("color:#74c0fc;")
        head.addWidget(self.title); head.addStretch(1); head.addWidget(self.cent)
        v.addLayout(head)
        self.view = SpectrogramView(); v.addWidget(self.view)
        self.btn = QPushButton(tr("spec.load", "Carica…")); self.btn.clicked.connect(self._pick)
        v.addWidget(self.btn)
        self.name = ""; self.centroid = 0.0

    def _pick(self):
        p, _ = QFileDialog.getOpenFileName(self, tr("spec.load", "Carica…"), "",
                                           "Audio (*.wav *.mp3 *.flac *.ogg *.m4a)")
        if p:
            self.load(p)

    def load(self, path: str):
        from pathlib import Path
        self.name = Path(path).name
        self.title.setText(f"<b>{self.name}</b>")
        self.centroid = self.view.load(path)
        self.cent.setText(tr("spec.centroid", "centroide: {hz:.0f} Hz").format(hz=self.centroid))


class SpectrogramCompare(QDialog):
    """Dialog non-modale: due spettrogrammi A/B + verdetto sul centroide (brillante vs cupo/rumore)."""

    def __init__(self, path_a: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("spec.title", "📊 Spettrogramma / confronto A-B"))
        self.setModal(False); self.resize(680, 380)
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        self.a = _Slot("A"); self.b = _Slot("B")
        row.addWidget(self.a); row.addWidget(self.b)
        v.addLayout(row)
        self.verdict = QLabel(""); self.verdict.setWordWrap(True)
        self.verdict.setStyleSheet("color:#e0a458; padding:4px;")
        v.addWidget(self.verdict)
        btn = QPushButton(tr("spec.compare", "⚖ Confronta")); btn.clicked.connect(self._compare)
        v.addWidget(btn)
        if path_a:
            self.a.load(path_a)

    def _compare(self):
        ca, cb = self.a.centroid, self.b.centroid
        if ca <= 0 or cb <= 0:
            self.verdict.setText(tr("spec.need2", "Carica due file A e B per confrontarli.")); return
        ratio = max(ca, cb) / max(min(ca, cb), 1e-6)
        brighter = "A" if ca > cb else "B"
        darker = "B" if ca > cb else "A"
        self.verdict.setText(tr("spec.verdict",
            "{br} è più brillante di {dk} (×{r:.2f}). Centroide più alto = suono più brillante o più rumoroso; "
            "più basso = più cupo/pulito. Per uno stile 'dark/classical' preferisci il centroide più basso.")
            .format(br=brighter, dk=darker, r=ratio))
