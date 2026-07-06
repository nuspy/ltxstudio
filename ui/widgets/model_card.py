"""Componenti UI riusabili per l'importer/libreria modelli (#17): badge + ModelCard.

Usati in tre punti: coda di Importa (compatta), griglia di Scopri e griglia di Libreria. Stile coerente con
la palette dell'app (#1a1a1a, accent #74c0fc). Nessuna logica di rete/catalogo qui: solo presentazione.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

# colori badge per "kind"
_BADGE_COLORS = {
    "nsfw": ("#ff6b6b", "#3a1414"),
    "sfw": ("#2f9e44", "#12240f"),
    "type": ("#74c0fc", "#0f2030"),
    "arch": ("#b197fc", "#1c1330"),
    "status": ("#ffd43b", "#302a0f"),
    "muted": ("#9aa0b4", "#1a1a1a"),
}


def badge(text: str, kind: str = "muted") -> QLabel:
    fg, bg = _BADGE_COLORS.get(kind, _BADGE_COLORS["muted"])
    lab = QLabel(text)
    lab.setStyleSheet(
        f"color:{fg}; background:{bg}; border:1px solid {fg}; border-radius:7px; "
        f"padding:1px 7px; font-size:10px; font-weight:600;")
    lab.setAlignment(Qt.AlignCenter)
    return lab


def _thumb_label(thumb, size: int) -> QLabel:
    lab = QLabel()
    lab.setFixedSize(size, size)
    lab.setAlignment(Qt.AlignCenter)
    lab.setStyleSheet("background:#111; border:1px solid #333; border-radius:6px; color:#555;")
    pm = None
    if isinstance(thumb, QPixmap) and not thumb.isNull():
        pm = thumb
    elif isinstance(thumb, str) and thumb:
        p = QPixmap(thumb)
        pm = p if not p.isNull() else None
    if pm is not None:
        lab.setPixmap(pm.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
    else:
        lab.setText("🖼")
    return lab


class ModelCard(QFrame):
    """Card modello. compact=True → riga orizzontale (coda import); False → tessera verticale (griglie)."""

    clicked = Signal()

    def __init__(self, *, title: str, subtitle: str = "", badges: list[tuple[str, str]] | None = None,
                 thumb=None, compact: bool = False, actions: list[tuple[str, object]] | None = None,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("modelCard")
        self.setStyleSheet(
            "#modelCard { background:#1a1a1a; border:1px solid #333; border-radius:8px; } "
            "#modelCard:hover { border-color:#74c0fc; }")
        self._status_lab: QLabel | None = None
        (self._build_compact if compact else self._build_full)(title, subtitle, badges or [], thumb,
                                                               actions or [])

    # ---- layout compatto (coda import) ----
    def _build_compact(self, title, subtitle, badges, thumb, actions):
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 6, 8, 6)
        h.setSpacing(8)
        h.addWidget(_thumb_label(thumb, 40))
        col = QVBoxLayout()
        col.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet("font-weight:600; font-size:12px;")
        t.setToolTip(title)
        col.addWidget(t)
        brow = QHBoxLayout()
        brow.setSpacing(4)
        for text, kind in badges:
            brow.addWidget(badge(text, kind))
        if subtitle:
            sub = QLabel(subtitle)
            sub.setStyleSheet("color:#888; font-size:10px;")
            brow.addWidget(sub)
        brow.addStretch(1)
        col.addLayout(brow)
        h.addLayout(col, 1)
        self._status_lab = QLabel("")
        self._status_lab.setStyleSheet("color:#ffd43b; font-size:10px;")
        h.addWidget(self._status_lab)
        for label, cb in actions:
            b = QPushButton(label)
            b.setFixedHeight(26)
            if cb:
                b.clicked.connect(cb)
            h.addWidget(b)

    # ---- layout tessera (griglie Scopri/Libreria) ----
    def _build_full(self, title, subtitle, badges, thumb, actions):
        self.setFixedWidth(180)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        v.addWidget(_thumb_label(thumb, 160), alignment=Qt.AlignCenter)
        t = QLabel(title)
        t.setWordWrap(True)
        t.setStyleSheet("font-weight:600; font-size:12px;")
        t.setToolTip(title)
        v.addWidget(t)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setStyleSheet("color:#888; font-size:10px;")
            v.addWidget(sub)
        brow = QHBoxLayout()
        brow.setSpacing(4)
        for text, kind in badges:
            brow.addWidget(badge(text, kind))
        brow.addStretch(1)
        v.addLayout(brow)
        for label, cb in actions:
            b = QPushButton(label)
            if cb:
                b.clicked.connect(cb)
            v.addWidget(b)

    def set_status(self, text: str, kind: str = "status"):
        if self._status_lab is not None:
            fg = _BADGE_COLORS.get(kind, _BADGE_COLORS["status"])[0]
            self._status_lab.setStyleSheet(f"color:{fg}; font-size:10px;")
            self._status_lab.setText(text)

    def mousePressEvent(self, e):                            # noqa: N802
        self.clicked.emit()
        super().mousePressEvent(e)


def flow_grid(cards: list[QWidget], columns: int = 4) -> QWidget:
    """Dispone le card in una griglia semplice a N colonne (per Scopri/Libreria)."""
    from PySide6.QtWidgets import QGridLayout
    host = QWidget()
    g = QGridLayout(host)
    g.setSpacing(10)
    g.setContentsMargins(4, 4, 4, 4)
    for i, c in enumerate(cards):
        g.addWidget(c, i // columns, i % columns, alignment=Qt.AlignTop | Qt.AlignLeft)
    g.setRowStretch((len(cards) // columns) + 1, 1)
    return host
