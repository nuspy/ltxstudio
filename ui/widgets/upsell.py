"""Widget di upsell per le feature bloccate dall'edizione.

- UpsellPanel: sostituisce il CONTENUTO di un tab bloccato (il tab resta visibile col lucchetto):
  lucchetto + nome feature + "Available on <tier>" + prezzo + bottone che apre il link Patreon.
- UpsellDialog: dialog mostrato quando l'utente aziona una capability bloccata (fallback
  universale via MainWindow._feature_gate).
La grafica è volutamente pulita/da vetrina: è una superficie di MARKETING, non un muro.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDialog,
)

from core import upsell
from ui.i18n_manager import tr

_ACCENT = "#8e6bff"


def _title_for(fid: str, i18n_key: str = "") -> str:
    """Nome umano della feature: i18n se disponibile, altrimenti l'id ripulito."""
    if i18n_key:
        t = tr(i18n_key, "")
        if t and t != i18n_key:
            return t
    return fid.split(".", 1)[-1].replace("_", " ").title()


def _unlock_row(fid: str, up: dict) -> QWidget:
    """Riga 'Available on <tier> (<price>)' + bottone di sblocco (condivisa da panel e dialog)."""
    w = QWidget(); h = QVBoxLayout(w); h.setSpacing(10); h.setContentsMargins(0, 0, 0, 0)
    tier = up.get("tier_name") or up.get("tier", "")
    price = up.get("price", "")
    line = tr("upsell.available_on", "Disponibile nel piano {tier}").format(tier=tier)
    if price:
        line += f"  ·  {price}"
    lab = QLabel(line); lab.setAlignment(Qt.AlignCenter)
    lab.setStyleSheet(f"color:{_ACCENT}; font-size:15px; font-weight:bold;")
    h.addWidget(lab)
    if up.get("url"):
        btn = QPushButton(tr("upsell.unlock_btn", "🔓 Sblocca su Patreon — {tier}").format(tier=tier))
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton {{ background:{_ACCENT}; color:white; font-weight:bold; font-size:14px;"
            "  border:none; border-radius:8px; padding:10px 26px; }"
            "QPushButton:hover { background:#7a54f0; }")
        btn.clicked.connect(lambda: upsell.open_url(fid))
        row = QHBoxLayout(); row.addStretch(1); row.addWidget(btn); row.addStretch(1)
        h.addLayout(row)
    return w


class UpsellPanel(QWidget):
    """Contenuto sostitutivo di un TAB bloccato: vetrina + CTA (il tab resta scopribile)."""

    def __init__(self, fid: str, i18n_key: str = "", parent=None):
        super().__init__(parent)
        up = upsell.info(fid) or {}
        v = QVBoxLayout(self)
        v.addStretch(2)
        lock = QLabel("🔒"); lock.setAlignment(Qt.AlignCenter)
        lock.setStyleSheet("font-size:52px;")
        v.addWidget(lock)
        name = QLabel(_title_for(fid, i18n_key)); name.setAlignment(Qt.AlignCenter)
        name.setStyleSheet("font-size:22px; font-weight:bold; color:#e8e4f6;")
        v.addWidget(name)
        sub = QLabel(tr("upsell.tab_teaser",
                        "Questa funzione è pronta: sbloccala e usala subito, nella stessa finestra."))
        sub.setAlignment(Qt.AlignCenter); sub.setWordWrap(True)
        sub.setStyleSheet("color:#9aa0b4; font-size:13px;")
        v.addWidget(sub)
        v.addSpacing(8)
        v.addWidget(_unlock_row(fid, up))
        v.addStretch(3)


class UpsellDialog(QDialog):
    """Dialog di upsell quando si aziona una capability bloccata (fallback universale)."""

    def __init__(self, fid: str, i18n_key: str = "", parent=None):
        super().__init__(parent)
        up = upsell.info(fid) or {}
        self.setWindowTitle(tr("upsell.title", "Funzione da sbloccare"))
        self.setModal(True)
        v = QVBoxLayout(self)
        head = QLabel("🔒  " + _title_for(fid, i18n_key))
        head.setStyleSheet("font-size:17px; font-weight:bold;")
        v.addWidget(head)
        body = QLabel(tr("upsell.dialog_msg",
                         "Questa funzione non è inclusa nel tuo piano attuale. È già installata e "
                         "pronta: passa al piano indicato per usarla immediatamente."))
        body.setWordWrap(True); body.setStyleSheet("color:#9aa0b4;")
        v.addWidget(body)
        v.addSpacing(6)
        v.addWidget(_unlock_row(fid, up))
        close = QPushButton(tr("upsell.later", "Più tardi"))
        close.clicked.connect(self.reject)
        row = QHBoxLayout(); row.addStretch(1); row.addWidget(close)
        v.addLayout(row)
        self.resize(430, 240)
