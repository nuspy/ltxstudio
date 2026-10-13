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


def available_text(up: dict) -> str:
    """«Disponibile solo nelle versioni: A, B, C» — TUTTE le edizioni che includono la feature (le versioni
    non sono una scala unica). Manifest vecchi senza `available_in`: ricade sul piano minimo."""
    eds = up.get("available_in") or []
    if eds:
        names = ", ".join(e.get("name") or e.get("id", "") for e in eds)
        return tr("upsell.available_in", "Disponibile solo nelle versioni: {editions}").format(editions=names)
    return tr("upsell.available_on", "Disponibile nel piano {tier}").format(
        tier=up.get("tier_name") or up.get("tier", ""))


def _unlock_row(fid: str, up: dict) -> QWidget:
    """Riga «Disponibile solo nelle versioni: …» + bottone di sblocco (condivisa da panel e dialog)."""
    w = QWidget(); h = QVBoxLayout(w); h.setSpacing(10); h.setContentsMargins(0, 0, 0, 0)
    tier = up.get("tier_name") or up.get("tier", "")
    price = up.get("price", "")
    line = available_text(up)
    if price and not up.get("available_in"):
        line += f"  ·  {price}"
    lab = QLabel(line); lab.setAlignment(Qt.AlignCenter); lab.setWordWrap(True)
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
        # Teaser visivo opzionale: la cella matrix può indicare params.teaser_image (path relativo
        # ad assets/ o assoluto) — un'anteprima di cosa si ottiene converte più del solo testo.
        try:
            from pathlib import Path as _P
            from core import features as _F
            _t = (_F.manifest().get("features", {}).get(fid, {}).get("params", {}) or {}).get("teaser_image")
            if _t:
                _p = _P(_t)
                if not _p.is_absolute():
                    _p = _P(__file__).resolve().parents[2] / "assets" / _t
                if _p.exists():
                    from PySide6.QtGui import QPixmap
                    pic = QLabel(); pic.setAlignment(Qt.AlignCenter)
                    pm = QPixmap(str(_p))
                    if not pm.isNull():
                        pic.setPixmap(pm.scaledToWidth(min(520, pm.width()), Qt.SmoothTransformation))
                        v.addSpacing(6); v.addWidget(pic)
        except Exception:                                    # noqa: BLE001
            pass
        v.addSpacing(8)
        v.addWidget(_unlock_row(fid, up))
        v.addStretch(3)


class UpsellStrip(QWidget):
    """Striscia compatta per una SEZIONE bloccata dentro un tab condiviso: sostituisce l'intero
    gruppo di widget della feature con UNA riga '🔒 <nome> — Disponibile nel piano <tier>' cliccabile.
    Evita decine di label ripetute quando una feature è fatta di molti componenti sparsi."""

    def __init__(self, fid: str, i18n_key: str = "", parent=None):
        super().__init__(parent)
        self._fid = fid
        self._i18n = i18n_key
        up = upsell.info(fid) or {}
        tier = up.get("tier_name") or up.get("tier", "")
        h = QHBoxLayout(self); h.setContentsMargins(8, 6, 8, 6)
        lab = QLabel(tr("upsell.strip", "🔒 {name} — Disponibile nel piano {tier}")
                     .format(name=_title_for(fid, i18n_key), tier=tier))
        lab.setStyleSheet(f"color:{_ACCENT}; font-weight:bold;")
        h.addWidget(lab); h.addStretch(1)
        btn = QPushButton(tr("upsell.strip_btn", "Sblocca →"))
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"QPushButton {{ color:{_ACCENT}; border:1px solid {_ACCENT};"
                          " border-radius:6px; padding:3px 12px; background:transparent; }")
        btn.clicked.connect(self._open)
        h.addWidget(btn)
        self.setStyleSheet(f"UpsellStrip {{ background:rgba(142,107,255,0.08);"
                           f" border:1px dashed {_ACCENT}; border-radius:8px; }}")

    def _open(self):
        UpsellDialog(self._fid, self._i18n, self).exec()


def lock_inline_item(combo, index: int, fid: str) -> None:
    """Lock-mode INLINE: disabilita una voce di QComboBox aggiungendo il badge '🔒 <tier>'.
    La voce resta VISIBILE (l'utente scopre che esiste) ma non selezionabile; tooltip con upsell."""
    up = upsell.info(fid)
    if not up:
        return
    from PySide6.QtCore import Qt as _Qt
    tier = up.get("tier_name") or up.get("tier", "")
    combo.setItemText(index, f"🔒 {combo.itemText(index)}  ({tier})")
    item = combo.model().item(index) if hasattr(combo.model(), "item") else None
    if item is not None:
        item.setFlags(item.flags() & ~_Qt.ItemIsEnabled)      # non selezionabile
    combo.setItemData(index, available_text(up), _Qt.ToolTipRole)


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
