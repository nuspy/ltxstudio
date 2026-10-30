"""Icona «vietato ai minori di 18 anni» sulle voci NSFW dei menu modello (Crea e Modifica immagini).

Come si riconosce un modello 18+ (stesse fonti dei filtri SFW/NSFW esistenti, verificate nel codice):
  1. flag del catalogo/override utente: `core.model_catalog.ModelCatalog.is_nsfw(name)` (usato da `_reload_gen`);
  2. registro pesi: voci con `sensitivity: adult` (`core.weights_catalog` / `is_adult_file`), voci di catalogo Qwen/Krea;
  3. euristica per parole nel nome (`core.nsfw_policy.looks_nsfw`).
L'icona NON è l'unico segnale: ogni voce marcata ha anche un tooltip testuale (i18n) con la stessa informazione.
L'icona sta nel DecorationRole: il testo della voce NON cambia (`currentText()` è il nome file usato dai job).
"""
from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QComboBox

from ui.i18n_manager import tr

_ICON: QIcon | None = None


def adult_icon() -> QIcon:
    """Cerchio rosso con «18» barrato da una diagonale: il pittogramma «vietato ai minori di 18 anni»."""
    global _ICON
    if _ICON is None:
        pm = QPixmap(32, 32)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QBrush(QColor("#c62828")))
        p.setPen(QColor("#ffffff"))
        p.drawEllipse(1, 1, 30, 30)
        f = QFont()
        f.setBold(True)
        f.setPixelSize(15)
        p.setFont(f)
        p.drawText(QRect(0, 0, 32, 32), Qt.AlignCenter, "18")
        p.setPen(QColor("#ffffff"))
        p.drawLine(7, 25, 25, 7)
        p.end()
        _ICON = QIcon(pm)
    return _ICON


def adult_tooltip() -> str:
    return tr("model.adult.tt", "Contenuto per adulti (18+, vietato ai minori). Compare solo in modalità NSFW e nelle "
                                "edizioni che lo consentono.")


def is_adult_name(name: str, mcat=None) -> bool:
    """Il file/nome è 18+ secondo catalogo (flag e override), registro pesi e parole del nome."""
    if not name:
        return False
    base = str(name).split("  ·")[0].split("  ⛔")[0].strip()
    try:
        if mcat is not None and mcat.is_nsfw(base):
            return True
    except Exception:                                         # noqa: BLE001
        pass
    from core import nsfw_policy
    return nsfw_policy.looks_nsfw(base)


def mark(combo: QComboBox, i: int) -> None:
    """Marca la voce `i`: icona + riga di tooltip testuale (accessibile anche senza vedere l'icona)."""
    combo.setItemIcon(i, adult_icon())
    old = combo.itemData(i, Qt.ToolTipRole)
    tip = adult_tooltip()
    combo.setItemData(i, f"{tip}\n{old}" if old else tip, Qt.ToolTipRole)


def mark_combo(combo: QComboBox, mcat=None, name_of=None, adult_of=None) -> int:
    """Marca tutte le voci 18+ del combo. `name_of(i)` = nome da testare (default: testo della voce); `adult_of(i)`
    = verdetto già noto (es. catalogo `sensitivity`). Le voci con «🔞» già nel testo hanno comunque il tooltip.
    Ritorna quante ne ha marcate."""
    n = 0
    for i in range(combo.count()):
        d = combo.itemData(i)
        if d == "__placeholder__" or (isinstance(d, dict) and d.get("placeholder")):
            continue
        adult = adult_of(i) if adult_of is not None else None
        if adult is None:
            adult = is_adult_name(name_of(i) if name_of else combo.itemText(i), mcat)
        if adult or "🔞" in combo.itemText(i):
            mark(combo, i)
            n += 1
    return n
