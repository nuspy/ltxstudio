"""Helper riutilizzabile per i dropdown modelli: filtro per precisione + colori VRAM (verde/giallo/rosso).

Usato da TUTTI i generatori (immagine: generazione/inpaint/outpaint/integra; video: tutte le sezioni).
- `make_precision_selector()` → QComboBox (Tutte / FP16 / FP8 / FP4 / GGUF).
- `populate(model_combo, names, prec_filter, *, name_to_path)` → riempie il combo filtrando per
  precisione e colorando ogni voce per fattibilità sulla GPU corrente. Mantiene userData = nome reale.
Vedi core/model_precision.py.
"""
from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox

from core import model_precision as mp
from ui.i18n_manager import tr

_FILTER_ORDER = ("all", "fp16", "fp8", "fp4", "gguf")


def make_precision_selector() -> QComboBox:
    cb = QComboBox()
    cb.addItem(tr("prec.all", "Tutte"), "all")
    for p in ("fp16", "fp8", "fp4", "gguf"):
        cb.addItem(mp.PRECISION_LABEL[p], p)
    cb.setToolTip(tr("prec.tt", "Filtra i modelli per precisione. Colori: verde=ottimale in VRAM, "
                                "giallo=offload parziale gestibile, rosso=troppo grande per la GPU."))
    return cb


def classify_name(name: str, free_gb: float | None = None, name_to_path=None) -> dict:
    if free_gb is None:
        free_gb = mp.gpu_free_gb()
    fp = name_to_path(name) if name_to_path else None
    return mp.classify(name, free_gb, file_path=fp)


def populate(combo: QComboBox, names: list[str], prec_filter: str = "all", *,
             name_to_path=None, prepend: list[tuple[str, object]] | None = None,
             free_gb: float | None = None) -> None:
    """Riempie `combo` con `names`, filtrando per precisione e colorando per fattibilità VRAM.
    prepend: voci fisse iniziali [(label, data)] (es. auto/none) non filtrate né colorate."""
    if free_gb is None:
        free_gb = mp.gpu_free_gb()
    combo.blockSignals(True)
    combo.clear()
    for label, data in (prepend or []):
        combo.addItem(label, data)
    for name in names:
        info = classify_name(name, free_gb, name_to_path)
        if prec_filter and prec_filter != "all" and info["precision"] != prec_filter:
            continue
        combo.addItem(name, name)
        idx = combo.count() - 1
        from PySide6.QtCore import Qt
        col = mp.QT_COLORS.get(info["color"])
        if col:
            combo.setItemData(idx, QColor(col), Qt.ForegroundRole)
        combo.setItemData(idx, f"{info['precision'].upper()} · ~{info['footprint_gb']}GB "
                               f"({_state_label(info['color'])})", Qt.ToolTipRole)
    combo.blockSignals(False)


def color_existing(combo: QComboBox, *, name_to_path=None, free_gb: float | None = None) -> None:
    """Colora le voci GIÀ presenti in `combo` (senza svuotarlo) per fattibilità VRAM, leggendo
    precisione/footprint dall'header del file. Usata dai dropdown popolati altrove (upscale/video).
    Se non si risolve un path file reale per una voce, NON la colora (evita colori fuorvianti)."""
    from PySide6.QtCore import Qt
    if free_gb is None:
        free_gb = mp.gpu_free_gb()
    for i in range(combo.count()):
        name = combo.itemText(i)
        fp = name_to_path(name) if name_to_path else (name if name else None)
        import os
        if not (fp and os.path.isfile(fp)):
            continue                                          # niente path reale → niente colore
        info = mp.classify(name, free_gb, file_path=fp)
        col = mp.QT_COLORS.get(info["color"])
        if col:
            combo.setItemData(i, QColor(col), Qt.ForegroundRole)
            combo.setItemData(i, f"{info['precision'].upper()} · ~{info['footprint_gb']}GB "
                                 f"({_state_label(info['color'])})", Qt.ToolTipRole)


def _state_label(color: str) -> str:
    return {"green": tr("prec.green", "ottimale in VRAM"),
            "yellow": tr("prec.yellow", "offload parziale, gestibile"),
            "red": tr("prec.red", "troppo grande per la GPU")}.get(color, "")
