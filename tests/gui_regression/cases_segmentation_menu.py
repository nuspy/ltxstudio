"""Fase 2 (Openjutsu, 03/11): menu «Segmentazione» (SAM2 predefinito / SAM 3.1), scelta ricordata, SAM 3.1 non utilizzabile = finestra di aiuto non
modale e NESSUN cambio di motore. Nessuna GPU, nessun modello caricato."""
from __future__ import annotations

import harness as H  # noqa: F401

from core import segmentation_engine as SE


def _menu(w):
    return w._seg_menu


def case_menu_has_both_engines_and_default_sam2(w, spy):
    m = _menu(w)
    items = {a.data(): a for a in m.actions() if a.data() in SE.ENGINES}
    assert set(items) == {"sam2", "sam31"}, items
    assert items["sam2"].isChecked() and not items["sam31"].isChecked()
    assert "SAM2" in items["sam2"].text()
    assert "SAM 3.1" in items["sam31"].text()


def case_sam31_unavailable_opens_nonmodal_help_and_keeps_sam2(w, spy):
    """Pesi presenti ma caricatore non pronto: si apre la guida non modale (cosa fa / quando sceglierlo), il motore NON cambia."""
    from PySide6.QtWidgets import QApplication
    from ui.widgets import sam31_help
    old_ready, old_status = SE.LOADER_READY, SE.sam31_status
    SE.LOADER_READY = False
    SE.sam31_status = lambda base=None: ("ok", [])
    try:
        m = _menu(w)
        a31 = next(a for a in m.actions() if a.data() == "sam31")
        before = list(sam31_help._open)
        a31.trigger()
        QApplication.processEvents()
        assert len(sam31_help._open) == len(before) + 1, "finestra di aiuto non aperta"
        dlg = sam31_help._open[-1]
        assert not dlg.isModal()
        assert "approva" not in dlg.text_label.text().lower().replace("approvazioni", ""), "il flusso di accesso a Meta non c'e piu"
        assert SE.current() == "sam2", "il motore e cambiato senza che SAM 3.1 fosse utilizzabile"
        assert next(a for a in m.actions() if a.data() == "sam2").isChecked()
        dlg.close()
    finally:
        SE.LOADER_READY, SE.sam31_status = old_ready, old_status


def case_sam31_missing_offers_the_catalog_download(w, spy):
    """Pesi mancanti: scegliere SAM 3.1 avvia il normale download del catalogo (id sam31-unsloth) e NON cambia motore."""
    from PySide6.QtWidgets import QApplication
    old_status = SE.sam31_status
    SE.sam31_status = lambda base=None: ("missing", list(SE.SAM31_FILES))
    asked = []
    orig = w._download_sam31
    w._download_sam31 = lambda: asked.append(SE.SAM31_CATALOG_ID)
    try:
        a31 = next(a for a in _menu(w).actions() if a.data() == "sam31")
        a31.trigger()
        QApplication.processEvents()
        assert asked == ["sam31-unsloth"], asked
        assert SE.current() == "sam2"
    finally:
        SE.sam31_status = old_status
        w._download_sam31 = orig


def case_choice_is_remembered_when_selectable(w, spy):
    from core import user_prefs
    SE.LOADER_READY = True
    try:
        if SE.sam31_status()[0] != "ok":
            return                                                       # SAM 3.1 non presente su questo PC: non applicabile
        m = _menu(w)
        a31 = next(a for a in m.actions() if a.data() == "sam31")
        a31.setChecked(True)
        a31.trigger()
        assert user_prefs.get(SE.PREF_KEY) == "sam31"
        next(a for a in m.actions() if a.data() == "sam2").trigger()
        assert user_prefs.get(SE.PREF_KEY) == "sam2"
    finally:
        SE.LOADER_READY = False


CASES = [
    ("seg_menu_engines", "menu Segmentazione: due motori, SAM2 predefinito.", case_menu_has_both_engines_and_default_sam2),
    ("seg_menu_sam31_unavailable", "SAM 3.1 non utilizzabile: aiuto non modale, motore invariato.", case_sam31_unavailable_opens_nonmodal_help_and_keeps_sam2),
    ("seg_menu_sam31_missing_download", "pesi mancanti: parte l'offerta di download del catalogo (sam31-unsloth), motore invariato.", case_sam31_missing_offers_the_catalog_download),
    ("seg_menu_choice_remembered", "la scelta si ricorda (user_prefs) quando SAM 3.1 e utilizzabile.", case_choice_is_remembered_when_selectable),
]
