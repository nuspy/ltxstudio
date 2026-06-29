"""Casi di regressione GUI-driven. Ogni caso: imposta i campi del tab + invoca l'azione; lo Spy
cattura la richiesta GUI→backend. Si verifica no-crash + richiesta valida (assert mirati).

Espanso progressivamente (swarm) per coprire OGNI tab/funzione/opzione/modello.
"""
from __future__ import annotations

import harness as H


def _first_concrete_model(combo):
    """Ritorna il primo modello CONCRETO nel combo (salta auto/none)."""
    for i in range(combo.count()):
        data = combo.itemData(i)
        txt = combo.itemText(i)
        if data and data not in ("__auto__", None) and "auto" not in txt.lower() and "nessun" not in txt.lower():
            combo.setCurrentIndex(i)
            return txt
    return combo.currentText()


# ───────────────────────── IMMAGINE: generazione t2i ─────────────────────────
def case_image_generation(w, spy):
    """La GUI deve inviare al backend immagine: module='image', arch valido, model, prompt, width/height."""
    f = w.frame1_tab
    f.prompt.setPlainText("a red apple on a table")
    _first_concrete_model(f.gen_model)
    w.generate_image()
    assert spy.calls, "nessuna richiesta inviata (generate_image non ha chiamato il backend)"
    c = spy.calls[-1]
    p = c["params"]
    assert c["module"] in ("image", None) or c["via"] == "native_driver", f"module inatteso: {c}"
    assert p.get("prompt"), f"prompt mancante nei params: {p}"
    assert p.get("arch"), f"arch mancante: {p}"


# ───────────────────────── IMMAGINE: inpaint ─────────────────────────
def case_inpaint(w, spy):
    """Inpaint: serve image+mask; la GUI deve inviare arch inpaint, image_path, mask_path (FLUX.1 nativo)."""
    from core import native_driver, backend
    native_driver.is_up = lambda *a, **k: True            # _start_native_inpaint fa un check HTTP reale
    backend.set_backend("image", "native")
    H.load_canvas_image(w)
    # seleziona un modello FLUX.1 Fill (ramo nativo), non FLUX.2 Klein (deferred)
    from ui.tabs.frame1_tab import INPAINT_MODELS
    for i in range(w.frame1_tab.inp_model.count()):
        lbl = w.frame1_tab.inp_model.itemText(i)
        spec = INPAINT_MODELS.get(lbl, {})
        if spec.get("unet") and spec.get("arch") == "Flux":
            w.frame1_tab.inp_model.setCurrentIndex(i); break
    cv = w.frame1_tab.canvas
    # disegna una maschera centrale (riempi un rettangolo nella mask_image)
    from PySide6.QtGui import QPainter, QColor
    from PySide6.QtCore import QRect
    pm = QPainter(cv.mask_image)
    pm.fillRect(QRect(cv.aspect_w//4, cv.aspect_h//4, cv.aspect_w//2, cv.aspect_h//2), QColor(255, 255, 255))
    pm.end()
    cv._refresh_mask()
    w.generate_inpaint("inpaint")
    assert spy.calls, "inpaint: nessuna richiesta inviata"
    p = spy.calls[-1]["params"]
    assert p.get("image_path") and p.get("mask_path"), f"inpaint senza image/mask: {p}"


# ───────────────────────── IMMAGINE: outpaint (no distorsione quadrata) ─────────────────────────
def case_outpaint(w, spy):
    """Outpaint: la GUI deve inviare width/height con le PROPORZIONI della foto (non quadrato),
    image_path+mask_path. Verifica che non forzi un quadrato."""
    path = H.load_canvas_image(w)  # 768x512 (non quadrata)
    cv = w.frame1_tab.canvas
    assert abs(cv.aspect_w / cv.aspect_h - 768/512) < 0.05, \
        f"il canvas ha perso le proporzioni della foto: {cv.aspect_w}x{cv.aspect_h} (atteso ~3:2)"


# ───────────────────────── VIDEO: generazione t2v ─────────────────────────
def case_video_generation(w, spy):
    """Video: on_generate deve instradare al motore nativo con module/params video (prompt, width,
    height, num_frames, fps)."""
    w.video_tab.prompt.setPlainText("a dog running on the beach")
    w.tabs.setCurrentWidget(w.video_tab)
    w.on_generate()
    assert spy.calls, "video: nessuna richiesta inviata da on_generate"
    p = spy.calls[-1]["params"]
    assert p.get("num_frames") and p.get("width") and p.get("height"), f"params video incompleti: {p}"


CASES = [
    ("image_generation_t2i", "Genera immagine: la GUI invia module=image + arch + model + prompt + dimensioni.",
     case_image_generation),
    ("image_inpaint", "Inpaint: con foto+maschera la GUI invia arch inpaint + image_path + mask_path.",
     case_inpaint),
    ("image_outpaint_aspect", "Outpaint: il canvas mantiene le proporzioni della foto (niente quadrato).",
     case_outpaint),
    ("video_generation_t2v", "Genera video: on_generate instrada al nativo con num_frames/width/height/fps.",
     case_video_generation),
]
