"""GUI-driven (qtbot) — contratto GUI→backend per "Krea 2 Edit" (editing per istruzione, single/multi).

Verifica, pilotando la GUI reale e catturando la richiesta con lo Spy (ZERO GPU), che scegliendo il motore
"Krea 2 Edit" nei pannelli esistenti (Componi / Cambia colore) parta un job con `arch=="krea2_edit"`, le
immagini giuste (images[] per il multi, image_path per il single), il `grounding_px` e la variante — cioè
che il nuovo motore sia instradato SENZA duplicare tab/controlli.

Esecuzione (venv GUI PySide6):
  set QT_QPA_PLATFORM=offscreen
  E:\\AI\\Comfy_Nexus.bak\\ComfyUI\\venv\\Scripts\\python.exe -m pytest \
      tests/gui_regression/test_qtbot_krea2_edit.py -v
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest                                                     # noqa: E402
import harness as H                                              # noqa: E402
import qtbot_harness as QB                                       # noqa: E402

PROMPT = "add mirrored aviator sunglasses on his face, keep everything else identical"


@pytest.fixture()
def win():
    """MainWindow offscreen ERMETICA: backend immagine 'nativo' finto, gui_state no-op, native ready."""
    from core import native_driver, gui_state, backend
    _orig = (native_driver.is_up, gui_state.restore, gui_state.store, backend.is_native)
    native_driver.is_up = lambda: True
    gui_state.restore = lambda _w, _s=None: {}
    gui_state.store = lambda _w: None
    backend.is_native = lambda *_a, **_k: True                   # ramo NATIVO immagine
    w = H.build_window()
    w._ensure_native_ready = lambda *a, **k: True                # niente spawn server nel test
    w._start_native_preview = lambda *a, **k: None               # niente dialog anteprima
    try:
        w._refresh_status()
    except Exception:                                            # noqa: BLE001
        pass
    w.tabs.setCurrentWidget(w.frame1_tab)
    QB.process_events()
    try:
        yield w
    finally:
        native_driver.is_up, gui_state.restore, gui_state.store, backend.is_native = _orig
        w.close(); QB.process_events()


def _dispatched(spy):
    for c in reversed(spy.calls):
        if c["via"] in ("enqueue", "native_driver"):
            return c
    return None


def _set_integrate_engine_krea2(win):
    win.frame1_tab.sub.setCurrentIndex(1)                       # sotto-tab "Modifica" (inpaint/integra/componi)
    QB.process_events()
    combo = win.frame1_tab.integrate_engine
    idx = combo.findText("Krea 2 Edit")
    if idx < 0:                                                  # tr potrebbe tradurre: Krea 2 Edit = index 2
        idx = 2
    QB.combo_select(combo, index=idx)
    QB.process_events()
    return idx


def test_grounding_slider_visible_only_for_krea2(win):
    """Lo slider grounding compare SOLO col motore Krea 2 Edit."""
    ft = win.frame1_tab
    ft.sub.setCurrentIndex(1)                                   # sotto-tab "Modifica" (rende visibile il combo)
    QB.process_events()
    QB.combo_select(ft.integrate_engine, index=0)               # Kontext
    QB.process_events()
    assert not ft._edit_grounding_row.isVisible(), "grounding visibile con Kontext (non dovrebbe)"
    _set_integrate_engine_krea2(win)
    assert ft._edit_grounding_row.isVisible(), "grounding NON visibile con Krea 2 Edit"


def test_compose_krea2_edit_routing_multi(win):
    """Componi + motore Krea 2 Edit + 2 foto → job arch='krea2_edit' con images[] (multi-reference vero)."""
    ft = win.frame1_tab
    _set_integrate_engine_krea2(win)
    # 2 immagini di riferimento (scena + soggetto)
    a = H.make_test_image(str(H.OUT_DIR / "_krea2_scene.png"), 768, 512)
    b = H.make_test_image(str(H.OUT_DIR / "_krea2_subject.png"), 512, 512)
    ft.compose_list.clear()
    ft.compose_list.addItem(a); ft.compose_list.addItem(b)
    ft.compose_prompt.setPlainText(PROMPT)
    ft.edit_grounding.setValue(1024)
    QB.process_events()
    with H.Spy(win) as spy:
        ft.request_compose.emit()
        QB.process_events()
    call = _dispatched(spy)
    assert call is not None, f"Componi Krea 2 Edit non ha dispacciato nulla (calls={spy.calls})"
    p = call["params"]
    assert str(p.get("arch")) == "krea2_edit", f"arch != krea2_edit: {p.get('arch')}"
    imgs = p.get("images") or ([p["image_path"]] if p.get("image_path") else [])
    assert len(imgs) == 2, f"multi-reference: attese 2 immagini, ho {imgs}"
    assert int(p.get("grounding_px", 0)) == 1024, f"grounding non trasmesso: {p.get('grounding_px')}"
    assert p.get("krea2_variant") in ("turbo", "raw")


def test_recolor_krea2_edit_routing_single(win):
    """Cambia colore + motore Krea 2 Edit → job arch='krea2_edit' single (image_path del canvas)."""
    ft = win.frame1_tab
    H.load_canvas_image(win)                                    # foto nel canvas → image_path
    ft.sub.setCurrentIndex(2)                                   # sotto-tab "Cambia colore"
    QB.process_events()
    combo = ft.color_engine
    idx = combo.findText("Krea 2 Edit")
    QB.combo_select(combo, index=idx if idx >= 0 else 2)        # Krea 2 Edit = index 2 (3 = Qwen-Image 2.1)
    ft.color_target.setText("#1e90ff")
    ft.color_subject.setText("the jacket")
    QB.process_events()
    with H.Spy(win) as spy:
        ft.request_recolor.emit()
        QB.process_events()
    call = _dispatched(spy)
    assert call is not None, f"Recolor Krea 2 Edit non ha dispacciato nulla (calls={spy.calls})"
    p = call["params"]
    assert str(p.get("arch")) == "krea2_edit", f"arch != krea2_edit: {p.get('arch')}"
    assert p.get("image_path"), f"single: image_path mancante ({sorted(p.keys())})"
    assert "recolor" in str(p.get("prompt", "")).lower() or "1e90ff" in str(p.get("prompt", "")).lower()
