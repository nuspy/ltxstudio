"""GUI-driven (qtbot) — CONTRATTO GUI→backend per OGNI architettura IMMAGINE e VIDEO.

Per ciascuna architettura selezionabile l'utente:
  1. naviga al tab (Immagini o Video),
  2. sceglie l'architettura dal combo REALE (video_arch / gen_arch / long_mode),
  3. digita un prompt e clicca REALMENTE «Genera»,
  4. lo Spy cattura la richiesta GUI→backend SENZA GPU e si verifica che il campo `arch`
     dispacciato sia quello atteso (routing corretto) e che il click NON sia stato inghiottito
     da una validazione (regressione «nessun modello krea 2 trovato»).

Questo intercetta i bug di ROUTING/VALIDAZIONE su tutte le arch in secondi, a zero GPU.
Il funzionamento a RUNTIME dei worker (load pesi, inferenza) è un test separato (live).

Esecuzione (venv GUI PySide6):
  set QT_QPA_PLATFORM=offscreen
  E:\\AI\\Comfy_Nexus.bak\\ComfyUI\\venv\\Scripts\\python.exe -m pytest \
      tests/gui_regression/test_qtbot_all_arches.py -v
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest                                                     # noqa: E402
import harness as H                                              # noqa: E402
import qtbot_harness as QB                                       # noqa: E402

PROMPT = ("un faro di ottone galleggia in un cielo di meduse dorate al tramonto, "
          "stile pittorico, luce cinematografica")


# ─────────────────────────── fixture ───────────────────────────
@pytest.fixture()
def win():
    """MainWindow offscreen ERMETICA: backend 'su' (finto), gui_state no-op."""
    from core import native_driver, gui_state
    _orig_up = native_driver.is_up
    _orig_restore, _orig_store = gui_state.restore, gui_state.store
    native_driver.is_up = lambda: True
    gui_state.restore = lambda _win, _state=None: {}
    gui_state.store = lambda _win: None
    w = H.build_window()
    try:
        w._refresh_status()
    except Exception:                                             # noqa: BLE001
        pass
    QB.process_events()
    try:
        yield w
    finally:
        native_driver.is_up = _orig_up
        gui_state.restore, gui_state.store = _orig_restore, _orig_store
        w.close()
        QB.process_events()


def _dispatch_call(spy):
    """Ultima richiesta GUI→backend (enqueue worker o native_driver); None se il click è stato inghiottito."""
    for c in reversed(spy.calls):
        if c["via"] in ("enqueue", "native_driver"):
            return c
    return None


# ─────────────────────────── VIDEO ───────────────────────────
VIDEO_ARCHES = [
    ("ltx", "ltx"),                    # (dato combo video_arch, arch atteso nei params)
    ("lingbot", "lingbot"),
    ("krea_rt", "krea_rt"),
]


@pytest.mark.parametrize("combo_data,expected_arch", VIDEO_ARCHES,
                         ids=[a[0] for a in VIDEO_ARCHES])
def test_video_arch_routing(win, combo_data, expected_arch):
    """Tab Video → seleziona l'architettura dal combo → Genera → params['arch'] corretto."""
    win.tabs.setCurrentWidget(win.video_tab)
    QB.process_events()
    idx = win.video_tab.video_arch.findData(combo_data)
    assert idx >= 0, f"arch {combo_data} assente nel combo video_arch"
    QB.combo_select(win.video_tab.video_arch, index=idx)
    assert (win.video_tab.video_arch.currentData() or "ltx") == combo_data
    win.video_tab.prompt.setPlainText(PROMPT)
    QB.process_events()
    with H.Spy(win) as spy:
        QB.click(win.gen_btn)
    call = _dispatch_call(spy)
    assert call is not None, (
        f"[{combo_data}] click su Genera NON ha dispacciato nulla "
        f"(validazione che inghiotte? calls={spy.calls})")
    # LTX è il default: l'arch resta IMPLICITO (params senza 'arch') e il worker fa `arch or "ltx"`.
    # lingbot/krea_rt sono espliciti. Normalizzo empty→ltx come il downstream.
    got = str(call["params"].get("arch", "") or "ltx").lower()
    assert got == expected_arch, f"[{combo_data}] arch dispacciato {got!r} != atteso {expected_arch!r}"


def test_video_continuous_routes_selfforcing(win):
    """Video lungo 'Continuo' → instrada al motore Self-Forcing (arch='self_forcing')."""
    win.tabs.setCurrentWidget(win.video_tab)
    QB.process_events()
    idx = win.video_tab.long_mode.findData("continuous")
    assert idx >= 0, "modalità 'continuous' assente in long_mode"
    QB.combo_select(win.video_tab.long_mode, index=idx)
    win.video_tab.prompt.setPlainText(PROMPT)
    QB.process_events()
    with H.Spy(win) as spy:
        QB.click(win.gen_btn)
    call = _dispatch_call(spy)
    assert call is not None, f"'Continuo' non ha dispacciato nulla (calls={spy.calls})"
    got = str(call["params"].get("arch", "")).lower()
    assert got == "self_forcing", f"'Continuo' arch {got!r} != 'self_forcing'"


# ─────────────────────────── IMMAGINI ───────────────────────────
IMAGE_ARCHES = [
    ("FLUX", "flux"),
    ("Krea 2", "krea2"),               # regressione «nessun modello krea 2 trovato»
    ("Z-Image", "zimage"),
    ("Ernie", "ernie"),
]


@pytest.mark.parametrize("arch_label,expected_arch", IMAGE_ARCHES,
                         ids=[a[0].replace(" ", "_") for a in IMAGE_ARCHES])
def test_image_arch_routing(win, arch_label, expected_arch):
    """Tab Immagini → seleziona l'architettura dal combo gen_arch → Genera → params['arch'] corretto.
    In particolare Krea 2 (snapshot senza file-modello) DEVE dispacciare, non essere bloccato."""
    win.tabs.setCurrentWidget(win.frame1_tab)
    QB.process_events()
    combo = win.frame1_tab.gen_arch
    idx = combo.findText(arch_label)
    assert idx >= 0, f"arch immagine {arch_label!r} assente nel combo gen_arch"
    QB.combo_select(combo, index=idx)
    assert combo.currentText() == arch_label
    QB.process_events()                                          # _reload_gen popola i modelli
    win.frame1_tab.prompt.setPlainText(PROMPT)
    with H.Spy(win) as spy:
        QB.click(win.frame1_tab.gen_btn)
        QB.process_events()
    call = _dispatch_call(spy)
    assert call is not None, (
        f"[{arch_label}] click su Genera NON ha dispacciato nulla "
        f"(validazione che inghiotte il dispatch? calls={spy.calls})")
    got = str(call["params"].get("arch", "")).lower()
    assert got == expected_arch, f"[{arch_label}] arch dispacciato {got!r} != atteso {expected_arch!r}"


def test_krea2_variant_reaches_backend(win):
    """Krea 2: la variante scelta (Turbo/Raw) arriva al backend come krea2_variant."""
    win.tabs.setCurrentWidget(win.frame1_tab)
    QB.process_events()
    combo = win.frame1_tab.gen_arch
    idx = combo.findText("Krea 2")
    assert idx >= 0
    QB.combo_select(combo, index=idx)
    QB.process_events()
    win.frame1_tab.prompt.setPlainText(PROMPT)
    with H.Spy(win) as spy:
        QB.click(win.frame1_tab.gen_btn)
        QB.process_events()
    call = _dispatch_call(spy)
    assert call is not None, f"Krea 2 non ha dispacciato (calls={spy.calls})"
    p = call["params"]
    assert str(p.get("arch", "")).lower() == "krea2"
    # krea2_variant deve essere presente e valorizzato (turbo/raw)
    var = str(p.get("krea2_variant", "")).lower()
    assert var in ("turbo", "raw"), f"krea2_variant assente/errato nei params: {p.get('krea2_variant')!r}"
