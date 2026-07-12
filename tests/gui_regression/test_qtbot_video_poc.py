"""F3 (#94) — PoC pytest(-qt) sul TAB VIDEO: interazioni utente REALI al posto delle
impostazioni programmatiche. Ogni test:

  1. costruisce la MainWindow offscreen con `harness.build_window()` (dialog patchati, no server),
  2. INTERAGISCE come l'utente via `qtbot_harness` (digitazione, frecce, click reali —
     incluso il click vero su `gen_btn`, che attraversa il wiring clicked→on_generate),
  3. verifica il PARAMETRO risultante nella richiesta GUI→backend catturata dallo Spy
     (contratto GUI→backend, ZERO GPU: nessuna generazione parte davvero).

Esecuzione (venv GUI):
  set QT_QPA_PLATFORM=offscreen
  E:\\AI\\Comfy_Nexus.bak\\ComfyUI\\venv\\Scripts\\python.exe -m pytest \
      tests/gui_regression/test_qtbot_video_poc.py -v

Note:
  • `native_driver.is_up` è patchato a True (App. A.2 della GUI_TEST_MATRIX): senza server,
    `_refresh_status` disabiliterebbe `gen_btn` e il click reale diventerebbe un no-op.
  • ERMETICITÀ (finding F3): la MainWindow ripristina lo stato GUI REALE dell'utente
    (`gui_state.json`, via QTimer.singleShot(0, _restore_gui_state)) — es. V2V="Canta" residuo
    dell'ultima sessione manda on_generate su MultiTalk e BLOCCA tutti i dispatch. Inoltre
    closeEvent → gui_state.store() SOVRASCRIVEREBBE lo stato reale dell'utente coi valori di
    test. Il fixture patcha gui_state.restore/store a no-op (ripristinati a fine test).
  • Il tab Video viene portato in primo piano nel fixture (= l'utente naviga al tab): serve
    perché le interazioni reali richiedono widget VISIBILI (guardia di qtbot_harness).
  • I casi storici (cases_video_tab.py, runner run_all/run_real) NON sono toccati.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # import harness/qtbot_harness

import pytest                                                     # noqa: E402

import harness as H                                               # noqa: E402
import qtbot_harness as QB                                        # noqa: E402

# prompt creativo (regola: mai gattini/cani), senza '\\' (tag-palette) e senza newline
PROMPT = ("un drago di cristallo pattina su un lago ghiacciato al tramonto, "
          "riflessi arancio, ripresa cinematografica")


# ─────────────────────────── fixture ───────────────────────────
@pytest.fixture()
def win():
    """MainWindow offscreen ERMETICA: backend 'su' (finto), stato GUI reale NON ripristinato né
    salvato (gui_state no-op), tab Video in primo piano, baseline V2V=Normale / storyboard OFF."""
    from core import native_driver, gui_state
    _orig_up = native_driver.is_up
    _orig_restore, _orig_store = gui_state.restore, gui_state.store
    native_driver.is_up = lambda: True          # gen_btn resta abilitato senza server reale
    gui_state.restore = lambda _win, _state=None: {}   # lo stato REALE dell'utente non entra nei test
    gui_state.store = lambda _win: None                # …e i valori di test non lo sovrascrivono al close
    w = H.build_window()
    try:
        w._refresh_status()                     # popola SUBITO l'inventario (combo modelli stabile)
    except Exception:                           # noqa: BLE001
        pass
    w.tabs.setCurrentWidget(w.video_tab)        # l'utente è sul tab Video (visibilità reale)
    QB.process_events()                         # consegna il singleShot(0) di _restore_gui_state (no-op)
    # baseline esplicita (difesa in profondità rispetto al no-op di restore)
    w.video_tab.v2v_panel.rb_normal.setChecked(True)
    w.video_tab.storyboard_panel.enable.setChecked(False)
    QB.process_events()
    try:
        yield w
    finally:
        native_driver.is_up = _orig_up
        gui_state.restore, gui_state.store = _orig_restore, _orig_store
        w.close()
        QB.process_events()


def _last_native(spy):
    """Ultima richiesta GUI→backend catturata (enqueue worker o native_driver)."""
    assert spy.calls, "nessuna richiesta catturata dallo Spy (il click non ha dispacciato nulla)"
    for c in reversed(spy.calls):
        if c["via"] in ("enqueue", "native_driver"):
            return c
    return spy.calls[-1]


def _first_concrete_index(combo):
    """Primo modello concreto gestito da get_values (diffusion/gguf) nel dropdown."""
    for i in range(combo.count()):
        d = combo.itemData(i) or {}
        if d.get("kind") in ("diffusion", "gguf") and d.get("model"):
            return i, d
    return None, {}


def _generate(win):
    """Click REALE su Genera dentro lo Spy; ritorna i params catturati."""
    with H.Spy(win) as spy:
        QB.click(win.gen_btn)
        return _last_native(spy)["params"]


# ─────────────────────────── test (interazioni reali) ───────────────────────────
def test_type_prompt_reaches_backend(win):
    """Digitare il prompt (click + tastiera) → il testo arriva IDENTICO nel param 'prompt'."""
    QB.type_text(win.video_tab.prompt, PROMPT)
    assert win.video_tab.prompt.toPlainText() == PROMPT
    p = _generate(win)
    assert p.get("prompt", "").startswith(PROMPT), \
        f"prompt digitato non trasmesso: {p.get('prompt')!r}"


def test_empty_prompt_typed_guard(win):
    """Svuotare il prompt da tastiera (Ctrl+A+Canc) + click su Genera → NESSUNA richiesta."""
    QB.type_text(win.video_tab.prompt, "")           # clear reale via tastiera
    assert win.video_tab.prompt.toPlainText() == ""
    with H.Spy(win) as spy:
        QB.click(win.gen_btn)
        assert not spy.calls, f"prompt vuoto ma richiesta inviata: {spy.calls}"


def test_combo_model_selected_with_keyboard(win):
    """Selezione modello dal combo con le FRECCE (segnali reali) → il modello scelto è
    inoltrato al backend (model / gguf_transformer)."""
    idx, d = _first_concrete_index(win.video_tab.model)
    if idx is None:
        pytest.skip("nessun modello concreto (diffusion/gguf) nel dropdown")
    QB.combo_select(win.video_tab.model, index=idx)
    assert (win.video_tab.model.currentData() or {}).get("model") == d["model"]
    win.video_tab.prompt.setPlainText(PROMPT)        # setup (non oggetto del test)
    p = _generate(win)
    sent = " ".join(str(p.get(k, "")) for k in ("model", "gguf_transformer", "model_gguf"))
    assert d["model"] in sent, (
        f"modello selezionato dal combo NON inoltrato al backend: {d['model']!r} "
        f"(chiavi params: {sorted(p.keys())})")


def test_spinbox_duration_and_steps_typed(win):
    """Durata/fps/steps DIGITATI negli spinbox (selectAll+cifre+Invio) → num_frames/fps/steps."""
    QB.spin_set(win.video_tab.fps, 24)
    QB.spin_set(win.video_tab.length, 2.0)
    QB.spin_set(win.video_tab.sampler_seed.steps, 12)
    win.video_tab.prompt.setPlainText(PROMPT)
    p = _generate(win)
    expected_nf = max(9, int(round(2.0 * 24)) // 8 * 8 + 1)     # = 49
    assert p.get("num_frames") == expected_nf, f"num_frames atteso {expected_nf}: {p.get('num_frames')}"
    assert p.get("fps") == 24, f"fps non trasmesso: {p.get('fps')}"
    assert p.get("steps") == 12, f"steps digitati non trasmessi: {p.get('steps')}"


def test_click_generate_captures_contract(win):
    """Click REALE su Genera (wiring clicked→on_generate) → lo Spy cattura una richiesta
    completa e coerente (prompt/width/height/num_frames/fps/steps/guidance/seed, W/H %64)."""
    win.video_tab.prompt.setPlainText(PROMPT)
    p = _generate(win)
    for k in ("prompt", "width", "height", "num_frames", "fps", "steps", "guidance", "seed"):
        assert p.get(k) not in (None, ""), f"param obbligatorio mancante/vuoto: {k} (params: {sorted(p.keys())})"
    assert p["width"] % 64 == 0 and p["height"] % 64 == 0, \
        f"risoluzione non multipla di 64: {p['width']}x{p['height']}"


def test_checkbox_fixed_seed_clicked(win):
    """CLICK reale sul checkbox 'seed fisso' + seed digitato → il valore arriva invariato."""
    ss = win.video_tab.sampler_seed
    QB.set_checked(ss.fixed_seed, True)              # click reale, non setChecked
    QB.spin_set(ss.seed, 424242)
    assert win.video_tab.wants_fixed_seed(), "fixed_seed non attivo dopo il click"
    win.video_tab.prompt.setPlainText(PROMPT)
    p = _generate(win)
    assert p.get("seed") == 424242, f"seed fisso non rispettato: {p.get('seed')}"


def test_slider_img_strength_keyboard(win):
    """Slider 'forza immagine' mosso con la tastiera (Home+frecce/pagine, valueChanged reali)
    → in I2V diventa ref_strength = valore/100."""
    H.load_canvas_image(win)                         # precondizione I2V (API, non oggetto del test)
    QB.slider_set(win.video_tab.img_strength, 80)
    assert win.video_tab.img_strength_lbl.text() == "0.80", \
        "label dello slider non aggiornata (valueChanged non emesso?)"
    win.video_tab.prompt.setPlainText(PROMPT)
    p = _generate(win)
    assert p.get("image_path"), f"I2V senza image_path: {sorted(p.keys())}"
    assert abs(float(p.get("ref_strength", -1)) - 0.80) < 0.01, \
        f"ref_strength atteso ~0.80: {p.get('ref_strength')}"


def test_lora_row_clicked_included_in_params(win):
    """CLICK reale sulla checkbox di una riga LoRA → la LoRA (nome+strength) entra nei params."""
    lt = win.video_tab.lora
    lt.set_available_loras(["LTX-2\\qtbot_fake_lora.safetensors"])   # riga garantita nel selettore
    QB.process_events()
    row = QB.lora_click_row(lt, "qtbot_fake_lora", checked=True)
    assert row["cb"].isChecked()
    names = [s["name"] for s in win.video_tab.selected_loras()]
    assert any("qtbot_fake_lora" in n for n in names), f"LoRA cliccata non selezionata: {names}"
    win.video_tab.prompt.setPlainText(PROMPT)
    p = _generate(win)
    sent = [l.get("name", "") for l in (p.get("loras") or [])]
    assert any("qtbot_fake_lora" in n for n in sent), \
        f"LoRA selezionata col click NON inoltrata al backend: loras={sent}"
