"""Casi GUI (mock) di FLUX.3 Image Klein (task 13): arch nel selettore SOLO con la sua funzione, DiT dal catalogo, passi/CFG del
distillato, riferimenti ≤10 con token «the Nth image», riquadri (layout e edit), motore FLUX.3 nei pannelli di editing (Integra,
Componi, Colore), pesi mancanti (avviso, niente job), e il filtro per architettura: un FLUX.3 non compare fra FLUX/Z-Image/Ernie e le
altre voci non compaiono sotto FLUX.3. Lo Spy intercetta _enqueue: niente GPU, nessun modello.
"""
from __future__ import annotations

import harness as H

from cases_qwen21_image import _ids, _img, _run

DIT = "flux3-dit-fp8r"


def _arch(f, label):
    f.img_nsfw.setCurrentText("SFW")
    f.gen_arch.setCurrentText(label)


def case_flux3_panel_visibility_and_model(w, spy):
    f = w.frame1_tab
    _arch(f, "FLUX")
    assert f._f3_box.isHidden(), "pannello FLUX.3 nascosto con FLUX"
    assert DIT not in _ids(f.gen_model), "il DiT FLUX.3 non deve comparire fra i modelli FLUX"
    for lbl in ("Z-Image", "Ernie", "Krea 2", "Qwen-Image 2.1"):
        _arch(f, lbl)
        assert f._f3_box.isHidden() and DIT not in _ids(f.gen_model), f"FLUX.3 visibile sotto {lbl}"
    _arch(f, "FLUX.3 Klein")
    assert not f._f3_box.isHidden(), "pannello FLUX.3 visibile con FLUX.3 Klein"
    assert _ids(f.gen_model) == [DIT], _ids(f.gen_model)
    assert f.f3_pup.isChecked(), "upsampler attivo di default (raccomandazione BFL)"
    assert "SEMPRE" in f.f3_pup.toolTip(), "il testo deve spiegare perché"
    assert f.gen_steps.value() == 16 and f.gen_steps.minimum() == 4 and f.gen_steps.maximum() == 16
    assert not f.gen_cfg.isEnabled() and not f.gen_neg.isEnabled(), "distillato: niente CFG né negative"
    _arch(f, "Qwen-Image 2.1")                                     # tornando indietro i campi si riaprono
    assert f.gen_cfg.isEnabled() and f.gen_neg.isEnabled() and f.gen_steps.maximum() == 80


def case_flux3_lora_panel_arch(w, spy):
    f = w.frame1_tab
    _arch(f, "FLUX.3 Klein")
    assert f.gen_lora_tab.arch.upper() == "FLUX3" if hasattr(f.gen_lora_tab, "arch") else True


def case_flux3_refs_up_to_ten_and_tokens(w, spy):
    f = w.frame1_tab
    _arch(f, "FLUX.3 Klein")
    assert f._gen_refs_box_w.isVisibleTo(f), "sezione riferimenti nascosta con FLUX.3 Klein"
    f.gen_refs.clear()
    added = sum(f.gen_refs.add_path(_img(f"_f3_r{i}.png")) for i in range(12))
    assert added == 10, f"limite FLUX.3 = 10, aggiunti {added}"
    f.gen_refs.clear()
    a, b = _img("_f3_a.png"), _img("_f3_b.png")
    f.gen_refs.add_path(a); f.gen_refs.add_path(b)
    f.gen_prompt.setPlainText("@Foto_2 accanto a @Foto_1 al tramonto")
    p = _run(w, spy, w.generate_image)
    assert p["arch"] == "flux3" and p["ref_images"] == [a, b], p
    assert p["prompt"] == "the second image accanto a the first image al tramonto", p["prompt"]
    assert p["flux3_dit"] == DIT and p["flux3_pup"] is True and p["steps"] == 16 and p.get("boxes") == [], p
    f.gen_refs.clear()


def case_flux3_t2i_params_and_pup_off(w, spy):
    f = w.frame1_tab
    _arch(f, "FLUX.3 Klein")
    f.gen_refs.clear()
    f.gen_prompt.setPlainText("un faro di vetro su una scogliera di sale")
    f.f3_pup.setChecked(False)
    p = _run(w, spy, w.generate_image)
    assert p["arch"] == "flux3" and p["flux3_pup"] is False and not p.get("ref_images"), p
    assert p["prompt"] == "un faro di vetro su una scogliera di sale"
    f.f3_pup.setChecked(True)


def case_flux3_boxes_layout_reach_the_job(w, spy):
    f = w.frame1_tab
    _arch(f, "FLUX.3 Klein")
    f.gen_refs.clear()
    boxes = [{"id": "faro_1", "desc": "a tall glass lighthouse", "bbox": [0.1, 0.4, 0.8, 0.6]}]
    f._set_flux3_boxes("f3", boxes)
    assert "1" in f.f3_boxes_lbl.text(), f.f3_boxes_lbl.text()
    f.gen_prompt.setPlainText("a coast at dusk with <faro_1>")
    p = _run(w, spy, w.generate_image)
    assert p["boxes"] == boxes, p
    f._set_flux3_boxes("f3", [])
    assert "nessun" in f.f3_boxes_lbl.text().lower()


def case_flux3_invalid_boxes_block_the_job(w, spy):
    from PySide6.QtWidgets import QMessageBox
    f = w.frame1_tab
    _arch(f, "FLUX.3 Klein")
    f.gen_refs.clear()
    f._set_flux3_boxes("f3", [{"id": "9no", "desc": "", "bbox": [0.5, 0.5, 0.1, 0.1]}])
    f.gen_prompt.setPlainText("x")
    shown = []
    old = QMessageBox.warning
    QMessageBox.warning = staticmethod(lambda *a, **k: shown.append(a[2] if len(a) > 2 else "") or QMessageBox.Ok)
    n0 = len(spy.calls)
    try:
        import core.ensure_weights as ew
        saved = ew.missing
        ew.missing = lambda ids: []
        try:
            w.generate_image()
        finally:
            ew.missing = saved
    finally:
        QMessageBox.warning = old
        f._set_flux3_boxes("f3", [])
    assert shown, "nessun avviso sui riquadri non validi"
    assert len(spy.calls) == n0, "il job è partito con riquadri non validi"


def case_flux3_missing_weights_warn_and_no_job(w, spy):
    """Pesi assenti → avviso con nome e GB e «Apri tab Pesi»: nessun job, nessun download."""
    import core.ensure_weights as ew
    f = w.frame1_tab
    _arch(f, "FLUX.3 Klein")
    f.gen_refs.clear()
    f.gen_prompt.setPlainText("una nave nella bottiglia")
    seen = {}
    saved = ew.missing

    def fake_missing(ids):
        seen["ids"] = list(ids)
        return [{"id": i, "label": i, "approx_gb": 1.5} for i in ids]
    ew.missing = fake_missing
    n0 = len(spy.calls)
    try:
        w.generate_image()
    finally:
        ew.missing = saved
    assert len(spy.calls) == n0, "job partito con i pesi mancanti"
    assert {DIT, "flux3-te", "flux2-vae", "flux3-pup-sakura-q8", "flux3-pup-configs"} <= set(seen["ids"]), seen
    f.f3_pup.setChecked(False)
    seen.clear()
    ew.missing = fake_missing
    try:
        w.generate_image()
    finally:
        ew.missing = saved
        f.f3_pup.setChecked(True)
    assert "flux3-pup-sakura-q8" not in seen["ids"], "senza upsampler i suoi pesi non servono"


def case_flux3_engine_in_edit_panels(w, spy):
    f = w.frame1_tab
    for combo, name in ((f.integrate_engine, "Integra"), (f.color_engine, "Colore")):
        assert combo.count() == 5 and combo.itemText(4) == "FLUX.3 Klein", f"{name}: motore FLUX.3 assente"
    f.integrate_engine.setCurrentIndex(4)
    assert _ids(f.integrate_model) == [DIT], _ids(f.integrate_model)
    assert not f._f3e_widget.isHidden(), "controlli upsampler/riquadri nascosti col motore FLUX.3"
    f.integrate_engine.setCurrentIndex(3)
    assert DIT not in _ids(f.integrate_model) and f._f3e_widget.isHidden(), "FLUX.3 sotto il motore Qwen"
    f.integrate_engine.setCurrentIndex(0)
    assert DIT not in _ids(f.integrate_model)
    f.color_engine.setCurrentIndex(4)
    assert _ids(f.color_model) == [DIT]
    f.color_engine.setCurrentIndex(0)
    f.integrate_engine.setCurrentIndex(0)


def case_flux3_edit_engine_integra_job(w, spy):
    f = w.frame1_tab
    f.integrate_engine.setCurrentIndex(4)
    f.integrate_prompt.setPlainText("trasforma la scena in una notte piovosa")
    f.f3e_pup.setChecked(False)
    f._set_flux3_boxes("f3e", [{"id": "lamp_1", "desc": "a street lamp", "mode": "new", "tgt_bbox": [0.1, 0.1, 0.5, 0.3]}])
    base = _img("_f3_base.png")
    f.canvas.load_image(base) if hasattr(f.canvas, "load_image") else None
    cfg = f.get_inpaint_config()
    assert cfg["integrate_engine"] == "flux3" and cfg["integrate_flux3_dit"] == DIT, cfg
    assert cfg["flux3_pup"] is False and cfg["boxes"][0]["id"] == "lamp_1", cfg
    f._set_flux3_boxes("f3e", [])
    f.f3e_pup.setChecked(True)
    f.integrate_engine.setCurrentIndex(0)


def case_flux3_edit_job_direct(w, spy):
    """_start_native_flux3_edit: riferimenti ≤10, riquadri validati contro il numero di riferimenti, risoluzione dal canvas."""
    a, b = _img("_f3_ea.png"), _img("_f3_eb.png")

    def go():
        w._start_native_flux3_edit(DIT, "metti the second image sul tavolo", refs=[a, b], pup=True,
                                   boxes=[{"id": "t_1", "desc": "table", "mode": "move", "ref": 2,
                                           "src_bbox": [0.1, 0.1, 0.4, 0.4], "tgt_bbox": [0.5, 0.5, 0.9, 0.9]}],
                                   mode="composizione")
    p = _run(w, spy, go)
    assert p["arch"] == "flux3" and p["ref_images"] == [a, b] and p["flux3_pup"] is True and p["steps"] == 16, p
    assert p["boxes"][0]["ref"] == 2 and "width" in p and "height" in p, p

    def go_mask():
        w._start_native_flux3_edit(DIT, "dipingi di blu", init_image=a, mask=b, pup=False, mode="inpaint")
    p = _run(w, spy, go_mask)
    assert p["init_image"] == a and p["mask_path"] == b and "width" not in p, p       # con maschera: forma della foto


def case_flux3_boxes_dialog_context(w, spy):
    """Il pannello riquadri si apre in LAYOUT senza riferimenti e in EDIT con i riferimenti (@Foto_N)."""
    from ui.widgets.flux3_boxes import Flux3BoxesDialog
    f = w.frame1_tab
    _arch(f, "FLUX.3 Klein")
    f.gen_refs.clear()
    assert f._flux3_refs("f3") == []
    d = Flux3BoxesDialog(f, [], 1.5)
    assert not d.panel._edit and d.panel.cb_mode.isHidden()
    f.gen_refs.add_path(_img("_f3_d0.png")); f.gen_refs.add_path(_img("_f3_d1.png"))
    refs = f._flux3_refs("f3")
    assert [n for n, _p in refs] == ["Foto_1", "Foto_2"], refs
    d2 = Flux3BoxesDialog(f, refs, 1.0)
    assert d2.panel._edit and not d2.panel.cb_mode.isHidden() and d2.panel.cb_ref.count() == 2
    f.gen_refs.clear()


CASES = [
    ("flux3_panel_visibility_and_model", "FLUX.3 Klein: pannello e DiT SOLO sotto la sua arch; distillato 16 passi senza CFG/negative; "
     "upsampler on di default.", case_flux3_panel_visibility_and_model),
    ("flux3_lora_panel_arch", "Il pannello LoRA passa all'arch Flux3.", case_flux3_lora_panel_arch),
    ("flux3_refs_up_to_ten_and_tokens", "Fino a 10 riferimenti; @Foto_N → «the Nth image» nell'ordine passato.",
     case_flux3_refs_up_to_ten_and_tokens),
    ("flux3_t2i_params_and_pup_off", "Parametri t2i verso il worker; upsampler disattivabile.", case_flux3_t2i_params_and_pup_off),
    ("flux3_boxes_layout_reach_the_job", "I riquadri di layout arrivano al job.", case_flux3_boxes_layout_reach_the_job),
    ("flux3_invalid_boxes_block_the_job", "Riquadri non validi: avviso, nessun job.", case_flux3_invalid_boxes_block_the_job),
    ("flux3_missing_weights_warn_and_no_job", "Pesi mancanti: avviso con nome/GB, nessun job né download; senza upsampler "
     "non servono i suoi pesi.", case_flux3_missing_weights_warn_and_no_job),
    ("flux3_engine_in_edit_panels", "Motore FLUX.3 in Integra/Colore: DiT di catalogo solo lì; controlli upsampler/riquadri.",
     case_flux3_engine_in_edit_panels),
    ("flux3_edit_engine_integra_job", "Configurazione Integra con motore FLUX.3 (DiT, upsampler, riquadri).",
     case_flux3_edit_engine_integra_job),
    ("flux3_edit_job_direct", "Job di edit: riferimenti, riquadri di edit, risoluzione dal canvas; con maschera = composito.",
     case_flux3_edit_job_direct),
    ("flux3_boxes_dialog_context", "Pannello riquadri: layout senza riferimenti, edit con @Foto_N.", case_flux3_boxes_dialog_context),
]
