"""Casi GUI (mock) del task 9: ControlNet FLUX.1/Z-Image, Redux e ControlNet Inpainting (Alimama) nel tab Immagini.
Verificano che ogni funzione compaia SOLO dove deve (architettura + famiglia del modello, dall'header) e che i parametri
arrivino al backend `image_control`. Lo Spy intercetta _enqueue: niente GPU, nessun peso caricato."""
from __future__ import annotations

import os

import harness as H

from cases_qwen21_image import _img, _run
from cases_frame1_tab import _draw_center_mask, _populate_inventory, _select_combo


def _model_items(f):
    return [f.gen_model.itemText(i) for i in range(f.gen_model.count()) if f.gen_model.itemData(i) not in ("__auto__", None)]


def _pick(f, want_flux2: bool):
    """Primo modello del menu FLUX che è (o non è) FLUX.2, dal file (core.arch_filter)."""
    from core import arch_filter as af
    for i, name in enumerate(_model_items(f)):
        if af.is_flux2(name) == want_flux2:
            f.gen_model.setCurrentIndex(f.gen_model.findText(name))
            return name
    return None


def _flux(f, flux2=False):
    f.img_nsfw.setCurrentText("NSFW")                  # menu completo (modelli 18+ inclusi)
    f.gen_arch.setCurrentText("FLUX")
    return _pick(f, flux2)


def case_ctl_visibility_by_arch_and_family(w, spy):
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("SFW")
    for lbl in ("Krea 2", "Qwen-Image 2.1", "Ernie"):
        f.gen_arch.setCurrentText(lbl)
        assert not (not f.cn_box.isHidden()), f"ControlNet visibile con {lbl}"
        assert not (not f.redux_box.isHidden()), f"Redux visibile con {lbl}"
    f.gen_arch.setCurrentText("Z-Image")
    assert (not f.cn_box.isHidden()), "ControlNet nascosto con Z-Image"
    assert not (not f.redux_box.isHidden()), "Redux visibile con Z-Image (è solo FLUX.1)"
    assert not (not f._cn_end_row.isHidden()), "«controllo fino a» non esiste per Z-Image"
    m1 = _flux(f, flux2=False)
    if m1:
        assert (not f.cn_box.isHidden()) and (not f.redux_box.isHidden()), f"ControlNet/Redux nascosti con FLUX.1 {m1}"
        assert (not f._cn_end_row.isHidden())
    m2 = _flux(f, flux2=True)
    if m2:
        assert not (not f.cn_box.isHidden()) and not (not f.redux_box.isHidden()), f"visibili con FLUX.2 {m2}"


def case_ctl_flux_params_to_backend(w, spy):
    _populate_inventory(w)
    f = w.frame1_tab
    m = _flux(f, flux2=False)
    if not m:
        return
    guide = _img("_ctl_guide.png")
    f.cn_box.setChecked(True)
    f.cn_photo["edit"].setText(guide)
    f.cn_kind.setCurrentIndex(f.cn_kind.findData("pose"))
    assert abs(f.cn_scale.value() / 100 - 0.9) < 1e-6 and abs(f.cn_end.value() / 100 - 0.65) < 1e-6, \
        "default della posa: forza 0.9, fino a 0.65 (model card Shakker)"
    f.gen_prompt.setPlainText("una gondola che diventa un'astronave, stessa sagoma")
    f.gen_seed_rand.setChecked(False); f.gen_seed.setValue(4242)
    p = _run(w, spy, w.generate_image)
    call = spy.calls[-1]
    assert call["module"] == "image_control", call["module"]
    assert p["mode"] == "flux_cn" and p["arch"] == "flux" and p["model"] == m, p
    assert p["control_image"] == guide and p["control_preprocess"] == "pose" and p["seed"] == 4242, p
    assert abs(p["control_scale"] - 0.9) < 1e-6 and abs(p["control_end"] - 0.65) < 1e-6, p


def case_ctl_zimage_params_to_backend(w, spy):
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("SFW")
    f.gen_arch.setCurrentText("Z-Image")
    names = _model_items(f)
    if not names:
        return
    f.gen_model.setCurrentIndex(f.gen_model.findText(names[0]))
    f.cn_box.setChecked(True)
    f.cn_photo["edit"].setText(_img("_ctl_guide.png"))
    f.cn_kind.setCurrentIndex(f.cn_kind.findData("canny"))
    f.gen_prompt.setPlainText("un pirata di cristallo")
    p = _run(w, spy, w.generate_image)
    assert spy.calls[-1]["module"] == "image_control" and p["mode"] == "zimage_cn" and p["arch"] == "zimage", p
    assert abs(p["control_scale"] - 0.75) < 1e-6 and "control_end" not in p, p


def case_ctl_redux_and_exclusivity(w, spy):
    _populate_inventory(w)
    f = w.frame1_tab
    m = _flux(f, flux2=False)
    if not m:
        return
    f.cn_box.setChecked(True)
    f.redux_box.setChecked(True)
    assert not f.cn_box.isChecked(), "Redux e ControlNet non si combinano: accendere Redux spegne il ControlNet"
    f.redux_photo["edit"].setText(_img("_ctl_ref.png"))
    f.redux_strength.setValue(80)
    f.gen_prompt.setPlainText("")
    f.gen_prompt.setPlainText("lo stesso mondo ma al tramonto")
    p = _run(w, spy, w.generate_image)
    assert spy.calls[-1]["module"] == "image_control" and p["mode"] == "flux_redux", p
    assert p["ref_images"] == [_img("_ctl_ref.png")] and abs(p["redux_strength"] - 0.8) < 1e-6, p
    f.cn_box.setChecked(True)
    assert not f.redux_box.isChecked()


def case_ctl_missing_weights_warns_and_does_not_generate(w, spy):
    from PySide6.QtWidgets import QMessageBox
    import core.ensure_weights as ew
    from core import native_driver
    _populate_inventory(w)
    f = w.frame1_tab
    m = _flux(f, flux2=False)
    if not m:
        return
    f.cn_box.setChecked(True)
    f.cn_photo["edit"].setText(_img("_ctl_guide.png"))
    f.gen_prompt.setPlainText("x")
    seen = []
    saved_exec, saved_missing, saved_up = QMessageBox.exec, ew.missing, native_driver.is_up
    QMessageBox.exec = lambda self, *a, **k: seen.append(self.text()) or 0
    ew.missing = lambda ids: [{"id": i, "label": i, "approx_gb": 4.3} for i in ids]
    native_driver.is_up = lambda: True
    n0 = len(spy.calls)
    try:
        w.generate_image()
    finally:
        QMessageBox.exec, ew.missing, native_driver.is_up = saved_exec, saved_missing, saved_up
    assert seen and "flux-cn-union-pro2" in seen[0], f"nessun avviso con nome del peso mancante: {seen}"
    assert len(spy.calls) == n0, "generazione partita con i pesi mancanti"


def case_ctl_cn_inpaint_only_with_flux1_base(w, spy):
    from core import arch_filter as af
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    base = klein = fill = None
    for i in range(f.inp_model.count()):
        label = f.inp_model.itemText(i)
        f.inp_model.setCurrentIndex(i)
        k = f._inp_kind_now()
        if k == "base" and base is None:
            base = label
        elif k == "klein" and klein is None:
            klein = label
        elif k == "fill" and fill is None:
            fill = label
    assert base, "nessun modello FLUX.1 base nel menu inpaint (inventario di test)"
    _select_combo(f.inp_model, base)
    assert (not f.inp_cn.isHidden()), f"ControlNet Inpainting nascosto con un FLUX.1 base ({base})"
    for other in (klein, fill):
        if other:
            _select_combo(f.inp_model, other)
            assert not (not f.inp_cn.isHidden()) and not f.inp_cn.isChecked(), f"visibile con {other}"
    _select_combo(f.inp_model, base)
    f.inp_cn.setChecked(True)
    f.inp_cn_scale.setValue(75)
    H.load_canvas_image(w)
    f.inpaint_prompt.setPlainText("una corona di fiori al posto del cappello")
    _draw_center_mask(f.canvas)
    f.inp_seed_rand.setChecked(False); f.inp_seed.setValue(99)
    p = _run(w, spy, lambda: w.generate_inpaint("inpaint"))
    assert spy.calls[-1]["module"] == "image_control" and p["mode"] == "flux_cn_inpaint" and p["arch"] == "flux", p
    assert os.path.isfile(p["image_path"]) and os.path.isfile(p["mask_path"]) and abs(p["control_scale"] - 0.75) < 1e-6, p


def _mq_ready(f):
    f.img_nsfw.setCurrentText("NSFW")
    for i in range(f.mq_model.count()):
        if not f.mq_model.itemText(i).startswith("("):
            f.mq_model.setCurrentIndex(i)
            return f.mq_model.itemText(i)
    return None


def case_mq_modes_and_params(w, spy):
    _populate_inventory(w)
    f = w.frame1_tab
    m = _mq_ready(f)
    if not m:
        return
    H.load_canvas_image(w)
    _draw_center_mask(f.canvas)
    seen = {}
    for mode, strength_key, want in (("local", "strength_local", 1.0), ("removal", "strength_local", 1.0),
                                     ("edge", "strength_edge", 0.6), ("color", "strength_color", 1.5)):
        f.mq_mode.setCurrentIndex(f.mq_mode.findData(mode))
        assert abs(f.mq_strength.value() - want) < 1e-6, f"{mode}: forza di default {f.mq_strength.value()} != {want}"
        f.mq_prompt.setPlainText("una corona d'alloro dorata" if mode != "removal" else "")
        f.mq_color.setText("#cc2244")
        p = _run(w, spy, w.generate_magicquill)
        call = spy.calls[-1]
        assert call["module"] == "image_mq" and p["mode"] == mode and p["arch"] == "kontext" and p["model"].endswith(m), p
        assert abs(p[strength_key] - want) < 1e-6, p
        seen[mode] = p
    assert "mask_path" in seen["local"] and seen["local"]["grow"] == 10
    assert "edge_path" in seen["edge"] and "erase_path" not in seen["edge"]
    assert seen["color"]["color_rgb"] == [204, 34, 68] and "mask_path" in seen["color"]
    f.mq_mode.setCurrentIndex(f.mq_mode.findData("edge")); f.mq_erase.setChecked(True)
    p = _run(w, spy, w.generate_magicquill)
    assert "erase_path" in p and "edge_path" not in p, p


def case_mq_guards_and_visibility(w, spy):
    from PySide6.QtWidgets import QMessageBox
    import core.ensure_weights as ew
    from core import native_driver
    from core import arch_filter as af
    _populate_inventory(w)
    f = w.frame1_tab
    m = _mq_ready(f)
    for i in range(f.mq_model.count()):                 # il menu contiene SOLO Kontext FLUX.1 (header)
        t = f.mq_model.itemText(i)
        if not t.startswith("("):
            assert af.is_kontext(t), f"{t} nel menu MagicQuill ma non e' un Kontext FLUX.1"
    f.mq_mode.setCurrentIndex(f.mq_mode.findData("removal"))
    assert f.mq_prompt.isHidden() and f._mq_color_row.isHidden() and not f.mq_grow.isHidden()
    f.mq_mode.setCurrentIndex(f.mq_mode.findData("color"))
    assert not f._mq_color_row.isHidden() and f.mq_grow.isHidden() and not f.mq_prompt.isHidden()
    f.mq_mode.setCurrentIndex(f.mq_mode.findData("edge"))
    assert not f.mq_erase.isHidden()
    if not m:
        return
    H.load_canvas_image(w)                              # senza maschera: avviso, nessun job
    f.canvas.clear_mask(silent=True)
    n0 = len(spy.calls)
    shown = []
    old = QMessageBox.information
    QMessageBox.information = staticmethod(lambda *a, **k: shown.append(a[2] if len(a) > 2 else "") or QMessageBox.Ok)
    try:
        w.generate_magicquill()
    finally:
        QMessageBox.information = old
    assert shown and len(spy.calls) == n0, "senza maschera: serve l'avviso e nessun job"
    _draw_center_mask(f.canvas)                         # pesi LoRA mancanti: avviso con nome, nessun job
    seen = []
    se, sm, su = QMessageBox.exec, ew.missing, native_driver.is_up
    QMessageBox.exec = lambda self, *a, **k: seen.append(self.text()) or 0
    ew.missing = lambda ids: [{"id": i, "label": i, "approx_gb": 0.3} for i in ids]
    native_driver.is_up = lambda: True
    try:
        w.generate_magicquill()
    finally:
        QMessageBox.exec, ew.missing, native_driver.is_up = se, sm, su
    assert seen and "mq2-edge-lora" in seen[0] and len(spy.calls) == n0, seen




def case_mq_component_missing_warns_no_crash(w, spy):
    """Edizione senza il codice MagicQuill (file esclusi dalla build): avviso «non inclusa», nessun job, nessun crash;
    vale per le 4 modalita' e per l'integrazione (casella puzzle di Integra)."""
    import os
    from PySide6.QtWidgets import QMessageBox
    _populate_inventory(w)
    f = w.frame1_tab
    m = _mq_ready(f)
    if not m:
        return
    H.load_canvas_image(w)
    _draw_center_mask(f.canvas)
    seen = []
    old_info = QMessageBox.information
    QMessageBox.information = staticmethod(lambda *a, **k: seen.append(a[2] if len(a) > 2 else "") or QMessageBox.Ok)
    os.environ["LTX_OPTIONAL_DISABLE"] = "magicquill"
    n0 = len(spy.calls)
    try:
        f.mq_mode.setCurrentIndex(f.mq_mode.findData("local"))
        f.mq_prompt.setPlainText("x")
        w.generate_magicquill()
        w._start_native_mq_foreground({"image": "a.png", "mask": "b.png"}, m, "x", 20, 1, 2.5)
    finally:
        os.environ.pop("LTX_OPTIONAL_DISABLE", None)
        QMessageBox.information = old_info
    assert len(seen) == 2 and all('inclusa' in t for t in seen), seen
    assert len(spy.calls) == n0, "job partito senza il componente"


CASES = [
    ("ctl_visibility_by_arch_and_family", "ControlNet/Redux solo con FLUX.1 e Z-Image (Redux solo FLUX.1); mai con FLUX.2, Krea 2, "
     "Qwen, Ernie.", case_ctl_visibility_by_arch_and_family),
    ("ctl_flux_params_to_backend", "ControlNet FLUX.1: modulo image_control, mode flux_cn, posa 0.9/0.65, foto guida e seed.",
     case_ctl_flux_params_to_backend),
    ("ctl_zimage_params_to_backend", "ControlNet Z-Image: mode zimage_cn, forza 0.75, nessun «fino a».",
     case_ctl_zimage_params_to_backend),
    ("ctl_redux_and_exclusivity", "Redux: mode flux_redux con riferimento e peso; Redux e ControlNet si escludono.",
     case_ctl_redux_and_exclusivity),
    ("ctl_missing_weights_warns", "Pesi ControlNet mancanti: avviso con nome e GB, nessuna generazione.",
     case_ctl_missing_weights_warns_and_does_not_generate),
    ("ctl_cn_inpaint_flux1_base_only", "ControlNet Inpainting solo con un FLUX.1 base (non Fill/Klein); params verso image_control.",
     case_ctl_cn_inpaint_only_with_flux1_base),
    ("mq_modes_and_params", "MagicQuill: 4 modalita' con forza di default dell'autore (0.6/1.5/1.0), parametri e file verso image_mq.",
     case_mq_modes_and_params),
    ("mq_component_missing", "Componente MagicQuill non installato: avviso + Apri tab Pesi, nessun job, nessun crash.",
     case_mq_component_missing_warns_no_crash),
    ("mq_guards_and_visibility", "MagicQuill: menu solo Kontext FLUX.1, controlli per modalita', avvisi senza maschera e pesi mancanti.",
     case_mq_guards_and_visibility),
]
