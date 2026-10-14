"""Casi GUI (mock) di Qwen-Image 2.1 nel tab Immagini: arch nel selettore, DiT dal catalogo (Turbo, 18+ solo in
NSFW), pannello dedicato (turbo, formato 2K, RGBA, encoder 18+, riferimenti ≤10 con token «image N», img2img),
motore Qwen nei pannelli di editing (Inpaint/Outpaint, Integra, Componi, Colore) e parametri che arrivano al
worker. Lo Spy intercetta _enqueue: niente GPU.
"""
from __future__ import annotations

import os

import harness as H

SFW_DITS = ["qwen21-dit-nvfp4", "qwen21-dit-fp8", "qwen21-dit-int8",
            "qwen21-turbo-dit-nvfp4", "qwen21-turbo-dit-int8"]
ADULT_DITS = ["qwen21-uc-dit-nvfp4", "qwen21-uc-dit-int8", "qwen21-noctq-v4-int8", "qwen21-noctq-anime-int8"]


def _img(name: str) -> str:
    p = str(H.OUT_DIR / name)
    if not os.path.exists(p):
        H.make_test_image(p, 512, 512)
    return p


def _ready(monkey: list):
    """Pesi «presenti» e backend nativo «su»: gli handler costruiscono e accodano il job (Spy)."""
    import core.ensure_weights as ew
    from core import native_driver
    saved, saved_up = ew.missing, native_driver.is_up
    ew.missing = lambda ids: []
    native_driver.is_up = lambda: True
    monkey.append(lambda: setattr(ew, "missing", saved))
    monkey.append(lambda: setattr(native_driver, "is_up", saved_up))


def _run(w, spy, fn):
    monkey: list = []
    _ready(monkey)
    try:
        fn()
    finally:
        for u in monkey:
            u()
    call = next((c for c in reversed(spy.calls) if c["via"] == "enqueue"), None)
    assert call, f"nessun job accodato: {spy.calls}"
    return call["params"]


def _ids(combo):
    return [combo.itemData(i) for i in range(combo.count())]


def _mask(cv):
    from PySide6.QtGui import QPainter, QColor
    from PySide6.QtCore import QRect
    pm = QPainter(cv.mask_image)
    pm.fillRect(QRect(cv.aspect_w // 4, cv.aspect_h // 4, cv.aspect_w // 2, cv.aspect_h // 2), QColor(255, 255, 255))
    pm.end()
    cv._refresh_mask()


def case_qwen21_panel_visibility(w, spy):
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("SFW")
    f.gen_arch.setCurrentText("FLUX")
    assert f._q21_box.isHidden(), "pannello Qwen nascosto con FLUX"
    f.gen_arch.setCurrentText("Qwen-Image 2.1")
    assert not f._q21_box.isHidden(), "pannello Qwen visibile con Qwen-Image 2.1"
    assert _ids(f.gen_model) == SFW_DITS, _ids(f.gen_model)
    assert f.gen_steps.value() == 40 and abs(f.gen_cfg.value() - 1.0) < 1e-6, "default: 40 passi, cfg 1.0"


def case_qwen21_adult_dits_only_nsfw(w, spy):
    f = w.frame1_tab
    f.gen_arch.setCurrentText("Qwen-Image 2.1")
    f.img_nsfw.setCurrentText("SFW")
    assert not set(ADULT_DITS) & set(_ids(f.gen_model)), "DiT 18+ visibili in SFW"
    f.integrate_engine.setCurrentIndex(3)
    assert not set(ADULT_DITS) & set(_ids(f.integrate_model)), "DiT 18+ visibili in Integra (SFW)"
    if f.img_nsfw.findText("NSFW") >= 0:
        f.img_nsfw.setCurrentText("NSFW")
        assert set(ADULT_DITS) <= set(_ids(f.gen_model)), f"DiT 18+ assenti in NSFW: {_ids(f.gen_model)}"
        assert set(ADULT_DITS) <= set(_ids(f.integrate_model)), "DiT 18+ assenti in Integra (NSFW)"
        f.gen_model.setCurrentIndex(f.gen_model.findData("qwen21-noctq-v4-int8"))
        assert f.gen_steps.value() == 25 and abs(f.gen_cfg.value() - 3.0) < 1e-6, "Noct-Q: 25 passi, CFG 3"
        f.img_nsfw.setCurrentText("SFW")                         # torna SFW: la voce 18+ sparisce
        assert f.gen_model.currentData() not in ADULT_DITS, "voce 18+ ancora selezionata in SFW"


def case_qwen21_turbo_locks_steps(w, spy):
    f = w.frame1_tab
    f.gen_arch.setCurrentText("Qwen-Image 2.1")
    f.gen_model.setCurrentIndex(f.gen_model.findData("qwen21-turbo-dit-int8"))
    assert f.gen_steps.value() == 8 and not f.gen_steps.isEnabled(), "Turbo: 8 passi bloccati"
    assert abs(f.gen_cfg.value() - 1.0) < 1e-6 and not f.gen_cfg.isEnabled(), "Turbo: CFG 1 bloccata"
    assert not f.q21_turbo.isEnabled() and f.q21_turbo.currentData() == "", "Turbo: niente LoRA turbo"
    f.gen_model.setCurrentIndex(f.gen_model.findData("qwen21-dit-int8"))
    assert f.gen_steps.isEnabled() and f.gen_steps.value() == 40 and f.q21_turbo.isEnabled(), "base: sbloccato"


def case_qwen21_heretic_only_nsfw(w, spy):
    f = w.frame1_tab
    f.gen_arch.setCurrentText("Qwen-Image 2.1")
    f.img_nsfw.setCurrentText("SFW")
    assert _ids(f.q21_te) == ["", "nvfp4", "fp8"], f"encoder in SFW: {_ids(f.q21_te)}"
    if f.img_nsfw.findText("NSFW") >= 0:
        f.img_nsfw.setCurrentText("NSFW")
        assert "heretic" in _ids(f.q21_te), "encoder 18+ assente in NSFW"
        f.q21_te.setCurrentIndex(f.q21_te.findData("heretic"))
        f.img_nsfw.setCurrentText("SFW")
        assert f.qwen21_te() == "" and "heretic" not in _ids(f.q21_te), "heretic ancora attivo in SFW"


def case_qwen21_controlnet_pe_params(w, spy):
    f = w.frame1_tab
    f.gen_arch.setCurrentText("Qwen-Image 2.1")
    f.q21_refs.enable.setChecked(False)
    f.q21_cn_box.setChecked(True)
    f.q21_cn_photo["edit"].setText(_img("_q21_cn.png"))
    f.q21_cn_kind.setCurrentIndex(f.q21_cn_kind.findData("lineart"))
    f.q21_cn_scale.setValue(80)
    f.gen_enhance.setChecked(True)
    f.q21_pe_mode.setCurrentIndex(f.q21_pe_mode.findData("think"))
    f.q21_te.setCurrentIndex(f.q21_te.findData("nvfp4"))
    f.gen_model.setCurrentIndex(f.gen_model.findData("qwen21-dit-int8"))
    f.gen_prompt.setPlainText("una cattedrale gotica fatta di ghiaccio, aurora boreale")
    p = _run(w, spy, w.generate_image)
    assert p["control_image"].endswith("_q21_cn.png") and p["control_preprocess"] == "lineart", p
    assert abs(p["control_scale"] - 0.8) < 1e-6 and p["qwen21_pe"] == "auto" and p["qwen21_pe_think"] is True, p
    assert p["qwen21_te"] == "nvfp4" and p["qwen21_dit"] == "qwen21-dit-int8", p
    f.q21_cn_box.setChecked(False); f.gen_enhance.setChecked(False)


def case_qwen21_i2i_to_reference(w, spy):
    f = w.frame1_tab
    f.gen_arch.setCurrentText("Qwen-Image 2.1")
    f.q21_i2i.setChecked(True)
    assert not f._q21_i2i_hint.isHidden(), "suggerimento img2img non visibile"
    f.q21_i2i_photo["edit"].setText(_img("_q21_toref.png"))
    f._q21_i2i_to_ref.click()
    refs = [r["path"] for r in f.q21_refs.get_references()]
    assert not f.q21_i2i.isChecked() and refs and refs[0].endswith("_q21_toref.png"), refs


def case_qwen21_refs_limit_and_tokens(w, spy):
    f = w.frame1_tab
    f.gen_arch.setCurrentText("Qwen-Image 2.1")
    f.q21_refs.enable.setChecked(True)
    added = sum(f.q21_refs.add_path(_img(f"_q21_ref{i}.png")) for i in range(12))
    assert added == 10, f"massimo 10 riferimenti, aggiunti {added}"
    toks = [r["token"] for r in f.q21_refs.get_references()]
    assert toks[0] == "image 1" and toks[-1] == "image 10", toks


def case_qwen21_generate_params(w, spy):
    f = w.frame1_tab
    f.gen_arch.setCurrentText("Qwen-Image 2.1")
    f.gen_model.setCurrentIndex(f.gen_model.findData("qwen21-dit-nvfp4"))
    f.q21_turbo.setCurrentIndex(f.q21_turbo.findData("6"))
    f.q21_aspect.setCurrentIndex(f.q21_aspect.findData("9:16"))
    f.q21_rgba.setChecked(True)
    f.q21_refs.enable.setChecked(True)
    f.q21_refs.add_path(_img("_q21_person.png"))
    f.gen_prompt.setPlainText("image 1 come astronauta che mangia un gelato su Marte, adesivo")
    p = _run(w, spy, w.generate_image)
    assert p["arch"] == "qwen_image21" and p["qwen21_dit"] == "qwen21-dit-nvfp4" and p["qwen21_turbo"] == "6", p
    assert p["transparent"] is True and p["aspect_2k"] == "9:16", p
    assert len(p["ref_images"]) == 1 and p["ref_images"][0].endswith("_q21_person.png"), p
    assert "init_image" not in p, "img2img non richiesto ma init_image presente"


def case_qwen21_img2img_params(w, spy):
    f = w.frame1_tab
    f.gen_arch.setCurrentText("Qwen-Image 2.1")
    f.q21_i2i.setChecked(True)
    assert not f._q21_strength_row.isHidden(), "slider forza img2img nascosto"
    f.q21_i2i_photo["edit"].setText(_img("_q21_i2i.png"))
    f.q21_strength.setValue(45)
    f.gen_prompt.setPlainText("la stessa scena come acquerello giapponese")
    p = _run(w, spy, w.generate_image)
    assert p["init_image"].endswith("_q21_i2i.png") and abs(p["strength"] - 0.45) < 1e-6, p


def case_qwen21_integrate_engine(w, spy):
    f = w.frame1_tab
    f.integrate_engine.setCurrentIndex(3)
    assert f._engine_val(3) == "qwen_image21"
    assert set(SFW_DITS) <= set(_ids(f.integrate_model)), _ids(f.integrate_model)
    assert f._edit_grounding_row.isHidden() and not f._int_pulid_group.isEnabled(), "grounding/PuLID solo altri motori"
    cfg = f.get_inpaint_config()
    assert cfg["integrate_engine"] == "qwen_image21" and cfg["integrate_qwen21_dit"] in SFW_DITS, cfg


def case_qwen21_compose_refs(w, spy):
    f = w.frame1_tab
    f.integrate_engine.setCurrentIndex(3)
    f.integrate_model.setCurrentIndex(f.integrate_model.findData("qwen21-turbo-dit-nvfp4"))
    f.compose_list.clear()
    for i in range(3):
        f.compose_list.addItem(_img(f"_q21_comp{i}.png"))
    f.compose_prompt.setPlainText("la persona di image 1 e il cane di image 2 nel salotto di image 3")
    p = _run(w, spy, w.generate_compose)
    assert p["arch"] == "qwen_image21" and len(p["ref_images"]) == 3, p
    assert p["qwen21_dit"] == "qwen21-turbo-dit-nvfp4" and p["steps"] == 8 and p["guidance"] == 1.0, p


def case_qwen21_recolor(w, spy):
    f = w.frame1_tab
    H.load_canvas_image(w)
    f.color_engine.setCurrentIndex(3)
    assert not f.color_model.isHidden() and set(SFW_DITS) <= set(_ids(f.color_model)), _ids(f.color_model)
    f.color_model.setCurrentIndex(f.color_model.findData("qwen21-dit-int8"))
    f.color_target.setText("#1e90ff"); f.color_subject.setText("the jacket")
    p = _run(w, spy, w.generate_recolor)
    assert p["qwen21_dit"] == "qwen21-dit-int8" and "Recolor the jacket" in p["prompt"], p
    assert p["ref_images"] == [f.image_path], p
    f.color_engine.setCurrentIndex(0)
    assert f.color_model.isHidden(), "combo modello Qwen visibile con Kontext"


def case_qwen21_inpaint_mask(w, spy):
    f = w.frame1_tab
    H.load_canvas_image(w)
    label = next((l for l, d in f._qwen21_inp.items() if d == "qwen21-dit-nvfp4"), None)
    assert label, f"voce Qwen assente dal combo Inpaint: {list(f._qwen21_inp)}"
    f.inp_model.setCurrentIndex(f.inp_model.findText(label))
    assert abs(f.inp_guidance.value() - 1.0) < 1e-6 and f.inp_steps.value() == 40, "Qwen: CFG 1, 40 passi"
    f.inpaint_prompt.setPlainText("a red vintage scooter")
    _mask(f.canvas)
    p = _run(w, spy, lambda: w.generate_inpaint("inpaint"))
    assert p["arch"] == "qwen_image21" and p["qwen21_dit"] == "qwen21-dit-nvfp4", p
    assert p.get("init_image") and p.get("mask_path") and 0.5 <= p["strength"] <= 1.0, p
    assert w._inpaint_mode == "inpaint", "il risultato deve tornare nel suo riquadro (bbox)"


CASES = [
    ("qwen21_panel_visibility", "Il pannello Qwen compare solo con quell'arch; in SFW 3 DiT base + 2 Turbo; 40/1.0.",
     case_qwen21_panel_visibility),
    ("qwen21_adult_dits_only_nsfw", "I DiT 18+ (Uncensored, Noct-Q) compaiono solo in NSFW, anche in Integra; "
     "Noct-Q imposta 25 passi/CFG 3.", case_qwen21_adult_dits_only_nsfw),
    ("qwen21_turbo_locks_steps", "Il Turbo ufficiale blocca 8 passi/CFG 1 e disattiva la LoRA turbo.",
     case_qwen21_turbo_locks_steps),
    ("qwen21_heretic_only_nsfw", "Encoder scelto esplicitamente (abbinato/NVFP4/FP8); «heretic» solo in NSFW.",
     case_qwen21_heretic_only_nsfw),
    ("qwen21_refs_limit_and_tokens", "Riferimenti: massimo 10, token «image 1»…«image 10».",
     case_qwen21_refs_limit_and_tokens),
    ("qwen21_generate_params", "Genera: DiT, turbo, formato 2K, RGBA e riferimenti arrivano al worker.",
     case_qwen21_generate_params),
    ("qwen21_img2img_params", "img2img: immagine iniziale e forza arrivano al worker.", case_qwen21_img2img_params),
    ("qwen21_integrate_engine", "Integra: motore Qwen-Image 2.1 con i suoi DiT, niente grounding/PuLID.",
     case_qwen21_integrate_engine),
    ("qwen21_compose_refs", "Componi: le foto della lista diventano riferimenti Qwen (Turbo = 8 passi, CFG 1).",
     case_qwen21_compose_refs),
    ("qwen21_recolor", "Cambia colore: motore Qwen col suo combo modello; istruzione di recolor sull'immagine 1.",
     case_qwen21_recolor),
    ("qwen21_controlnet_pe_params", "ControlNet (foto, preprocessore, forza), arricchitore e encoder esplicito "
     "arrivano al worker.", case_qwen21_controlnet_pe_params),
    ("qwen21_i2i_to_reference", "Il suggerimento img2img è visibile e «usa come riferimento» sposta la foto in image 1.",
     case_qwen21_i2i_to_reference),
    ("qwen21_inpaint_mask", "Inpaint: voce Qwen nel combo modelli → foto + maschera + forza al worker.",
     case_qwen21_inpaint_mask),
]


def case_qwen21_model_disappears_placeholder(w, spy):
    """Il DiT scelto sparisce dal menu (riga della matrice spenta): il combo NON ricade sul primo elemento ma resta
    su «— scegli un modello —» e generate_image si ferma senza inviare nulla."""
    from core import features
    f = w.frame1_tab
    saved = (features._MANIFEST, features._FEATURES)
    try:
        f.img_nsfw.setCurrentText("SFW")
        f.gen_arch.setCurrentText("Qwen-Image 2.1")
        i = f.gen_model.findData("qwen21-dit-fp8")
        assert i >= 0, _ids(f.gen_model)
        f.gen_model.setCurrentIndex(i)
        feats = {"cap.image_qwen21": {"enabled": True, "params": {}},
                 "model.qwen21-dit-fp8": {"enabled": False, "params": {}}}
        features._MANIFEST = {"features": feats, "allows_adult": False}
        features._FEATURES = feats
        f._reload_gen_qwen21()
        assert f.gen_model_is_placeholder(), f"atteso segnaposto, trovato {f.gen_model.currentData()}"
        assert f.gen_model.currentIndex() == 0 and "qwen21-dit-fp8" not in _ids(f.gen_model)
        f.gen_prompt.setPlainText("una medusa di cristallo sopra una citta' in tempesta")
        before = len(spy.calls)
        w.generate_image()
        assert len(spy.calls) == before, "con il segnaposto la generazione NON deve partire"
    finally:
        features._MANIFEST, features._FEATURES = saved
        f._reload_gen_qwen21()


CASES.append(("qwen21_model_disappears_placeholder",
              "DiT Qwen sparito dal menu (matrice): segnaposto «scegli un modello», generazione bloccata.",
              case_qwen21_model_disappears_placeholder))
