"""Casi GUI (mock) di «Immagini di riferimento» in Crea immagini (task 6, 17/10): sezione solo coi modelli
multi-riferimento, nomi @image_N stabili, completamento con «@», avviso su token inesistente, traduzione nella sintassi
del modello con le immagini nell'ordine giusto, «dal canvas», stato salvato. Lo Spy intercetta _enqueue: niente GPU."""
from __future__ import annotations

import os

import harness as H

from cases_qwen21_image import _img, _run


def _arch(f, label):
    f.img_nsfw.setCurrentText("SFW")
    f.gen_arch.setCurrentText(label)


def _names(f):
    return [r["name"] for r in f.gen_refs.get_references()]


def case_genrefs_section_only_for_multiref_models(w, spy):
    f = w.frame1_tab
    for lbl in ("FLUX", "Z-Image", "Ernie"):
        _arch(f, lbl)
        assert not f._gen_refs_box_w.isVisibleTo(f), f"sezione riferimenti visibile con {lbl} (non li accetta da «Genera»)"
    _arch(f, "Krea 2")                                          # task 9: Krea 2 con riferimenti = Krea 2 Edit (≤2)
    assert f._gen_refs_box_w.isVisibleTo(f), "sezione riferimenti nascosta con Krea 2 (Edit, max 2)"
    f.gen_refs.clear()
    assert sum(f.gen_refs.add_path(_img(f"_gr_k{i}.png")) for i in range(4)) == 2, "limite Krea 2 Edit = 2"
    f.gen_refs.clear()
    _arch(f, "Qwen-Image 2.1")
    assert f._gen_refs_box_w.isVisibleTo(f), "sezione riferimenti nascosta con Qwen-Image 2.1"
    f.gen_refs.clear()
    added = sum(f.gen_refs.add_path(_img(f"_gr_lim{i}.png")) for i in range(12))
    assert added == 10, f"limite Qwen = 10, aggiunti {added}"
    assert _names(f)[0] == "image_1" and _names(f)[-1] == "image_10", _names(f)
    f.gen_refs.clear()


def case_genrefs_krea2_edit_from_create(w, spy):
    """Krea 2 + riferimenti → job Krea 2 Edit con le immagini nell'ordine, token tradotti, e il DiT/TE single-file
    scelti in «Crea» (task 9: prima l'Edit arrivava solo dai pannelli di modifica e non li riceveva)."""
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    f.gen_arch.setCurrentText("Krea 2")
    i = f.gen_model.findData("krea2-noct-v4")
    if i < 0:
        return                                                  # edizione senza voci 18+
    f.gen_model.setCurrentIndex(i)
    f.gen_refs.clear()
    a, b = _img("_gr_k0.png"), _img("_gr_k1.png")
    f.gen_refs.add_path(a); f.gen_refs.add_path(b)
    f.gen_prompt.setPlainText("@Foto_1 con il cappello di @Foto_2")
    p = _run(w, spy, w.generate_image)
    assert p["arch"] == "krea2_edit" and p["images"] == [a, b], p
    assert p["prompt"] == "image 1 con il cappello di image 2", p["prompt"]
    assert p["krea2_dit"] == "krea2-noct-v4", p
    f.gen_refs.clear()
    f.gen_refs.add_path(a)                                      # una sola foto → single
    f.gen_prompt.setPlainText("@Foto_1 in smoking")
    p = _run(w, spy, w.generate_image)
    assert p["arch"] == "krea2_edit" and p.get("image_path") == a and "images" not in p, p
    f.gen_refs.clear()
    f.gen_prompt.setPlainText("un'aragosta in smoking")         # senza riferimenti resta testo→immagine
    p = _run(w, spy, w.generate_image)
    assert p["arch"] == "krea2", p


def case_genrefs_names_stable(w, spy):
    f = w.frame1_tab
    _arch(f, "Qwen-Image 2.1")
    f.gen_refs.clear()
    for i in range(3):
        f.gen_refs.add_path(_img(f"_gr_n{i}.png"))
    assert _names(f) == ["image_1", "image_2", "image_3"], _names(f)
    f.gen_refs._remove_row(f.gen_refs._rows[1])                 # via image_2
    assert _names(f) == ["image_1", "image_3"], f"rinumerazione: {_names(f)}"
    f.gen_refs.add_path(_img("_gr_n9.png"))
    assert _names(f) == ["image_1", "image_2", "image_3"], f"la nuova deve prendere il primo numero libero: {_names(f)}"
    assert f.gen_refs.get_references()[1]["path"].endswith("_gr_n9.png")
    f.gen_refs.clear()


def case_genrefs_prompt_completion(w, spy):
    from PySide6.QtTest import QTest
    f = w.frame1_tab
    _arch(f, "Qwen-Image 2.1")
    f.gen_refs.clear()
    for i in range(2):
        f.gen_refs.add_path(_img(f"_gr_c{i}.png"))
    f.gen_prompt.clear()
    f.gen_prompt.setFocus()
    QTest.keyClicks(f.gen_prompt, "la tartaruga in @")
    h = f._gen_prompt_refs
    assert h.tokens() == ["@image_1", "@image_2"], h.tokens()
    assert not h.model.item(0).icon().isNull() and "_gr_c0.png" in h.model.item(0).text(), "miniatura e nome nell'elenco"
    assert h.completer.completionCount() == 2
    h._insert("@image_1")
    assert f.gen_prompt.toPlainText().endswith("@image_1 "), f.gen_prompt.toPlainText()
    f.gen_refs.clear()


def case_genrefs_unknown_token_warns_before_generating(w, spy):
    from PySide6.QtWidgets import QMessageBox
    f = w.frame1_tab
    _arch(f, "Qwen-Image 2.1")
    f.gen_model.setCurrentIndex(f.gen_model.findData("qwen21-dit-nvfp4"))
    f.gen_refs.clear()
    f.gen_refs.add_path(_img("_gr_u0.png"))
    f.gen_prompt.setPlainText("@image_1 abbraccia @image_7")
    shown = []
    old = QMessageBox.warning
    QMessageBox.warning = staticmethod(lambda *a, **k: shown.append(a[2] if len(a) > 2 else "") or QMessageBox.Ok)
    n0 = len(spy.calls)
    try:
        w.generate_image()
    finally:
        QMessageBox.warning = old
    assert shown and "@image_7" in str(shown[0]), f"nessun avviso sul token inesistente: {shown}"
    assert len(spy.calls) == n0, f"generazione partita nonostante @image_7 inesistente: {spy.calls[n0:]}"
    f.gen_refs.clear()


def case_genrefs_cfg_order_and_translation(w, spy):
    f = w.frame1_tab
    _arch(f, "Qwen-Image 2.1")
    f.gen_model.setCurrentIndex(f.gen_model.findData("qwen21-dit-nvfp4"))
    f.gen_refs.clear()
    paths = [_img(f"_gr_o{i}.png") for i in range(3)]
    for p in paths:
        f.gen_refs.add_path(p)
    f.gen_refs._remove_row(f.gen_refs._rows[1])                  # restano image_1 e image_3
    f.gen_prompt.setPlainText("la tartaruga in @image_3 mangia la lattuga in @image_1")
    p = _run(w, spy, w.generate_image)
    assert [os.path.basename(x) for x in p["ref_images"]] == ["_gr_o0.png", "_gr_o2.png"], p["ref_images"]
    assert p["prompt"] == "la tartaruga in image 2 mangia la lattuga in image 1", p["prompt"]
    # una foto caricata ma NON citata viene passata comunque
    f.gen_prompt.setPlainText("solo @image_1")
    p = _run(w, spy, w.generate_image)
    assert len(p["ref_images"]) == 2 and p["prompt"] == "solo image 1", p
    f.gen_refs.clear()


def case_genrefs_from_canvas_and_state(w, spy):
    from core import gui_state
    f = w.frame1_tab
    _arch(f, "Qwen-Image 2.1")
    f.gen_refs.clear()
    path = H.load_canvas_image(w)
    f.gen_refs.canvas_btn.click()
    assert [r["path"] for r in f.gen_refs.get_references()] == [path], f.gen_refs.get_references()
    f.gen_refs.add_path(_img("_gr_s1.png"))
    f.gen_refs._remove_row(f.gen_refs._rows[0])                  # resta solo image_2
    st = gui_state.collect(w)
    key = next((k for k in st if k.endswith("gen_refs")), None)
    assert key and st[key]["t"] == "custom", f"stato dei riferimenti non salvato da gui_state: {[k for k in st if 'ref' in k]}"
    f.gen_refs.clear()
    gui_state.restore(w, st)
    assert _names(f) == ["image_2"], f"stato ripristinato: {_names(f)}"
    f.gen_refs.clear()


CASES = [
    ("genrefs_section_only_for_multiref_models", "«Immagini di riferimento» visibile solo con Qwen-Image 2.1; massimo 10.",
     case_genrefs_section_only_for_multiref_models),
    ("genrefs_krea2_edit_from_create", "Krea 2 + riferimenti → Krea 2 Edit (≤2) con DiT/TE single-file; senza → t2i.",
     case_genrefs_krea2_edit_from_create),
    ("genrefs_names_stable", "Nomi @image_N assegnati al caricamento; togliere image_2 non rinumera; la nuova prende il "
     "primo libero.", case_genrefs_names_stable),
    ("genrefs_prompt_completion", "Digitando «@» nel prompt compare l'elenco dei nomi e l'inserimento sostituisce.",
     case_genrefs_prompt_completion),
    ("genrefs_unknown_token_warns", "Un @image_N inesistente nel prompt avvisa e NON genera.",
     case_genrefs_unknown_token_warns_before_generating),
    ("genrefs_cfg_order_and_translation", "Alla generazione: ref_images nell'ordine, token tradotti in «image N», "
     "foto non citata passata comunque.", case_genrefs_cfg_order_and_translation),
    ("genrefs_from_canvas_and_state", "«Dal canvas» aggiunge la foto; lo stato (nomi compresi) è salvato da gui_state.",
     case_genrefs_from_canvas_and_state),
]
