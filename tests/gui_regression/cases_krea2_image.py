"""Casi GUI (mock) di Krea 2 nel tab Immagini: varianti dal catalogo, finetune single-file 18+ (Krea Noct V4) solo in
NSFW, passi/guidance della variante (Turbo guidance-distillato: 8 passi a guidance 0, altrimenti reticolo sul volto)
e parametri che arrivano al worker. Lo Spy intercetta _enqueue: niente GPU.
"""
from __future__ import annotations

from cases_qwen21_image import _ids, _run

NOCT = "krea2-noct-v4"


def case_krea2_noct_only_nsfw(w, spy):
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("SFW")
    f.gen_arch.setCurrentText("Krea 2")
    ids = _ids(f.gen_model)
    assert "krea2_turbo" in ids and "krea2_raw" in ids, ids
    assert NOCT not in ids, f"Krea Noct (18+) visibile in SFW: {ids}"
    if f.img_nsfw.findText("NSFW") >= 0:
        f.img_nsfw.setCurrentText("NSFW")
        assert NOCT in _ids(f.gen_model), f"Krea Noct assente in NSFW: {_ids(f.gen_model)}"
        f.img_nsfw.setCurrentText("SFW")


def case_krea2_steps_guidance(w, spy):
    f = w.frame1_tab
    f.gen_arch.setCurrentText("Krea 2")
    f.gen_model.setCurrentIndex(f.gen_model.findData("krea2_raw"))
    assert f.gen_steps.value() == 52 and abs(f.gen_cfg.value() - 3.5) < 1e-6, "Raw: 52 passi, guidance 3.5"
    f.gen_model.setCurrentIndex(f.gen_model.findData("krea2_turbo"))
    assert f.gen_steps.value() == 8 and f.gen_cfg.value() == 0.0, "Turbo: 8 passi, guidance 0"
    f.gen_prompt.setPlainText("un faro sotto una tempesta di meduse luminose, foto notturna")
    p = _run(w, spy, w.generate_image)
    assert p["arch"] == "krea2" and p["krea2_variant"] == "turbo" and not p.get("krea2_dit"), p
    assert p["steps"] == 8 and p["guidance"] == 0.0, f"guidance 0 deve arrivare come 0 (non 3.5): {p}"


def case_krea2_noct_params(w, spy):
    f = w.frame1_tab
    if f.img_nsfw.findText("NSFW") < 0:
        return
    f.img_nsfw.setCurrentText("NSFW")
    f.gen_arch.setCurrentText("Krea 2")
    f.gen_model.setCurrentIndex(f.gen_model.findData(NOCT))
    assert f.gen_steps.value() == 8 and f.gen_cfg.value() == 0.0, "Noct (merge Turbo): 8 passi, guidance 0"
    f.gen_prompt.setPlainText("ritratto in controluce di una ballerina di tango in un mercato coperto")
    p = _run(w, spy, w.generate_image)
    assert p["arch"] == "krea2" and p["krea2_variant"] == "turbo" and p["krea2_dit"] == NOCT, p
    f.img_nsfw.setCurrentText("SFW")


def case_krea2_te_only_nsfw(w, spy):
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("SFW")
    f.gen_arch.setCurrentText("Krea 2")
    assert f._k2_box.isHidden(), "riquadro encoder Krea 2 visibile in SFW (unica alternativa è 18+)"
    f.gen_prompt.setPlainText("una cattedrale di ghiaccio abitata da pinguini monaci")
    p = _run(w, spy, w.generate_image)
    assert not p.get("krea2_te"), f"encoder 18+ inviato in SFW: {p}"
    if f.img_nsfw.findText("NSFW") < 0:
        return
    f.img_nsfw.setCurrentText("NSFW")
    assert not f._k2_box.isHidden(), "riquadro encoder Krea 2 assente in NSFW"
    f.k2_te.setCurrentIndex(f.k2_te.findData("krea2-te-uncensored"))
    p = _run(w, spy, w.generate_image)
    assert p.get("krea2_te") == "krea2-te-uncensored", p
    f.img_nsfw.setCurrentText("SFW")                              # torna SFW: l'encoder 18+ si azzera
    assert f.k2_te.currentData() == "" and f._k2_box.isHidden()


def case_krea2_sickrealism(w, spy):
    """Krea SickRealism v2 fp8 (Civitai, 18+): solo in NSFW, 8 passi / guidance 0, arriva come krea2_dit."""
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("SFW")
    f.gen_arch.setCurrentText("Krea 2")
    assert "krea2-sickrealism-v2" not in _ids(f.gen_model), "SickRealism (18+) visibile in SFW"
    if f.img_nsfw.findText("NSFW") < 0:
        return
    f.img_nsfw.setCurrentText("NSFW")
    f.gen_model.setCurrentIndex(f.gen_model.findData("krea2-sickrealism-v2"))
    assert f.gen_model.currentData() == "krea2-sickrealism-v2", _ids(f.gen_model)
    assert f.gen_steps.value() == 8 and f.gen_cfg.value() == 0.0
    f.gen_prompt.setPlainText("foto analogica di un astronauta che fa surf su una nebulosa, grana 35mm")
    p = _run(w, spy, w.generate_image)
    assert p["krea2_dit"] == "krea2-sickrealism-v2" and p["krea2_variant"] == "turbo", p
    f.img_nsfw.setCurrentText("SFW")


CASES = [
    ("krea2_sickrealism", "Krea SickRealism v2 fp8: solo NSFW, 8/0, krea2_dit al worker.", case_krea2_sickrealism),
    ("krea2_te_only_nsfw", "Krea 2: encoder uncensored selezionabile solo in NSFW e azzerato tornando in SFW.",
     case_krea2_te_only_nsfw),
    ("krea2_noct_only_nsfw", "Krea 2: Turbo/Raw sempre, Krea Noct V4 (18+) solo in NSFW.", case_krea2_noct_only_nsfw),
    ("krea2_steps_guidance", "Krea 2: Raw 52/3.5, Turbo 8/0 — la guidance 0 arriva al worker come 0.",
     case_krea2_steps_guidance),
    ("krea2_noct_params", "Krea Noct V4: krea2_dit + variante turbo + 8 passi/guidance 0 al worker.",
     case_krea2_noct_params),
]
