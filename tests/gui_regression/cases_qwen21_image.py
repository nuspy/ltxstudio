"""Casi GUI (mock) di Qwen-Image 2.1 nel tab Immagini: arch nel selettore, profili di peso dal catalogo,
pannello dedicato (turbo, formato 2K, RGBA, encoder 18+, riferimenti ≤10 con token «image N») e parametri
che arrivano al worker. Lo Spy intercetta _enqueue: niente GPU.
"""
from __future__ import annotations

import os

import harness as H


def _img(name: str) -> str:
    p = str(H.OUT_DIR / name)
    if not os.path.exists(p):
        H.make_test_image(p, 512, 512)
    return p


def _ready(monkey: list):
    import core.ensure_weights as ew
    saved = ew.missing
    ew.missing = lambda ids: []
    monkey.append(lambda: setattr(ew, "missing", saved))


def case_qwen21_panel_visibility(w, spy):
    f = w.frame1_tab
    f.gen_arch.setCurrentText("FLUX")
    assert f._q21_box.isHidden(), "pannello Qwen nascosto con FLUX"
    f.gen_arch.setCurrentText("Qwen-Image 2.1")
    assert not f._q21_box.isHidden(), "pannello Qwen visibile con Qwen-Image 2.1"
    ids = [f.gen_model.itemData(i) for i in range(f.gen_model.count())]
    assert ids == ["qwen21-dit-nvfp4", "qwen21-dit-fp8", "qwen21-dit-int8"], ids
    assert f.gen_steps.value() == 40 and abs(f.gen_cfg.value() - 1.0) < 1e-6, "default: 40 passi, cfg 1.0"


def case_qwen21_heretic_only_nsfw(w, spy):
    f = w.frame1_tab
    f.gen_arch.setCurrentText("Qwen-Image 2.1")
    f.img_nsfw.setCurrentText("SFW")
    assert f.q21_te.isHidden(), "encoder 18+ nascosto in SFW"
    if f.img_nsfw.findText("NSFW") >= 0:
        f.img_nsfw.setCurrentText("NSFW")
        assert not f.q21_te.isHidden(), "encoder 18+ visibile in NSFW"


def case_qwen21_refs_limit_and_tokens(w, spy):
    f = w.frame1_tab
    f.gen_arch.setCurrentText("Qwen-Image 2.1")
    f.q21_refs.enable.setChecked(True)
    added = sum(f.q21_refs.add_path(_img(f"_q21_ref{i}.png")) for i in range(12))
    assert added == 10, f"massimo 10 riferimenti, aggiunti {added}"
    toks = [r["token"] for r in f.q21_refs.get_references()]
    assert toks[0] == "image 1" and toks[-1] == "image 10", toks


def case_qwen21_generate_params(w, spy):
    monkey: list = []
    _ready(monkey)
    try:
        f = w.frame1_tab
        f.gen_arch.setCurrentText("Qwen-Image 2.1")
        f.gen_model.setCurrentIndex(f.gen_model.findData("qwen21-dit-nvfp4"))
        f.q21_turbo.setCurrentIndex(f.q21_turbo.findData("6"))
        f.q21_aspect.setCurrentIndex(f.q21_aspect.findData("9:16"))
        f.q21_rgba.setChecked(True)
        f.q21_refs.enable.setChecked(True)
        f.q21_refs.add_path(_img("_q21_person.png"))
        f.gen_prompt.setPlainText("image 1 come astronauta che mangia un gelato su Marte, adesivo")
        w.generate_image()
    finally:
        for u in monkey:
            u()
    call = next((c for c in spy.calls if c["via"] == "enqueue"), None)
    assert call, f"nessun job accodato: {spy.calls}"
    p = call["params"]
    assert p["arch"] == "qwen_image21" and p["qwen21_profile"] == "nvfp4" and p["qwen21_turbo"] == "6", p
    assert p["transparent"] is True and p["aspect_2k"] == "9:16", p
    assert len(p["ref_images"]) == 1 and p["ref_images"][0].endswith("_q21_person.png"), p


CASES = [
    ("qwen21_panel_visibility", "Il pannello Qwen compare solo con quell'arch; profili nvfp4/fp8/int8; default 40/1.0.",
     case_qwen21_panel_visibility),
    ("qwen21_heretic_only_nsfw", "L'encoder «heretic» si vede solo in modalità NSFW.", case_qwen21_heretic_only_nsfw),
    ("qwen21_refs_limit_and_tokens", "Riferimenti: massimo 10, token «image 1»…«image 10».",
     case_qwen21_refs_limit_and_tokens),
    ("qwen21_generate_params", "Genera: profilo, turbo, formato 2K, RGBA e riferimenti arrivano al worker.",
     case_qwen21_generate_params),
]
