"""Riferimenti nominati @Foto_N: numerazione stabile, scansione dei token, traduzione per modello, alias."""
from __future__ import annotations

import pytest

from core import image_refs as R


def test_names_are_stable_and_reuse_first_free():
    reg = R.NameRegistry()
    assert [reg.assign("image") for _ in range(3)] == [1, 2, 3]
    reg.release("image", 2)                      # togliere Foto_2 NON rinumera Foto_3
    assert reg.assign("image") == 2              # la nuova prende il primo numero libero
    assert reg.assign("image") == 4
    assert reg.assign("video") == 1              # numerazione per tipo
    assert R.name_of("image", 7) == "Foto_7" and R.token_of("image", 7) == "@Foto_7"


def test_scan_finds_tokens_and_legacy():
    p = "la tartaruga in @Foto_1 mangia la lattuga in @Foto_12, <Picture 3> e <Audio 1>; @Foto_1x no"
    toks = R.scan(p)
    assert [(t.kind, t.n, t.legacy) for t in toks] == [("image", 1, False), ("image", 12, False), ("image", 3, True),
                                                       ("audio", 1, True)]
    assert p[toks[0].start:toks[0].end] == "@Foto_1"
    assert R.scan("email a@foto.it, @Fotografia_1") == []


def test_unknown_and_uncited():
    have = {("image", 1), ("image", 3)}
    bad = R.unknown_tokens("@Foto_1 con @Foto_2", have)
    assert [t.n for t in bad] == [2]
    assert R.uncited("solo @Foto_1", have) == [("image", 3)]


def test_translate_uses_position_not_number():
    # Foto_2 è stata tolta: passate [1, 3] → Foto_3 è la 2ª immagine
    out = R.translate("@Foto_1 abbraccia @Foto_3", {"image": [1, 3]}, "qwen21")
    assert out == "image 1 abbraccia image 2"
    out = R.translate("@Foto_3 e @Foto_1", {"image": [1, 3]}, "h3")
    assert out == "<Picture 2> e <Picture 1>"
    out = R.translate("<Picture 3> vecchio prompt", {"image": [1, 3]}, "ltx")      # legacy letto e tradotto
    assert out == "Image 2 vecchio prompt"


def test_translate_leaves_unknown_tokens_and_other_kinds():
    out = R.translate("@Foto_9 e @Video_1 e @Audio_1", {"image": [1], "video": [1]}, "h3")
    assert out == "@Foto_9 e <Video 1> e @Audio_1"
    out = R.translate("@Video_1", {"video": [1]}, "qwen21")       # qwen: solo immagini → resta (nessun formato)
    assert out == "@Video_1"


def test_every_style_has_source_and_flag():
    for name, st in R.MODEL_SYNTAX.items():
        assert st["source"] and isinstance(st["verified"], bool) and st["fmt"].get("image"), name
    assert R.MODEL_SYNTAX["qwen21"]["max_images"] == 10 and R.MODEL_SYNTAX["krea2_edit"]["max_images"] == 2


def test_alias_extension_point(monkeypatch):
    """Tradurre il prefisso per lingua = aggiungere alias: il parser li accetta, il nome mostrato resta PREFIX."""
    monkeypatch.setitem(R.ALIASES, "image", ["Foto", "Photo"])
    assert [(t.kind, t.n) for t in R.scan("@Photo_2 e @Foto_1")] == [("image", 2), ("image", 1)]
    assert R.translate("@Photo_2", {"image": [2]}, "qwen21") == "image 1"
    assert R.token_of("image", 2) == "@Foto_2"
