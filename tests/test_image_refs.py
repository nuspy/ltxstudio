"""Riferimenti nominati @image_N: numerazione stabile, scansione dei token, traduzione per modello, alias."""
from __future__ import annotations

import pytest

from core import image_refs as R


def test_names_are_stable_and_reuse_first_free():
    reg = R.NameRegistry()
    assert [reg.assign("image") for _ in range(3)] == [1, 2, 3]
    reg.release("image", 2)                      # togliere image_2 NON rinumera image_3
    assert reg.assign("image") == 2              # la nuova prende il primo numero libero
    assert reg.assign("image") == 4
    assert reg.assign("video") == 1              # numerazione per tipo
    assert R.name_of("image", 7) == "image_7" and R.token_of("image", 7) == "@image_7"
    assert R.token_of("video", 2) == "@video_2" and R.token_of("audio", 1) == "@audio_1"


def test_prefix_unico_minuscolo_non_tradotto():
    assert R.PREFIX == {"image": "image", "video": "video", "audio": "audio"}


def test_scan_finds_tokens_and_legacy():
    p = "la tartaruga in @image_1 mangia la lattuga in @image_12, <Picture 3> e <Audio 1>; @image_1x no"
    toks = R.scan(p)
    assert [(t.kind, t.n, t.legacy) for t in toks] == [("image", 1, False), ("image", 12, False), ("image", 3, True),
                                                       ("audio", 1, True)]
    assert p[toks[0].start:toks[0].end] == "@image_1"
    assert R.scan("email a@foto.it, @Fotografia_1, @images_2") == []


def test_vecchi_nomi_letti_come_alias():
    """@Foto_N, @Video_N, @Audio_N (e le maiuscole) restano leggibili e tradotti; la GUI mostra solo i nuovi."""
    toks = R.scan("@Foto_2 @Video_1 @Audio_3 @IMAGE_4 @VIDEO_5")
    assert [(t.kind, t.n) for t in toks] == [("image", 2), ("video", 1), ("audio", 3), ("image", 4), ("video", 5)]
    assert R.translate("@Foto_1 e @image_2", {"image": [1, 2]}, "qwen21") == "image 1 e image 2"
    assert R.translate("@Video_1 e @Audio_1", {"video": [1], "audio": [1]}, "h3") == "<Video 1> e <Audio 1>"


def test_unknown_and_uncited():
    have = {("image", 1), ("image", 3)}
    bad = R.unknown_tokens("@image_1 con @image_2", have)
    assert [t.n for t in bad] == [2]
    assert R.uncited("solo @image_1", have) == [("image", 3)]
    assert [t.n for t in R.unknown_tokens("@Foto_2", have)] == [2]         # anche con il vecchio nome


def test_translate_uses_position_not_number():
    # image_2 è stata tolta: passate [1, 3] → image_3 è la 2ª immagine
    out = R.translate("@image_1 abbraccia @image_3", {"image": [1, 3]}, "qwen21")
    assert out == "image 1 abbraccia image 2"
    out = R.translate("@image_3 e @image_1", {"image": [1, 3]}, "h3")
    assert out == "<Picture 2> e <Picture 1>"
    out = R.translate("<Picture 3> vecchio prompt", {"image": [1, 3]}, "ltx")      # legacy letto e tradotto
    assert out == "Image 2 vecchio prompt"


def test_translate_leaves_unknown_tokens_and_other_kinds():
    out = R.translate("@image_9 e @video_1 e @audio_1", {"image": [1], "video": [1]}, "h3")
    assert out == "@image_9 e <Video 1> e @audio_1"
    out = R.translate("@video_1", {"video": [1]}, "qwen21")       # qwen: solo immagini → resta (nessun formato)
    assert out == "@video_1"


def test_every_style_has_source_and_flag():
    for name, st in R.MODEL_SYNTAX.items():
        assert st["source"] and isinstance(st["verified"], bool) and st["fmt"].get("image"), name
    assert R.MODEL_SYNTAX["qwen21"]["max_images"] == 10 and R.MODEL_SYNTAX["krea2_edit"]["max_images"] == 2


def test_alias_extension_point(monkeypatch):
    """Tradurre il prefisso per lingua = aggiungere alias: il parser li accetta, il nome mostrato resta PREFIX."""
    monkeypatch.setitem(R.ALIASES, "image", ["image", "Foto", "Photo"])
    assert [(t.kind, t.n) for t in R.scan("@Photo_2 e @Foto_1")] == [("image", 2), ("image", 1)]
    assert R.translate("@Photo_2", {"image": [2]}, "qwen21") == "image 1"
    assert R.token_of("image", 2) == "@image_2"
