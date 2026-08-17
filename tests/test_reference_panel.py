"""Pannello «Riferimenti multipli» — offscreen, nessuna GPU.

I riferimenti NON sono keyframe: non hanno posizione temporale e si citano nel prompt col
proprio token. I limiti sono capacità del MODELLO (H3: 9 img + 3 video + 3 audio, max 12;
LTX con LoRA MSR: 2–5 immagini), quindi il pannello li applica, non li suggerisce.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

from ui.widgets.reference_panel import ReferencePanel, kind_of  # noqa: E402


def test_kind_of():
    assert kind_of("a.png") == "image" and kind_of("b.JPG") == "image"
    assert kind_of("c.mp4") == "video" and kind_of("d.mov") == "video"
    assert kind_of("e.wav") == "audio" and kind_of("f.mp3") == "audio"
    assert kind_of("g.txt") is None and kind_of("h") is None


def test_token_per_tipo(qtbot):
    p = ReferencePanel(); qtbot.addWidget(p)
    assert p.get_references() == []                      # disattivo di default
    p.enable.setChecked(True)
    for f in ("a.png", "b.png", "clip.mp4", "voce.wav"):
        assert p.add_path(f), f"{f} rifiutato"
    toks = [r["token"] for r in p.get_references()]
    assert toks == ["<Picture 1>", "<Picture 2>", "<Video 1>", "<Audio 1>"], toks
    # rimuovendo la prima immagine la numerazione si compatta (i token restano coerenti col prompt)
    p._remove_row(p._rows[0])
    assert [r["token"] for r in p.get_references()] == ["<Picture 1>", "<Video 1>", "<Audio 1>"]


def test_limiti_h3(qtbot):
    p = ReferencePanel(); qtbot.addWidget(p)
    p.enable.setChecked(True)
    p.set_limits(images=9, videos=3, audios=3, total=12, min_items=1)
    for i in range(9):
        assert p.add_path(f"img{i}.png")
    assert not p.add_path("img9.png"), "superato il tetto di 9 immagini"
    for i in range(3):
        assert p.add_path(f"v{i}.mp4")
    assert not p.add_path("v3.mp4"), "superato il tetto di 3 video"
    # 9 immagini + 3 video = 12 = tetto TOTALE: l'audio non entra più, pur avendo quota libera
    assert len(p.get_references()) == 12
    assert p._count("audio") == 0 and p._lim["audio"] == 3
    assert not p.add_path("a0.wav"), "il tetto totale di 12 deve prevalere sulla quota per tipo"


def test_limiti_ltx_msr_scarta_i_tipi_non_ammessi(qtbot):
    """Passando a LTX+MSR (solo immagini) i riferimenti video/audio già inseriti vanno rimossi."""
    p = ReferencePanel(); qtbot.addWidget(p)
    p.enable.setChecked(True)
    p.add_path("a.png"); p.add_path("clip.mp4"); p.add_path("voce.wav")
    p.set_limits(images=5, videos=0, audios=0, total=5, min_items=2)
    kinds = {r["kind"] for r in p.get_references()}
    assert kinds == {"image"}, f"tipi non ammessi rimasti: {kinds}"
    assert not p.add_path("clip2.mp4")
    assert p.validation_errors(), "1 sola immagine con min_items=2 deve dare errore"
    p.add_path("b.png")
    assert p.validation_errors() == []


def test_insert_token_signal(qtbot):
    p = ReferencePanel(); qtbot.addWidget(p)
    p.enable.setChecked(True)
    p.add_path("a.png")
    with qtbot.waitSignal(p.insert_token, timeout=1000) as blocker:
        p._rows[0].ins_btn.click()
    assert blocker.args == ["<Picture 1>"]


def test_token_finisce_nel_prompt_del_tab(qtbot):
    """Giro completo nella GUI vera: aggiungo un riferimento, premo «↳ prompt», il token
    compare nella casella del prompt al punto del cursore."""
    from ui.tabs.video_tab import VideoTab
    t = VideoTab(); qtbot.addWidget(t)
    from PySide6.QtGui import QTextCursor
    from core.model_catalog import video_caps
    # gating: LTX ha multiref (LoRA MSR) → attivo; un'arch che non ce l'ha lo spegne
    assert t.reference_panel.isEnabled(), "con LTX il pannello dovrebbe essere attivo"
    t._apply_reference_limits("lingbot", video_caps("lingbot"))
    t.reference_panel.setEnabled(video_caps("lingbot").get("multiref", False))
    assert not t.reference_panel.isEnabled(), "senza multiref il pannello va spento"
    t.reference_panel.setEnabled(True)
    t.prompt.setPlainText("un gorilla in smoking suona il theremin,")
    cur = t.prompt.textCursor(); cur.movePosition(QTextCursor.End); t.prompt.setTextCursor(cur)
    t.reference_panel.enable.setChecked(True)
    assert t.reference_panel.add_path("gorilla.png")
    t.reference_panel._rows[0].ins_btn.click()
    assert "<Picture 1>" in t.prompt.toPlainText()


@pytest.mark.parametrize("arch,expect_images", [("minimax_h3", 9), ("ltx", 5)])
def test_video_tab_applica_i_limiti(qtbot, arch, expect_images):
    from core.model_catalog import video_caps
    from ui.tabs.video_tab import VideoTab
    t = VideoTab(); qtbot.addWidget(t)
    caps = dict(video_caps(arch)); caps["multiref"] = True     # forza il ramo LTX+MSR per 'ltx'
    t._apply_reference_limits(arch, caps)
    assert t.reference_panel._lim["image"] == expect_images
