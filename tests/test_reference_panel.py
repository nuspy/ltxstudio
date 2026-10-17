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
    # rimuovendo la prima immagine i NOMI non cambiano (Foto_2 resta Foto_2); il token del modello è posizionale
    p._remove_row(p._rows[0])
    assert [r["name"] for r in p.get_references()] == ["Foto_2", "Video_1", "Audio_1"]
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
    assert blocker.args == ["@Foto_1"]


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
    # nel prompt si inserisce SEMPRE @Foto_N (unica convenzione); la sintassi del modello (con LTX «Image 1», con
    # H3 `<Picture 1>`) si applica alla generazione con translate()
    assert "@Foto_1" in t.prompt.toPlainText(), t.prompt.toPlainText()
    assert t.reference_panel.translate(t.prompt.toPlainText()).endswith("Image 1 ") or         "Image 1" in t.reference_panel.translate(t.prompt.toPlainText())


@pytest.mark.parametrize("arch,expect_images", [("minimax_h3", 9), ("ltx", 4)])
def test_video_tab_applica_i_limiti(qtbot, arch, expect_images):
    from core.model_catalog import video_caps
    from ui.tabs.video_tab import VideoTab
    t = VideoTab(); qtbot.addWidget(t)
    caps = dict(video_caps(arch)); caps["multiref"] = True     # forza il ramo LTX+MSR per 'ltx'
    t._apply_reference_limits(arch, caps)
    assert t.reference_panel._lim["image"] == expect_images


def test_nomi_stabili_e_primo_libero(qtbot):
    p = ReferencePanel(checkable=False); qtbot.addWidget(p)
    p.set_limits(images=10, videos=0, audios=0, total=10)
    for f in ("a.png", "b.png", "c.png"):
        assert p.add_path(f)
    assert [r["name"] for r in p.get_references()] == ["Foto_1", "Foto_2", "Foto_3"]
    p._remove_row(p._rows[1])                                   # via Foto_2: Foto_3 NON diventa Foto_2
    assert [r["name"] for r in p.get_references()] == ["Foto_1", "Foto_3"]
    assert p.add_path("d.png")                                  # la nuova prende il primo libero: Foto_2
    assert [r["name"] for r in p.get_references()] == ["Foto_1", "Foto_2", "Foto_3"]
    assert p.get_references()[1]["path"] == "d.png"


def test_traduzione_ordine_e_token_inesistente(qtbot):
    p = ReferencePanel(checkable=False); qtbot.addWidget(p)
    p.set_token_style("qwen21")
    p.set_limits(images=10, videos=0, audios=0, total=10)
    for f in ("a.png", "b.png", "c.png"):
        p.add_path(f)
    p._remove_row(p._rows[1])                                   # restano Foto_1, Foto_3
    assert p.translate("la tartaruga in @Foto_1 mangia la lattuga in @Foto_3") ==         "la tartaruga in image 1 mangia la lattuga in image 2"
    assert p.prompt_errors("@Foto_1 e @Foto_3") == []
    errs = p.prompt_errors("@Foto_2 non c'è più")
    assert errs and "@Foto_2" in errs[0]
    assert p.uncited("solo @Foto_1") == ["@Foto_3"]             # caricata ma non citata: passata comunque


def test_stato_salvato_ripristina_nomi(qtbot, tmp_path):
    from PIL import Image
    files = []
    for i in range(3):
        f = tmp_path / f"i{i}.png"; Image.new("RGB", (8, 8)).save(f); files.append(str(f))
    p = ReferencePanel(checkable=False); qtbot.addWidget(p)
    p.set_limits(images=10, videos=0, audios=0, total=10)
    for f in files:
        p.add_path(f)
    p._remove_row(p._rows[0])                                   # restano Foto_2, Foto_3
    st = p.gui_get_state()
    q = ReferencePanel(checkable=False); qtbot.addWidget(q)
    q.set_limits(images=10, videos=0, audios=0, total=10)
    q.gui_set_state(st)
    assert [r["name"] for r in q.get_references()] == ["Foto_2", "Foto_3"]


def test_drop_di_file(qtbot, tmp_path):
    from PySide6.QtCore import QMimeData, QPoint, QUrl, Qt
    from PySide6.QtGui import QDropEvent
    from PIL import Image
    f = tmp_path / "x.png"; Image.new("RGB", (8, 8)).save(f)
    p = ReferencePanel(checkable=False); qtbot.addWidget(p)
    p.set_limits(images=10, videos=0, audios=0, total=10)
    md = QMimeData(); md.setUrls([QUrl.fromLocalFile(str(f)), QUrl.fromLocalFile(str(tmp_path / "x.txt"))])
    ev = QDropEvent(QPoint(5, 5), Qt.CopyAction, md, Qt.LeftButton, Qt.NoModifier)
    p.dropEvent(ev)
    assert [r["name"] for r in p.get_references()] == ["Foto_1"], "il file non-immagine va ignorato"
