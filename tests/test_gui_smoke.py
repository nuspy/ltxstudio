"""Smoke test della GUI (pytest-qt, offscreen, NESSUNA GPU): costruzione widget + config.

Verifica che i pannelli si costruiscano e che i config delle feature nuove (modalità Canta/Talk,
face-swap, generatore liriche) siano corretti — senza eseguire generazioni. Veloce, ripetibile.
"""
import pytest


# ───────────────────────── V2V panel: modalità lip-sync ─────────────────────────
def test_v2v_modes(qtbot):
    from ui.widgets.v2v_panel import V2VPanel
    p = V2VPanel(); qtbot.addWidget(p)
    for rb, mode in [(p.rb_normal, "normal"), (p.rb_talk, "talk"),
                     (p.rb_sing, "sing"), (p.rb_retake, "retake")]:
        rb.setChecked(True)
        assert p.current_mode() == mode


def test_v2v_sing_config(qtbot):
    from ui.widgets.v2v_panel import V2VPanel
    p = V2VPanel(); qtbot.addWidget(p)
    p.rb_sing.setChecked(True)
    cfg = p.get_v2v_config()
    assert cfg["v2v_mode"] == "sing"
    assert "sing_separate_vocals" in cfg and "sing_audio_scale" in cfg
    # senza canzone selezionata → la validazione segnala l'audio mancante
    assert p.validation_errors()


def test_v2v_sing_with_audio(qtbot):
    from ui.widgets.v2v_panel import V2VPanel
    p = V2VPanel(); qtbot.addWidget(p)
    p.rb_sing.setChecked(True)
    p._sing_audio_path = "X.mp3"
    assert p.validation_errors() == []
    assert p.get_v2v_config()["sing_audio_path"] == "X.mp3"


# ───────────────────────── Music tab: generatore liriche LLM ─────────────────────────
def test_music_lyrics_button(qtbot):
    from ui.tabs.music_tab import MusicTab, _LYRIC_LANGS
    m = MusicTab(); qtbot.addWidget(m)
    assert hasattr(m, "btn_lyr_ai") and hasattr(m, "lyr_prompt")
    assert m.lyr_lang.count() == len(_LYRIC_LANGS) == 15


def test_lyrics_prompt_enrichment():
    from core import lyrics_llm
    pr = lyrics_llm.build_prompt("amore d'estate", "it", "pop, allegro")
    assert "Italian" in pr and "amore d'estate" in pr and "pop" in pr
    assert "[Verse]" in lyrics_llm._SYSTEM and "[Chorus]" in lyrics_llm._SYSTEM


# ───────────────────────── Video tab: face-swap post-process ─────────────────────────
def test_video_faceswap_config(qtbot):
    from ui.tabs.video_tab import VideoTab
    t = VideoTab(); qtbot.addWidget(t)
    assert t.get_face_swap_auto() is None            # disattivo di default
    t.fs_auto.setChecked(True)
    t.fs_face.setText("face.png")
    assert t.get_face_swap_auto() == {"face": "face.png"}


def test_faceswap_module_api():
    from core import faceswap
    assert isinstance(faceswap.available(), bool)
    assert hasattr(faceswap, "swap_image") and hasattr(faceswap, "swap_video")
