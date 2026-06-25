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


# ───────────────────────── Storyboard multi-scena ─────────────────────────
def test_storyboard_panel(qtbot):
    from ui.widgets.storyboard_panel import StoryboardPanel
    p = StoryboardPanel(); qtbot.addWidget(p)
    assert p.get_storyboard() == []                  # disattivo di default
    p.enable.setChecked(True)
    p._rows[0].image_path = "a.png"; p._rows[0].second.setValue(0.0)
    p._rows[1].image_path = "b.png"; p._rows[1].second.setValue(5.0)
    sb = p.get_storyboard()
    assert len(sb) == 2 and sb[0]["second"] == 0.0 and sb[1]["image"] == "b.png"
    assert p.validation_errors() == []               # 2 scene → valido


def test_storyboard_chain():
    """apply_storyboard costruisce una catena di LTXVAddLatentGuide correttamente connessa."""
    from core.template_engine import TemplateEngine
    prompt = {
        "cc": {"class_type": "LTXVConcatAVLatent", "inputs": {}},
        "cfg": {"class_type": "CFGGuider", "inputs": {}},
        "el": {"class_type": "EmptyLTXVLatentVideo", "inputs": {}},
        "pc": {"class_type": "LTXVConditioning", "inputs": {}},
        "vae": {"class_type": "VAELoader", "inputs": {}},
        "cg": {"class_type": "LTXVCropGuides", "inputs": {}},
    }
    eng = TemplateEngine.__new__(TemplateEngine)
    eng.params = {}
    ok = eng.apply_storyboard(prompt, {"fps": 25, "storyboard": [
        {"image": "s1.png", "frame": 0, "strength": 0.8},
        {"image": "s2.png", "second": 5.0, "strength": 0.6},
        {"image": "s3.png", "frame": 260, "strength": 0.5},
    ]})
    assert ok
    guides = sorted(k for k, v in prompt.items() if v["class_type"] == "LTXVAddLatentGuide")
    assert len(guides) == 3
    # frame→latent_idx (//8): 0→0, 125→15, 260→32
    assert prompt[guides[1]]["inputs"]["latent_idx"] == 15
    assert prompt[guides[2]]["inputs"]["latent_idx"] == 32
    # catena: guide1 prende il latent dell'uscita di guide0
    assert prompt[guides[1]]["inputs"]["latent"] == [guides[0], 2]
    # uscita finale cablata su concat/guider/crop
    assert prompt["cc"]["inputs"]["video_latent"] == [guides[2], 2]
    assert prompt["cfg"]["inputs"]["positive"] == [guides[2], 0]
    assert prompt["cg"]["inputs"]["negative"] == [guides[2], 1]
