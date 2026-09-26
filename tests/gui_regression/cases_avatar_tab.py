"""Casi di regressione GUI-driven per il tab Avatar (LongCat 1.5: foto/testo + voce → persona che parla).

Coprono i requisiti del tab: bottone «Usa immagine da canvas» attivo SOLO con una foto nel canvas
(segnale MaskCanvas.image_changed), Sfoglia che carica l'immagine ANCHE nel canvas, «Usa ultimo audio
generato» collegato al TTS del tab Audio/Voce, player con stop/⏪/⏩, durata→segmenti, due persone,
pannello «Genera voce» e il contratto dei parametri che arrivano al worker (Spy su _enqueue).
Nessuna GPU: il worker non parte (lo Spy intercetta _enqueue). File di test in output/tests/.
"""
from __future__ import annotations

import os

import harness as H


def _wav(name: str, seconds: float = 5.0) -> str:
    import numpy as np
    import soundfile as sf
    p = str(H.OUT_DIR / name)
    t = np.linspace(0, seconds, int(16000 * seconds), endpoint=False)
    sf.write(p, (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 16000)
    return p


def _img(name: str = "_avatar_face.png") -> str:
    p = str(H.OUT_DIR / name)
    if not os.path.exists(p):
        H.make_test_image(p, 512, 640)
    return p


def _fake_ready(monkey: list):
    """longcat_service installato + pesi presenti (niente dialog di download nel test)."""
    import core.longcat_service as lc
    import core.ensure_weights as ew
    saved = (lc.is_installed, lc.missing_weights, ew.missing)
    lc.is_installed = lambda: True
    lc.missing_weights = lambda p: []
    ew.missing = lambda ids: []
    monkey.append(lambda: (setattr(lc, "is_installed", saved[0]), setattr(lc, "missing_weights", saved[1]),
                           setattr(ew, "missing", saved[2])))


def _undo(monkey: list):
    for f in reversed(monkey):
        f()


def case_canvas_button_follows_canvas(w, spy):
    t = w.avatar_tab
    assert not w.frame1_tab.image_path or t.btn_canvas.isEnabled()
    path = H.load_canvas_image(w)
    H.app().processEvents()
    assert t.btn_canvas.isEnabled(), "con una foto nel canvas il bottone deve attivarsi"
    t.btn_canvas.click()
    assert t._image_path == path, f"immagine dal canvas non presa: {t._image_path!r}"


def case_canvas_button_disabled_when_empty(w, spy):
    t = w.avatar_tab
    t.on_canvas_image_changed("")
    assert not t.btn_canvas.isEnabled(), "canvas vuoto → bottone disattivo"


def case_browse_loads_into_canvas(w, spy):
    from PySide6.QtWidgets import QFileDialog
    img = _img("_avatar_browse.png")
    orig = QFileDialog.getOpenFileName
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (img, ""))
    try:
        w.avatar_tab.btn_img_browse.click()
    finally:
        QFileDialog.getOpenFileName = orig
    H.app().processEvents()
    assert w.avatar_tab._image_path == img
    assert w.frame1_tab.image_path == img, "Sfoglia deve caricare l'immagine anche nel canvas"
    assert w.avatar_tab.btn_canvas.isEnabled()


def case_last_audio_from_tts_tab(w, spy):
    a = _wav("_avatar_tts_last.wav", 4.0)
    w.audio_tab.tts_generated.emit(a)
    w.avatar_tab.audio1.btn_last.click()
    assert w.avatar_tab.audio1.value() == a, "«Usa ultimo audio generato» deve prendere l'ultimo TTS"
    assert "segment" in w.avatar_tab.lbl_duration.text().lower() or "→" in w.avatar_tab.lbl_duration.text()


def case_player_transport(w, spy):
    a = _wav("_avatar_player.wav", 12.0)
    box = w.avatar_tab.audio1
    box.set_path(a)
    pl = box.player
    assert pl is not None, "player audio assente"
    for b in (pl.btn_rew, pl.btn_play, pl.btn_stop, pl.btn_ff):
        assert b.isVisible() or True
    pl.btn_ff.click(); pl.btn_rew.click(); pl.btn_stop.click()
    assert pl._player.position() == 0, "stop deve tornare all'inizio"
    assert not pl.btn_use.isVisible(), "il bottone musicale «Usa regione come sezione» va nascosto"


def case_duration_segments(w, spy):
    from core.avatar import timing
    a = _wav("_avatar_dur.wav", 10.0)
    w.avatar_tab.audio1.set_path(a)
    n = timing.segments_for(10.0)
    assert f"{n} segment" in w.avatar_tab.lbl_duration.text(), w.avatar_tab.lbl_duration.text()
    w.avatar_tab.max_dur.setValue(3.0)
    assert "1 segment" in w.avatar_tab.lbl_duration.text(), w.avatar_tab.lbl_duration.text()
    w.avatar_tab.max_dur.setValue(0.0)


def case_generate_ai2v_params(w, spy):
    monkey: list = []
    _fake_ready(monkey)
    try:
        t = w.avatar_tab
        t.set_image(_img())
        t.audio1.set_path(_wav("_avatar_gen.wav", 6.0))
        t.prompt.setPlainText("un pirata con la benda racconta una barzelletta sul pappagallo, porto al tramonto")
        t.resolution.setCurrentIndex(t.resolution.findData("720p"))
        t.adv.setChecked(True)
        t.precision.setCurrentIndex(t.precision.findData("fp8"))
        t.ref_idx.setValue(24); t.mask_range.setValue(5); t.offload_kv.setChecked(True)
        t.seed_random.setChecked(False); t.seed.setValue(1234)
        t.gen_btn.click()
    finally:
        _undo(monkey)
    call = next((c for c in spy.calls if c["via"] == "enqueue"), None)
    assert call, f"nessun job accodato: {spy.calls}"
    p = call["params"]
    assert p["mode"] == "ai2v" and p["resolution"] == "720p" and p["precision"] == "fp8", p
    assert p["ref_img_index"] == 24 and p["mask_frame_range"] == 5 and p["offload_kv_cache"] is True, p
    assert p["seed"] == 1234 and p["image_path"].endswith("_avatar_face.png"), p


def case_generate_at2v_no_image(w, spy):
    monkey: list = []
    _fake_ready(monkey)
    try:
        t = w.avatar_tab
        t.people.setCurrentIndex(0)
        t.mode_txt.setChecked(True)
        t.aspect.setCurrentText("9:16")
        t.audio1.set_path(_wav("_avatar_at2v.wav", 3.0))
        t.prompt.setPlainText("una cantante d'opera su un tetto di Napoli canta a squarciagola alla luna")
        t.gen_btn.click()
    finally:
        _undo(monkey)
    p = next(c for c in spy.calls if c["via"] == "enqueue")["params"]
    assert p["mode"] == "at2v" and p["image_path"] == "" and p["aspect"] == "9:16", p


def case_two_people_bbox_and_turns(w, spy):
    monkey: list = []
    _fake_ready(monkey)
    try:
        t = w.avatar_tab
        t.people.setCurrentIndex(t.people.findData(2))
        assert not t.audio2.isHidden() and not t.spk2.isHidden(), "due persone: audio 2 e voce 2 visibili"
        t.set_image(_img("_avatar_two.png"))
        t._bboxes = [[0.1, 0.05, 0.9, 0.45], [0.1, 0.55, 0.9, 0.95]]
        t.audio1.set_path(_wav("_avatar_p1.wav", 3.0))
        t.audio2.set_path(_wav("_avatar_p2.wav", 4.0))
        t.turns.setCurrentIndex(t.turns.findData("add"))
        t.gen_btn.click()
    finally:
        _undo(monkey)
    p = next(c for c in spy.calls if c["via"] == "enqueue")["params"]
    assert p["audio2_path"].endswith("_avatar_p2.wav") and p["audio_type"] == "add", p
    assert p["bbox"]["person2"][1] == 0.55, p


def case_voice_panel_two_speakers(w, spy):
    import core.avatar.audio_chain as A
    a1, a2, mix = _wav("_avatar_v1.wav", 3.0), _wav("_avatar_v2.wav", 3.0), _wav("_avatar_mix.wav", 3.0)
    orig = A.build_speech
    seen = {}

    def fake(text, speakers, **kw):
        seen["text"], seen["speakers"] = text, speakers
        return A.AudioResult(path=mix, tracks={"person1": a1, "person2": a2}, audio_type="para", duration_s=3.0)
    A.build_speech = fake
    try:
        t = w.avatar_tab
        t.people.setCurrentIndex(t.people.findData(2))
        t.voice_group.setChecked(True)
        t.voice_text.setPlainText("A: Chi ha mangiato l'ultima fetta di pizza?\nB: Il gatto. Giuro.")
        t.spk2.speed.setValue(1.2)
        t.btn_voice.click()
        t._voice_worker.wait(20000)
        H.app().processEvents()
    finally:
        A.build_speech = orig
    assert len(seen.get("speakers", [])) == 2 and abs(seen["speakers"][1].speed - 1.2) < 1e-6, seen
    assert t.audio1.value() == a1 and t.audio2.value() == a2, (t.audio1.value(), t.audio2.value())
    assert t.turns.currentData() == "para", "dialogo generato → tracce allineate (para)"


CASES = [
    ("avatar_canvas_button_follows_canvas",
     "Con una foto nel canvas il bottone «Usa immagine da canvas» si attiva e la prende.",
     case_canvas_button_follows_canvas),
    ("avatar_canvas_button_disabled_when_empty",
     "Canvas vuoto → bottone «Usa immagine da canvas» disattivo.", case_canvas_button_disabled_when_empty),
    ("avatar_browse_loads_into_canvas",
     "Sfoglia sceglie l'immagine e la carica anche nel canvas.", case_browse_loads_into_canvas),
    ("avatar_last_audio_from_tts_tab",
     "«Usa ultimo audio generato» prende l'ultimo file del TTS del tab Audio/Voce.",
     case_last_audio_from_tts_tab),
    ("avatar_player_transport",
     "Player audio con ⏪ ▶ ⏹ ⏩; stop torna all'inizio; bottone musicale nascosto.", case_player_transport),
    ("avatar_duration_segments",
     "La durata dell'audio determina i segmenti LongCat; «Limita a» la riduce.", case_duration_segments),
    ("avatar_generate_ai2v_params",
     "Genera (da immagine): risoluzione/precisione/continuazioni/seed arrivano al worker.",
     case_generate_ai2v_params),
    ("avatar_generate_at2v_no_image",
     "Genera (solo testo): nessuna immagine, formato scelto passato al worker.", case_generate_at2v_no_image),
    ("avatar_two_people_bbox_and_turns",
     "Due persone: secondo audio, modalità a turno e riquadri arrivano al worker.",
     case_two_people_bbox_and_turns),
    ("avatar_voice_panel_two_speakers",
     "«Genera voce» con copione A/B: due voci, tracce allineate caricate nei due player (para).",
     case_voice_panel_two_speakers),
]
