"""Casi GUI (mock) della «Correzione intonazione» (DiffPitcher) del tab Musica: modo template (voce-guida) e modo
spartito MIDI. Niente subprocess: si cattura la chiamata a diffpitcher_service.make_job.
"""
from __future__ import annotations

import harness as H
from cases_music_tab import _FakeJob, _Patch, _music_tab


def _files():
    src = H.OUT_DIR / "_pitch_src.wav"
    ref = H.OUT_DIR / "_pitch_ref.wav"
    mid = H.OUT_DIR / "_pitch_score.mid"
    for p in (src, ref, mid):
        if not p.exists():
            p.write_bytes(b"x")
    return str(src), str(ref), str(mid)


def _patch(spy, pt):
    from core import diffpitcher_service as dp
    pt.set(dp, "is_installed", lambda mode="template": True)

    def fake(source, pitch_ref="", out_name="vocal", **k):
        spy.calls.append({"via": "music", "label": "dp.make_job", "module": "diffpitcher",
                          "params": {"source": source, "pitch_ref": pitch_ref, "name": out_name, **{
                              x: k.get(x) for x in ("steps", "shift_semi", "mode", "midi")}}})
        return _FakeJob()
    pt.set(dp, "make_job", fake)


def case_pitch_panel_modes_visibility(w, spy):
    t = _music_tab(w)
    p = t.pitch_panel
    assert p.mode.currentData() == "template", "il default deve restare il modo template"
    assert not p.ref.isHidden() or p.isHidden() is False
    p.mode.setCurrentIndex(p.mode.findData("score"))
    assert p.ref.isHidden() and not p.midi.isHidden(), "modo spartito: deve comparire il campo MIDI e sparire la voce-guida"
    p.mode.setCurrentIndex(p.mode.findData("template"))
    assert not p.ref.isHidden() and p.midi.isHidden()


def case_pitch_template_params(w, spy):
    t = _music_tab(w)
    pt = _Patch()
    try:
        _patch(spy, pt)
        src, ref, _mid = _files()
        p = t.pitch_panel
        p.mode.setCurrentIndex(p.mode.findData("template"))
        p.src.setPath(src); p.ref.setPath(ref); p.steps.setValue(30); p.shift.setValue(2)
        t._process_pitch()
    finally:
        pt.undo()
    c = next(c for c in reversed(spy.calls) if c["label"] == "dp.make_job")["params"]
    assert c["mode"] == "template" and c["pitch_ref"] == ref and c["steps"] == 30 and c["shift_semi"] == 2, c


def case_pitch_score_params_and_guard(w, spy):
    t = _music_tab(w)
    pt = _Patch()
    try:
        _patch(spy, pt)
        src, ref, mid = _files()
        p = t.pitch_panel
        p.mode.setCurrentIndex(p.mode.findData("score"))
        p.src.setPath(src); p.ref.setPath(ref); p.midi.setPath(""); p.steps.setValue(40); p.shift.setValue(-3)
        n = len([c for c in spy.calls if c["label"] == "dp.make_job"])
        t._process_pitch()                                       # MIDI mancante: nessun job (avviso)
        assert len([c for c in spy.calls if c["label"] == "dp.make_job"]) == n, "senza MIDI il job non deve partire"
        p.midi.setPath(ref)                                       # un .wav al posto del MIDI: rifiutato
        t._process_pitch()
        assert len([c for c in spy.calls if c["label"] == "dp.make_job"]) == n, "un file non-MIDI non deve passare"
        p.midi.setPath(mid)
        t._process_pitch()
    finally:
        pt.undo()
    c = next(c for c in reversed(spy.calls) if c["label"] == "dp.make_job")["params"]
    assert c["mode"] == "score" and c["midi"] == mid and c["steps"] == 40 and c["shift_semi"] == -3, c


CASES = [
    ("pitch_panel_modes_visibility", "Intonazione: modo template di default; lo spartito mostra il campo MIDI.",
     case_pitch_panel_modes_visibility),
    ("pitch_template_params", "Intonazione template: voce-guida, passi e trasposizione arrivano al servizio.",
     case_pitch_template_params),
    ("pitch_score_params_and_guard", "Intonazione spartito: serve un .mid/.midi; parametri al servizio.",
     case_pitch_score_params_and_guard),
]
