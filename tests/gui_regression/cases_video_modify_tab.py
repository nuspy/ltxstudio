"""Casi GUI (mock) del sotto-tab «Modifica» nel tab Video (task 6, 17/10).

Decisione utente: Talk/Canta/ReTake/Ascolta/V2V stanno nel sotto-tab «Modifica», accanto a «Genera»; architettura,
checkpoint, LoRA e impostazioni comuni restano SOPRA la barra (condivisi). Talk e Canta devono funzionare IDENTICI:
`golden/video_modify_cfg.json` è stato catturato sul layout PRECEDENTE (commit 886032b, pannello V2V in colonna) con lo
stesso scenario; qui si confronta la cfg emessa (get_v2v_config + params dei job).

Rigenerare il golden (solo su un albero col layout vecchio): python tests/gui_regression/cases_video_modify_tab.py --dump F
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GOLDEN = HERE / "golden" / "video_modify_cfg.json"

if __name__ == "__main__":                                  # esecuzione diretta per catturare il golden
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))

import harness as H  # noqa: E402


def _dummy(name: str, content: bytes = b"\x00\x00\x00\x00") -> str:
    p = H.OUT_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(content)
    return str(p)


def _norm(o):
    """Percorsi dell'albero → <ROOT> (il golden non dipende da dove gira il test); via i campi volatili."""
    root = str(ROOT)
    if isinstance(o, dict):
        return {k: _norm(v) for k, v in o.items() if k not in ("output", "parent_pid")}
    if isinstance(o, (list, tuple)):
        return [_norm(x) for x in o]
    if isinstance(o, str):
        return o.replace(root, "<ROOT>").replace(root.replace("\\", "/"), "<ROOT>")
    return o


def _ready():
    import core.ensure_weights as ew
    from core import native_driver
    saved = (ew.missing, native_driver.is_up)
    ew.missing = lambda ids: []
    native_driver.is_up = lambda: True
    return saved


def _unready(saved):
    import core.ensure_weights as ew
    from core import native_driver
    ew.missing, native_driver.is_up = saved


def scenario(w, spy) -> dict:
    """Talk (foto + audio utente) e Canta (foto + canzone): cfg del pannello e params del job."""
    vt = w.video_tab
    out: dict = {}
    H.load_canvas_image(w)
    vt.prompt.setPlainText("una tartaruga cantante in smoking, cinematografico")
    vt.sampler_seed.fixed_seed.setChecked(True)
    vt.sampler_seed.seed.setValue(1234)
    panel = vt.v2v_panel
    saved = _ready()
    try:
        # ---- Canta ----
        panel.rb_sing.setChecked(True)
        panel._sing_audio_path = _dummy("_vmod_song.wav", b"RIFF0000WAVEfmt ")
        out["sing_cfg"] = panel.get_v2v_config()
        n0 = len(spy.calls)
        w.on_generate()
        c = [c for c in spy.calls[n0:] if c["via"] in ("enqueue", "native_driver")]
        out["sing_job"] = c[-1]["params"] if c else None
        # ---- Talk (foto + audio utente dal tab Audio) ----
        panel.rb_talk.setChecked(True)
        audio = _dummy("_vmod_voice.wav", b"RIFF0000WAVEfmt ")
        w.audio_tab._audio_file = audio
        w.audio_tab.file_lbl.setText(Path(audio).name)
        w.audio_tab.rb_user.setChecked(True)
        w.audio_tab._on_mode()
        out["talk_cfg"] = panel.get_v2v_config()
        n0 = len(spy.calls)
        w.on_generate()
        c = [c for c in spy.calls[n0:] if c["via"] in ("enqueue", "native_driver")]
        out["talk_job"] = c[-1]["params"] if c else None
        # ---- ReTake e Ascolta: solo cfg ----
        panel.rb_retake.setChecked(True)
        out["retake_cfg"] = panel.get_v2v_config()
        panel.rb_listen.setChecked(True)
        out["listen_cfg"] = panel.get_listen_config()
    finally:
        _unready(saved)
        panel.rb_normal.setChecked(True)
    return _norm(out)


def case_vmod_layout(w, spy):
    """Sotto-tab Genera|Modifica: V2V in «Modifica»; storyboard/riferimenti/post in «Genera»; il resto SOPRA la barra."""
    vt = w.video_tab
    tabs = vt.mode_tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == ["Genera", "Modifica"] or tabs.count() == 2
    gen, edit = tabs.widget(0), tabs.widget(1)
    assert edit.isAncestorOf(vt.v2v_panel), "pannello V2V (Talk/Canta/ReTake/Ascolta) non nel sotto-tab Modifica"
    for name in ("storyboard_panel", "reference_panel", "fr_auto", "rife_auto", "img_strength", "i2v_method",
                 "pulid_kf_enable"):
        assert gen.isAncestorOf(getattr(vt, name)), f"{name} non nel sotto-tab Genera"
    for name in ("video_arch", "model", "nsfw", "width", "fps", "length", "prompt", "lora", "sampler_seed",
                 "precision", "attn_backend", "decode_lowmem"):
        wdg = getattr(vt, name)
        assert not gen.isAncestorOf(wdg) and not edit.isAncestorOf(wdg), f"{name} deve stare SOPRA la barra (condiviso)"
        assert vt.isAncestorOf(wdg)


def case_vmod_mode_sync(w, spy):
    """Scegliere una modalità di modifica porta in «Modifica»; tornare in «Genera» rimette Normale."""
    vt = w.video_tab
    p = vt.v2v_panel
    vt.mode_tabs.setCurrentIndex(0)
    p.rb_talk.setChecked(True)
    assert vt.is_edit_tab(), "Talk scelto ma il sotto-tab resta «Genera»"
    vt.mode_tabs.setCurrentIndex(0)
    assert p.current_mode() == "normal", f"in «Genera» la modalità deve essere Normale: {p.current_mode()}"
    vt.show_edit_tab()
    assert vt.is_edit_tab() and p.current_mode() == "normal"
    p.rb_sing.setChecked(True)
    assert vt.is_edit_tab() and p.current_mode() == "sing"
    p.rb_normal.setChecked(True)


def case_vmod_open_from_vedit(w, spy):
    """Il rimando «Apri Talk/Canta» di Modifica Video AI apre il tab Video sul sotto-tab «Modifica» con la modalità."""
    vt = w.video_tab
    vt.mode_tabs.setCurrentIndex(0)
    song = _dummy("_vmod_song2.wav", b"RIFF0000WAVEfmt ")
    w._open_v2v_from_vedit("sing", "", song)
    assert w.tabs.currentWidget() is vt and vt.is_edit_tab(), "tab Video/«Modifica» non aperto"
    assert vt.v2v_panel.current_mode() == "sing" and vt.v2v_panel._sing_audio_path == song
    vt.mode_tabs.setCurrentIndex(0)
    w._open_v2v_from_vedit("talk", "", "")
    assert vt.is_edit_tab() and vt.v2v_panel.current_mode() == "talk"
    vt.v2v_panel.rb_normal.setChecked(True)


def case_vmod_talk_sing_cfg_identical(w, spy):
    """Talk e Canta emettono la STESSA cfg di prima (golden catturato col layout precedente)."""
    got = scenario(w, spy)
    assert GOLDEN.exists(), f"golden mancante: {GOLDEN}"
    want = json.loads(GOLDEN.read_text("utf-8"))
    for k in sorted(set(got) | set(want)):
        assert got.get(k) == want.get(k), f"{k} diverso dal layout precedente:\n  prima : {want.get(k)}\n  adesso: {got.get(k)}"


CASES = [
    ("vmod_layout", "Sotto-tab Genera|Modifica: V2V in Modifica; storyboard/riferimenti/post in Genera; arch/modello/LoRA "
     "sopra la barra.", case_vmod_layout),
    ("vmod_mode_sync", "Talk/Canta scelti → sotto-tab Modifica; tornare in Genera → modalità Normale.", case_vmod_mode_sync),
    ("vmod_open_from_vedit", "«Apri Talk/Canta» di Modifica Video AI apre il sotto-tab Modifica con la modalità.",
     case_vmod_open_from_vedit),
    ("vmod_talk_sing_cfg_identical", "Talk e Canta emettono la stessa cfg del layout precedente (golden).",
     case_vmod_talk_sing_cfg_identical),
]


if __name__ == "__main__":
    dest = sys.argv[sys.argv.index("--dump") + 1]
    win = H.build_window()
    with H.Spy(win) as sp:
        data = scenario(win, sp)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text(json.dumps(data, indent=1, ensure_ascii=False, sort_keys=True, default=str), "utf-8")
    print("dump ok", dest, {k: (v is not None) for k, v in data.items()})
    os._exit(0)
