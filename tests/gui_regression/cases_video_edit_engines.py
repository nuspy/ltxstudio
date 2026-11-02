"""Fase 1 (Openjutsu, 02/11): dropdown «Motore» del tab «Modifica Video AI» (tabella core/video_edit_engines.py) e motore H3.

Si verifica: (1) il menu mostra SOLO i motori che fanno l'azione della modalita; (2) un motore senza pesi resta visibile, marcato e con
«Apri tab Pesi», e non parte; (3) corrispondenza widget -> backend: il motore scelto nel menu e quello che il worker lancia (python e
worker H3, `h3_edit.action`, profilo scelto). Subprocess FINTI: nessuna GPU, nessun SAM2.
"""
from __future__ import annotations

import json
import os
import subprocess
import types

import harness as H

from core import video_edit_engines as veng
from core import ltx_video_job as lvj


def _tab(w):
    return w.video_edit_ai_tab


def _ids(t):
    return [t.engine.itemData(i) for i in range(t.engine.count())]


def _prep(w, mode: str):
    from ui.tabs.video_edit_ai_tab import VideoEditAITab
    VideoEditAITab._dev_ack_session = True
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    path = str(H.OUT_DIR / "_vedit_eng_input.mp4")
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(b"\x00\x00\x00\x18ftypmp42")
    t.set_video(path)
    t.select_mode(mode)
    return t, path


class _Weights:
    """Rende «presenti» o «assenti» tutti i pesi della tabella (la funzione e cercata a CHIAMATA, quindi basta sostituirla)."""

    def __init__(self, present: bool):
        self.present = present

    def __enter__(self):
        self.old = veng._present
        veng._present = lambda item, _p=self.present: _p
        return self

    def __exit__(self, *a):
        veng._present = self.old


def case_engine_menu_shows_only_engines_for_the_action(w, spy):
    with _Weights(True):
        expect = {"edit": ["ltx_edit_anything"], "inpaint": ["ltx_inpaint"], "replace_char": ["ltx_union_canny", "h3_edit"],
                  "change_background": ["h3_edit"]}
        for mode, want in expect.items():
            t, _ = _prep(w, mode)
            assert _ids(t) == want, (mode, _ids(t))
            assert t.engine.currentData() == veng.default_engine(veng.action_of_mode(mode)), mode
        t, _ = _prep(w, "refocus")                               # modalita senza azione nella tabella: nessun menu motore
        assert t.engine.count() == 0 and t.engine.isHidden()


def case_engine_not_installed_stays_visible_marked_and_refused(w, spy):
    with _Weights(False):
        t, _ = _prep(w, "change_background")
        assert _ids(t) == ["h3_edit"]
        assert "non installato" in t.engine.itemText(0).lower() or "not installed" in t.engine.itemText(0).lower(), t.engine.itemText(0)
        assert not t.engine_info_row.isHidden() and not t.engine_open_weights.isHidden()      # messaggio + «Apri tab Pesi»
        t.obj.setText("person")
        t.prompt.setPlainText("a rainy neon street at night")
        opened = []
        t.request_open_weights_tab.connect(lambda: opened.append(1))
        t._ask_missing_weight = lambda sp, msg: "weights"            # il dialogo e sostituito: «Apri tab Pesi»
        t._go()
        assert not spy.calls and opened, "senza pesi il lavoro non deve partire e deve offrire il tab Pesi"


def case_engine_choice_switches_fields_and_is_remembered(w, spy):
    with _Weights(True):
        t, _ = _prep(w, "replace_char")
        i_ltx, i_h3 = t.engine.findData("ltx_union_canny"), t.engine.findData("h3_edit")
        t.engine.setCurrentIndex(i_ltx)
        assert t.h3_profile.isHidden() and t.h3_edge.isHidden()
        t.engine.setCurrentIndex(i_h3)
        assert not t.h3_profile.isHidden() and not t.h3_edge.isHidden()
        assert veng.remembered("replace_character") == "h3_edit"
        t2, _ = _prep(w, "edit")
        t3, _ = _prep(w, "replace_char")                            # ricordo l'ultima scelta dell'utente per l'azione
        assert t3.engine.currentData() == "h3_edit"
        t3.engine.setCurrentIndex(t3.engine.findData("ltx_union_canny"))   # ripristino il default per i test successivi


def _run_h3_worker(w, spy, mode: str, *, with_photo: bool):
    from ui import main_window as mw
    with _Weights(True):
        t, path = _prep(w, mode)
        t.engine.setCurrentIndex(t.engine.findData("h3_edit"))
        t.obj.setText("person")
        t.prompt.setPlainText("a rainy neon street at night, puddles reflecting signs")
        if with_photo or mode == "replace_char":
            img = str(H.OUT_DIR / "_vedit_eng_photo.png")
            H.make_test_image(img, 640, 640)
            t.ref_img.setText(img)
        t.h3_profile.setCurrentIndex(max(0, t.h3_profile.findData("int8")))
        if t.h3_profile.currentData() is None:                          # nessun profilo H3 installato su questa macchina: lo aggiungo al menu
            t.h3_profile.addItem("INT8", "int8")
            t.h3_profile.setCurrentIndex(t.h3_profile.count() - 1)
        t._go()
        assert spy.calls, "nessuna richiesta accodata"
        wk = getattr(w, "_edit_ai_worker", None)
        assert wk is not None
        cfg = dict(wk.cfg)
        done, seen = [], {}
        wk2 = mw._VideoEditAIWorker(cfg, str(H.OUT_DIR))
        wk2.done.connect(lambda r: done.append(r))
        real_run = subprocess.run

        def fake_run(cmd, *a, **k):
            cmd = [str(c) for c in cmd]
            if cmd and cmd[0].lower().endswith("python.exe") and any(c.endswith("video_segment.py") for c in cmd):   # SAM2
                os.makedirs(os.path.join(os.path.splitext(path)[0] + "_track", "masks"), exist_ok=True)
                return types.SimpleNamespace(stdout="GUIDE:g.mp4\n", stderr="", returncode=0)
            if cmd and cmd[0].lower().endswith("python.exe") and any(c.endswith("minimax_h3_infer.py") for c in cmd):
                seen["py"], seen["worker"] = cmd[0], cmd[-2]
                seen["params"] = json.load(open(cmd[-1], encoding="utf-8"))
                return types.SimpleNamespace(stdout="", stderr="", returncode=1)
            return types.SimpleNamespace(stdout="", stderr="", returncode=1)

        from core import mem_coordinator as _mc
        _o_fw, _o_fv = _mc.free_and_wait, lvj.free_video_vram
        _mc.free_and_wait = lambda *a, **k: 0
        lvj.free_video_vram = lambda *a, **k: 0
        subprocess.run = fake_run
        try:
            wk2.run()
        finally:
            subprocess.run = real_run
            _mc.free_and_wait, lvj.free_video_vram = _o_fw, _o_fv
        return cfg, seen, done


def case_engine_h3_background_reaches_h3_worker(w, spy):
    """Cambia sfondo col motore H3: il worker lanciato e quello di MiniMax H3 (venv H3), con action/profilo/maschere scelti nel tab."""
    cfg, seen, done = _run_h3_worker(w, spy, "change_background", with_photo=False)
    assert cfg["engine"] == "h3_edit" and cfg["mode"] == "change_background" and cfg["h3_profile"] == "int8", cfg
    assert seen, f"il worker H3 non e' stato lanciato: {done}"
    want_py, want_worker = lvj.runner_for({"arch": "minimax_h3"})
    assert seen["py"].lower() == want_py.lower() and seen["worker"].lower() == want_worker.lower(), seen
    p = seen["params"]
    assert p["engine"] == "h3_edit" and p["engine_kind"] == veng.KIND_H3_EDIT and p["engine_action"] == "change_background"
    e = p["h3_edit"]
    assert e["action"] == "change_background" and e["keep_text"] == "person" and e["mask_dir"].endswith(os.path.join("_track", "masks"))
    assert e["clip_s"] == 6.0 and e["overlap"] == 39 and e["picture"] == ""
    assert "edit_lora" not in p and "gguf_transformer" not in p        # nessun resto del percorso LTX


def case_engine_h3_replace_character_reaches_h3_worker_with_photo(w, spy):
    cfg, seen, done = _run_h3_worker(w, spy, "replace_char", with_photo=True)
    assert seen, f"il worker H3 non e' stato lanciato: {done}"
    e = seen["params"]["h3_edit"]
    assert e["action"] == "replace_character" and e["picture"].endswith("_vedit_eng_photo.png")
    assert seen["params"]["engine_action"] == "replace_character"
    veng.remember("replace_character", "ltx_union_canny")


def case_engine_ltx_choice_keeps_todays_backend(w, spy):
    """Con il motore LTX (default di oggi) il percorso resta quello di sempre: nessun parametro H3 nel cfg."""
    with _Weights(True):
        t, _ = _prep(w, "edit")
        t.prompt.setPlainText("make it a watercolor painting")
        t._go()
        assert spy.calls
        cfg = dict(w._edit_ai_worker.cfg)
        assert "engine" not in cfg or cfg["engine"] != "h3_edit"
        assert "h3_profile" not in cfg and cfg["mode"] == "edit" and cfg["edit_lora"]


CASES = [
    ("vengine_menu_filtered", "motore: il menu mostra SOLO i motori che fanno l'azione.", case_engine_menu_shows_only_engines_for_the_action),
    ("vengine_not_installed", "motore non installato: visibile, marcato, «Apri tab Pesi», non parte.",
     case_engine_not_installed_stays_visible_marked_and_refused),
    ("vengine_switch_remember", "motore: la scelta cambia i campi e si ricorda per azione.", case_engine_choice_switches_fields_and_is_remembered),
    ("vengine_h3_background_backend", "motore H3: Cambia sfondo arriva al worker H3 con i parametri del tab.",
     case_engine_h3_background_reaches_h3_worker),
    ("vengine_h3_replace_backend", "motore H3: Sostituisci personaggio arriva al worker H3 con la foto.",
     case_engine_h3_replace_character_reaches_h3_worker_with_photo),
    ("vengine_ltx_unchanged", "motore LTX (default): percorso di sempre, nessun parametro H3.", case_engine_ltx_choice_keeps_todays_backend),
]
