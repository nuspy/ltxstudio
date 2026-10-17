"""Casi GUI della Fase 1a IC-LoRA del tab «Modifica Video AI» (famiglie/modalità, campi, checkpoint, tetti, gate,
esclusione IC-LoRA dal pannello LoRA). Complemento di cases_video_edit_ai_tab.py (stesso stile e helper).

Nessuna GPU: i worker NON vengono eseguiti (lo Spy intercetta _enqueue; la config reale è in w._edit_ai_worker.cfg).
I dialog Qt sono già non-bloccanti (harness.patch_dialogs); il dialog sul checkpoint è comunque sovrascritto
dai casi (`t._ask_checkpoint`).
"""
from __future__ import annotations

import os

import harness as H
from cases_video_edit_ai_tab import _ack_dev, _cfg_of, _select_mode, _set_video, _tab

_DIST = {"model_kind": "diffusion", "model":
         r"E:\AI\Models\diffusion_models\LTXVideo\v2\ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"}
_DEV = {"model_kind": "diffusion", "model": r"E:\AI\Models\diffusion_models\LTX\ltx-2.3-22b-dev-fp8.safetensors"}
_AUTO = {"model_kind": "auto", "model": "auto"}
_REL25 = {"model_kind": "diffusion", "model": r"E:\AI\Models\diffusion_models\ltx-2.5-distilled.safetensors"}


def _getter(w, vals):
    _tab(w)._model_getter = (lambda: vals)
    _tab(w).refresh_checkpoint_hint()


def _mk_image(name: str = "_vedit_style.png") -> str:
    path = str(H.OUT_DIR / name)
    H.make_test_image(path, 128, 96)
    return path


def _dummy_video(name: str) -> str:
    path = str(H.OUT_DIR / name)
    with open(path, "wb") as fh:
        fh.write(b"\x00\x00\x00\x18ftypmp42")
    return path


def _fill_mode(w, key: str):
    """Imposta i campi minimi validi per `key` e ritorna la tab."""
    from ui.tabs.video_edit_ai_tab import VideoEditAITab
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    _keep = VideoEditAITab._dev_ack_session
    _set_video(w)                                        # (gli helper storici impostano l'ack: qui si ripristina)
    VideoEditAITab._dev_ack_session = _keep
    _select_mode(w, key)
    if key == "restyle":
        t.ref_img.setText(_mk_image())
        t.style_text.setText("Disney 2D Animation")
        t.prompt.setPlainText("")
    elif key == "cameraman":
        t.cam_ref.setText(_dummy_video("_vedit_camref.mp4"))
    return t


_NEW_MODES = {          # key -> (kind atteso, cap atteso)
    "restyle": ("restyle", "cap.video_restyle"),
    "refocus": ("refocus", "cap.video_restore_iclora"),
    "uncompress": ("uncompress", "cap.video_restore_iclora"),
    "deinterlace": ("deinterlace", "cap.video_restore_iclora"),
    "motiondeblur": ("motiondeblur", "cap.video_restore_iclora"),
    "detailer": ("detailer", "cap.video_restore_iclora"),
    "outpaint": ("outpaint", "cap.video_outpaint"),
    "cameraman": ("cameraman", "cap.video_camera_ic"),
}


def _make_new_mode_case(key: str):
    kind, cap = _NEW_MODES[key]

    def _case(w, spy):
        from core import video_edit_modes as vem
        _ack_dev(False)
        t = _fill_mode(w, key)
        _getter(w, _DIST)                                # distilled scelto nel tab Video: nessun dialog
        t._go()
        assert spy.calls, f"{key}: nessuna richiesta accodata (log: {t.log.toPlainText()[-300:]})"
        cfg = _cfg_of(w)
        assert cfg.get("mode") == key, cfg.get("mode")
        sp = vem.get_spec(key)
        assert sp.kind == kind and sp.cap == cap, (sp.kind, sp.cap)
        assert (cfg.get("edit_lora") or "") == "", f"edit_lora deve restare vuota per i kind a registro: {cfg.get('edit_lora')}"
        assert cfg.get("model") == _DIST["model"], f"model non inoltrato ESATTAMENTE come scelto: {cfg.get('model')}"
        assert cfg.get("ckpt_state") == "distilled", cfg.get("ckpt_state")
        assert "conditioning_attention" in cfg
        if key == "restyle":
            assert cfg["style_image"] and cfg["prompt"] == "Make it Disney 2D Animation style.", cfg["prompt"]
        if key == "outpaint":
            assert cfg["out_format"] == "16:9" and cfg["out_paste"] is True and "out_margins" in cfg
        if key == "cameraman":
            assert cfg["cam_ref"] and cfg["width"] == 960 and 0.3 <= cfg["cam_img_strength"] <= 1.0
        if sp.family == vem.FAMILY_REPAIR:
            assert cfg["res_mode"] == "source" and cfg["prompt"], "il restauro parte col prompt-guida precompilato"
            assert "keep_audio" in cfg
    return _case


def case_families_and_modes(w, spy):
    """Due livelli: famiglie (Lip dub dalla fase 1b, HDR in «Ripara video» dalla 1c), modalità per famiglia."""
    from core import video_edit_modes as vem
    t = _tab(w)
    fams = [t.family.itemData(i) for i in range(t.family.count())]
    assert fams == ["edit", "style", "repair", "outpaint", "lipdub", "camera"], fams
    expect = {"edit": ["edit", "inpaint", "replace_char"], "style": ["anime2real", "real2anime", "restyle"],
              "repair": ["refocus", "uncompress", "deinterlace", "motiondeblur", "detailer", "hdr"],
              "outpaint": ["outpaint"], "lipdub": ["lipdub", "lipdub_audio"], "camera": ["cameraman", "motion_track"]}
    for fam, modes in expect.items():
        t.family.setCurrentIndex(t.family.findData(fam))
        got = [t.mode.itemData(i) for i in range(t.mode.count())]
        assert got == modes, f"{fam}: {got} != {modes}"
    for k in ("edit", "restyle", "detailer", "outpaint", "cameraman", "replace_char", "lipdub", "lipdub_audio", "hdr", "motion_track"):
        t.select_mode(k)                                 # da qualunque famiglia
        assert t.mode.currentData() == k and t.family.currentData() == vem.family_of(k)
    from ui.tabs.video_edit_ai_tab import MODES
    assert MODES == vem.legacy_modes()


def case_fields_per_mode(w, spy):
    """Campi abilitati/visibili per modalità."""
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    _select_mode(w, "restyle")
    assert t.ref_img.isEnabled() and not t.ref_strength.isEnabled() and not t.style_text.isHidden()
    _select_mode(w, "replace_char")
    assert t.ref_img.isEnabled() and t.ref_strength.isEnabled() and t.style_text.isHidden()
    _select_mode(w, "refocus")
    assert not t.ref_img.isEnabled() and not t.obj.isEnabled() and t.cam_ref.parentWidget().isHidden()
    assert not t.res_mode.parentWidget().isHidden() and not t.attn.isHidden() and not t.keep_audio.isHidden()
    _select_mode(w, "outpaint")
    assert not t.out_format.isHidden() and t.res_mode.parentWidget().isHidden() and t.out_m["top"].parentWidget().isHidden()
    t.out_format.setCurrentIndex(t.out_format.findData("custom"))
    assert not t.out_m["top"].parentWidget().isHidden() and not t.out_anchor.isEnabled()
    _select_mode(w, "cameraman")
    assert not t.cam_ref.parentWidget().isHidden() and not t.cam_img_strength.isHidden()
    assert t.note.text(), "ogni modalità ha una nota"
    _select_mode(w, "edit")
    assert t.cam_ref.parentWidget().isHidden() and t.attn.isHidden()


def case_repair_no_prompt_ok(w, spy):
    """Restauro: prompt vuoto accettato (opzionale); edit con prompt vuoto blocca (esistente)."""
    _ack_dev(False)
    t = _fill_mode(w, "uncompress")
    _getter(w, _DIST)
    t.prompt.setPlainText("")
    t._go()
    assert spy.calls, "il restauro con prompt vuoto deve accodare"
    n = len(spy.calls)
    _select_mode(w, "edit")
    t.prompt.setPlainText("")
    t._go()
    assert len(spy.calls) == n, "edit senza prompt deve bloccare"


def case_ckpt_hint_states(w, spy):
    """Riga di stato del checkpoint: distilled / dev / auto / release 2.5 / modello non LTX, con icona+testo."""
    t = _tab(w)
    _select_mode(w, "refocus")
    rows = [(_DIST, "✔", "distilled"), (_DEV, "ℹ", "dev"), (_AUTO, "ℹ", ""),
            (_REL25, "⛔", ""), ({"model_kind": "minimax_h3", "model": ""}, "ℹ", "")]
    for vals, icon, word in rows:
        _getter(w, vals)
        txt = t.ckpt_lbl.text()
        assert icon in txt and word in txt, f"{vals['model_kind']} {vals['model'][-30:]}: «{txt}»"
        assert not t.ckpt_lbl.isHidden()
    _getter(w, _DIST)
    assert "ltx-2.3-22b-distilled" in t.ckpt_lbl.text(), "il nome del checkpoint scelto deve comparire"


def case_dev_warning_dialog_choices(w, spy):
    """Checkpoint dev: dialog a 3 scelte. Annulla non accoda; Apri tab Video emette il
    segnale e non accoda; Continua accoda con model esattamente come scelto (mai cambiato)."""
    _ack_dev(False)
    t = _fill_mode(w, "refocus")
    _getter(w, _DEV)
    asked, opened = [], []
    t.request_open_video_tab.connect(lambda: opened.append(1))
    for ans in ("cancel", "open", "continue"):
        t._ask_checkpoint = (lambda sp, info, blocking, _a=ans: asked.append((sp.key, blocking)) or _a)
        before = len(spy.calls)
        t._go()
        grew = len(spy.calls) - before
        assert grew == (1 if ans == "continue" else 0), f"{ans}: richieste accodate +{grew}"
    assert asked == [("refocus", False)] * 3, asked
    assert opened == [1], f"'open' deve emettere request_open_video_tab una volta: {opened}"
    assert _cfg_of(w).get("model") == _DEV["model"], "il checkpoint dev scelto non va sostituito"
    asked.clear()
    _getter(w, _DIST)                                    # con distilled: nessun dialog
    t._go()
    assert not asked, "con un distilled non deve comparire alcun dialog"
    _getter(w, _DEV)
    _ack_dev(True)                                       # «non ripetere»: niente dialog neanche col dev
    t._go()
    assert not asked, "con ack di sessione non deve richiedere di nuovo"


def case_dev_never_blocks_adds_dlora(w, spy):
    """Col dev NESSUNA modalità blocca, nemmeno una con distilled=required (simulata su refocus): il dialog AVVISA e la
    generazione parte («Procedi»); il checkpoint scelto non viene mai sostituito; il testo dice che verrà aggiunta
    la LoRA distilled."""
    import dataclasses
    from core import video_edit_modes as vem
    _ack_dev(False)
    orig = vem.MODE_SPECS["refocus"]
    vem.MODE_SPECS["refocus"] = dataclasses.replace(orig, distilled=vem.DISTILLED_REQUIRED)
    try:
        t = _fill_mode(w, "refocus")
        _getter(w, _DEV)
        txt = t.ckpt_lbl.text()
        assert "⛔" not in txt and "LoRA" in txt and "distilled" in txt, txt
        got = []
        t._ask_checkpoint = lambda sp, info, blocking=False: got.append((blocking, info.get("dlora_ok"))) or "continue"
        t._go()
        assert got and got[0][0] is False, f"deve solo avvisare (blocking=False): {got}"
        assert len(spy.calls) == 1, "col dev 'Procedi' deve far partire la generazione"
        assert _cfg_of(w).get("model") == _DEV["model"], "il checkpoint scelto non va sostituito"
    finally:
        vem.MODE_SPECS["refocus"] = orig


def case_dev_real_dialog_warns_and_runs(w, spy):
    """Dialog REALE (non sovrascritto): QMessageBox di sola informazione con testo «verrà aggiunta la LoRA distilled»,
    bottone «Procedi comunque» → la generazione parte. Bottoni: Procedi / Apri tab Video / Annulla (niente «blocca»)."""
    from PySide6.QtWidgets import QMessageBox
    _ack_dev(False)
    t = _fill_mode(w, "uncompress")
    _getter(w, _DEV)
    seen = {}
    orig_exec = QMessageBox.exec

    def fake_exec(box):
        seen["text"] = box.text()
        seen["icon"] = box.icon()
        seen["buttons"] = [b.text() for b in box.buttons()]
        for b in box.buttons():
            if box.buttonRole(b) == QMessageBox.AcceptRole:
                b.click()
        return 0
    QMessageBox.exec = fake_exec
    try:
        before = len(spy.calls)
        t._go()
    finally:
        QMessageBox.exec = orig_exec
    assert seen, "il dialog sul checkpoint dev deve comparire"
    assert "distilled" in seen["text"] and "LoRA" in seen["text"], seen["text"]
    assert seen["icon"] == QMessageBox.Information, "con la LoRA su disco è solo un'informazione"
    assert len(seen["buttons"]) == 3, seen["buttons"]
    assert len(spy.calls) == before + 1, "«Procedi comunque» deve far partire la generazione"


def case_dev_dlora_missing_warns_with_weights_link(w, spy):
    """LoRA distilled ASSENTE su disco: riga di stato e dialog avvisano (icona warning) con bottone «Apri tab Pesi
    modelli» che emette il segnale; nessun download; «Procedi comunque» fa partire comunque la generazione."""
    from core import video_edit_ic as vic
    _ack_dev(True)                                       # con la LoRA mancante l'avviso compare SEMPRE
    t = _fill_mode(w, "refocus")
    real_info = vic.checkpoint_info
    vic.checkpoint_info = lambda vals: {**real_info(vals), "dlora_ok": False}
    try:
        _getter(w, _DEV)
        from ui.i18n_manager import tr
        want = tr("vedit.ckpt.dlora_missing", "").split("{f}")[0][:20]   # testo nella lingua della GUI
        assert "⚠" in t.ckpt_lbl.text() and want and want in t.ckpt_lbl.text(), t.ckpt_lbl.text()
        asked, opened = [], []
        t.request_open_weights_tab.connect(lambda: opened.append(1))
        t._ask_checkpoint = lambda sp, info, blocking=False: asked.append(info["dlora_ok"]) or "weights"
        before = len(spy.calls)
        t._go()
        assert asked == [False], "avviso anche con «non ripetere»"
        assert opened == [1] and len(spy.calls) == before, "«Apri tab Pesi» emette il segnale e non accoda"
        t._ask_checkpoint = lambda sp, info, blocking=False: "continue"
        t._go()
        assert len(spy.calls) == before + 1, "«Procedi comunque» accoda comunque"
    finally:
        vic.checkpoint_info = real_info


def case_release25_blocks(w, spy):
    """Checkpoint di release 2.5: le IC-LoRA locali sono per 2.3 → blocco con messaggio, nessuna richiesta."""
    _ack_dev(True)
    t = _fill_mode(w, "refocus")
    _getter(w, _REL25)
    t._go()
    assert not spy.calls, "release 2.5 deve bloccare le modalità IC-LoRA nuove"
    assert "2.3" in t.log.toPlainText()


def case_restyle_requires_image(w, spy):
    """ReStyle senza immagine di stile blocca; con immagine (+ stile a parole) accoda con style_image."""
    _ack_dev(True)
    t = _fill_mode(w, "restyle")
    t.ref_img.setText("")
    t._go()
    assert not spy.calls, "restyle senza immagine di stile deve bloccare"
    t.ref_img.setText(_mk_image())
    t.style_text.setText("")
    t.prompt.setPlainText("")
    t._go()
    assert not spy.calls, "restyle senza stile né prompt deve bloccare"
    t.style_text.setText("watercolor")
    t._go()
    assert spy.calls and _cfg_of(w)["prompt"] == "Make it watercolor style."


def case_outpaint_canvas_and_margins(w, spy):
    """Outpaint: anteprima/canvas dal sorgente noto (multipli di 64, area nuova>0), margini personalizzati,
    avviso area troppo grande, blocco senza area nuova."""
    from types import SimpleNamespace
    _ack_dev(True)
    t = _fill_mode(w, "outpaint")
    t._vinfo = SimpleNamespace(width=1080, height=1920, nb_frames=100, fps=25.0)   # verticale
    t.out_format.setCurrentIndex(t.out_format.findData("16:9"))
    t._on_outpaint_change()                              # (il sorgente è stato impostato a mano dopo la selezione)
    lay = t._outpaint_layout()
    assert lay.canvas_w % 64 == 0 and lay.canvas_h % 64 == 0 and lay.canvas_w > lay.canvas_h
    assert lay.new_area > 0.5 and lay.src_w <= lay.canvas_w and lay.src_h <= lay.canvas_h
    assert "canvas" in t.out_info.text().lower()
    assert lay.too_much_new_area and "60" in t.out_info.text(), "avviso area nuova > 60% atteso"
    t._go()
    assert spy.calls and "60" in t.log.toPlainText()
    n = len(spy.calls)
    t.out_format.setCurrentIndex(t.out_format.findData("custom"))   # tutti i margini a 0 → blocca
    t._go()
    assert len(spy.calls) == n, "margini tutti a zero: deve bloccare"
    t.out_m["left"].setValue(200)
    t.out_m["right"].setValue(200)
    lay = t._outpaint_layout()
    assert lay.canvas_w > lay.src_w and lay.x > 0 and lay.canvas_w % 64 == 0
    t._go()
    cfg = _cfg_of(w)
    assert len(spy.calls) == n + 1 and cfg["out_format"] == "custom" and cfg["out_margins"][1] == 200


def case_outpaint_narrow_bar_warning(w, spy):
    """Barre nuove strette (< 15% del canvas): avviso in info e nel log; con margini larghi niente avviso;
    il sorgente non viene rimpicciolito (512x288 -> 21:9 = canvas 704x320, scala 1.0)."""
    from types import SimpleNamespace
    from core import video_ic_prep as prep
    _ack_dev(True)
    t = _fill_mode(w, "outpaint")
    t._vinfo = SimpleNamespace(width=512, height=288, nb_frames=49, fps=25.0)
    t.out_format.setCurrentIndex(t.out_format.findData("21:9"))
    t._on_outpaint_change()
    lay = t._outpaint_layout()
    assert lay.canvas == (704, 320) and lay.scale == 1.0 and lay.narrow_bar
    assert t._narrow_bar_text() in t.out_info.text(), "avviso barra stretta atteso nell'info"
    n = len(spy.calls)
    t._go()
    assert len(spy.calls) == n + 1 and t._narrow_bar_text() in t.log.toPlainText(), "avviso nel log (non blocca)"
    t.out_format.setCurrentIndex(t.out_format.findData("custom"))
    t.out_m["left"].setValue(200)
    t.out_m["right"].setValue(200)
    t._on_outpaint_change()
    assert not t._outpaint_layout().narrow_bar and t._narrow_bar_text() not in t.out_info.text()


def case_outpaint_formats_direction_feather(w, spy):
    """Outpaint: tutti i formati (21:9, 1:1, 4:3, 3:4, 9:16) danno canvas validi; la riga info dice DOVE si aggiunge
    (sopra/sotto/lati); «Estendi verso» sposta l'area nuova da una parte; sfumatura e barre minime finiscono nella cfg."""
    from types import SimpleNamespace
    _ack_dev(True)
    from ui.i18n_manager import tr

    def side(name):                                   # frammento localizzato «sopra {n} px» senza il numero
        return tr(f"vedit.out.side.{name}", name).split("{n}")[0]
    t = _fill_mode(w, "outpaint")
    t._vinfo = SimpleNamespace(width=1280, height=720, nb_frames=49, fps=25.0)
    for fmt in ("21:9", "1:1", "4:3", "3:4", "9:16"):
        t.out_format.setCurrentIndex(t.out_format.findData(fmt))
        t._on_outpaint_change()
        lay = t._outpaint_layout()
        assert lay is not None and lay.canvas_w % 64 == 0 and lay.canvas_h % 64 == 0 and lay.new_area > 0, fmt
        assert "canvas" in t.out_info.text().lower(), fmt
    # 4:3 da 16:9: barre SOPRA e SOTTO, dette nell'info
    t.out_format.setCurrentIndex(t.out_format.findData("4:3"))
    t._on_outpaint_change()
    txt = t.out_info.text()
    assert side("top") in txt and side("bottom") in txt and side("left") not in txt and "120 px" in txt, txt
    # «Estendi verso» in alto: tutta l'area nuova sopra (video in basso)
    t.out_anchor.setCurrentIndex(t.out_anchor.findData("bottom"))
    t._on_outpaint_change()
    lay = t._outpaint_layout()
    txt = t.out_info.text()
    assert lay.y > 0 and lay.y + lay.src_h == lay.canvas_h and side("top") in txt and side("bottom") not in txt, txt
    # 21:9 da 16:9: lati
    t.out_anchor.setCurrentIndex(t.out_anchor.findData("center"))
    t.out_format.setCurrentIndex(t.out_format.findData("21:9"))
    t._on_outpaint_change()
    assert side("left") in t.out_info.text() and side("right") in t.out_info.text()
    # sfumatura: default 24, attiva solo con «rimetti l'originale»; barre minime opt-in
    assert t.out_feather.value() == 24 and t.out_feather.isEnabled() and not t.out_minbar.isChecked()
    t.out_paste.setChecked(False)
    assert not t.out_feather.isEnabled()
    t.out_paste.setChecked(True)
    t.out_feather.setValue(40)
    n = len(spy.calls)
    t._go()
    cfg = _cfg_of(w)
    assert len(spy.calls) == n + 1 and cfg["out_feather"] == 40 and cfg["out_min_bar"] == 0.0 and cfg["out_paste"] is True
    t.out_format.setCurrentIndex(t.out_format.findData("4:3"))
    plain = t._outpaint_layout()
    t.out_minbar.setChecked(True)
    t._on_outpaint_change()
    shrunk = t._outpaint_layout()
    assert plain.narrow_bar and not shrunk.narrow_bar and shrunk.scale < 1.0
    t._go()
    assert _cfg_of(w)["out_min_bar"] == 0.15
    t.out_format.setCurrentIndex(t.out_format.findData("custom"))
    assert not t.out_minbar.isEnabled()
    t.out_minbar.setChecked(False)


def case_outpaint_dark_autodetect(w, spy):
    """Sorgente scuro (video sintetico nero) → il checkbox gamma è suggerito in automatico (e dichiarato nel log)."""
    import subprocess
    from core import video_ic_prep as prep
    dark = str(H.OUT_DIR / "_vedit_dark.mp4")
    light = str(H.OUT_DIR / "_vedit_light.mp4")
    for path, col in ((dark, "black"), (light, "white")):
        subprocess.run([prep.ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={col}:s=160x96:r=25:d=2",
                        "-pix_fmt", "yuv420p", path], check=True, capture_output=True)
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    _select_mode(w, "outpaint")
    t.out_dark.setChecked(False)
    t.set_video(dark)
    assert t.out_dark.isChecked(), "video scuro: gamma suggerito"
    assert "gamma" in t.log.toPlainText().lower()
    t.out_dark.setChecked(False)
    t.set_video(light)
    assert not t.out_dark.isChecked(), "video chiaro: nessun suggerimento"


def case_cameraman_validation(w, spy):
    """Cameraman: senza riferimento blocca (anche con video principale); sotto 960x512 avvisa ma non blocca."""
    _ack_dev(True)
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    _set_video(w)
    _select_mode(w, "cameraman")
    t.cam_ref.setText("")
    t._go()
    assert not spy.calls, "cameraman senza riferimento deve bloccare"
    ref = _dummy_video("_vedit_camref.mp4")
    t.cam_ref.setText(ref)
    t._video = ""                                         # il video principale è opzionale per cameraman
    t.cam_w.setValue(640)
    t.cam_h.setValue(384)
    t._go()
    assert spy.calls, "sotto 960x512: avviso, non blocco"
    assert "960" in t.log.toPlainText()
    cfg = _cfg_of(w)
    assert cfg["video"] == ref and cfg["cam_ref"] == ref and (cfg["width"], cfg["height"]) == (640, 384)


def case_weight_missing_blocks(w, spy):
    """Peso IC-LoRA assente: il worker GUI dà errore chiaro (nessun degrado a ti2v, nessuna esecuzione)."""
    from core import mem_coordinator, video_edit_ic as vic
    from ui.main_window import _VideoEditAIWorker
    _ack_dev(True)
    t = _fill_mode(w, "refocus")
    _getter(w, _DIST)
    t._go()
    cfg = _cfg_of(w)
    cfg["video"] = _dummy_video("_vedit_test_input.mp4")
    got = {}
    orig_missing, orig_fw = vic.missing_weight, mem_coordinator.free_and_wait
    vic.missing_weight = lambda spec: "Peso «refocus» mancante: X:\\nope.safetensors"
    mem_coordinator.free_and_wait = lambda *a, **k: None
    try:
        wk = _VideoEditAIWorker(cfg, str(H.OUT_DIR))
        wk.done.connect(lambda r: got.update(r))
        wk.run()                                         # sincrono: build_job fallisce PRIMA di lanciare il worker
    finally:
        vic.missing_weight, mem_coordinator.free_and_wait = orig_missing, orig_fw
    assert got.get("status") == "error" and "mancante" in got.get("error", ""), got


def case_gate_blocks_when_cap_off(w, spy):
    """Edizione senza la cap: nessun accodamento (l'upsell lo mostra il gate); le storiche non sono gated.
    (L'harness sostituisce `_feature_gate` con un sì incondizionato: qui lo si rimpiazza con uno che nega le cap.video_*
    e registra l'id richiesto, per verificare il COLLEGAMENTO modalità → cap → blocco.)"""
    _ack_dev(True)
    t = _fill_mode(w, "refocus")
    _getter(w, _DIST)
    seen = []
    w._feature_gate = lambda fid: (seen.append(fid) or False)
    t._go()
    assert not spy.calls, "cap spenta: nessun accodamento"
    assert seen == ["cap.video_restore_iclora"], seen
    for key, cap in (("outpaint", "cap.video_outpaint"), ("restyle", "cap.video_restyle")):
        seen.clear()
        _fill_mode(w, key)
        t._go()
        assert seen == [cap] and not spy.calls, (key, seen)
    seen.clear()
    _select_mode(w, "edit")                              # le modalità storiche non sono gated
    t.prompt.setPlainText("rain")
    t._go()
    assert not seen and len(spy.calls) == 1, "edit (storica) non deve essere gated"


def case_legacy_modes_pass_model(w, spy):
    """Anche le modalità storiche inoltrano il checkpoint scelto (prima il tab non passava `model`)."""
    _ack_dev(True)
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    _set_video(w)
    _select_mode(w, "edit")
    _getter(w, _DEV)
    t.prompt.setPlainText("make it night")
    t._go()
    assert spy.calls and _cfg_of(w).get("model") == _DEV["model"]
    assert _cfg_of(w).get("ckpt_state") == "dev"


def case_ic_lora_hidden_from_lora_panel(w, spy):
    """Le IC-LoRA (ltx-*ic-lora*, Cameraman, ReStyle, scene-emb) NON compaiono nel pannello LoRA normale; le LoRA
    camera-control / CameraControls restano visibili."""
    from pathlib import Path
    from core.lora_catalog import is_ic_lora_file
    w.video_tab.nsfw.setCurrentText("NSFW")
    w.lora_tab.refresh()
    names = [r["entry"].filename for r in w.lora_tab._rows.values()]
    assert names, "il pannello LoRA LTX non deve essere vuoto"
    bad = [n for n in names if is_ic_lora_file(n)]
    assert not bad, f"IC-LoRA visibili nel pannello LoRA normale: {bad}"
    cat = [e.filename for e in w.lora_tab.catalog.loras(allow_nsfw=True)]
    assert not [n for n in cat if is_ic_lora_file(n)], "nemmeno nel catalogo"
    for ic in ("ltx-2.3-22b-ic-lora-outpaint.safetensors", "LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors",
               "852654_LTX2.3-22B_ReStyle_IC-LoRA_8000_v0.1.safetensors", "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"):
        assert is_ic_lora_file(ic), ic
    for ok in ("ltx-2-19b-lora-camera-control-dolly-left.safetensors", "LTX2.3_CameraControls.safetensors"):
        assert not is_ic_lora_file(ok), ok
        if (Path(r"E:\AI\Models\LoRA\LTX-2") / ok).exists():
            assert ok in cat, f"{ok} (LoRA normale) deve restare visibile"


# ── Lip dub (fase 1b) ────────────────────────────────────────────────────────────────────────────
def _lipdub_video(name: str, audio: bool = True, frames: int = 50) -> str:
    """Video REALE minuscolo (ffmpeg, CPU) con o senza traccia audio: il tab lo sonda con ffprobe."""
    import subprocess
    from core import video_ic_prep as prep
    path = str(H.OUT_DIR / name)
    cmd = [prep.ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=25"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=330:sample_rate=16000", "-shortest"]
    cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac"] if audio else []) + [path]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _lipdub_tab(w, video: str):
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    t.set_video(video)
    _select_mode(w, "lipdub")
    t.lip_line.setPlainText("")
    t.lip_scene.setText("")
    t.lip_lang.setCurrentIndex(0)
    t.prompt.setPlainText("")
    return t


def case_lipdub_fields(w, spy):
    """Lip dub: campi propri visibili (scena, battuta, lingua, info), strength/attenzione/mantieni audio/enhance nascosti,
    risoluzione visibile; tornando a un'altra modalità tutto si ripristina."""
    t = _lipdub_tab(w, _lipdub_video("_vedit_lipdub_a.mp4"))
    for fld in (t.lip_scene, t.lip_line, t.lip_lang, t.lip_info):
        assert not fld.isHidden(), fld
    for fld in (t.strength, t.attn, t.keep_audio, t.vedit_enhance, t.obj):
        assert fld.isHidden(), f"nascosto in lipdub: {fld}"
    assert not t.res_mode.parentWidget().isHidden()
    assert t.family.currentData() == "lipdub" and t.note.text(), "nota della modalità"
    ph_lip = t.prompt.placeholderText()
    _select_mode(w, "refocus")
    for fld in (t.lip_scene, t.lip_line, t.lip_lang, t.lip_info):
        assert fld.isHidden(), fld
    assert not t.strength.isHidden() and not t.vedit_enhance.isHidden() and not t.attn.isHidden()
    assert t.prompt.placeholderText() != ph_lip, "il segnaposto del prompt torna quello normale"
    codes = [t.lip_lang.itemData(i) for i in range(t.lip_lang.count())]
    from core import video_edit_ic as vic
    assert codes == [""] + list(vic.SPOKEN_CODES), codes            # SOLO le lingue con prove di supporto
    assert not t.lip_lang.isEditable()
    _select_mode(w, "lipdub")
    assert not t.lip_translate.isHidden() and not t.lip_src_text.isHidden() and not t.lip_open_talk.parentWidget().isHidden()
    assert t.lip_audio.parentWidget().isHidden() and t.lip_exp.isHidden() and t.lip_audio_ref.isHidden()
    assert "Doppiaggio" in t.mode.currentText() or t.mode.currentText(), t.mode.currentText()


def case_lipdub_cfg(w, spy):
    """Lip dub con distilled: la cfg porta battuta/lingua/scena (il prompt si compone nel core), enhance spento,
    nessun mantieni-audio, checkpoint inoltrato come scelto."""
    _ack_dev(False)
    t = _lipdub_tab(w, _lipdub_video("_vedit_lipdub_a.mp4"))
    _getter(w, _DIST)
    t.lip_line.setPlainText("Buongiorno a tutti, benvenuti.")
    t.lip_scene.setText("a chef in a kitchen")
    t.lip_lang.setCurrentIndex(t.lip_lang.findData("it"))
    t.vedit_enhance.setChecked(True)                      # anche se spuntato altrove: per lipdub resta spento
    t._go()
    assert spy.calls, f"nessuna richiesta accodata (log: {t.log.toPlainText()[-300:]})"
    cfg = _cfg_of(w)
    assert cfg["mode"] == "lipdub" and cfg["lipdub_line"] == "Buongiorno a tutti, benvenuti.", cfg
    assert cfg["lipdub_lang"] == "it" and cfg["lipdub_scene"] == "a chef in a kitchen"
    assert cfg["enhance"] is False and cfg["keep_audio"] is False
    assert cfg["model"] == _DIST["model"] and cfg["ckpt_state"] == "distilled"
    assert cfg["prompt"] == "" and (cfg.get("edit_lora") or "") == ""
    # prompt libero: viene inoltrato e il core lo usera' al posto della composizione
    t.prompt.setPlainText("custom full prompt")
    t._go()
    assert _cfg_of(w)["prompt"] == "custom full prompt"
    t.vedit_enhance.setChecked(False)


def case_lipdub_validation(w, spy):
    """Blocchi: niente battuta, video senza audio, clip oltre il tetto; col dev avvisa ma non blocca."""
    _ack_dev(False)
    t = _lipdub_tab(w, _lipdub_video("_vedit_lipdub_a.mp4"))
    _getter(w, _DIST)
    t._go()
    assert not spy.calls and t.log.toPlainText().strip(), "senza battuta né prompt: blocco con messaggio"
    t.log.clear()
    mute = _lipdub_video("_vedit_lipdub_mute.mp4", audio=False)
    t = _lipdub_tab(w, mute)
    t.lip_line.setPlainText("ciao")
    assert "⛔" in t.lip_info.text(), t.lip_info.text()
    t._go()
    assert not spy.calls, "video senza traccia audio: blocco"
    t.log.clear()
    longv = _lipdub_video("_vedit_lipdub_long.mp4", frames=300)
    t = _lipdub_tab(w, longv)
    t.lip_line.setPlainText("ciao")
    t._go()
    assert not spy.calls and "249" in t.log.toPlainText(), t.log.toPlainText()
    # dev: avviso (dialog sovrascritto = «Procedi comunque»), non blocca
    t = _lipdub_tab(w, _lipdub_video("_vedit_lipdub_a.mp4"))
    t.lip_line.setPlainText("ciao a tutti")
    _getter(w, _DEV)
    asked = []
    t._ask_checkpoint = lambda sp, info, blocking=False: asked.append(sp.key) or "continue"
    t._go()
    assert asked == ["lipdub"] and len(spy.calls) == 1, (asked, len(spy.calls))
    assert _cfg_of(w)["ckpt_state"] == "dev"


def case_lipdub_info_fit(w, spy):
    """Riga informativa: durata del video, stima del parlato, avviso se la battuta è lunga/corta (mai blocco)."""
    t = _lipdub_tab(w, _lipdub_video("_vedit_lipdub_a.mp4"))        # 2 s
    assert "2.0" in t.lip_info.text(), t.lip_info.text()
    t.lip_line.setPlainText(" ".join(["parola"] * 5))               # ~2 s: ok
    assert "⚠" not in t.lip_info.text() and "≈" in t.lip_info.text(), t.lip_info.text()
    t.lip_line.setPlainText(" ".join(["parola"] * 30))              # ~12 s su 2 s
    assert "⚠" in t.lip_info.text(), t.lip_info.text()
    t.lip_line.setPlainText("ok")                                   # 0.4 s su 2 s
    assert "⚠" in t.lip_info.text(), t.lip_info.text()
    _select_mode(w, "refocus")
    assert t.lip_info.text() == "", "fuori da lipdub la riga e' vuota"


def case_lipdub_weight_missing_dialog(w, spy):
    """LoRA lipdub assente: avviso con «Apri tab Pesi modelli» (segnale), nessun accodamento, nessun download."""
    from core import video_edit_ic as vic
    _ack_dev(True)
    t = _lipdub_tab(w, _lipdub_video("_vedit_lipdub_a.mp4"))
    _getter(w, _DIST)
    t.lip_line.setPlainText("ciao a tutti")
    orig = vic.missing_weight
    vic.missing_weight = lambda spec: "Peso «lipdub» mancante: X:\\nope.safetensors"
    try:
        asked, opened = [], []
        t.request_open_weights_tab.connect(lambda: opened.append(1))
        t._ask_missing_weight = lambda sp, msg: asked.append(sp.key) or "weights"
        before = len(spy.calls)
        t._go()
        assert asked == ["lipdub"] and opened == [1] and len(spy.calls) == before
        t._ask_missing_weight = lambda sp, msg: "cancel"
        t._go()
        assert opened == [1] and len(spy.calls) == before, "Annulla: nessun segnale e nessun accodamento"
    finally:
        vic.missing_weight = orig


def case_lipdub_gate(w, spy):
    """cap.video_lipdub spenta: nessun accodamento (l'upsell lo mostra il gate)."""
    _ack_dev(True)
    t = _lipdub_tab(w, _lipdub_video("_vedit_lipdub_a.mp4"))
    _getter(w, _DIST)
    t.lip_line.setPlainText("ciao a tutti")
    seen = []
    w._feature_gate = lambda fid: (seen.append(fid) or False)
    t._go()
    assert seen == ["cap.video_lipdub"] and not spy.calls, (seen, len(spy.calls))


def case_lipdub_audio_fields(w, spy):
    """Doppiaggio da audio (sperimentale): riga audio, riferimento voce e banner visibili; niente traduzione/Talk-Canta."""
    t = _lipdub_tab(w, _lipdub_video("_vedit_lipdub_a.mp4"))
    _select_mode(w, "lipdub_audio")
    assert not t.lip_audio.parentWidget().isHidden() and not t.lip_audio_ref.isHidden() and not t.lip_exp.isHidden()
    assert t.lip_translate.isHidden() and t.lip_src_text.isHidden() and t.lip_open_talk.parentWidget().isHidden()
    assert "SPERIMENTAL" in t.lip_exp.text().upper() or "EXPERIMENTAL" in t.lip_exp.text().upper(), t.lip_exp.text()
    assert not t.lip_scene.isHidden() and not t.lip_line.isHidden() and t.strength.isHidden() and t.keep_audio.isHidden()
    assert "sperimental" in t.mode.currentText().lower() or "experimental" in t.mode.currentText().lower(), t.mode.currentText()
    _select_mode(w, "lipdub")
    assert t.lip_audio.parentWidget().isHidden() and t.lip_exp.isHidden() and not t.lip_translate.isHidden()


def _lipdub_wav(name: str, seconds: float) -> str:
    import subprocess
    from core import video_ic_prep as prep
    path = str(H.OUT_DIR / name)
    subprocess.run([prep.ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=300:sample_rate=22050:duration={seconds}", path], check=True, capture_output=True)
    return path


def case_lipdub_audio_cfg_validation(w, spy):
    """Da audio: serve il file; durate (audio piu' corto/lungo) avvisano; cfg con audio e riferimento voce; ultimo audio."""
    _ack_dev(False)
    t = _lipdub_tab(w, _lipdub_video("_vedit_lipdub_a.mp4"))        # 2 s
    _select_mode(w, "lipdub_audio")
    _getter(w, _DIST)
    t._go()
    assert not spy.calls and t.log.toPlainText().strip(), "senza audio: blocco con messaggio"
    t.log.clear()
    t.lip_audio.setText(_lipdub_wav("_vedit_lipdub_short.wav", 0.8))
    assert "⚠" in t.lip_info.text(), t.lip_info.text()
    t.lip_audio.setText(_lipdub_wav("_vedit_lipdub_long.wav", 5.0))
    assert "⚠" in t.lip_info.text(), t.lip_info.text()
    ok = _lipdub_wav("_vedit_lipdub_ok.wav", 2.0)
    t.lip_audio.setText(ok)
    assert "⚠" not in t.lip_info.text() and "2.0" in t.lip_info.text(), t.lip_info.text()
    t.lip_audio_ref.setCurrentIndex(t.lip_audio_ref.findData("original"))
    t.lip_scene.setText("a singer")
    t._go()
    assert spy.calls, t.log.toPlainText()
    cfg = _cfg_of(w)
    assert cfg["mode"] == "lipdub_audio" and cfg["lipdub_audio"] == ok and cfg["lipdub_audio_ref"] == "original", cfg
    assert cfg["lipdub_scene"] == "a singer" and cfg["enhance"] is False and cfg["keep_audio"] is False
    assert cfg["model"] == _DIST["model"] and cfg["ckpt_state"] == "distilled"
    # «ultimo audio generato» dal getter di main_window
    t.lip_audio.setText("")
    t.set_last_audio_getter(lambda: ok)
    t._use_last_audio()
    assert t.lip_audio.text() == ok
    t.set_last_audio_getter(lambda: "")
    t.lip_audio.setText("")
    t._use_last_audio()
    assert t.lip_audio.text() == "" and t.log.toPlainText().strip()
    # video muto + riferimento «originale»: blocco (con «audio scelto» e' ammesso)
    t = _lipdub_tab(w, _lipdub_video("_vedit_lipdub_mute.mp4", audio=False))
    _select_mode(w, "lipdub_audio")
    t.lip_audio.setText(ok)
    t.lip_audio_ref.setCurrentIndex(t.lip_audio_ref.findData("external"))
    n = len(spy.calls)
    t._go()
    assert len(spy.calls) == n + 1, "video muto + riferimento «audio scelto»: ammesso"
    t.lip_audio_ref.setCurrentIndex(t.lip_audio_ref.findData("original"))
    t._go()
    assert len(spy.calls) == n + 1, "video muto + riferimento «originale»: blocco"


def case_lipdub_open_talk_sing(w, spy):
    """«Hai un tuo audio?»: i bottoni emettono (modalita', video, audio); l'handler di main_window apre Talk/Canta nel tab
    Video con video/audio impostati. Talk/Canta non sono modificate: si usano i loro controlli esistenti."""
    from ui.tabs.video_edit_ai_tab import VideoEditAITab  # noqa: F401
    vid = _lipdub_video("_vedit_lipdub_a.mp4")
    wav = _lipdub_wav("_vedit_lipdub_ok.wav", 2.0)
    t = _lipdub_tab(w, vid)
    got = []
    t.request_open_v2v.connect(lambda m, v, a: got.append((m, v, a)))
    t.set_last_audio_getter(lambda: wav)
    t.lip_open_talk.click()
    t.lip_open_sing.click()
    assert got == [("talk", vid, wav), ("sing", vid, wav)], got
    # handler reale di main_window (collegato dal costruttore)
    panel = w.video_tab.v2v_panel
    w.tabs.setCurrentWidget(t)
    w._open_v2v_from_vedit("talk", vid, wav)
    assert w.tabs.currentWidget() is w.video_tab and panel.current_mode() == "talk"
    assert panel._video_path == vid and w.audio_tab._audio_file == wav and w.audio_tab.rb_user.isChecked()
    w.tabs.setCurrentWidget(t)
    w._open_v2v_from_vedit("sing", vid, wav)
    assert w.tabs.currentWidget() is w.video_tab and panel.current_mode() == "sing" and panel._sing_audio_path == wav
    panel.rb_normal.setChecked(True)
    w._open_v2v_from_vedit("talk", "", "")                              # senza video/audio: apre solo la modalita'
    assert panel.current_mode() == "talk"
    panel.rb_normal.setChecked(True)


def case_lipdub_translate_flow(w, spy):
    """Estrai e traduci (STT+LLM MOCK): riempie battuta/testo originale/lingua rilevata; riuso del testo originale; errori
    chiari senza toccare la battuta; senza lingua o video blocca."""
    from core import video_dub_translate as vdt
    t = _lipdub_tab(w, _lipdub_video("_vedit_lipdub_a.mp4"))
    calls = []
    real = vdt.run

    def fake_run(video, target, *, source_text="", video_seconds=0.0, log=None):
        calls.append((target, source_text, round(video_seconds, 1)))
        return {"text": "Hello everybody, welcome!", "original_text": "Ciao a tutti, benvenuti!", "original_language": "it",
                "source_language": "it", "backend": "mock", "fit": "ok", "est_s": 1.6, "warning": ""}
    vdt.run = fake_run
    try:
        t.lip_lang.setCurrentIndex(0)
        t._translate_start(sync=True)
        assert not calls and t.log.toPlainText().strip(), "senza lingua di destinazione: blocco"
        t.log.clear()
        t.lip_lang.setCurrentIndex(t.lip_lang.findData("en"))
        t.lip_line.setPlainText("vecchio testo")
        t._translate_start(sync=True)
        assert calls == [("en", "", 2.0)], calls
        assert t.lip_line.toPlainText() == "Hello everybody, welcome!" and t.lip_src_text.toPlainText().startswith("Ciao")
        assert "Italian" in t.lip_src_lang.text() and t.lip_translate.isEnabled()
        # secondo giro, altra lingua: riusa il testo originale (niente nuovo STT) anche se corretto a mano
        t.lip_src_text.setPlainText("Ciao a tutti!")
        t.lip_lang.setCurrentIndex(t.lip_lang.findData("es"))
        t._translate_start(sync=True)
        assert calls[-1] == ("es", "Ciao a tutti!", 2.0), calls
        # il testo e' modificabile e finisce nella cfg
        _ack_dev(False)
        _getter(w, _DIST)
        t.lip_line.setPlainText("Hola a todos")
        t._go()
        assert _cfg_of(w)["lipdub_line"] == "Hola a todos" and _cfg_of(w)["lipdub_lang"] == "es"
        # video cambiato: l'STT riparte (testo originale non riusato)
        t.set_video(_lipdub_video("_vedit_lipdub_b.mp4"))
        t._translate_start(sync=True)
        assert calls[-1][1] == "", calls
        # errore chiaro: la battuta resta com'e'
        def boom(*a, **k):
            raise vdt.DubTranslateError("vedit.dub.err_llm", "LLM locale non attivo")
        vdt.run = boom
        t.lip_line.setPlainText("resta cosi'")
        t.log.clear()
        t._translate_start(sync=True)
        assert t.lip_line.toPlainText() == "resta cosi'" and "⚠" in t.log.toPlainText() and t.lip_translate.isEnabled()
        # percorso con THREAD vero (stesso mock)
        vdt.run = fake_run
        t._translate_clicked()
        import time
        end = time.time() + 10
        while t.lip_line.toPlainText() != "Hello everybody, welcome!" and time.time() < end:
            H.app().processEvents(); time.sleep(0.05)
        assert t.lip_line.toPlainText() == "Hello everybody, welcome!", "il thread non ha consegnato il risultato"
    finally:
        vdt.run = real
    t2 = _tab(w)
    t2._video = ""
    t2._translate_start(sync=True)                                      # senza video: blocco


# ── HDR (fase 1c) ────────────────────────────────────────────────────────────────────────────────
def _hdr_tab(w, frames: int = 50):
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    t.set_video(_lipdub_video("_vedit_hdr_a.mp4", audio=True, frames=frames))
    _select_mode(w, "hdr")
    t.prompt.setPlainText("")
    return t


def case_hdr_fields(w, spy):
    """HDR: nessun prompt/enhance/forza/attenzione; campi propri (formato, esposizione, alta qualità, info); EXR half solo
    se il formato include l'EXR; risoluzione e «mantieni audio» visibili; tornando a refocus tutto si ripristina."""
    t = _hdr_tab(w)
    assert t.family.currentData() == "repair" and t.note.text(), "nota della modalità"
    for fld in (t.prompt, t.vedit_enhance, t.strength, t.attn, t.obj, t.lip_line):
        assert fld.isHidden(), f"nascosto in hdr: {fld}"
    for fld in (t.hdr_format, t.hdr_exposure, t.hdr_hq, t.hdr_info):
        assert not fld.isHidden(), fld
    assert not t.res_mode.parentWidget().isHidden() and not t.keep_audio.isHidden()
    assert [t.hdr_format.itemData(i) for i in range(t.hdr_format.count())] == ["pq", "both", "exr"]
    assert t.hdr_format.currentData() == "pq" and t.hdr_exr_half.isHidden(), "default HEVC PQ, EXR half nascosto"
    t.hdr_format.setCurrentIndex(t.hdr_format.findData("both"))
    assert not t.hdr_exr_half.isHidden()
    assert "_HDR_PQ" in t.hdr_info.text() and "_EXR" in t.hdr_info.text(), t.hdr_info.text()
    _select_mode(w, "refocus")
    for fld in (t.hdr_format, t.hdr_exposure, t.hdr_exr_half, t.hdr_hq, t.hdr_info):
        assert fld.isHidden(), fld
    assert not t.prompt.isHidden() and not t.vedit_enhance.isHidden() and not t.strength.isHidden() and not t.attn.isHidden()
    _select_mode(w, "hdr")
    assert t.prompt.isHidden() and not t.hdr_format.isHidden()


def case_hdr_cfg(w, spy):
    """HDR col distilled: parte con prompt vuoto; la cfg porta formato/esposizione/EXR half/alta qualità, model come
    scelto, audio mantenuto; i valori della GUI arrivano invariati."""
    _ack_dev(False)
    t = _hdr_tab(w)
    _getter(w, _DIST)
    t._go()
    assert spy.calls, f"nessuna richiesta accodata (log: {t.log.toPlainText()[-300:]})"
    cfg = _cfg_of(w)
    assert cfg["mode"] == "hdr" and cfg["prompt"] == "" and (cfg.get("edit_lora") or "") == "", cfg
    assert cfg["hdr_format"] == "pq" and cfg["hdr_exposure_ev"] == 0.0 and cfg["hdr_exr_half"] is True and cfg["hdr_hq"] is False
    assert cfg["model"] == _DIST["model"] and cfg["ckpt_state"] == "distilled" and cfg["keep_audio"] is True
    assert cfg["res_mode"] == "source"
    t.hdr_format.setCurrentIndex(t.hdr_format.findData("both"))
    t.hdr_exposure.setValue(1.5)
    t.hdr_exr_half.setChecked(False)
    t.hdr_hq.setChecked(True)
    t.keep_audio.setChecked(False)
    t._go()
    cfg = _cfg_of(w)
    assert (cfg["hdr_format"], cfg["hdr_exposure_ev"], cfg["hdr_exr_half"], cfg["hdr_hq"], cfg["keep_audio"]) == \
           ("both", 1.5, False, True, False), cfg
    t.keep_audio.setChecked(True)
    t.hdr_hq.setChecked(False)
    t.hdr_exposure.setValue(0.0)
    t.hdr_exr_half.setChecked(True)
    t.hdr_format.setCurrentIndex(t.hdr_format.findData("pq"))


def case_hdr_dev_warns_but_runs(w, spy):
    """HDR col dev: nessun blocco (distilled raccomandato, non richiesto): avviso con la distilled-LoRA 0.5 e parte;
    il checkpoint resta quello scelto."""
    _ack_dev(False)
    t = _hdr_tab(w)
    _getter(w, _DEV)
    asked = []
    t._ask_checkpoint = lambda sp, info, blocking=False: asked.append(sp.key) or "continue"
    t._go()
    assert asked == ["hdr"] and spy.calls, (asked, t.log.toPlainText()[-200:])
    cfg = _cfg_of(w)
    assert cfg["model"] == _DEV["model"] and cfg["ckpt_state"] == "dev"
    del t._ask_checkpoint


def case_hdr_weight_missing_dialog(w, spy):
    """LoRA HDR o file scene-emb assenti: avviso con «Apri tab Pesi modelli» (segnale), nessun accodamento, nessun download."""
    from core import video_edit_ic as vic
    _ack_dev(True)
    t = _hdr_tab(w)
    _getter(w, _DIST)
    orig = vic.missing_weight
    vic.missing_weight = lambda spec: "File scene-emb HDR mancante: X:\\nope.safetensors"
    try:
        asked, opened = [], []
        t.request_open_weights_tab.connect(lambda: opened.append(1))
        t._ask_missing_weight = lambda sp, msg: asked.append((sp.key, msg)) or "weights"
        before = len(spy.calls)
        t._go()
        assert [a[0] for a in asked] == ["hdr"] and "scene-emb" in asked[0][1] and opened == [1] and len(spy.calls) == before
    finally:
        vic.missing_weight = orig


def case_hdr_gate(w, spy):
    """cap.video_hdr_ic spenta: nessun accodamento (l'upsell lo mostra il gate)."""
    _ack_dev(True)
    t = _hdr_tab(w)
    _getter(w, _DIST)
    seen = []
    w._feature_gate = lambda fid: (seen.append(fid) or False)
    t._go()
    assert seen == ["cap.video_hdr_ic"] and not spy.calls, (seen, len(spy.calls))


def case_hdr_info_10bit_and_segments(w, spy):
    """Riga info: avviso se il sorgente e' gia' a 10 bit (non va dato HDR in ingresso); info segmenti per clip lunghi."""
    t = _hdr_tab(w, frames=200)
    assert "⚠" not in t.hdr_info.text(), t.hdr_info.text()
    assert "3" in t.seg_info.text(), f"200 frame con tetto 97 = 3 segmenti: {t.seg_info.text()!r}"
    from types import SimpleNamespace
    t._vinfo = SimpleNamespace(width=160, height=96, nb_frames=50, pix_fmt="yuv420p10le", has_audio=True, fps=25.0,
                               duration=2.0)
    t._refresh_hdr_info()
    assert "⚠" in t.hdr_info.text() and "yuv420p10le" in t.hdr_info.text(), t.hdr_info.text()


# ── Movimento da tracce (motion-track, fase 1d) ────────────────────────────────────────────────────
def _mt_tab(w, with_image: bool = True, name: str = "_vedit_mt_start.png"):
    """Tab in modalità motion_track con un'immagine di partenza vera (16:9) e nessuna traccia."""
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    _select_mode(w, "motion_track")
    t.mt_editor.clear_all()
    t.prompt.setPlainText("")
    if with_image:
        path = str(H.OUT_DIR / name)
        H.make_test_image(path, 640, 360)
        t.set_track_image(path)
    else:
        t.set_track_image("")
    return t


def _mt_click(t, x: float, y: float):
    """Clic sul punto (x,y) dello spazio di lavoro dell'editor, come farebbe l'utente."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    p = t.mt_editor.canvas.to_screen(x, y)
    QTest.mouseClick(t.mt_editor.canvas, Qt.LeftButton, Qt.NoModifier, QPoint(int(round(p.x())), int(round(p.y()))))


def _mt_draw(t):
    """Due tracce: una curva a 3 punti e un punto fermo."""
    t.mt_editor.canvas.resize(900, 560)                                    # geometria utile anche se la finestra non e' mostrata
    for xy in ((200, 400), (500, 250), (800, 330)):
        _mt_click(t, *xy)
    t.mt_editor.new_track()
    _mt_click(t, 300, 100)


def case_motion_fields(w, spy):
    """Motion-track: immagine+editor+dimensioni/durata/seme visibili; niente video obbligatorio, forza/oggetto/camera nascosti;
    famiglia «Camera / Movimento»; tornando a un'altra modalità i campi spariscono."""
    t = _mt_tab(w)
    assert t.family.currentData() == "camera" and t.note.text(), "famiglia e nota"
    for fld in (t.mt_editor, t.mt_info, t.mt_img_strength, t.mt_seed):
        assert not fld.isHidden(), fld
    for fld in (t.mt_img.parentWidget(), t.mt_w.parentWidget(), t.mt_seconds.parentWidget()):
        assert not fld.isHidden(), fld
    for fld in (t.strength, t.obj, t.cam_ref.parentWidget(), t.lip_line, t.keep_audio, t.cfg):
        assert fld.isHidden(), f"nascosto in motion_track: {fld}"
    assert not t.prompt.isHidden() and not t.attn.isHidden()
    assert (t.mt_w.value(), t.mt_h.value()) == (1024, 576), "dimensioni dall'aspetto dell'immagine 16:9"
    assert t.mt_fps.currentData() == 24 and abs(t.mt_seconds.value() - 5.0) < 1e-6 and t._mt_frames() == 121
    assert "121" in t.mt_info.text() and "5.0" in t.mt_info.text(), t.mt_info.text()
    assert t.mt_editor.model.width == 1024 and t.mt_editor.canvas.bg is not None, "sfondo dell'editor = l'immagine"
    assert t.mt_editor.canvas.toolTip() and "TEMPO" in t.mt_editor.canvas.toolTip(), "il tooltip spiega l'ordine = tempo"
    _select_mode(w, "refocus")
    for fld in (t.mt_editor, t.mt_info, t.mt_img_strength, t.mt_seed, t.mt_img.parentWidget()):
        assert fld.isHidden(), fld
    assert not t.strength.isHidden()
    _select_mode(w, "cameraman")
    assert t.mt_editor.isHidden() and not t.cam_ref.parentWidget().isHidden()
    assert [t.mode.itemData(i) for i in range(t.mode.count())] == ["cameraman", "motion_track"]


def case_motion_image_sources(w, spy):
    """Immagine di partenza: da file, dal canvas del Frame 1, dal primo fotogramma del video caricato; portrait → 576x1024."""
    t = _mt_tab(w, with_image=False)
    assert t.mt_editor.canvas.bg is None and "disegna" in t.mt_info.text().lower(), t.mt_info.text()
    # dal canvas
    cv = str(H.OUT_DIR / "_vedit_mt_canvas.png")
    H.make_test_image(cv, 400, 700)                                       # verticale
    t.set_canvas_image_getter(lambda: cv)
    t._mt_from_canvas()
    assert t.mt_img.text() == cv and (t.mt_w.value(), t.mt_h.value()) == (576, 1024), (t.mt_w.value(), t.mt_h.value())
    assert t.mt_editor.model.width == 576 and t.mt_editor.canvas.bg is not None
    # canvas vuoto: nessuna modifica, messaggio nel log
    t.set_canvas_image_getter(lambda: "")
    t._mt_from_canvas()
    assert t.mt_img.text() == cv and "canvas" in t.log.toPlainText().lower()
    # dal primo fotogramma del video (160x96 → aspetto 5:3 → lato corto 576)
    t.set_video(_lipdub_video("_vedit_mt_src.mp4", audio=False, frames=30))
    assert t.mt_img_frame.isEnabled()
    t._mt_from_video()
    assert t.mt_img.text().endswith("_frame0.png") and os.path.isfile(t.mt_img.text())
    assert t.mt_h.value() == 576 and t.mt_w.value() in (960, 896, 1024), (t.mt_w.value(), t.mt_h.value())
    # tracce già disegnate seguono il cambio di risoluzione/immagine (proporzionali)
    _mt_draw(t)
    x0 = t.mt_editor.model.splines[0][0][0]
    w0 = t.mt_editor.model.width
    t.mt_w.setValue(w0 + 64)
    assert abs(t.mt_editor.model.splines[0][0][0] - x0 * (w0 + 64) / w0) < 0.5
    t.set_video("")


def case_motion_cfg(w, spy):
    """Motion-track col distilled: la cfg porta immagine, tracce (JSON nel formato del nodo con size), 121 frame a 24 fps,
    dimensioni 1024x576, seme, forza immagine; model come scelto; prompt libero; nessun video richiesto."""
    import json
    _ack_dev(False)
    t = _mt_tab(w)
    _mt_draw(t)
    t.prompt.setPlainText("A paper boat drifts across a neon puddle while two drones circle above")
    t.mt_seed.setValue(1234)
    _getter(w, _DIST)
    t._go()
    assert spy.calls, f"nessuna richiesta accodata (log: {t.log.toPlainText()[-300:]})"
    cfg = _cfg_of(w)
    assert cfg["mode"] == "motion_track" and cfg["track_image"] == t.mt_img.text() == cfg["video"], cfg
    assert (cfg["width"], cfg["height"], cfg["fps"], cfg["num_frames"], cfg["seed"]) == (1024, 576, 24, 121, 1234), cfg
    assert cfg["strength"] == 1.0 and cfg["image_strength"] == 1.0 and cfg["tracks_size"] == [1024, 576]
    assert cfg["model"] == _DIST["model"] and cfg["ckpt_state"] == "distilled" and "paper boat" in cfg["prompt"]
    d = json.loads(cfg["tracks"])
    assert len(d["points_store"]) == 2 and len(d["points_store"][0]) == 3 and d["points_store"][1] == [{"x": d["points_store"][1][0]["x"], "y": d["points_store"][1][0]["y"]}]
    assert len(d["coordinates"][0]) == 121 and d["size"] == [1024, 576]
    from core import video_edit_ic as vic
    ij = vic.build_job(dict(cfg), str(H.OUT_DIR), 77, str(H.OUT_DIR / "_vedit_mt_out.mp4"))   # la stessa cfg e' valida per il job
    assert ij.params["control_type"] == "motion-track" and ij.params["num_frames"] == 121 and os.path.isfile(ij.params["control_video"])
    # durata 10 s a 24 fps = 241 frame (clip lunghi)
    t.mt_seconds.setValue(10.0)
    assert t._mt_frames() == 241 and "241" in t.mt_info.text() and "10.0" in t.mt_info.text()
    t._go()
    cfg = _cfg_of(w)
    assert cfg["num_frames"] == 241 and len(json.loads(cfg["tracks"])["coordinates"][0]) == 241
    t.mt_fps.setCurrentIndex(t.mt_fps.findData(30))
    assert t.mt_seconds.maximum() <= 8.6 and t._mt_frames() <= 257, "tetto sui frame: i secondi massimi scendono con gli fps"
    t.mt_fps.setCurrentIndex(t.mt_fps.findData(24)); t.mt_seconds.setValue(5.0)


def case_motion_validation(w, spy):
    """Senza immagine o senza tracce il pulsante blocca (nessun accodamento) con messaggio chiaro."""
    _ack_dev(True)
    t = _mt_tab(w, with_image=False)
    _getter(w, _DIST)
    before = len(spy.calls)
    t._go()
    assert len(spy.calls) == before and "immagine" in t.log.toPlainText().lower()
    t = _mt_tab(w)
    t.log.clear()
    t._go()
    assert len(spy.calls) == before and "traccia" in t.log.toPlainText().lower()
    _mt_draw(t)
    t.mt_editor.clear_all()
    t._go()
    assert len(spy.calls) == before


def case_motion_dev_weight_gate(w, spy):
    """Col dev: avviso (distilled-LoRA 0.5) e parte; peso assente → avviso + «Apri tab Pesi modelli», niente accodamento;
    gate spento → niente accodamento."""
    from core import video_edit_ic as vic
    _ack_dev(False)
    t = _mt_tab(w); _mt_draw(t)
    _getter(w, _DEV)
    asked = []
    t._ask_checkpoint = lambda sp, info, blocking=False: asked.append(sp.key) or "continue"
    t._go()
    assert asked == ["motion_track"] and spy.calls and _cfg_of(w)["model"] == _DEV["model"]
    del t._ask_checkpoint
    # peso mancante
    _ack_dev(True)
    _getter(w, _DIST)
    orig = vic.missing_weight
    vic.missing_weight = lambda spec: "Peso «motion-track» mancante: X:\\nope.safetensors"
    try:
        asked2, opened = [], []
        t.request_open_weights_tab.connect(lambda: opened.append(1))
        t._ask_missing_weight = lambda sp, msg: asked2.append((sp.key, msg)) or "weights"
        before = len(spy.calls)
        t._go()
        assert [a[0] for a in asked2] == ["motion_track"] and opened == [1] and len(spy.calls) == before
    finally:
        vic.missing_weight = orig
    # gate
    seen = []
    w._feature_gate = lambda fid: (seen.append(fid) or False)
    before = len(spy.calls)
    t._go()
    assert seen == ["cap.video_camera_ic"] and len(spy.calls) == before, (seen,)


def case_motion_video_tab_handoff(w, spy):
    """Tab Video → «Guida il movimento con tracce…»: apre Modifica Video AI in motion_track con l'immagine del canvas Frame 1
    già impostata (e senza immagine avvisa); il resto del tab Video non cambia."""
    img = str(H.OUT_DIR / "_vedit_mt_handoff.png")
    H.make_test_image(img, 800, 450)
    w.frame1_tab.canvas.load_image(img, fit_mode="canvas")
    w.tabs.setCurrentWidget(w.video_tab)
    n_before = (w.video_tab.length.value(), w.video_tab.fps.currentText() if hasattr(w.video_tab.fps, "currentText") else None)
    assert w.video_tab.motion_tracks_btn.isEnabled() and w.video_tab.motion_tracks_btn.minimumHeight() >= 32
    w.video_tab.motion_tracks_btn.click()
    t = _tab(w)
    assert w.tabs.currentWidget() is t and t.mode.currentData() == "motion_track" and t.family.currentData() == "camera"
    assert t.mt_img.text() == w.frame1_tab.image_path and t.mt_editor.canvas.bg is not None
    assert (w.video_tab.length.value(), w.video_tab.fps.currentText() if hasattr(w.video_tab.fps, "currentText") else None) == n_before
    # senza immagine nel canvas: apre comunque la modalità e lo dice nel log
    w.frame1_tab.canvas._image_path = ""
    t.set_track_image("")
    w.tabs.setCurrentWidget(w.video_tab)
    w.video_tab.motion_tracks_btn.click()
    assert t.mode.currentData() == "motion_track" and w.tabs.currentWidget() is t


def case_track_editor_in_tab_save_load(w, spy):
    """Editor dentro il tab: clic/annulla/nuova traccia/cancella; salva e ricarica le tracce (JSON del nodo) anche in un'altra
    risoluzione; carica il JSON del workflow ufficiale Lightricks, se presente."""
    import json
    t = _mt_tab(w)
    _mt_draw(t)
    m = t.mt_editor.model
    assert len(m.splines) == 2 and m.n_points == 4 and "4" in t.mt_editor.status.text()
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    QTest.mouseClick(t.mt_editor.canvas, Qt.RightButton, Qt.NoModifier, t.mt_editor.canvas.rect().center())
    assert m.n_points == 3
    t.mt_editor.btn_undo.click()
    assert m.n_points == 2 and len(m.splines) == 1
    f = str(H.OUT_DIR / "_vedit_mt_tracks.json")
    assert t.mt_editor.save_file(f)
    t.mt_editor.btn_clear.click()
    assert not t.mt_editor.has_tracks()
    t.mt_w.setValue(1280)                                                  # altra risoluzione: le tracce caricate si riscalano
    assert t.mt_editor.load_file(f) and t.mt_editor.has_tracks()
    d = json.load(open(f, encoding="utf-8"))
    assert abs(t.mt_editor.model.splines[0][0][0] - d["points_store"][0][0]["x"] * 1280 / 1024) < 0.6
    wf = r"E:\AI\workflows\LTX-2.3-curated\_Lightricks-official\example_workflows\2.3\LTX-2.3_ICLoRA_Motion_Track_Distilled.json"
    if os.path.exists(wf):
        t.mt_editor.btn_clear.click()
        assert t.mt_editor.load_file(wf) and len(t.mt_editor.model.splines) == 2
    assert not t.mt_editor.load_file(str(H.OUT_DIR / "_vedit_mt_nonesiste.json"))
    t.mt_editor.clear_all()


CASES = (
    [("vedit_family_modes", "Due livelli famiglia/modalità: contenuti, select_mode (lipdub e hdr compresi).",
      case_families_and_modes),
     ("vedit_fields_per_mode", "Campi abilitati/visibili per modalità (restyle, restauro, outpaint, cameraman).",
      case_fields_per_mode),
     ("vedit_repair_no_prompt_ok", "Restauro con prompt vuoto accoda; edit con prompt vuoto blocca.",
      case_repair_no_prompt_ok),
     ("vedit_ckpt_hint_states", "Riga checkpoint: distilled/dev/auto/2.5/non-LTX con icona e testo.",
      case_ckpt_hint_states),
     ("vedit_dev_warning_dialog", "Dialog a 3 scelte col dev: Annulla/Apri tab Video/Procedi; ack di sessione.",
      case_dev_warning_dialog_choices),
     ("vedit_dev_never_blocks", "Col dev nessuna modalità blocca (anche distilled=required): avviso e via; modello invariato.",
      case_dev_never_blocks_adds_dlora),
     ("vedit_dev_real_dialog", "Dialog dev reale: informazione «verrà aggiunta la LoRA distilled», Procedi → parte.",
      case_dev_real_dialog_warns_and_runs),
     ("vedit_dev_dlora_missing", "LoRA distilled assente: avviso con «Apri tab Pesi», nessun download, Procedi parte.",
      case_dev_dlora_missing_warns_with_weights_link),
     ("vedit_release25_blocks", "Checkpoint release 2.5: modalità IC-LoRA bloccate con messaggio.",
      case_release25_blocks),
     ("vedit_restyle_requires_image", "ReStyle: immagine di stile obbligatoria; prompt «Make it … style.».",
      case_restyle_requires_image),
     ("vedit_outpaint_canvas_margins", "Outpaint: canvas multipli di 64, area nuova, margini, avviso e blocchi.",
      case_outpaint_canvas_and_margins),
     ("vedit_outpaint_narrow_bar", "Outpaint: barra nuova < 15% del canvas avvisa (info+log); sorgente a scala 1.0.",
      case_outpaint_narrow_bar_warning),
     ("vedit_outpaint_formats_dir_feather", "Outpaint: tutti i formati, info «Aggiunge: sopra/sotto/lati», Estendi verso, sfumatura e barre minime in cfg.",
      case_outpaint_formats_direction_feather),
     ("vedit_outpaint_dark_autodetect", "Outpaint: sorgente scuro → gamma suggerito (e dichiarato nel log).",
      case_outpaint_dark_autodetect),
     ("vedit_cameraman_validation", "Cameraman: riferimento obbligatorio, <960x512 avvisa, video principale opzionale.",
      case_cameraman_validation),
     ("vedit_weight_missing_blocks", "Peso IC-LoRA assente: errore chiaro dal worker GUI, niente degrado.",
      case_weight_missing_blocks),
     ("vedit_gate_blocks_when_cap_off", "Cap spenta: nessun accodamento; le storiche non sono gated.",
      case_gate_blocks_when_cap_off),
     ("vedit_legacy_modes_pass_model", "Le modalità storiche inoltrano il checkpoint scelto.",
      case_legacy_modes_pass_model),
     ("vedit_ic_lora_hidden_from_lora_panel", "IC-LoRA escluse dal pannello LoRA normale; camera-control visibili.",
      case_ic_lora_hidden_from_lora_panel),
     ("vedit_lipdub_fields", "Lip dub: campi propri visibili, strength/attenzione/audio/enhance nascosti, ripristino.",
      case_lipdub_fields),
     ("vedit_lipdub_cfg", "Lip dub: cfg con battuta/lingua/scena, enhance spento, model come scelto, prompt libero.",
      case_lipdub_cfg),
     ("vedit_lipdub_validation", "Lip dub: blocchi (no battuta, no audio, troppo lungo); col dev avvisa e parte.",
      case_lipdub_validation),
     ("vedit_lipdub_info_fit", "Lip dub: riga info durata/stima parlato, avvisi battuta lunga/corta.",
      case_lipdub_info_fit),
     ("vedit_lipdub_weight_missing_dialog", "Lip dub: LoRA assente → avviso + «Apri tab Pesi» (segnale), nessun accodamento.",
      case_lipdub_weight_missing_dialog),
     ("vedit_lipdub_gate", "Lip dub: cap.video_lipdub spenta blocca l'accodamento.", case_lipdub_gate),
     ("vedit_lipdub_audio_fields", "Doppiaggio da audio (sperimentale): campi, banner, alternanza con da-testo.",
      case_lipdub_audio_fields),
     ("vedit_lipdub_audio_cfg_validation", "Da audio: file obbligatorio, avvisi durata, cfg, ultimo audio, video muto.",
      case_lipdub_audio_cfg_validation),
     ("vedit_lipdub_open_talk_sing", "Rimando a Talk/Canta: segnale e handler main_window (modalita', video, audio).",
      case_lipdub_open_talk_sing),
     ("vedit_lipdub_translate_flow", "Estrai e traduci (mock): riempie battuta, riuso testo, errori chiari, thread.",
      case_lipdub_translate_flow),
     ("vedit_hdr_fields", "HDR: niente prompt/forza/attenzione, campi propri (formato, esposizione, hq), EXR half condizionato.",
      case_hdr_fields),
     ("vedit_hdr_cfg", "HDR: cfg con formato/esposizione/EXR half/alta qualità, prompt vuoto ok, model come scelto.",
      case_hdr_cfg),
     ("vedit_hdr_dev_warns_runs", "HDR col dev: avviso (distilled-LoRA 0.5) e parte, checkpoint invariato.",
      case_hdr_dev_warns_but_runs),
     ("vedit_hdr_weight_missing_dialog", "HDR: LoRA/scene-emb assenti → avviso + «Apri tab Pesi», nessun accodamento.",
      case_hdr_weight_missing_dialog),
     ("vedit_hdr_gate", "HDR: cap.video_hdr_ic spenta blocca l'accodamento.", case_hdr_gate),
     ("vedit_hdr_info_10bit_segments", "HDR: avviso sorgente 10 bit e info segmenti per clip lunghi.",
      case_hdr_info_10bit_and_segments),
     ("vedit_motion_fields", "Motion-track: campi propri (immagine, editor, dimensioni, durata, seme), niente forza/camera, tooltip tempo.",
      case_motion_fields),
     ("vedit_motion_image_sources", "Motion-track: immagine da canvas / primo fotogramma del video / file; verticale 576x1024; riscala le tracce.",
      case_motion_image_sources),
     ("vedit_motion_cfg", "Motion-track: cfg con immagine, tracce JSON del nodo, 121 frame 24 fps, 241 a 10 s, seme, model come scelto.",
      case_motion_cfg),
     ("vedit_motion_validation", "Motion-track: senza immagine o senza tracce blocca con messaggio.", case_motion_validation),
     ("vedit_motion_dev_weight_gate", "Motion-track: dev avvisa e parte, peso mancante → Pesi modelli, gate cap spento blocca.",
      case_motion_dev_weight_gate),
     ("vedit_motion_video_tab_handoff", "Tab Video → «Guida il movimento con tracce…» apre la modalità con l'immagine del canvas.",
      case_motion_video_tab_handoff),
     ("vedit_track_editor_save_load", "Editor nel tab: clic/annulla/cancella, salva e ricarica JSON, altra risoluzione, workflow ufficiale.",
      case_track_editor_in_tab_save_load)]
    + [(f"vedit_newmode_{k}", f"Modalità IC-LoRA {k}: cfg emessa (kind a registro, model come scelto, campi).",
        _make_new_mode_case(k)) for k in _NEW_MODES]
)
