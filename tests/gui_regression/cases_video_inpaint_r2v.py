"""Casi GUI del task 8: Inpaint video con FOTO di riferimento (LoRA r2v) nel tab «Modifica Video AI».

Come cases_video_edit_ai_tab.py: l'azione (VideoEditAITab._go → request_edit → MainWindow._gen_video_edit_ai → _enqueue
intercettato dallo Spy) e, dove serve, l'esecuzione sincrona di `_VideoEditAIWorker.run` con i subprocess FINTI (nessuna GPU,
nessun SAM2): si verificano i `params` che arriverebbero al worker LTX.
"""
from __future__ import annotations

import json
import os
import subprocess
import types

import harness as H

from core import video_inpaint as vinp


def _tab(w):
    return w.video_edit_ai_tab


def _prep(w, *, use_ref: bool, photo: bool = True, obj: str = "umbrella"):
    from ui.tabs.video_edit_ai_tab import VideoEditAITab
    VideoEditAITab._dev_ack_session = True
    t = _tab(w)
    w.tabs.setCurrentWidget(t)
    path = str(H.OUT_DIR / "_vedit_test_input.mp4")
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(b"\x00\x00\x00\x18ftypmp42")
    t.set_video(path)
    t.select_mode("inpaint")
    t.obj.setText(obj)
    t.prompt.setPlainText("a glowing jellyfish umbrella in neon rain")
    img = str(H.OUT_DIR / "_vedit_r2v_photo.png")
    H.make_test_image(img, 640, 640)
    t.ref_img.setText(img if photo else "")
    t.inp_use_ref.setChecked(use_ref)
    return t, img


def _cfg_of(w) -> dict:
    wk = getattr(w, "_edit_ai_worker", None)
    assert wk is not None, "nessun _VideoEditAIWorker: l'azione non e' arrivata a _gen_video_edit_ai"
    return dict(wk.cfg)


def case_r2v_off_uses_t2v_even_with_stale_photo(w, spy):
    """Casella spenta: la foto residua nel campo (condiviso con replace_char/restyle) NON attiva r2v: LoRA t2v come sempre."""
    t, _img = _prep(w, use_ref=False)
    t._go()
    assert spy.calls, "nessuna richiesta accodata"
    cfg = _cfg_of(w)
    assert "inpaint_masked_t2v" in cfg["edit_lora"], cfg["edit_lora"]
    assert not cfg.get("ref_image") and not cfg.get("r2v"), cfg


def case_r2v_on_uses_r2v_lora_and_photo(w, spy):
    """Casella accesa + foto: LoRA r2v, foto e forza in cfg, flag r2v."""
    t, img = _prep(w, use_ref=True)
    t.ref_strength.setValue(0.65)
    t._go()
    assert spy.calls, "nessuna richiesta accodata"
    cfg = _cfg_of(w)
    assert "inpaint_masked_r2v_rank32" in cfg["edit_lora"], cfg["edit_lora"]
    assert cfg["ref_image"] == img and cfg.get("r2v") is True
    assert abs(float(cfg["ref_strength"]) - 0.65) < 1e-6


def case_r2v_on_without_photo_blocks(w, spy):
    t, _img = _prep(w, use_ref=True, photo=False)
    t._go()
    assert not spy.calls, "senza foto la casella r2v deve bloccare l'azione"


def case_r2v_ui_visibility(w, spy):
    """La casella compare SOLO in inpaint; riga foto/forza solo con la casella accesa (e restano per replace_char/restyle)."""
    t, _img = _prep(w, use_ref=False)
    # QWidget.isVisible() dipende dalla finestra mostrata: si guarda isHidden() del campo e l'abilitazione
    assert not t.inp_use_ref.isHidden() and t.inp_use_ref.isEnabled()
    assert t.ref_img.parentWidget().isHidden() and t.ref_strength.isHidden()
    t.inp_use_ref.setChecked(True)
    assert not t.ref_img.parentWidget().isHidden() and t.ref_img.isEnabled() and not t.ref_strength.isHidden()
    for key in ("edit", "anime2real", "outpaint", "motion_track"):
        t.select_mode(key)
        assert t.inp_use_ref.isHidden(), f"la casella r2v non deve comparire in {key}"
        assert t.ref_img.parentWidget().isHidden(), f"la riga foto non deve restare visibile in {key}"
    t.select_mode("replace_char")
    assert not t.ref_img.parentWidget().isHidden() and t.ref_img.isEnabled(), "replace_char deve avere ancora la sua foto"
    assert t.inp_use_ref.isHidden()
    t.select_mode("restyle")
    assert not t.ref_img.parentWidget().isHidden() and t.inp_use_ref.isHidden()


def case_r2v_missing_weight_asks_and_never_downloads(w, spy):
    """Peso r2v assente: avviso con «Apri tab Pesi modelli», nessun accodamento, nessun download."""
    t, _img = _prep(w, use_ref=True)
    orig = vinp.missing_weight
    vinp.missing_weight = lambda cfg: "Peso r2v mancante: X:\\nope.safetensors"
    try:
        asked, opened = [], []
        t.request_open_weights_tab.connect(lambda: opened.append(1))
        t._ask_missing_r2v = lambda msg: asked.append(msg) or "weights"
        t._go()
        assert asked and opened == [1] and not spy.calls, (asked, opened, spy.calls)
        asked.clear()
        t._ask_missing_r2v = lambda msg: asked.append(msg) or "cancel"
        t._go()
        assert asked and not spy.calls
    finally:
        vinp.missing_weight = orig


def case_r2v_gate_blocks_only_r2v(w, spy):
    """Gate d'edizione proprio per r2v: negato, il r2v non parte; l'inpaint a parole (t2v) resta libero."""
    t, _img = _prep(w, use_ref=True)
    seen = []
    w._feature_gate = lambda fid: (seen.append(fid) or fid != "cap.video_inpaint_r2v")
    t._go()
    assert "cap.video_inpaint_r2v" in seen and not spy.calls, (seen, spy.calls)
    t.inp_use_ref.setChecked(False)
    seen.clear()
    t._go()
    assert spy.calls and "cap.video_inpaint_r2v" not in seen, (seen, spy.calls)


def _run_worker(w, *, use_ref: bool, nb_frames: int):
    """Esegue `_VideoEditAIWorker.run` in modo sincrono con subprocess FINTI; ritorna (params del worker LTX, esito done)."""
    from ui import main_window as mw
    t, img = _prep(w, use_ref=use_ref)
    t._go()
    cfg = _cfg_of(w)
    cfg.update(width=1280, height=704, fps=25, num_frames=nb_frames)
    wk = mw._VideoEditAIWorker(cfg, str(H.OUT_DIR))
    done, captured = [], {}
    wk.done.connect(lambda r: done.append(r))
    real_run = subprocess.run

    def fake_run(cmd, *a, **k):
        cmd = [str(c) for c in cmd]
        if cmd and cmd[0].lower().endswith("python.exe") and any(c.endswith("ltx2_infer.py") for c in cmd):
            captured.update(json.load(open(cmd[-1], encoding="utf-8")))
            return types.SimpleNamespace(stdout="", stderr="finto", returncode=1)
        if cmd and cmd[0].lower().endswith("python.exe"):                      # SAM2 (video_segment.py)
            return types.SimpleNamespace(stdout="GUIDE:g.mp4\n", stderr="", returncode=0)
        return types.SimpleNamespace(stdout="", stderr="", returncode=1)        # ffprobe & co.

    # NIENTE /free reali: mem_coordinator.free_and_wait e free_video_vram parlano coi server GPU persistenti (:8189-8192) e
    # potrebbero scaricare modelli di un'altra sessione (campagna live in corso): qui si stubbano.
    from core import ltx_video_job as _lvj
    from core import mem_coordinator as _mc
    _o_fw, _o_fv = _mc.free_and_wait, _lvj.free_video_vram
    _mc.free_and_wait = lambda *a, **k: 0
    _lvj.free_video_vram = lambda *a, **k: 0
    subprocess.run = fake_run
    try:
        wk.run()
    finally:
        subprocess.run = real_run
        _mc.free_and_wait, _lvj.free_video_vram = _o_fw, _o_fv
    return captured, done, img


def case_r2v_worker_params_short(w, spy):
    cap, done, img = _run_worker(w, use_ref=True, nb_frames=97)
    assert cap, f"il worker LTX non e' stato invocato: {done}"
    assert "inpaint_masked_r2v_rank32" in cap["edit_lora"]
    assert os.path.basename(cap["image_path"]).startswith("_r2vref_"), cap["image_path"]       # foto 1:1 su 16:9 = con bordi
    assert os.path.isfile(cap["image_path"]) and cap["r2v_every_segment"] is True and cap["ref_strength"] > 0
    assert cap["inpaint_masks_dir"].endswith("masks") and cap["control_video"].endswith("_vedit_test_input.mp4")
    assert "chunk_frames" not in cap, "97 frame: nessuna segmentazione"


def case_r2v_worker_params_long_segments_keep_photo(w, spy):
    cap, done, img = _run_worker(w, use_ref=True, nb_frames=249)
    assert cap["chunk_frames"] == vinp.SEGMENT_FRAMES and cap["overlap_frames"] == vinp.SEGMENT_OVERLAP
    assert cap["image_path"] and cap["r2v_every_segment"] is True and cap["num_frames"] == 249


def case_t2v_worker_params_unchanged(w, spy):
    """Senza foto: stessi params di prima (LoRA t2v, nessuna immagine), + segmentazione solo se il clip e' lungo."""
    cap, done, _img = _run_worker(w, use_ref=False, nb_frames=97)
    assert "inpaint_masked_t2v_rank128" in cap["edit_lora"]
    assert "image_path" not in cap and "r2v_mode" not in cap
    assert cap["cfg_scale"] == 6.0 and cap["mask_dilate"] == 24 and cap["control_strength"] == 0.4


CASES = [
    ("vinpaint_r2v_off_uses_t2v", "r2v: casella spenta = LoRA t2v anche con foto residua nel campo.",
     case_r2v_off_uses_t2v_even_with_stale_photo),
    ("vinpaint_r2v_on_uses_r2v", "r2v: casella accesa + foto = LoRA r2v, foto e forza in cfg.",
     case_r2v_on_uses_r2v_lora_and_photo),
    ("vinpaint_r2v_on_without_photo_blocks", "r2v: casella accesa senza foto blocca.",
     case_r2v_on_without_photo_blocks),
    ("vinpaint_r2v_ui_visibility", "r2v: casella/riga foto compaiono solo dove devono (inpaint; replace_char/restyle intatti).",
     case_r2v_ui_visibility),
    ("vinpaint_r2v_missing_weight", "r2v: peso assente = avviso + Apri tab Pesi, niente download.",
     case_r2v_missing_weight_asks_and_never_downloads),
    ("vinpaint_r2v_gate", "r2v: gate d'edizione proprio, il t2v resta libero.", case_r2v_gate_blocks_only_r2v),
    ("vinpaint_r2v_worker_params_short", "r2v: params del worker LTX (clip corto).", case_r2v_worker_params_short),
    ("vinpaint_r2v_worker_params_long", "r2v: clip lungo = segmenti, la foto vale per ogni segmento.",
     case_r2v_worker_params_long_segments_keep_photo),
    ("vinpaint_t2v_worker_params_unchanged", "t2v: params del worker invariati rispetto a prima.",
     case_t2v_worker_params_unchanged),
]
