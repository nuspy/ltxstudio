"""Casi di regressione GUI-driven per il tab "Prolunga video" (VideoExtendTab).

Azione coperta: MainWindow.generate_video_extend (segnale tab.request_extend).
Backend attivo per "video" = NATIVO (data/backend_config.json) → il percorso reale è
MainWindow._extend_video_native, che:
  - legge video_extend_tab.get_config() (video, prompt, add_seconds, seg/overlap/context/steps),
  - estrae l'ULTIMO frame del video sorgente con ffmpeg,
  - eredita width/height/fps dal video_tab e steps/cfg/seed dal video_tab.sampler_seed,
  - accoda un _NativeLtx2Worker via _enqueue (catturato dallo Spy: worker.params).

NOTA: questo tab NON ha un dropdown modello proprio — il modello/diffusion è quello del tab Video
(video_tab). Quindi non esiste "un caso per ogni modello" qui; la copertura modelli è nel tab Video.
Si copre invece: routing nativo/comfy, precondizioni (video assente, prompt assente), mappatura dei
parametri (num_frames in funzione di add_seconds*fps, chunking per i lunghi), opzioni del form e API.

Ogni caso è no-crash + assert mirati su spy.calls. I setup incerti sono in try/except DENTRO il caso.
"""
from __future__ import annotations

import os
import subprocess
import time

import harness as H


# ── helper: crea un piccolo mp4 reale (serve a ffmpeg per estrarre l'ultimo frame) ──
def _make_test_video(seconds: float = 1.0, fps: int = 24, w: int = 512, h: int = 512) -> str | None:
    """Genera un mp4 di test da un'immagine fissa via ffmpeg. Ritorna il path o None se ffmpeg manca."""
    img = str(H.OUT_DIR / "_ext_src_frame.png")
    if not os.path.exists(img):
        H.make_test_image(img, w, h)
    out = str(H.OUT_DIR / "_ext_src_video.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-loop", "1", "-i", img, "-t", str(seconds),
             "-r", str(fps), "-pix_fmt", "yuv420p", "-c:v", "libx264", out, "-y"],
            check=False, timeout=60)
    except Exception:                                          # noqa: BLE001
        return None
    return out if os.path.exists(out) else None


def _set_video(w, path: str):
    """Imposta il video sorgente sia via API tab sia sul cfg (robustezza)."""
    w.tabs.setCurrentWidget(w.video_extend_tab)
    w.video_extend_tab.set_video(path)


# ───────────────────────── PRECONDIZIONI: nessun video ─────────────────────────
def case_extend_no_video_no_backend_call(w, spy):
    """Senza video sorgente: la GUI NON deve inviare nulla al backend (solo warning)."""
    t = w.video_extend_tab
    t._video = None                                           # nessun video caricato
    t.prompt.setPlainText("continue the scene, the camera pans right")
    w.tabs.setCurrentWidget(t)
    w.generate_video_extend()
    assert not spy.calls, f"extend senza video NON deve chiamare il backend: {spy.calls}"


# ───────────────────────── PRECONDIZIONI: nessun prompt ─────────────────────────
def case_extend_no_prompt_no_backend_call(w, spy):
    """Con video ma prompt vuoto: la GUI NON deve inviare nulla (l'estensione richiede un prompt)."""
    vid = _make_test_video()
    if not vid:
        return                                                # ffmpeg assente: salta (no-crash)
    _set_video(w, vid)
    w.video_extend_tab.prompt.setPlainText("")               # prompt vuoto
    w.generate_video_extend()
    assert not spy.calls, f"extend senza prompt NON deve chiamare il backend: {spy.calls}"


# ───────────────────────── HAPPY PATH (nativo) ─────────────────────────
def case_extend_native_happy(w, spy):
    """Video + prompt → percorso nativo: deve accodare un worker con image_path(ultimo frame),
    prompt, num_frames, width, height, fps."""
    vid = _make_test_video()
    if not vid:
        return
    _set_video(w, vid)
    w.video_extend_tab.prompt.setPlainText("the subject keeps walking forward")
    w.generate_video_extend()
    assert spy.calls, "extend (nativo) non ha accodato alcuna richiesta"
    p = spy.calls[-1]["params"]
    assert p.get("prompt"), f"prompt mancante nei params: {p}"
    assert p.get("image_path") and os.path.exists(p["image_path"]), \
        f"image_path (ultimo frame) mancante/non estratto: {p.get('image_path')}"
    assert p.get("num_frames") and p.get("width") and p.get("height") and p.get("fps"), \
        f"params video incompleti: {p}"


# ───────────────────────── add_seconds → num_frames ─────────────────────────
def case_extend_add_seconds_drives_num_frames(w, spy):
    """num_frames deve scalare con add_seconds*fps (formula //8*8+1). add_seconds maggiore ⇒ più frame."""
    vid = _make_test_video()
    if not vid:
        return
    _set_video(w, vid)
    t = w.video_extend_tab
    t.prompt.setPlainText("extend the shot")
    t.add_seconds.setValue(2.0)
    w.generate_video_extend()
    assert spy.calls, "extend: nessuna richiesta (add_seconds=2)"
    nf_small = spy.calls[-1]["params"].get("num_frames")
    spy.calls.clear()
    t.add_seconds.setValue(8.0)
    w.generate_video_extend()
    assert spy.calls, "extend: nessuna richiesta (add_seconds=8)"
    nf_big = spy.calls[-1]["params"].get("num_frames")
    assert nf_big and nf_small and nf_big > nf_small, \
        f"num_frames non scala con add_seconds: {nf_small} -> {nf_big}"
    assert (nf_big - 1) % 8 == 0, f"num_frames non allineato a 8k+1: {nf_big}"


# ───────────────────────── chunking sui video lunghi ─────────────────────────
def case_extend_long_sets_chunk_frames(w, spy):
    """Per estensioni lunghe (num_frames>150) la GUI deve impostare chunk_frames per il memory-bound."""
    vid = _make_test_video()
    if not vid:
        return
    _set_video(w, vid)
    t = w.video_extend_tab
    t.prompt.setPlainText("a very long continuation")
    # fps del video_tab ~24/25 → 10s * fps > 150 frame
    t.add_seconds.setValue(10.0)
    w.generate_video_extend()
    assert spy.calls, "extend lungo: nessuna richiesta"
    p = spy.calls[-1]["params"]
    if p.get("num_frames", 0) > 150:
        assert p.get("chunk_frames"), f"video lungo senza chunk_frames: {p}"


# ───────────────────────── inerita seed/steps/cfg dal video_tab ─────────────────────────
def case_extend_inherits_sampler_from_video_tab(w, spy):
    """L'estensione eredita steps/guidance/seed dal video_tab.sampler_seed (stesso spazio visivo)."""
    vid = _make_test_video()
    if not vid:
        return
    _set_video(w, vid)
    w.video_extend_tab.prompt.setPlainText("continue")
    w.video_tab.sampler_seed.steps.setValue(7)
    w.video_tab.sampler_seed.seed.setValue(12345)
    w.generate_video_extend()
    assert spy.calls, "extend: nessuna richiesta"
    p = spy.calls[-1]["params"]
    assert p.get("steps") == 7, f"steps non ereditati dal video_tab: {p.get('steps')}"
    assert p.get("seed") == 12345, f"seed non ereditato dal video_tab: {p.get('seed')}"


# ───────────────────────── opzioni del form (get_config) ─────────────────────────
def case_extend_config_reflects_form_options(w, spy):
    """get_config() deve riflettere TUTTE le opzioni del form: add_seconds, seg_seconds, overlap,
    context, steps (anche se nel percorso nativo non tutte vengono usate, devono essere esposte)."""
    t = w.video_extend_tab
    t.add_seconds.setValue(30.0)
    t.seg_seconds.setValue(12.0)
    t.overlap.setValue(10)
    t.context.setValue(24)
    t.steps.setValue(15)
    cfg = t.get_config()
    assert cfg["add_seconds"] == 30.0 and cfg["seg_seconds"] == 12.0, f"durate non mappate: {cfg}"
    assert cfg["overlap_frames"] == 10, f"overlap non mappato: {cfg}"
    assert cfg["context_frames"] == 24, f"context non mappato: {cfg}"
    assert cfg["steps"] == 15, f"steps non mappato: {cfg}"


# ───────────────────────── stima segmenti (single vs multi) ─────────────────────────
def case_extend_segment_estimate_single_vs_multi(w, spy):
    """<=20s ⇒ 1 spezzone (label 'single'); >20s ⇒ multi-spezzone con conteggio. Verifica no-crash
    e che l'etichetta cambi tra i due regimi (logica _update_seg_estimate)."""
    t = w.video_extend_tab
    t.add_seconds.setValue(10.0)
    t._update_seg_estimate()
    single_txt = t.seg_estimate.text()
    t.add_seconds.setValue(60.0)
    t.seg_seconds.setValue(15.0)
    t._update_seg_estimate()
    multi_txt = t.seg_estimate.text()
    assert single_txt != multi_txt, \
        f"la stima segmenti non distingue single/multi: {single_txt!r} == {multi_txt!r}"


# ───────────────────────── set_video API ─────────────────────────
def case_extend_set_video_updates_label(w, spy):
    """set_video() deve memorizzare il path e aggiornare l'etichetta col nome file (no-crash)."""
    t = w.video_extend_tab
    t.set_video(r"E:\some\dir\my_clip.mp4")
    assert t._video and t._video.endswith("my_clip.mp4"), f"_video non impostato: {t._video}"
    assert "my_clip.mp4" in t.video_lbl.text(), f"etichetta non aggiornata: {t.video_lbl.text()}"


# ───────────────────────── routing COMFY (fallback non-nativo) ─────────────────────────
def case_extend_routes_to_comfy_when_not_native(w, spy):
    """Se il backend video NON è nativo e il driver ComfyUI è su, generate_video_extend deve passare
    dal percorso ComfyUI (extender.build + driver.submit). Si verifica che raggiunga driver.submit."""
    vid = _make_test_video()
    if not vid:
        return
    try:
        from core import backend
        orig = backend.is_native
        backend.is_native = lambda module: False             # forza fallback comfy
        w.driver.is_up = lambda: True                        # driver "su"
        # neutralizza le dipendenze da file/template del builder comfy
        w.extender.plan_segments = lambda *a, **k: (1, 18.0)
        w.extender.build = lambda *a, **k: {"1": {"class_type": "x", "inputs": {}}}
        w._current_pid = None
        w._inpaint_pid = None
        _set_video(w, vid)
        w.video_extend_tab.prompt.setPlainText("continue via comfy path")
        w.generate_video_extend()
        assert spy.calls and spy.calls[-1]["via"] == "comfy", \
            f"percorso comfy non ha chiamato driver.submit: {spy.calls}"
    finally:
        try:
            backend.is_native = orig
        except Exception:                                     # noqa: BLE001
            pass


CASES = [
    ("video_extend_no_video_no_call",
     "Senza video sorgente: la GUI non invia nulla al backend (solo warning).",
     case_extend_no_video_no_backend_call),
    ("video_extend_no_prompt_no_call",
     "Con video ma prompt vuoto: la GUI non invia nulla (l'estensione richiede un prompt).",
     case_extend_no_prompt_no_backend_call),
    ("video_extend_native_happy",
     "Video+prompt → nativo: accoda worker con image_path(ultimo frame)+prompt+num_frames+w/h/fps.",
     case_extend_native_happy),
    ("video_extend_add_seconds_drives_num_frames",
     "num_frames scala con add_seconds*fps (allineato 8k+1).",
     case_extend_add_seconds_drives_num_frames),
    ("video_extend_long_sets_chunk_frames",
     "Estensione lunga (num_frames>150) → la GUI imposta chunk_frames (memory-bound).",
     case_extend_long_sets_chunk_frames),
    ("video_extend_inherits_sampler_from_video_tab",
     "L'estensione eredita steps/seed dal video_tab.sampler_seed.",
     case_extend_inherits_sampler_from_video_tab),
    ("video_extend_config_reflects_form_options",
     "get_config() espone add_seconds/seg_seconds/overlap/context/steps dal form.",
     case_extend_config_reflects_form_options),
    ("video_extend_segment_estimate_single_vs_multi",
     "Stima segmenti: <=20s single, >20s multi (etichetta cambia).",
     case_extend_segment_estimate_single_vs_multi),
    ("video_extend_set_video_updates_label",
     "set_video() memorizza il path e aggiorna l'etichetta col nome file.",
     case_extend_set_video_updates_label),
    ("video_extend_routes_to_comfy_when_not_native",
     "Fallback non-nativo + driver su → percorso ComfyUI (driver.submit).",
     case_extend_routes_to_comfy_when_not_native),
]
