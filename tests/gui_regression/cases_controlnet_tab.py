"""Casi di regressione GUI-driven per il tab ControlNet (ui/tabs/controlnet_tab.py).

Il tab ControlNet NON ha un pulsante proprio: alimenta `on_generate()` del pannello video.
`controlnet_tab.get_values()` ritorna {controlnet_enabled, controlnet_ref_video, controlnet_type}
dove controlnet_type è l'INDICE UI 0=canny, 1=depth, 2=pose (TYPES).

Flusso (backend video = NATIVO in data/backend_config.json):
  on_generate() → _generate_video_native() → se controlnet attivo+video:
    core.controlnet_preprocess.preprocess_video(ref, kind, out, fps) e poi
    params["control_video"]/["control_type"]/["control_strength"], e POP di image_path.
  kind = {0:'canny',1:'depth',2:'pose'}[type_idx].

Regola di conflitto (core/conflict_rules.py): "cnet_needs_video" = se controlnet_enabled
allora controlnet_ref_video è OBBLIGATORIO (severity blocking) → on_generate si ferma.

I "modelli" di questo tab sono i 3 TIPI di ControlNet (canny/depth/pose): un caso per OGNUNO
per scoprire se "funziona solo un tipo" (mappatura indice→kind sbagliata).

Lo Spy cattura le richieste GUI→backend; il worker video passa da w._enqueue (via='enqueue',
params=worker.params). Il preprocess reale (cv2/DepthAnything/DWpose) è MOCKATO nei casi che
testano l'instradamento, così si verifica il WIRING della GUI senza GPU/codec.
"""
from __future__ import annotations

import os

import harness as H


# ───────────────────────── helper locali ─────────────────────────
def _dummy_video(name: str = "_cn_ref.mp4") -> str:
    """Crea un VERO mini-video (12 frame 64x64) come riferimento: nella run REALE il preprocess
    ControlNet lo legge davvero (l'header fittizio produceva una guida da 1 byte → worker exit 1)."""
    p = str(H.OUT_DIR / name)
    if not os.path.exists(p) or os.path.getsize(p) < 1024:
        import subprocess
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=64x64:rate=12:duration=1", "-pix_fmt", "yuv420p", p, "-y"],
                       check=False)
        if not os.path.exists(p) or os.path.getsize(p) < 1024:   # fallback senza ffmpeg: cv2
            import cv2
            import numpy as np
            vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 64))
            for i in range(12):
                fr = np.full((64, 64, 3), i * 20 % 255, np.uint8)
                cv2.circle(fr, (32, 32), 6 + i, (255, 255, 255), 2)
                vw.write(fr)
            vw.release()
    return p


def _enable_cn(w, type_idx: int, with_video: bool = True):
    """Attiva il ControlNet sul tab, sceglie il tipo e (opz.) imposta il video di riferimento."""
    cn = w.controlnet_tab
    cn.enable.setChecked(True)
    cn.type.setCurrentIndex(type_idx)
    cn._video = _dummy_video() if with_video else None
    return cn


def _prep_video_prompt(w, text="a person walking, cinematic"):
    """Precondizione minima per on_generate: prompt nel tab Video + tab Video attivo."""
    w.video_tab.prompt.setPlainText(text)
    try:
        w.tabs.setCurrentWidget(w.video_tab)
    except Exception:
        pass


class _MockPreprocess:
    """Context manager: sostituisce controlnet_preprocess.preprocess_video con uno stub che
    crea il file di output e ritorna il path (nessun cv2/modello). Ripristina all'uscita."""
    def __init__(self, fail: bool = False):
        self.fail = fail
        self._orig = None

    def __enter__(self):
        from core import controlnet_preprocess as cnp
        self._orig = cnp.preprocess_video

        def stub(video_path, kind, out_video, fps=None):
            if self.fail:
                raise RuntimeError("preprocess mock failure")
            os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
            with open(out_video, "wb") as fh:
                fh.write(b"\x00")
            return out_video
        cnp.preprocess_video = stub
        return self

    def __exit__(self, *a):
        from core import controlnet_preprocess as cnp
        cnp.preprocess_video = self._orig
        return False


def _last_video_call(spy):
    """Ritorna l'ultima richiesta col profilo video (params con num_frames), o None."""
    for c in reversed(spy.calls):
        if isinstance(c.get("params"), dict) and c["params"].get("num_frames"):
            return c
    return spy.calls[-1] if spy.calls else None


# ───────────────────────── UI: toggle / combo / get_values ─────────────────────────
def case_cn_enable_toggle(w, spy):
    """Toggle 'Abilita ControlNet': lo stato del checkbox si riflette in get_values (no crash)."""
    cn = w.controlnet_tab
    cn.enable.setChecked(False)
    assert cn.get_values()["controlnet_enabled"] is False, "enable off non riflesso"
    cn.enable.setChecked(True)
    assert cn.get_values()["controlnet_enabled"] is True, "enable on non riflesso"


def case_cn_type_combo_all(w, spy):
    """Combo tipo ControlNet: selezionare OGNI voce (canny/depth/pose) aggiorna l'hint senza crash,
    e l'indice è riportato in get_values.controlnet_type."""
    cn = w.controlnet_tab
    assert cn.type.count() == 3, f"attesi 3 tipi (canny/depth/pose), trovati {cn.type.count()}"
    for i in range(cn.type.count()):
        cn.type.setCurrentIndex(i)
        cn._on_type_changed(i)                    # aggiorna l'etichetta hint
        assert cn.get_values()["controlnet_type"] == i, f"indice tipo non propagato: {i}"
        assert cn.type_hint.text(), f"hint vuoto per tipo {i}"


def case_cn_get_values_keys(w, spy):
    """get_values() espone esattamente le 3 chiavi che alimentano on_generate."""
    v = w.controlnet_tab.get_values()
    for k in ("controlnet_enabled", "controlnet_ref_video", "controlnet_type"):
        assert k in v, f"chiave mancante in get_values: {k}"


def case_cn_pick_video_sets_path(w, spy):
    """Selezione manuale del video di riferimento: get_values riporta il path scelto."""
    cn = w.controlnet_tab
    cn._video = _dummy_video()
    assert cn.get_values()["controlnet_ref_video"] == cn._video, "ref_video non riportato in get_values"


# ───────────────────────── instradamento per OGNI tipo (modello) ─────────────────────────
def _run_type(w, spy, type_idx, expected_kind):
    _prep_video_prompt(w)
    _enable_cn(w, type_idx, with_video=True)
    with _MockPreprocess():
        w.on_generate()
    assert spy.calls, f"controlnet {expected_kind}: nessuna richiesta inviata da on_generate"
    p = _last_video_call(spy)["params"]
    assert p.get("control_video"), f"{expected_kind}: control_video mancante nei params: {p}"
    assert p.get("control_type") == expected_kind, \
        f"{expected_kind}: control_type errato = {p.get('control_type')} (mappatura indice→kind rotta)"
    assert "control_strength" in p, f"{expected_kind}: control_strength mancante: {p}"


def case_cn_generate_canny(w, spy):
    """Tipo CANNY (indice 0): on_generate instrada al nativo con control_type='canny' + control_video."""
    _run_type(w, spy, 0, "canny")


def case_cn_generate_depth(w, spy):
    """Tipo DEPTH (indice 1): on_generate instrada al nativo con control_type='depth' + control_video."""
    _run_type(w, spy, 1, "depth")


def case_cn_generate_pose(w, spy):
    """Tipo POSE (indice 2): on_generate instrada al nativo con control_type='pose' + control_video."""
    _run_type(w, spy, 2, "pose")


# ───────────────────────── precondizioni / conflitti / fallback ─────────────────────────
def case_cn_enabled_without_video_blocked(w, spy):
    """ControlNet attivo SENZA video di riferimento = conflitto blocking (cnet_needs_video):
    on_generate deve fermarsi e NON inviare alcuna richiesta video."""
    _prep_video_prompt(w)
    _enable_cn(w, 0, with_video=False)
    with _MockPreprocess():
        w.on_generate()
    vc = _last_video_call(spy)
    assert vc is None or not vc["params"].get("control_video"), \
        f"atteso blocco (no ref video) ma è partita una richiesta: {spy.calls}"


def case_cn_disabled_normal_video(w, spy):
    """ControlNet DISABILITATO: la generazione video parte normalmente, SENZA param control_*."""
    _prep_video_prompt(w)
    w.controlnet_tab.enable.setChecked(False)
    w.on_generate()
    assert spy.calls, "video normale: nessuna richiesta inviata"
    p = _last_video_call(spy)["params"]
    assert not p.get("control_video") and not p.get("control_type"), \
        f"controlnet disabilitato ma control_* presenti: {p}"


def case_cn_replaces_image(w, spy):
    """ControlNet attivo + immagine caricata: la guida video SOSTITUISCE l'immagine
    (image_path deve essere rimosso dai params, control_video presente)."""
    H.load_canvas_image(w)
    _prep_video_prompt(w)
    _enable_cn(w, 0, with_video=True)
    with _MockPreprocess():
        w.on_generate()
    assert spy.calls, "controlnet+immagine: nessuna richiesta inviata"
    p = _last_video_call(spy)["params"]
    assert p.get("control_video"), f"control_video mancante: {p}"
    assert not p.get("image_path"), f"image_path doveva essere rimosso dalla guida controlnet: {p}"


def case_cn_preprocess_failure_fallback(w, spy):
    """Se il preprocess del video di controllo fallisce, on_generate NON deve crashare:
    ripiega sulla generazione senza guida (nessun control_video, ma la richiesta parte)."""
    _prep_video_prompt(w)
    _enable_cn(w, 1, with_video=True)             # depth → forza il ramo modello
    with _MockPreprocess(fail=True):
        w.on_generate()
    assert spy.calls, "fallback: nessuna richiesta inviata dopo errore preprocess"
    p = _last_video_call(spy)["params"]
    assert not p.get("control_video"), f"dopo fallimento preprocess control_video non deve esserci: {p}"


CASES = [
    ("controlnet_enable_toggle",
     "Toggle Abilita ControlNet: lo stato si riflette in get_values (no crash).",
     case_cn_enable_toggle),
    ("controlnet_type_combo_all",
     "Combo tipo: ogni voce (canny/depth/pose) aggiorna l'hint e propaga l'indice in get_values.",
     case_cn_type_combo_all),
    ("controlnet_get_values_keys",
     "get_values espone controlnet_enabled/ref_video/type per alimentare on_generate.",
     case_cn_get_values_keys),
    ("controlnet_pick_video_sets_path",
     "Video di riferimento selezionato: get_values riporta il path.",
     case_cn_pick_video_sets_path),
    ("controlnet_generate_canny",
     "Tipo canny: on_generate invia control_type='canny' + control_video al nativo.",
     case_cn_generate_canny),
    ("controlnet_generate_depth",
     "Tipo depth: on_generate invia control_type='depth' + control_video al nativo.",
     case_cn_generate_depth),
    ("controlnet_generate_pose",
     "Tipo pose: on_generate invia control_type='pose' + control_video al nativo.",
     case_cn_generate_pose),
    ("controlnet_enabled_without_video_blocked",
     "ControlNet attivo senza video di riferimento = conflitto blocking: nessuna richiesta.",
     case_cn_enabled_without_video_blocked),
    ("controlnet_disabled_normal_video",
     "ControlNet off: il video parte normalmente, senza param control_*.",
     case_cn_disabled_normal_video),
    ("controlnet_replaces_image",
     "ControlNet + immagine: la guida video sostituisce l'immagine (image_path rimosso).",
     case_cn_replaces_image),
    ("controlnet_preprocess_failure_fallback",
     "Preprocess fallito: nessun crash, generazione senza guida (no control_video).",
     case_cn_preprocess_failure_fallback),
]
