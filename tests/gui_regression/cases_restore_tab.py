"""Casi di regressione GUI-driven per il tab «Restauro» (RestoreTab, sotto «Migliora qualità → Foto»).

Copre le QUATTRO operazioni del tab + ogni opzione/toggle rilevante, e genera un caso per OGNI
modello concreto dei due dropdown (upscale ESRGAN e FLUX.2 colorizzazione) così da scoprire se
"funziona solo un modello".

Flusso reale esercitato:
  restore_tab._on_process()  →  emette request_restore(cfg)  →  MainWindow.generate_restore(cfg)
    • operation="restore"             → backend NATIVO (ESRGAN/spandrel) via _start_native_upscale → _enqueue
    • operation="colorize"            → ComfyUI: restorer.build_colorize  → driver.submit
    • operation="restore_and_colorize"→ ComfyUI: restorer.build_restore_and_colorize → driver.submit
    • operation="faceswap"            → NATIVO (InsightFace) via _faceswap_photo → _enqueue

Lo Spy dell'harness cattura le richieste GUI→backend SENZA eseguirle (no GPU/rete).
NB: per i path NATIVI serve native_driver.is_up()==True; per i path ComfyUI serve driver.is_up()==True
e un upload fittizio: questi gate (server offline nei test) vengono neutralizzati DENTRO il caso.
"""
from __future__ import annotations

import os

import harness as H


# ── inventario reale dei modelli (per generare 1 caso per modello) ───────────────
def _real_upscale_models() -> list[str]:
    try:
        from core import offline_inventory as oi
        ms = oi.list_models("upscale_models")
        return ms or ["RealESRGAN_x4plus.pth"]
    except Exception:                                       # noqa: BLE001
        return ["RealESRGAN_x4plus.pth"]


def _real_flux2_models() -> list[str]:
    try:
        from core import offline_inventory as oi
        diff = oi.list_models("diffusion_models")
        fl = [m for m in diff if "flux2" in m.lower() or "flux_2" in m.lower() or "flux 2" in m.lower()]
        return fl or ["Flux\\FLUX_2_Dev_FP8_Mixed_Scaled.safetensors"]
    except Exception:                                       # noqa: BLE001
        return ["Flux\\FLUX_2_Dev_FP8_Mixed_Scaled.safetensors"]


# ── helper di precondizione ──────────────────────────────────────────────────────
def _test_image(name: str = "_restore_in.png", w: int = 640, h: int = 480) -> str:
    path = str(H.OUT_DIR / name)
    if not os.path.exists(path):
        H.make_test_image(path, w, h)
    return path


def _force_servers_up(w):
    """Neutralizza i gate 'server offline' (nei test non gira né ComfyUI né l'InferenceServer).
    - native_driver.is_up()  → True   (path NATIVO: restore/faceswap)
    - driver.is_up()         → True   (generate_restore lo richiede per QUALSIASI operazione)
    - driver.upload_image()  → fittizio (evita la POST reale a ComfyUI in colorize/both)
    """
    try:
        from core import native_driver
        native_driver.is_up = lambda: True
    except Exception:                                       # noqa: BLE001
        pass
    try:
        w.driver.is_up = lambda: True
        w.driver.upload_image = lambda p: {"name": os.path.basename(p), "subfolder": ""}
    except Exception:                                       # noqa: BLE001
        pass


def _goto(w):
    """Porta la GUI sul tab Restauro (no-crash dello switch + realismo)."""
    try:
        w.tabs.setCurrentWidget(w.quality_tab)
        w.quality_tab.setCurrentWidget(w.restore_tab)
    except Exception:                                       # noqa: BLE001
        pass


def _select_op(rt, op: str):
    """Seleziona il radio dell'operazione (scatena _on_op_changed via toggled)."""
    {"restore": rt.radio_restore, "colorize": rt.radio_colorize,
     "both": rt.radio_both, "faceswap": rt.radio_faceswap}[op].setChecked(True)


# ───────────────────────── RESTORE (ESRGAN nativo) ─────────────────────────
def case_restore_basic(w, spy):
    """Restauro base: la GUI deve mandare al backend NATIVO module='upscale' con arch='upscale',
    model, image_path e target_scale."""
    rt = w.restore_tab
    _goto(w); _force_servers_up(w)
    rt.set_upscale_models(_real_upscale_models())
    rt._input_path = _test_image()
    _select_op(rt, "restore")
    rt._on_process()
    assert spy.calls, "restore: nessuna richiesta inviata al backend"
    c = spy.calls[-1]; p = c["params"]
    assert c["module"] == "upscale", f"module atteso 'upscale': {c}"
    assert p.get("arch") == "upscale", f"arch atteso 'upscale': {p}"
    assert p.get("model"), f"model upscale mancante: {p}"
    assert p.get("image_path"), f"image_path mancante: {p}"
    assert p.get("target_scale"), f"target_scale mancante: {p}"


def _make_restore_model_case(model: str):
    def case(w, spy):
        rt = w.restore_tab
        _goto(w); _force_servers_up(w)
        rt.set_upscale_models([model])            # combo = solo questo modello
        rt._input_path = _test_image()
        _select_op(rt, "restore")
        rt._on_process()
        assert spy.calls, f"restore[{model}]: nessuna richiesta inviata"
        p = spy.calls[-1]["params"]
        assert p.get("model") == model, f"modello inviato != selezionato: {p.get('model')!r} != {model!r}"
    return case


def case_restore_scale_4x(w, spy):
    """Restauro a 4×: target_scale deve propagarsi (4.0) nei params nativi."""
    rt = w.restore_tab
    _goto(w); _force_servers_up(w)
    rt.set_upscale_models(_real_upscale_models())
    rt._input_path = _test_image()
    _select_op(rt, "restore")
    rt.scale_spin.setValue(4.0)
    rt._on_process()
    assert spy.calls, "restore 4x: nessuna richiesta"
    assert abs(float(spy.calls[-1]["params"].get("target_scale", 0)) - 4.0) < 1e-6, \
        f"target_scale 4.0 non propagato: {spy.calls[-1]['params']}"


def case_restore_scale_1x(w, spy):
    """Restauro a 1× (estremo basso del range): la GUI non deve crashare e deve inviare scale=1.0."""
    rt = w.restore_tab
    _goto(w); _force_servers_up(w)
    rt.set_upscale_models(_real_upscale_models())
    rt._input_path = _test_image()
    _select_op(rt, "restore")
    rt.scale_spin.setValue(1.0)
    rt._on_process()
    assert spy.calls, "restore 1x: nessuna richiesta"
    assert abs(float(spy.calls[-1]["params"].get("target_scale", 0)) - 1.0) < 1e-6


def case_restore_denoise_on(w, spy):
    """Restauro con denoise attivo: lo spinbox forza viene abilitato e la richiesta parte comunque
    (il backend nativo upscale ignora il denoise, ma la GUI non deve crashare)."""
    rt = w.restore_tab
    _goto(w); _force_servers_up(w)
    rt.set_upscale_models(_real_upscale_models())
    rt._input_path = _test_image()
    _select_op(rt, "restore")
    assert not rt.denoise_str.isEnabled(), "denoise_str dovrebbe nascere disabilitato"
    rt.denoise_chk.setChecked(True)
    assert rt.denoise_str.isEnabled(), "denoise_str deve abilitarsi col check (interlock rotto)"
    rt.denoise_str.setValue(0.35)
    rt._on_process()
    assert spy.calls, "restore+denoise: nessuna richiesta inviata"


# ───────────────────────── COLORIZE (FLUX.2 ComfyUI) ─────────────────────────
def case_colorize_basic(w, spy):
    """Colorizzazione: con un modello FLUX.2 selezionato la GUI deve costruire il prompt e fare
    driver.submit (ramo ComfyUI)."""
    rt = w.restore_tab
    _goto(w); _force_servers_up(w)
    rt.set_flux2_models(_real_flux2_models())
    rt._input_path = _test_image()
    rt.subject_edit.setText("a vintage portrait of a woman")
    rt.style_edit.setText("warm Kodachrome film colors")
    rt.color_steps.setValue(18)
    _select_op(rt, "colorize")
    rt._on_process()
    assert spy.calls, "colorize: nessuna richiesta inviata (atteso driver.submit ComfyUI)"
    assert spy.calls[-1]["via"] == "comfy", f"colorize deve passare da ComfyUI: {spy.calls[-1]}"


def _make_colorize_model_case(model: str):
    def case(w, spy):
        rt = w.restore_tab
        _goto(w); _force_servers_up(w)
        rt.set_flux2_models([model])             # combo FLUX.2 = solo questo
        # verifica che il filtro flux2 non lo abbia scartato (altrimenti resta il placeholder)
        assert rt.flux2_model.currentText() == model, \
            f"il modello FLUX.2 {model!r} non è stato accettato dal combo: {rt.flux2_model.currentText()!r}"
        rt._input_path = _test_image()
        rt.subject_edit.setText("old photo")
        _select_op(rt, "colorize")
        rt._on_process()
        assert spy.calls, f"colorize[{model}]: nessuna richiesta inviata"
        assert spy.calls[-1]["via"] == "comfy", f"colorize[{model}] non è andato su ComfyUI: {spy.calls[-1]}"
    return case


def case_colorize_no_model_sends_placeholder(w, spy):
    """REGRESSIONE: senza modelli FLUX.2 il combo mostra un PLACEHOLDER non-vuoto; la GUI lo invia
    comunque come nome modello (build_colorize non lo intercetta perché non è ''). Caso che
    documenta il comportamento: parte una submit con un 'modello' fittizio invece di bloccare."""
    rt = w.restore_tab
    _goto(w); _force_servers_up(w)
    rt.set_flux2_models([])                       # nessun modello → placeholder
    rt._input_path = _test_image()
    _select_op(rt, "colorize")
    placeholder = rt.flux2_model.currentText()
    assert placeholder, "atteso un placeholder non-vuoto quando non ci sono modelli FLUX.2"
    rt._on_process()
    # no-crash è sufficiente; se è partita una submit, è la submit 'fittizia' documentata sopra.


# ───────────────────────── RESTORE + COLORIZE (combinato) ─────────────────────────
def case_restore_and_colorize(w, spy):
    """Restauro+Colorizzazione: serve sia upscale_model sia flux2_model; ramo ComfyUI (driver.submit)."""
    rt = w.restore_tab
    _goto(w); _force_servers_up(w)
    rt.set_upscale_models(_real_upscale_models())
    rt.set_flux2_models(_real_flux2_models())
    rt._input_path = _test_image()
    rt.subject_edit.setText("a family on a beach")
    _select_op(rt, "both")
    rt._on_process()
    assert spy.calls, "restore+colorize: nessuna richiesta inviata"
    assert spy.calls[-1]["via"] == "comfy", f"restore+colorize deve passare da ComfyUI: {spy.calls[-1]}"


# ───────────────────────── FACE-SWAP foto (InsightFace nativo) ─────────────────────────
def case_faceswap_ok(w, spy):
    """Face-swap foto: con foto target + foto sorgente valide la GUI deve accodare il worker nativo
    (_enqueue), senza passare da ComfyUI."""
    rt = w.restore_tab
    _goto(w); _force_servers_up(w)
    rt._input_path = _test_image("_restore_target.png")
    rt.fs_source.setText(_test_image("_restore_face.png", 320, 320))
    _select_op(rt, "faceswap")
    rt._on_process()
    assert spy.calls, "faceswap: nessun worker accodato (atteso _enqueue nativo)"
    assert spy.calls[-1]["via"] == "enqueue", f"faceswap deve usare _enqueue nativo: {spy.calls[-1]}"


def case_faceswap_missing_source(w, spy):
    """Face-swap senza foto sorgente: la GUI deve avvisare e NON accodare nulla (no-crash)."""
    rt = w.restore_tab
    _goto(w); _force_servers_up(w)
    rt._input_path = _test_image("_restore_target.png")
    rt.fs_source.setText("")                      # nessuna sorgente
    _select_op(rt, "faceswap")
    rt._on_process()
    assert not spy.calls, f"faceswap senza sorgente non deve inviare nulla: {spy.calls}"


def case_faceswap_missing_target(w, spy):
    """Face-swap senza foto target valida: avviso + nessun invio."""
    rt = w.restore_tab
    _goto(w); _force_servers_up(w)
    rt._input_path = ""                           # nessuna immagine target
    rt.fs_source.setText(_test_image("_restore_face.png", 320, 320))
    _select_op(rt, "faceswap")
    rt._on_process()
    assert not spy.calls, f"faceswap senza target non deve inviare nulla: {spy.calls}"


# ───────────────────────── precondizioni input mancanti ─────────────────────────
def case_restore_missing_image(w, spy):
    """Restauro senza immagine selezionata: avviso 'nessuna immagine' e nessun invio (no-crash)."""
    rt = w.restore_tab
    _goto(w); _force_servers_up(w)
    rt.set_upscale_models(_real_upscale_models())
    rt._input_path = ""
    _select_op(rt, "restore")
    rt._on_process()
    assert not spy.calls, f"restore senza immagine non deve inviare nulla: {spy.calls}"


def case_use_canvas_input(w, spy):
    """Pulsante 'Usa immagine dal canvas': imposta image_path='<from_canvas>'; generate_restore deve
    risolverlo sul frame1_tab.image_path e procedere col restauro nativo."""
    rt = w.restore_tab
    _goto(w); _force_servers_up(w)
    rt.set_upscale_models(_real_upscale_models())
    # carica un'immagine nel canvas del tab Frame1 e usala come sorgente
    H.load_canvas_image(w)
    rt._use_canvas_image()                        # → _input_path = "<from_canvas>"
    assert rt._input_path == "<from_canvas>", "il bottone canvas non ha impostato il placeholder"
    _select_op(rt, "restore")
    rt._on_process()
    assert spy.calls, "restore da canvas: nessuna richiesta inviata"
    p = spy.calls[-1]["params"]
    assert p.get("image_path") and p["image_path"] != "<from_canvas>", \
        f"il placeholder canvas non è stato risolto in un path reale: {p}"


# ───────────────────────── logica UI (abilitazioni gruppi) ─────────────────────────
def case_op_enable_states(w, spy):
    """_on_op_changed: per ogni operazione i gruppi giusti sono abilitati/disabilitati."""
    rt = w.restore_tab
    _goto(w)
    _select_op(rt, "restore")
    assert rt.restore_grp.isEnabled() and not rt.color_grp.isEnabled() and not rt.faceswap_grp.isEnabled(), \
        "stato gruppi errato per 'restore'"
    _select_op(rt, "colorize")
    assert rt.color_grp.isEnabled() and not rt.restore_grp.isEnabled() and not rt.faceswap_grp.isEnabled(), \
        "stato gruppi errato per 'colorize'"
    _select_op(rt, "both")
    assert rt.color_grp.isEnabled() and rt.restore_grp.isEnabled() and not rt.faceswap_grp.isEnabled(), \
        "stato gruppi errato per 'restore_and_colorize'"
    _select_op(rt, "faceswap")
    assert rt.faceswap_grp.isEnabled() and not rt.color_grp.isEnabled() and not rt.restore_grp.isEnabled(), \
        "stato gruppi errato per 'faceswap'"


def case_get_config_roundtrip(w, spy):
    """get_config() deve riflettere i widget (operation + scelte) senza cliccare Processa."""
    rt = w.restore_tab
    _goto(w)
    rt.set_upscale_models(_real_upscale_models())
    rt._input_path = _test_image()
    rt.scale_spin.setValue(3.0)
    _select_op(rt, "restore")
    cfg = rt.get_config()
    assert cfg["operation"] == "restore", f"operation errata: {cfg}"
    assert cfg["image_path"], "image_path mancante in get_config"
    assert abs(cfg["target_scale"] - 3.0) < 1e-6, f"target_scale non riflesso: {cfg}"
    assert "upscale_model" in cfg and "flux2_model" in cfg, f"chiavi config mancanti: {cfg}"


# ───────────────────────── registro casi ─────────────────────────
CASES = [
    ("restore_basic_native",
     "Restauro: la GUI invia al backend nativo module=upscale + arch + model + image_path + target_scale.",
     case_restore_basic),
    ("restore_scale_4x",
     "Restauro 4×: target_scale=4.0 propagato nei params nativi.",
     case_restore_scale_4x),
    ("restore_scale_1x",
     "Restauro 1× (estremo del range): no-crash, target_scale=1.0 inviato.",
     case_restore_scale_1x),
    ("restore_denoise_on",
     "Restauro con denoise: interlock checkbox→spinbox + richiesta inviata senza crash.",
     case_restore_denoise_on),
    ("colorize_basic",
     "Colorizzazione: con FLUX.2 selezionato costruisce il prompt e fa driver.submit (ComfyUI).",
     case_colorize_basic),
    ("colorize_no_model_sends_placeholder",
     "REGRESSIONE: senza FLUX.2 il combo invia il PLACEHOLDER come modello invece di bloccare.",
     case_colorize_no_model_sends_placeholder),
    ("restore_and_colorize",
     "Restauro+Colorizzazione: usa upscale_model + flux2_model, ramo ComfyUI (driver.submit).",
     case_restore_and_colorize),
    ("faceswap_ok",
     "Face-swap foto: target+sorgente validi → accoda worker nativo (_enqueue), no ComfyUI.",
     case_faceswap_ok),
    ("faceswap_missing_source",
     "Face-swap senza foto sorgente: avviso + nessun invio (no-crash).",
     case_faceswap_missing_source),
    ("faceswap_missing_target",
     "Face-swap senza foto target: avviso + nessun invio (no-crash).",
     case_faceswap_missing_target),
    ("restore_missing_image",
     "Restauro senza immagine: avviso 'nessuna immagine' + nessun invio (no-crash).",
     case_restore_missing_image),
    ("use_canvas_input",
     "Usa immagine dal canvas: '<from_canvas>' risolto sul path reale del frame1 + restauro inviato.",
     case_use_canvas_input),
    ("op_enable_states",
     "Radio operazione: abilita/disabilita i gruppi restauro/colore/faceswap correttamente.",
     case_op_enable_states),
    ("get_config_roundtrip",
     "get_config() riflette operation + target_scale + image_path senza cliccare Processa.",
     case_get_config_roundtrip),
]

# Un caso per OGNI modello upscale concreto (scopre 'funziona solo un modello').
for _m in _real_upscale_models():
    _safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in _m)
    CASES.append((f"restore_model_{_safe}",
                  f"Restauro col modello upscale '{_m}': la GUI invia ESATTAMENTE quel model al backend.",
                  _make_restore_model_case(_m)))

# Un caso per OGNI modello FLUX.2 concreto (colorizzazione).
for _m in _real_flux2_models():
    _safe = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in _m)
    CASES.append((f"colorize_model_{_safe}",
                  f"Colorizzazione col modello FLUX.2 '{_m}': accettato dal combo + submit ComfyUI.",
                  _make_colorize_model_case(_m)))
