"""Casi di regressione GUI-driven per il tab Animazione/Mocap (anim_tab).

Particolarita': le azioni di questo tab NON passano dal backend nativo/Comfy (lo Spy
non le cattura). Vanno invece verso due servizi locali:
  - mocap  → start_mocap()  → core.mocap_service (worker webcam→BVH in venv isolato);
  - retarget → run_retarget() → core.blender_service (Blender headless, retarget.py).
Quindi qui catturiamo le richieste monkeypatchando, DENTRO il caso, i punti d'ingresso
di quei servizi (mai a import-time): cosi' verifichiamo che la GUI passi parametri
corretti senza avviare webcam/Blender reali.

Pattern identico a cases.py: CASES = [(nome, descrizione, fn), ...]; ogni fn(w, spy).
"""
from __future__ import annotations

import os
from pathlib import Path

import harness as H


# ───────────────────────── helper locali ─────────────────────────
def _scratch(name: str) -> str:
    """Crea un file fittizio (esistente su disco) per le precondizioni rig/anim."""
    p = H.OUT_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(b"dummy")
    return str(p)


def _set_combo_text(combo, text: str) -> bool:
    """Seleziona nel combo la voce con testo == text (case-insensitive)."""
    for i in range(combo.count()):
        if combo.itemText(i).strip().lower() == text.strip().lower():
            combo.setCurrentIndex(i)
            return True
    return False


def _prep_rig_anim(w, *, rig: bool = True, anim: bool = True):
    """Imposta rig (.glb) e/o anim (.bvh) come file realmente esistenti sul canvas del tab."""
    a = w.anim_tab
    if rig:
        a._rig_path = _make_existing(_scratch("_anim_rig.glb"))
        a.rig_edit.setText(a._rig_path)
    if anim:
        a._anim_path = _make_existing(_scratch("_anim_clip.bvh"))
        a.anim_edit.setText(a._anim_path)


def _make_existing(path: str) -> str:
    if not os.path.exists(path):
        Path(path).write_bytes(b"dummy")
    return path


def _capture_blender(w):
    """Sostituisce _start_blender + blender_available per catturare la richiesta di retarget
    SENZA avviare Blender. Ritorna la lista che raccoglie (script, args)."""
    captured: list = []
    w._start_blender = lambda script, args, on_ok, on_err, timeout=900: captured.append(
        {"script": script, "args": dict(args)})
    try:
        from core import blender_service
        blender_service.blender_available = lambda: True
    except Exception:                                          # noqa: BLE001
        pass
    return captured


def _capture_mocap(w, *, env_ready: bool = True):
    """Patcha mocap_env_ready + w.mocap.start per catturare l'avvio cattura senza webcam.
    Ritorna la lista che raccoglie i kwargs passati a mocap.start."""
    captured: list = []
    try:
        from core import mocap_service
        mocap_service.mocap_env_ready = lambda: env_ready
    except Exception:                                          # noqa: BLE001
        pass
    w.mocap.start = lambda out_bvh, preview, camera=0, **k: (
        captured.append({"out": out_bvh, "preview": preview, "camera": camera}) or True)
    return captured


# ───────────────────────── MOCAP: avvio cattura webcam ─────────────────────────
def case_mocap_start_emits(w, spy):
    """Avvia cattura: con env pronto e camera scelta, la GUI deve chiamare mocap.start con
    l'indice camera selezionato e mettere il tab in stato 'in cattura'."""
    a = w.anim_tab
    a.camera.setValue(2)
    cap = _capture_mocap(w, env_ready=True)
    a._on_start()                                              # emette request_mocap_start → start_mocap
    assert cap, "mocap.start non e' stato invocato"
    assert cap[-1]["camera"] == 2, f"indice camera errato: {cap[-1]}"
    # stato 'in cattura': start disabilitato, stop abilitato
    assert not a.start_btn.isEnabled() and a.stop_btn.isEnabled(), "set_capturing(True) non applicato"


def case_mocap_start_env_missing(w, spy):
    """Env mocap assente: la GUI non deve crashare ne' avviare la cattura (avviso + ritorno)."""
    a = w.anim_tab
    cap = _capture_mocap(w, env_ready=False)
    a._on_start()
    assert not cap, "con env assente non deve partire mocap.start"
    assert a.start_btn.isEnabled(), "start_btn deve restare abilitato (cattura non avviata)"


def case_mocap_stop(w, spy):
    """Stop cattura: emette request_mocap_stop → mocap.stop; nessun crash anche senza cattura attiva."""
    stopped = []
    w.mocap.stop = lambda: stopped.append(True)
    w.anim_tab._on_stop()
    assert stopped, "mocap.stop non invocato"


# ───────────────────────── RETARGET: applica animazione al rig ─────────────────────────
def _retarget_format_case(fmt: str):
    def _case(w, spy):
        _prep_rig_anim(w)
        a = w.anim_tab
        assert _set_combo_text(a.fmt, fmt), f"formato '{fmt}' assente nel combo"
        cap = _capture_blender(w)
        a._on_apply()                                          # emette request_retarget → run_retarget
        assert cap, f"retarget {fmt}: nessuna chiamata a Blender (run_retarget non ha proceduto)"
        args = cap[-1]["args"]
        assert cap[-1]["script"] == "retarget.py", f"script errato: {cap[-1]}"
        assert args.get("model") and args.get("anim"), f"model/anim mancanti: {args}"
        assert args.get("format") == fmt, f"format non propagato: {args}"
        out = str(args.get("output", ""))
        assert out.lower().endswith("." + fmt), f"estensione output errata per {fmt}: {out}"
    return _case


def _retarget_template_case(tmpl: str):
    def _case(w, spy):
        _prep_rig_anim(w)
        a = w.anim_tab
        assert _set_combo_text(a.template, tmpl), f"template '{tmpl}' assente nel combo"
        cap = _capture_blender(w)
        a._on_apply()
        assert cap, f"retarget template {tmpl}: nessuna chiamata a Blender"
        assert cap[-1]["args"].get("template") == tmpl, f"template non propagato: {cap[-1]['args']}"
    return _case


def case_retarget_missing_rig(w, spy):
    """Apply senza rig: avviso + nessun avvio Blender, nessun crash."""
    _prep_rig_anim(w, rig=False, anim=True)
    cap = _capture_blender(w)
    w.anim_tab._on_apply()
    assert not cap, "senza rig non deve avviare il retarget"


def case_retarget_missing_anim(w, spy):
    """Apply senza animazione: avviso + nessun avvio Blender, nessun crash."""
    _prep_rig_anim(w, rig=True, anim=False)
    cap = _capture_blender(w)
    w.anim_tab._on_apply()
    assert not cap, "senza animazione non deve avviare il retarget"


def case_retarget_missing_files(w, spy):
    """Apply con path inesistenti (impostati ma non su disco): avviso, nessun Blender, no-crash."""
    a = w.anim_tab
    a._rig_path = str(H.OUT_DIR / "_nope_rig.glb")
    a._anim_path = str(H.OUT_DIR / "_nope_clip.bvh")
    cap = _capture_blender(w)
    a._on_apply()
    assert not cap, "con file inesistenti non deve avviare il retarget"


# ───────────────────────── UI: rig sorgente, import, preview ─────────────────────────
def case_use_last_rig(w, spy):
    """'Usa ultimo rig': set_last_rig abilita il bottone e _use_last_rig popola il path rig."""
    a = w.anim_tab
    last = _make_existing(_scratch("_anim_last_rig.glb"))
    a.set_last_rig(last)
    assert a.last_btn.isEnabled(), "set_last_rig non ha abilitato il bottone"
    a._use_last_rig()
    assert a._rig_path == last and a.rig_edit.text() == last, "ultimo rig non applicato"


def case_browse_rig_cancel(w, spy):
    """Browse rig annullato (QFileDialog mockato vuoto): nessun crash, path invariato."""
    a = w.anim_tab
    a._browse_rig()                                            # getOpenFileName → ("","")
    assert a._rig_path == "", "browse annullato non deve impostare un path"


def case_browse_anim_cancel(w, spy):
    """Browse animazione annullato: nessun crash, path invariato."""
    a = w.anim_tab
    a._browse_anim()
    assert a._anim_path == "", "browse animazione annullato non deve impostare un path"


def case_set_anim_source(w, spy):
    """set_anim (es. BVH catturato dalla webcam) popola la sorgente animazione."""
    a = w.anim_tab
    bvh = _make_existing(_scratch("_anim_captured.bvh"))
    a.set_anim(bvh)
    assert a._anim_path == bvh and a.anim_edit.text() == bvh, "set_anim non ha aggiornato la sorgente"


def case_set_preview(w, spy):
    """set_preview con un PNG valido non deve crashare e aggiorna il tooltip con il n. frame."""
    a = w.anim_tab
    png = str(H.OUT_DIR / "_anim_preview.png")
    H.make_test_image(png, 320, 240)
    a.set_preview(png, 42)                                     # no-crash + pixmap impostata


def case_capturing_toggle(w, spy):
    """set_capturing alterna lo stato dei bottoni start/stop coerentemente."""
    a = w.anim_tab
    a.set_capturing(True)
    assert not a.start_btn.isEnabled() and a.stop_btn.isEnabled(), "stato 'in cattura' errato"
    a.set_capturing(False)
    assert a.start_btn.isEnabled() and not a.stop_btn.isEnabled(), "stato 'fermo' errato"


def case_src_tabs_switch(w, spy):
    """Cambio sorgente webcam→import: nessun crash (sotto-tab interno al tab anim)."""
    a = w.anim_tab
    n = a.src_tabs.count()
    assert n >= 2, f"attese 2 sorgenti (webcam/import), trovate {n}"
    for i in range(n):
        a.src_tabs.setCurrentIndex(i)


def case_format_combo_values(w, spy):
    """Regressione opzioni: il combo formato deve esporre fbx/glb/bvh."""
    a = w.anim_tab
    vals = {a.fmt.itemText(i).lower() for i in range(a.fmt.count())}
    assert {"fbx", "glb", "bvh"} <= vals, f"formati export mancanti: {vals}"


def case_template_combo_values(w, spy):
    """Regressione opzioni: il combo template deve esporre auto/Mixamo/Unreal Engine 5."""
    a = w.anim_tab
    vals = {a.template.itemText(i).lower() for i in range(a.template.count())}
    assert {"auto", "mixamo", "unreal engine 5"} <= vals, f"template mancanti: {vals}"


def case_select_anim_tab(w, spy):
    """Selezione del tab Animazione nella finestra principale: nessun crash."""
    w.tabs.setCurrentWidget(w.anim_tab)
    assert w.tabs.currentWidget() is w.anim_tab


# ───────────────────────── registro ─────────────────────────
CASES = [
    ("anim_mocap_start_emits",
     "Avvio cattura webcam: chiama mocap.start con l'indice camera scelto + stato 'in cattura'.",
     case_mocap_start_emits),
    ("anim_mocap_start_env_missing",
     "Env mocap assente: avviso, nessun avvio cattura, nessun crash.",
     case_mocap_start_env_missing),
    ("anim_mocap_stop",
     "Stop cattura: emette request_mocap_stop → mocap.stop senza crash.",
     case_mocap_stop),

    ("anim_retarget_format_fbx",
     "Retarget formato FBX: avvia Blender retarget.py con output .fbx + model/anim.",
     _retarget_format_case("fbx")),
    ("anim_retarget_format_glb",
     "Retarget formato GLB: avvia Blender retarget.py con output .glb + model/anim.",
     _retarget_format_case("glb")),
    ("anim_retarget_format_bvh",
     "Retarget formato BVH: avvia Blender retarget.py con output .bvh + model/anim.",
     _retarget_format_case("bvh")),

    ("anim_retarget_template_auto",
     "Retarget template 'auto': il template viene propagato negli args Blender.",
     _retarget_template_case("auto")),
    ("anim_retarget_template_mixamo",
     "Retarget template 'Mixamo': il template viene propagato negli args Blender.",
     _retarget_template_case("Mixamo")),
    ("anim_retarget_template_unreal",
     "Retarget template 'Unreal Engine 5': il template viene propagato negli args Blender.",
     _retarget_template_case("Unreal Engine 5")),

    ("anim_retarget_missing_rig",
     "Apply senza rig: avviso, nessun avvio Blender, no-crash.",
     case_retarget_missing_rig),
    ("anim_retarget_missing_anim",
     "Apply senza animazione: avviso, nessun avvio Blender, no-crash.",
     case_retarget_missing_anim),
    ("anim_retarget_missing_files",
     "Apply con path inesistenti: avviso, nessun avvio Blender, no-crash.",
     case_retarget_missing_files),

    ("anim_use_last_rig",
     "'Usa ultimo rig': abilita il bottone e popola il path rig.",
     case_use_last_rig),
    ("anim_browse_rig_cancel",
     "Browse rig annullato: nessun crash, path invariato.",
     case_browse_rig_cancel),
    ("anim_browse_anim_cancel",
     "Browse animazione annullato: nessun crash, path invariato.",
     case_browse_anim_cancel),
    ("anim_set_anim_source",
     "set_anim (BVH catturato) popola la sorgente animazione.",
     case_set_anim_source),
    ("anim_set_preview",
     "set_preview con PNG valido: nessun crash, tooltip frame aggiornato.",
     case_set_preview),
    ("anim_capturing_toggle",
     "set_capturing alterna correttamente i bottoni start/stop.",
     case_capturing_toggle),
    ("anim_src_tabs_switch",
     "Cambio sorgente webcam↔import: nessun crash.",
     case_src_tabs_switch),
    ("anim_format_combo_values",
     "Opzioni formato export: fbx/glb/bvh presenti.",
     case_format_combo_values),
    ("anim_template_combo_values",
     "Opzioni template: auto/Mixamo/Unreal Engine 5 presenti.",
     case_template_combo_values),
    ("anim_select_tab",
     "Selezione del tab Animazione nella finestra principale: nessun crash.",
     case_select_anim_tab),
]
