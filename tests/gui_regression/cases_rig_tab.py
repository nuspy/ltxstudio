"""Casi di regressione GUI-driven per il tab "Rig 3D" (RigTab → MainWindow.generate_rig).

Il tab Rig prende una mesh (.glb/.obj, di norma quella prodotta da TRELLIS) e ne predice
scheletro + skinning (SkinTokens/TokenRig), esportando un GLB/FBX rigged.

Flusso GUI→backend:  RigTab._on_generate() → request_rig.emit(get_config()) → MainWindow.generate_rig(cfg)
  - ramo NATIVO  : se core.rig_service.is_installed() → _rig_native() → self._enqueue(worker)
  - ramo ComfyUI : altrimenti, se driver.is_up() e il nodo SkinTokenRigTrimesh è presente e c'è Blender
                   → self.rigger.build(**cfg) → self.driver.submit(prompt)

Lo Spy dell'harness intercetta sia _enqueue (ramo nativo) sia driver.submit (ramo comfy), quindi
in entrambi i rami una richiesta valida produce spy.calls non vuoto.

Per le verifiche "param non perso tra GUI e pipeline" forziamo il ramo ComfyUI e avvolgiamo
RigPipeline.build per catturare i kwargs effettivamente passati (cap dict).
"""
from __future__ import annotations

from pathlib import Path

import harness as H


class _SkipComfyBranch(Exception):
    """Segnala lo skip dei casi comfy-mock in run reale (intercettata dai casi stessi)."""


def _skip_in_real(fn):
    """Decoratore: il caso ritorna subito (skip) se _force_comfy segnala REAL_MODE."""
    def _w(w, spy):
        try:
            return fn(w, spy)
        except _SkipComfyBranch:
            return                                    # skip esplicito in run_real
    _w.__doc__ = fn.__doc__
    _w.__name__ = getattr(fn, "__name__", "case")
    return _w


# ───────────────────────── helper ─────────────────────────
def _open_rig(w):
    """Porta in primo piano il tab Rig (no-op se non in un QTabWidget)."""
    try:
        w.tabs.setCurrentWidget(w.rig_tab)
    except Exception:                                  # noqa: BLE001
        pass
    return w.rig_tab


def _set_mesh(w, name: str = "_rig_test_mesh.glb") -> str:
    """Precondizione: una mesh di input esistente (generate_rig richiede un file esistente)."""
    p = Path(H.OUT_DIR) / name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"glTF\x02\x00\x00\x00")          # stub: basta che il file esista
    t = _open_rig(w)
    t._input_path = str(p)
    t.path_edit.setText(str(p))
    return str(p)


def _force_comfy(w):
    """Forza il ramo ComfyUI (SkinTokens nativo NON installato, driver up, nodo presente, Blender ok)
    e cattura i kwargs passati a RigPipeline.build. Ritorna il dict `cap`.
    SOLO SUITE MOCK: in run_real (H.REAL_MODE) questo ramo non è eseguibile (RealSpy non
    intercetta driver.submit e ComfyUI è dismesso) → i casi che lo usano skippano."""
    if getattr(H, "REAL_MODE", False):
        raise _SkipComfyBranch()
    from core import rig_service
    import core.blender_installer as bi
    rig_service.is_installed = lambda: False
    bi.find_blender = lambda: "blender"               # bypassa il pre-check Blender
    w.driver.is_up = lambda: True
    w.driver.object_info = lambda *a, **k: {"SkinTokenRigTrimesh": {}}

    cap: dict = {}
    _orig = w.rigger.build

    def _wrapped(mesh_path, **kw):
        cap["mesh_path"] = mesh_path
        cap.update(kw)
        return _orig(mesh_path, **kw)

    w.rigger.build = _wrapped
    return cap


def _force_native(w):
    """Forza il ramo NATIVO (SkinTokens installato + Blender presente)."""
    from core import rig_service
    import core.blender_installer as bi
    rig_service.is_installed = lambda: True
    bi.find_blender = lambda: "blender"


def _select_combo_text(combo, text: str) -> bool:
    for i in range(combo.count()):
        if combo.itemText(i) == text:
            combo.setCurrentIndex(i)
            return True
    return False


def _run(w):
    """Invoca l'azione passando dal segnale del tab (request_rig → generate_rig)."""
    w.rig_tab._on_generate()


# ───────────────────────── azione base / wiring ─────────────────────────
def case_generate_default(w, spy):
    """Rig con config di default: la GUI deve inviare una richiesta al backend (ramo comfy)
    con la mesh selezionata e il template di default."""
    cap = _force_comfy(w)
    mesh = _set_mesh(w)
    _run(w)
    assert spy.calls, "rig: nessuna richiesta inviata (generate_rig non ha chiamato il backend)"
    assert spy.calls[-1]["via"] == "comfy", f"atteso submit ComfyUI, ottenuto: {spy.calls[-1]}"
    assert cap.get("mesh_path") == mesh, f"mesh non propagata a build: {cap}"
    assert cap.get("skeleton_template") == "Keep model names", f"template default errato: {cap}"


def case_signal_wiring(w, spy):
    """Il segnale request_rig del tab deve essere collegato a generate_rig (emettendolo parte una richiesta)."""
    _force_comfy(w)
    _set_mesh(w)
    w.rig_tab.request_rig.emit(w.rig_tab.get_config())
    assert spy.calls, "rig: request_rig non instradato a generate_rig"


# ───────────────────────── scheletro: un caso per OGNI template ─────────────────────────
def _case_skeleton(template_text):
    def _c(w, spy):
        cap = _force_comfy(w)
        _set_mesh(w)
        ok = _select_combo_text(w.rig_tab.skeleton, template_text)
        assert ok, f"template scheletro non presente nel combo: {template_text!r}"
        _run(w)
        assert spy.calls, f"rig[{template_text}]: nessuna richiesta inviata (build potrebbe rifiutare il template)"
        assert cap.get("skeleton_template") == template_text, \
            f"template non propagato: atteso {template_text!r}, cap={cap.get('skeleton_template')!r}"
    return _c


# ───────────────────────── formato export: un caso per OGNI formato ─────────────────────────
def _case_format(fmt):
    def _c(w, spy):
        cap = _force_comfy(w)
        _set_mesh(w)
        ok = _select_combo_text(w.rig_tab.file_format, fmt)
        assert ok, f"formato non presente nel combo: {fmt!r}"
        _run(w)
        assert spy.calls, f"rig[{fmt}]: nessuna richiesta inviata"
        assert cap.get("file_format") == fmt, f"formato non propagato: atteso {fmt!r}, cap={cap.get('file_format')!r}"
    return _c


# ───────────────────────── toggle / opzioni ─────────────────────────
def case_transfer_off(w, spy):
    """use_transfer (default ON) → OFF: il valore deve arrivare a build come False."""
    cap = _force_comfy(w)
    _set_mesh(w)
    w.rig_tab.transfer.setChecked(False)
    _run(w)
    assert spy.calls, "rig: nessuna richiesta inviata"
    assert cap.get("use_transfer") is False, f"use_transfer non propagato (off): {cap}"


def case_transfer_on(w, spy):
    """use_transfer ON esplicito: arriva a build come True."""
    cap = _force_comfy(w)
    _set_mesh(w)
    w.rig_tab.transfer.setChecked(True)
    _run(w)
    assert spy.calls, "rig: nessuna richiesta inviata"
    assert cap.get("use_transfer") is True, f"use_transfer non propagato (on): {cap}"


def case_postprocess_on(w, spy):
    """use_postprocess ON: arriva a build come True."""
    cap = _force_comfy(w)
    _set_mesh(w)
    w.rig_tab.postprocess.setChecked(True)
    _run(w)
    assert spy.calls, "rig: nessuna richiesta inviata"
    assert cap.get("use_postprocess") is True, f"use_postprocess non propagato: {cap}"


def case_bottom_origin_on(w, spy):
    """bottom_center_origin ON (workflow game engine): arriva a build come True."""
    cap = _force_comfy(w)
    _set_mesh(w)
    w.rig_tab.bottom_origin.setChecked(True)
    _run(w)
    assert spy.calls, "rig: nessuna richiesta inviata"
    assert cap.get("bottom_center_origin") is True, f"bottom_center_origin non propagato: {cap}"


def case_group_per_vertex(w, spy):
    """group_per_vertex (influenze ossee/vertice) propagato come int impostato."""
    cap = _force_comfy(w)
    _set_mesh(w)
    w.rig_tab.gpv.setValue(8)
    _run(w)
    assert spy.calls, "rig: nessuna richiesta inviata"
    assert cap.get("group_per_vertex") == 8, f"group_per_vertex non propagato: {cap}"


def case_smooth_angle(w, spy):
    """smooth_angle propagato come float impostato."""
    cap = _force_comfy(w)
    _set_mesh(w)
    w.rig_tab.smooth.setValue(30.0)
    _run(w)
    assert spy.calls, "rig: nessuna richiesta inviata"
    assert abs(float(cap.get("smooth_angle", -1)) - 30.0) < 1e-6, f"smooth_angle non propagato: {cap}"


def case_advanced_sampling(w, spy):
    """Parametri di campionamento autoregressivo (beams/temperature/top_k/top_p/rep_pen) propagati a build."""
    cap = _force_comfy(w)
    _set_mesh(w)
    t = w.rig_tab
    t.num_beams.setValue(6)
    t.temperature.setValue(0.7)
    t.top_k.setValue(20)
    t.top_p.setValue(0.80)
    t.rep_pen.setValue(1.5)
    _run(w)
    assert spy.calls, "rig: nessuna richiesta inviata"
    assert cap.get("num_beams") == 6, f"num_beams non propagato: {cap}"
    assert abs(float(cap.get("temperature", -1)) - 0.7) < 1e-6, f"temperature non propagato: {cap}"
    assert cap.get("top_k") == 20, f"top_k non propagato: {cap}"
    assert abs(float(cap.get("top_p", -1)) - 0.80) < 1e-6, f"top_p non propagato: {cap}"
    assert abs(float(cap.get("repetition_penalty", -1)) - 1.5) < 1e-6, f"repetition_penalty non propagato: {cap}"


# ───────────────────────── ramo NATIVO (zero ComfyUI) ─────────────────────────
def case_native_branch(w, spy):
    """Se SkinTokens nativo è installato, l'azione deve passare per _enqueue (worker nativo, zero ComfyUI)."""
    _force_native(w)
    _set_mesh(w)                                        # file reale (preflight nativo richiede is_file)
    _run(w)
    assert spy.calls, "rig nativo: nessuna richiesta accodata (_enqueue non chiamato)"
    assert spy.calls[-1]["via"] == "enqueue", f"atteso ramo nativo via enqueue, ottenuto: {spy.calls[-1]}"


# ───────────────────────── precondizioni / negativi ─────────────────────────
def case_no_mesh_no_request(w, spy):
    """Senza mesh selezionata la GUI NON deve inviare nulla (deve avvisare e tornare)."""
    _force_comfy(w)
    _open_rig(w)
    w.rig_tab._input_path = ""
    w.rig_tab.path_edit.clear()
    _run(w)
    assert not spy.calls, f"rig senza mesh non doveva inviare richieste: {spy.calls}"


def case_node_missing_no_request(w, spy):
    """Ramo ComfyUI ma nodo SkinTokenRigTrimesh assente: niente submit (pre-check del nodo)."""
    from core import rig_service
    import core.blender_installer as bi
    rig_service.is_installed = lambda: False
    bi.find_blender = lambda: "blender"
    w.driver.is_up = lambda: True
    w.driver.object_info = lambda *a, **k: {}          # nodo NON presente
    _set_mesh(w)
    _run(w)
    assert not spy.calls, f"nodo mancante: non doveva inviare submit: {spy.calls}"


def case_backend_offline_no_request(w, spy):
    """Né nativo installato né ComfyUI up: nessuna richiesta (avviso offline)."""
    from core import rig_service
    rig_service.is_installed = lambda: False
    w.driver.is_up = lambda: False
    _set_mesh(w)
    _run(w)
    assert not spy.calls, f"backend offline: non doveva inviare richieste: {spy.calls}"


# ───────────────────────── UI-only (no backend) ─────────────────────────
def case_use_last_mesh(w, spy):
    """'Usa ultimo modello': set_last_mesh abilita il pulsante e _use_last_mesh imposta l'input."""
    t = _open_rig(w)
    fake = str(Path(H.OUT_DIR) / "_rig_last.glb")
    t.set_last_mesh(fake)
    assert t.last_btn.isEnabled(), "set_last_mesh non ha abilitato il pulsante 'usa ultimo'"
    t._use_last_mesh()
    assert t._input_path == fake, f"_use_last_mesh non ha impostato l'input: {t._input_path!r}"
    assert not spy.calls, "use_last_mesh è UI-only, non deve toccare il backend"


def case_get_config_shape(w, spy):
    """get_config() deve esporre tutte le chiavi attese dalla pipeline (contratto GUI→backend)."""
    t = _open_rig(w)
    cfg = t.get_config()
    expected = {
        "mesh_path", "skeleton_template", "file_format", "bottom_center_origin",
        "use_transfer", "use_postprocess", "group_per_vertex", "smooth_angle",
        "num_beams", "temperature", "top_k", "top_p", "repetition_penalty",
    }
    missing = expected - set(cfg)
    assert not missing, f"get_config() chiavi mancanti: {missing}"


_RAW_CASES = [
    ("rig_generate_default",
     "Rig (default): la GUI invia una richiesta backend con mesh + template di default.",
     case_generate_default),
    ("rig_signal_wiring",
     "request_rig deve essere collegato a generate_rig (emettendolo parte una richiesta).",
     case_signal_wiring),

    # un caso per OGNI template scheletro (scopre 'funziona solo un template')
    ("rig_skeleton_keep_model_names",
     "Scheletro 'Keep model names': template propagato a build e richiesta inviata.",
     _case_skeleton("Keep model names")),
    ("rig_skeleton_mixamo",
     "Scheletro 'Mixamo': template propagato a build e richiesta inviata.",
     _case_skeleton("Mixamo")),
    ("rig_skeleton_unreal_engine_5",
     "Scheletro 'Unreal Engine 5': template propagato a build e richiesta inviata.",
     _case_skeleton("Unreal Engine 5")),

    # un caso per OGNI formato export
    ("rig_format_glb",
     "Formato export 'glb' propagato a build.",
     _case_format("glb")),
    ("rig_format_fbx",
     "Formato export 'fbx' propagato a build.",
     _case_format("fbx")),

    # toggle / opzioni
    ("rig_transfer_off",
     "use_transfer OFF propagato come False.", case_transfer_off),
    ("rig_transfer_on",
     "use_transfer ON propagato come True.", case_transfer_on),
    ("rig_postprocess_on",
     "use_postprocess ON propagato come True.", case_postprocess_on),
    ("rig_bottom_origin_on",
     "bottom_center_origin ON propagato come True.", case_bottom_origin_on),
    ("rig_group_per_vertex",
     "group_per_vertex propagato come int impostato.", case_group_per_vertex),
    ("rig_smooth_angle",
     "smooth_angle propagato come float impostato.", case_smooth_angle),
    ("rig_advanced_sampling",
     "Parametri campionamento (beams/temp/top_k/top_p/rep_pen) propagati a build.",
     case_advanced_sampling),

    # ramo nativo
    ("rig_native_branch",
     "Con SkinTokens nativo installato l'azione passa per _enqueue (zero ComfyUI).",
     case_native_branch),

    # precondizioni / negativi
    ("rig_no_mesh_no_request",
     "Senza mesh la GUI non invia richieste (avviso e ritorno).", case_no_mesh_no_request),
    ("rig_node_missing_no_request",
     "Nodo SkinTokenRigTrimesh assente: niente submit (pre-check nodo).", case_node_missing_no_request),
    ("rig_backend_offline_no_request",
     "Né nativo né ComfyUI disponibili: nessuna richiesta (offline).", case_backend_offline_no_request),

    # UI-only
    ("rig_use_last_mesh",
     "'Usa ultimo modello' abilita il pulsante e imposta l'input (UI-only).", case_use_last_mesh),
    ("rig_get_config_shape",
     "get_config() espone tutte le chiavi attese dalla pipeline.", case_get_config_shape),
]

# In run_real i casi comfy-mock skippano (via _SkipComfyBranch dentro _force_comfy); il wrap è
# innocuo per la suite mock. Il ramo NATIVO (rig_native_branch) resta eseguito ovunque.
CASES = [(n, d, _skip_in_real(f)) for n, d, f in _RAW_CASES]
