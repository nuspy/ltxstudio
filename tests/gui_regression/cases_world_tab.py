"""Casi di regressione GUI-driven per il tab «Mondo 3D» (ui/tabs/world_tab.py → w.world_tab).

PARTICOLARITÀ di questo tab: NON passa dal backend nativo (native_driver/_enqueue) che lo Spy
dell'harness intercetta. La generazione va attraverso `core.world_service` (WorldGen):
    is_installed() → preflight() → make_job(p, mesh=…) → Job.start()  (subprocess venv isolato)
e diverse azioni hanno effetti collaterali REALI (subprocess installer, http.server, Esplora
risorse via os.startfile, browser via webbrowser.open). Per testare l'ISTRUZIONE senza eseguirla,
ogni caso che genera/installa patcha `world_service`/QProcess/os.startfile DENTRO il caso e
ripristina in finally — il job catturato viene anche annotato in `spy.calls` per il report.

Azioni coperte:
  • Genera scena (_gen) per OGNI modalità di output del combo: mesh / splat / both (+ chaining both→splat)
  • text-to-scene vs image-to-scene; opzioni low_vram, risoluzione, HF token, nome progetto
  • preflight che BLOCCA (input vuoti, immagine inesistente, modulo non installato)
  • Stop (con/senza job)
  • Installa modulo (senza spawnare il processo reale)
  • Apri cartella, Player web (guardia "nessuna scena"), Esporta FBX/OBJ (guardie), Sfoglia immagine
"""
from __future__ import annotations

import os

import harness as H


# ── infrastruttura di patch (cattura job, niente subprocess/Explorer/browser) ────
class _FakeJob:
    """Sostituto di world_service.Job: non spawna nulla; espone l'API usata dal tab."""
    def __init__(self):
        self.running = False
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class _patch_world:
    """Patcha core.world_service per i casi che generano: is_installed() controllabile e
    make_job() che CATTURA i parametri (mesh, hf_token, WorldParams) invece di spawnare.
    I job catturati sono in `.jobs` e anche in `spy.calls` (per il report)."""

    def __init__(self, spy, installed: bool = True):
        self.spy = spy
        self.installed = installed
        self.jobs: list[dict] = []
        self._orig: dict = {}

    def __enter__(self):
        from core import world_service as ws
        self._ws = ws
        self._orig["is_installed"] = ws.is_installed
        self._orig["make_job"] = ws.make_job
        ws.is_installed = lambda: self.installed

        def fake_make_job(p, *, mesh, hf_token=None, **cbs):
            rec = {
                "via": "world_service", "label": "make_job", "module": "worldgen",
                "params": {
                    "name": getattr(p, "name", None),
                    "image_path": getattr(p, "image_path", None),
                    "prompt": getattr(p, "prompt", None),
                    "resolution": getattr(p, "resolution", None),
                    "low_vram": getattr(p, "low_vram", None),
                    "mesh": mesh, "hf_token": hf_token,
                },
            }
            self.jobs.append(rec)
            self.spy.calls.append(rec)
            return _FakeJob()

        ws.make_job = fake_make_job
        return self

    def __exit__(self, *a):
        self._ws.is_installed = self._orig["is_installed"]
        self._ws.make_job = self._orig["make_job"]
        return False


def _set_mode(tab, value: str) -> bool:
    """Seleziona la voce del combo 'Output' (mesh/splat/both) per itemData. True se trovata."""
    cb = tab.mode
    for i in range(cb.count()):
        if cb.itemData(i) == value:
            cb.setCurrentIndex(i)
            return True
    return False


def _img(name: str = "_world_test_input.png") -> str:
    """Crea (se serve) una foto di test esistente su disco e ne ritorna il path."""
    path = str(H.OUT_DIR / name)
    if not os.path.exists(path):
        H.make_test_image(path, 1024, 576)
    return path


# ═══════════════════════ GENERAZIONE — una per modalità di output ════════════════
def case_world_gen_mode_mesh(w, spy):
    """Genera scena modalità MESH: la GUI deve creare un job make_job(mesh=True) con i WorldParams."""
    t = w.world_tab
    w.tabs.setCurrentWidget(t)
    with _patch_world(spy, installed=True) as pw:
        t.image.setText(_img())
        t.prompt.setText("a sunny mountain valley")
        assert _set_mode(t, "mesh"), "combo output: voce 'mesh' assente"
        t._gen()
        assert pw.jobs, "mesh: nessun job creato (_gen non ha chiamato make_job)"
        j = pw.jobs[-1]["params"]
        assert j["mesh"] is True, f"mesh: atteso mesh=True, ricevuto {j}"
        assert j["image_path"], f"mesh: image_path mancante: {j}"


def case_world_gen_mode_splat(w, spy):
    """Genera scena modalità SPLAT: primo (e unico) job make_job(mesh=False)."""
    t = w.world_tab
    with _patch_world(spy, installed=True) as pw:
        t.image.setText(_img())
        t.prompt.setText("a photorealistic forest")
        assert _set_mode(t, "splat"), "combo output: voce 'splat' assente"
        t._gen()
        assert pw.jobs, "splat: nessun job creato"
        assert pw.jobs[-1]["params"]["mesh"] is False, f"splat: atteso mesh=False: {pw.jobs[-1]}"


def case_world_gen_mode_both(w, spy):
    """Genera scena modalità ENTRAMBI: primo job mesh=True e _pending_splat impostato (chaining)."""
    t = w.world_tab
    with _patch_world(spy, installed=True) as pw:
        t.image.setText(_img())
        t.prompt.setText("a desert canyon at dusk")
        assert _set_mode(t, "both"), "combo output: voce 'both' assente"
        t._gen()
        assert pw.jobs and pw.jobs[-1]["params"]["mesh"] is True, \
            f"both: primo job deve essere mesh=True: {pw.jobs}"
        assert t._pending_splat is True, "both: _pending_splat deve essere True per il chaining splat"


def case_world_gen_both_chains_splat(w, spy):
    """ENTRAMBI: a mesh completata (_on_done code=0, was_mesh=True) parte il 2° job splat (mesh=False)."""
    t = w.world_tab
    with _patch_world(spy, installed=True) as pw:
        t.image.setText(_img())
        t.prompt.setText("a coastal cliff")
        _set_mode(t, "both")
        t._gen()
        assert pw.jobs, "both: primo job assente"
        t._on_done(0, True)            # mesh ok → deve innescare lo splat
        assert len(pw.jobs) == 2, f"both: atteso chaining a 2 job, trovati {len(pw.jobs)}"
        assert pw.jobs[-1]["params"]["mesh"] is False, f"both: 2° job deve essere splat: {pw.jobs[-1]}"
        assert t._pending_splat is False, "both: _pending_splat deve azzerarsi dopo il chaining"


# ═══════════════════════ INPUT: testo vs immagine ═══════════════════════════════
def case_world_gen_text_only(w, spy):
    """text-to-scene: senza immagine ma con prompt, il job parte con image_path vuoto."""
    t = w.world_tab
    with _patch_world(spy, installed=True) as pw:
        t.image.setText("")            # nessuna foto
        t.prompt.setText("an alien planet with two moons")
        _set_mode(t, "mesh")
        t._gen()
        assert pw.jobs, "text-only: nessun job creato"
        assert not pw.jobs[-1]["params"]["image_path"], \
            f"text-only: image_path doveva essere vuoto: {pw.jobs[-1]}"
        assert pw.jobs[-1]["params"]["prompt"], "text-only: prompt mancante nel job"


def case_world_gen_image_input(w, spy):
    """image-to-scene: con foto esistente, il job riceve l'image_path corretto."""
    t = w.world_tab
    with _patch_world(spy, installed=True) as pw:
        path = _img()
        t.image.setText(path)
        t.prompt.setText("")
        _set_mode(t, "mesh")
        t._gen()
        assert pw.jobs, "image-input: nessun job creato"
        assert pw.jobs[-1]["params"]["image_path"] == path, \
            f"image-input: image_path errato: {pw.jobs[-1]}"


# ═══════════════════════ OPZIONI / TOGGLE / CAMPI ═══════════════════════════════
def case_world_gen_lowvram_flag(w, spy):
    """Toggle Low VRAM: deve propagarsi a WorldParams.low_vram=True nel job."""
    t = w.world_tab
    with _patch_world(spy, installed=True) as pw:
        t.image.setText(_img())
        t.low_vram.setChecked(True)
        _set_mode(t, "mesh")
        t._gen()
        assert pw.jobs, "lowvram: nessun job creato"
        assert pw.jobs[-1]["params"]["low_vram"] is True, f"lowvram non propagato: {pw.jobs[-1]}"


def case_world_gen_resolution_passed(w, spy):
    """Lo spinbox Risoluzione deve passare il valore impostato (es. 2048) al job."""
    t = w.world_tab
    with _patch_world(spy, installed=True) as pw:
        t.image.setText(_img())
        t.res.setValue(2048)
        _set_mode(t, "mesh")
        t._gen()
        assert pw.jobs, "resolution: nessun job creato"
        assert pw.jobs[-1]["params"]["resolution"] == 2048, f"resolution non propagata: {pw.jobs[-1]}"


def case_world_gen_hf_token_passed(w, spy):
    """Il campo HF token (FLUX.1-dev gated) deve arrivare a make_job(hf_token=…)."""
    t = w.world_tab
    with _patch_world(spy, installed=True) as pw:
        t.image.setText(_img())
        t.hf.setText("hf_dummy_token")
        _set_mode(t, "mesh")
        t._gen()
        assert pw.jobs, "hf_token: nessun job creato"
        assert pw.jobs[-1]["params"]["hf_token"] == "hf_dummy_token", \
            f"hf_token non propagato: {pw.jobs[-1]}"


def case_world_gen_custom_name(w, spy):
    """Il nome progetto digitato deve finire in WorldParams.name (cartella output dedicata)."""
    t = w.world_tab
    with _patch_world(spy, installed=True) as pw:
        t.name.setText("my_scene_42")
        t.image.setText(_img())
        _set_mode(t, "mesh")
        t._gen()
        assert pw.jobs, "name: nessun job creato"
        assert pw.jobs[-1]["params"]["name"] == "my_scene_42", f"name non propagato: {pw.jobs[-1]}"


# ═══════════════════════ PREFLIGHT — casi che DEVONO bloccare ═══════════════════
def case_world_preflight_empty_inputs_blocks(w, spy):
    """Senza immagine NÉ prompt: il preflight blocca → nessun job (warning, no crash)."""
    t = w.world_tab
    with _patch_world(spy, installed=True) as pw:
        t.image.setText("")
        t.prompt.setText("")
        _set_mode(t, "mesh")
        t._gen()
        assert not pw.jobs, f"preflight: con input vuoti NON deve partire alcun job: {pw.jobs}"


def case_world_preflight_missing_image_blocks(w, spy):
    """Con un path immagine inesistente il preflight blocca → nessun job."""
    t = w.world_tab
    with _patch_world(spy, installed=True) as pw:
        t.image.setText(r"E:\__does_not_exist__\nope.png")
        t.prompt.setText("ignored")
        _set_mode(t, "mesh")
        t._gen()
        assert not pw.jobs, f"preflight: immagine inesistente deve bloccare: {pw.jobs}"


def case_world_not_installed_blocks_gen(w, spy):
    """Modulo NON installato: il preflight blocca la generazione (nessun job spawnato)."""
    t = w.world_tab
    with _patch_world(spy, installed=False) as pw:
        t.image.setText(_img())
        t.prompt.setText("a valley")
        _set_mode(t, "mesh")
        t._gen()
        assert not pw.jobs, f"not-installed: non deve creare job: {pw.jobs}"


# ═══════════════════════ STOP ═══════════════════════════════════════════════════
def case_world_stop_no_job(w, spy):
    """Stop senza job attivo: non deve sollevare eccezioni."""
    t = w.world_tab
    t._stop()
    assert t._pending_splat is False, "stop: _pending_splat deve restare/azzerarsi a False"


def case_world_stop_with_running_job(w, spy):
    """Stop con job attivo: chiama job.stop() e azzera _pending_splat."""
    t = w.world_tab
    with _patch_world(spy, installed=True):
        t.image.setText(_img())
        _set_mode(t, "both")
        t._gen()
        job = t._job
        assert job is not None, "stop: il job avrebbe dovuto essere creato"
        t._stop()
        assert getattr(job, "stopped", False) is True, "stop: job.stop() non invocato"
        assert t._pending_splat is False, "stop: _pending_splat non azzerato"


# ═══════════════════════ INSTALLA MODULO (senza spawnare) ═══════════════════════
def case_world_install_action_no_spawn(w, spy):
    """Installa modulo: con conferma=Yes la GUI avvia il QProcess installer (qui neutralizzato)
    e disabilita il pulsante. Verifica no-crash e che parta il processo (senza eseguirlo davvero)."""
    from PySide6.QtCore import QProcess
    t = w.world_tab
    orig_start = QProcess.start
    QProcess.start = lambda self, *a, **k: None      # neutralizza lo spawn reale dell'installer
    try:
        t._install()
        assert t._install_proc is not None, "install: il QProcess installer non è stato avviato"
        assert t.btn_install.isEnabled() is False, "install: il pulsante deve disabilitarsi durante l'install"
    finally:
        QProcess.start = orig_start


# ═══════════════════════ AZIONI POST-GEN / UTILITY ══════════════════════════════
def case_world_open_folder_no_crash(w, spy):
    """Apri cartella: apre la dir output via os.startfile (qui neutralizzato) senza crash."""
    t = w.world_tab
    orig = os.startfile if hasattr(os, "startfile") else None
    os.startfile = lambda *a, **k: None
    try:
        t._open_folder()
    finally:
        if orig is not None:
            os.startfile = orig


def case_world_web_player_no_scene_guard(w, spy):
    """Player web senza scena generata: deve mostrare l'avviso e NON avviare l'http.server."""
    t = w.world_tab
    assert t._cur is None, "precondizione: nessuna scena corrente"
    t._open_web_player()
    assert t._http_proc is None, "web player: senza scena non deve avviare alcun server http"


def case_world_export_fbx_no_scene_guard(w, spy):
    """Esporta FBX senza scena: guardia → info dialog, nessun crash (Blender non invocato)."""
    t = w.world_tab
    assert t._cur is None, "precondizione: nessuna scena corrente"
    t._export("fbx")           # _cur None → info "Genera prima una scena mesh."


def case_world_export_obj_no_glb_guard(w, spy):
    """Esporta OBJ con _cur ma senza GLB su disco: guardia → info dialog, nessun crash."""
    from core import world_service as ws
    t = w.world_tab
    t._cur = ws.WorldParams(name="__no_glb_on_disk__", prompt="x")
    assert not t._cur.glb().exists(), "precondizione: il GLB non deve esistere"
    t._export("obj")           # glb mancante → info "Nessun GLB: genera in modalità Mesh."


def case_world_pick_image_no_crash(w, spy):
    """Sfoglia immagine: con QFileDialog patchato (ritorna stringa vuota) non deve crashare."""
    t = w.world_tab
    t._pick_image()
    # dialog patchato → path vuoto → il campo resta invariato
    assert t.image.text() == "", "pick_image: con scelta annullata il campo non deve cambiare"


def case_world_refresh_install_state(w, spy):
    """Refresh stato installazione: abilita/disabilita gen vs install coerentemente con is_installed()."""
    from core import world_service as ws
    t = w.world_tab
    orig = ws.is_installed
    try:
        ws.is_installed = lambda: True
        t._refresh_install_state()
        assert t.btn_gen.isEnabled() is True and t.btn_install.isEnabled() is False, \
            "refresh(installed): gen abilitato + install disabilitato"
        ws.is_installed = lambda: False
        t._refresh_install_state()
        assert t.btn_gen.isEnabled() is False and t.btn_install.isEnabled() is True, \
            "refresh(not-installed): gen disabilitato + install abilitato"
    finally:
        ws.is_installed = orig


CASES = [
    ("world_gen_mode_mesh", "Genera scena (Mesh): make_job(mesh=True) con i WorldParams (image_path).",
     case_world_gen_mode_mesh),
    ("world_gen_mode_splat", "Genera scena (Splat): make_job(mesh=False).",
     case_world_gen_mode_splat),
    ("world_gen_mode_both", "Genera scena (Entrambi): 1° job mesh=True + _pending_splat per il chaining.",
     case_world_gen_mode_both),
    ("world_gen_both_chains_splat", "Entrambi: a mesh completata parte il 2° job splat (mesh=False).",
     case_world_gen_both_chains_splat),
    ("world_gen_text_only", "text-to-scene: senza foto ma con prompt, job con image_path vuoto.",
     case_world_gen_text_only),
    ("world_gen_image_input", "image-to-scene: la foto esistente arriva al job come image_path.",
     case_world_gen_image_input),
    ("world_gen_lowvram_flag", "Toggle Low VRAM → WorldParams.low_vram=True nel job.",
     case_world_gen_lowvram_flag),
    ("world_gen_resolution_passed", "Spinbox Risoluzione → resolution propagata al job (2048).",
     case_world_gen_resolution_passed),
    ("world_gen_hf_token_passed", "Campo HF token → make_job(hf_token=…) (FLUX.1-dev gated).",
     case_world_gen_hf_token_passed),
    ("world_gen_custom_name", "Nome progetto → WorldParams.name (cartella output dedicata).",
     case_world_gen_custom_name),
    ("world_preflight_empty_inputs_blocks", "Preflight: input vuoti (no foto/no prompt) → nessun job.",
     case_world_preflight_empty_inputs_blocks),
    ("world_preflight_missing_image_blocks", "Preflight: immagine inesistente → nessun job.",
     case_world_preflight_missing_image_blocks),
    ("world_not_installed_blocks_gen", "Preflight: modulo non installato → nessun job.",
     case_world_not_installed_blocks_gen),
    ("world_stop_no_job", "Stop senza job attivo: no-crash.",
     case_world_stop_no_job),
    ("world_stop_with_running_job", "Stop con job attivo: job.stop() + _pending_splat azzerato.",
     case_world_stop_with_running_job),
    ("world_install_action_no_spawn", "Installa modulo: avvia il QProcess installer (neutralizzato) e disabilita il pulsante.",
     case_world_install_action_no_spawn),
    ("world_open_folder_no_crash", "Apri cartella: os.startfile (neutralizzato) senza crash.",
     case_world_open_folder_no_crash),
    ("world_web_player_no_scene_guard", "Player web senza scena: avviso + nessun http.server avviato.",
     case_world_web_player_no_scene_guard),
    ("world_export_fbx_no_scene_guard", "Esporta FBX senza scena: guardia info, no-crash.",
     case_world_export_fbx_no_scene_guard),
    ("world_export_obj_no_glb_guard", "Esporta OBJ con _cur ma senza GLB: guardia info, no-crash.",
     case_world_export_obj_no_glb_guard),
    ("world_pick_image_no_crash", "Sfoglia immagine (dialog annullato): no-crash, campo invariato.",
     case_world_pick_image_no_crash),
    ("world_refresh_install_state", "Refresh stato installazione: abilita gen vs install coerentemente.",
     case_world_refresh_install_state),
]
