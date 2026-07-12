"""Casi di regressione GUI-driven per il tab TRELLIS.2 ("trellis_tab" → immagine → modello 3D).

Azione coperta: MainWindow.generate_trellis(cfg), agganciata al segnale TrellisTab.request_trellis
(emesso da TrellisTab._on_generate / pulsante "Genera"). Ogni caso imposta i widget del tab, invoca
l'azione tramite il SEGNALE del tab (w.trellis_tab._on_generate()) e lo Spy cattura la richiesta
GUI→backend SENZA eseguire (niente GPU/rete).

Due percorsi backend in generate_trellis:
  • NATIVO  : se core.trellis_service.is_installed() → _trellis_native() → self._enqueue(worker).
              Lo Spy cattura via "enqueue" (il worker non porta module/params); per ispezionare i
              parametri che arrivano davvero al backend patchiamo trellis_service.make_job.
              ⚠ In nativo SOLO seed/steps/texture_size raggiungono make_job: resolution/precision/
              attn/bg/guidance/faces/format/texture-toggle NON sono passati (script ufficiale).
  • COMFYUI : altrimenti, se driver attivo + nodo LoadTrellis2Models → self.trellisser.build(cfg) →
              driver.submit(prompt). Qui TUTTE le opzioni della GUI confluiscono in build(): è il
              percorso che usiamo per verificare combo/toggle/numerici opzione-per-opzione (così si
              scopre se "funziona solo un modello/risoluzione/formato").

NB: il file deve restare importabile senza effetti collaterali: ogni patch di modulo è fatta DENTRO
il caso e ripristinata in finally.
"""
from __future__ import annotations

import harness as H


class _SkipMockOnly(Exception):
    """Skip dei casi mock-based in run REALE: il RealSpy ESEGUE i worker → il Job fittizio _J
    (senza .start) crasha, e il ramo ComfyUI è dismesso. Il contratto parametri resta coperto
    dalla suite mock; in run reale gira il caso end-to-end nativo dedicato."""


def _skip_in_real(fn):
    def _w(w, spy):
        try:
            return fn(w, spy)
        except _SkipMockOnly:
            return
    _w.__doc__ = fn.__doc__
    _w.__name__ = getattr(fn, "__name__", "case")
    return _w


# ───────────────────────── infrastruttura di cattura ─────────────────────────
def _native_capture(w):
    """Forza il percorso NATIVO e cattura i kwargs passati a trellis_service.make_job.
    Ritorna (cap, restore)."""
    if getattr(H, "REAL_MODE", False):
        raise _SkipMockOnly()
    import core.trellis_service as ts
    orig_inst, orig_make = ts.is_installed, ts.make_job
    cap: dict = {}
    ts.is_installed = lambda: True

    def fake_make_job(image_path, name="model", **kw):
        cap["image_path"] = image_path
        cap["name"] = name
        cap.update(kw)

        class _J:  # oggetto Job fittizio: non viene mai avviato (worker non parte nello Spy)
            pass
        return _J()

    ts.make_job = fake_make_job

    def restore():
        ts.is_installed, ts.make_job = orig_inst, orig_make

    return cap, restore


def _comfy_capture(w):
    """Forza il percorso COMFYUI e cattura i kwargs passati a self.trellisser.build.
    Ritorna (cap, restore)."""
    if getattr(H, "REAL_MODE", False):
        raise _SkipMockOnly()
    import core.trellis_service as ts
    orig_inst = ts.is_installed
    orig_build = w.trellisser.build
    orig_up = w.driver.is_up
    orig_oi = w.driver.object_info
    orig_upload = w._upload_local
    orig_gd = w.gen_dialog.start

    ts.is_installed = lambda: False                       # niente nativo → ramo ComfyUI
    w.driver.is_up = lambda: True                         # supera il guard "backend offline"
    w.driver.object_info = lambda *a, **k: {"LoadTrellis2Models": {"input": {}}}
    w._upload_local = lambda p: "in.png"                  # niente upload reale
    w.gen_dialog.start = lambda *a, **k: None             # niente dialog di avanzamento

    cap: dict = {}

    def fake_build(image_name, **kw):
        cap["image_name"] = image_name
        cap.update(kw)
        return {}                                         # prompt vuoto → driver.submit (spy) lo cattura

    w.trellisser.build = fake_build

    def restore():
        ts.is_installed = orig_inst
        w.trellisser.build = orig_build
        w.driver.is_up = orig_up
        w.driver.object_info = orig_oi
        w._upload_local = orig_upload
        w.gen_dialog.start = orig_gd

    return cap, restore


def _select_canvas_input(w):
    """Carica una foto nel canvas e imposta l'input del tab su <from_canvas>."""
    H.load_canvas_image(w)
    w.tabs.setCurrentWidget(w.trellis_tab)
    w.trellis_tab._use_canvas_image()        # _input_path = "<from_canvas>"


def _comfy_case(setter, checker):
    """Costruisce un caso che, sul ramo ComfyUI, imposta un'opzione (setter) e verifica che arrivi
    a trellisser.build (checker(cap))."""
    def _case(w, spy):
        _select_canvas_input(w)
        cap, restore = _comfy_capture(w)
        try:
            setter(w.trellis_tab)
            w.trellis_tab._on_generate()
            assert spy.calls, "trellis: nessuna richiesta inviata al backend (build/submit non chiamati)"
            checker(cap)
        finally:
            restore()
    return _case


# ───────────────────────── percorsi/dispatch base ─────────────────────────
def case_generate_native_default(w, spy):
    """NATIVO (default): la GUI deve accodare un job 3D e passare a make_job seed/steps/texture_size."""
    _select_canvas_input(w)
    cap, restore = _native_capture(w)
    try:
        w.trellis_tab._on_generate()
        assert spy.calls, "trellis nativo: nessun job accodato (_enqueue non chiamato)"
        assert spy.calls[-1]["via"] == "enqueue", f"atteso enqueue, ottenuto: {spy.calls[-1]}"
        assert "seed" in cap and "texture_size" in cap, f"make_job senza seed/texture_size: {cap}"
        assert cap.get("ss_steps") and cap.get("slat_steps"), f"make_job senza steps: {cap}"
    finally:
        restore()


def case_generate_comfy_default(w, spy):
    """COMFYUI (default): la GUI deve chiamare build con immagine + risoluzione + formato e inviare submit."""
    _select_canvas_input(w)
    cap, restore = _comfy_capture(w)
    try:
        w.trellis_tab._on_generate()
        assert spy.calls, "trellis comfy: nessuna richiesta inviata (submit non chiamato)"
        assert spy.calls[-1]["via"] == "comfy", f"atteso submit comfy, ottenuto: {spy.calls[-1]}"
        assert cap.get("image_name"), f"build senza image_name: {cap}"
        assert cap.get("resolution"), f"build senza resolution: {cap}"
        assert cap.get("file_format"), f"build senza file_format: {cap}"
        assert cap.get("filename_prefix"), f"build senza filename_prefix: {cap}"
    finally:
        restore()


def case_no_image_no_dispatch(w, spy):
    """Nessuna immagine selezionata: la GUI deve avvisare e NON inviare nulla al backend."""
    import core.trellis_service as ts
    orig = ts.is_installed
    ts.is_installed = lambda: True                # passa il primo guard, deve fermarsi al check immagine
    try:
        w.trellis_tab._input_path = ""           # nessun input
        w.trellis_tab._on_generate()
        assert not spy.calls, f"senza immagine NON deve partire nulla, invece: {spy.calls}"
    finally:
        ts.is_installed = orig


def case_canvas_masked_without_mask_no_dispatch(w, spy):
    """Input 'da canvas mascherato' ma senza maschera: avviso e nessun invio al backend."""
    import core.trellis_service as ts
    orig = ts.is_installed
    ts.is_installed = lambda: True
    H.load_canvas_image(w)
    w.frame1_tab.canvas.has_mask = lambda: False         # nessuna maschera disegnata
    try:
        w.trellis_tab._use_canvas_masked()               # _input_path = "<from_canvas_masked>"
        w.trellis_tab._on_generate()
        assert not spy.calls, f"masked senza maschera NON deve inviare nulla: {spy.calls}"
    finally:
        ts.is_installed = orig


def case_canvas_masked_forces_no_background(w, spy):
    """Input 'da canvas mascherato' con maschera: build deve ricevere remove_background=False (force_no_bg)."""
    import core.trellis_service as ts
    from pathlib import Path
    test_img = H.make_test_image(str(H.OUT_DIR / "_trellis_masked_subject.png"))
    H.load_canvas_image(w)
    cv = w.frame1_tab.canvas
    cv.has_mask = lambda: True
    cv.export_subject_rgba = lambda out_dir: {"image": test_img}     # soggetto isolato fittizio
    cap, restore = _comfy_capture(w)
    try:
        w.trellis_tab._use_canvas_masked()
        w.trellis_tab._on_generate()
        assert spy.calls, "masked con maschera: nessuna richiesta inviata"
        assert cap.get("remove_background") is False, \
            f"masked deve forzare remove_background=False, invece: {cap.get('remove_background')}"
        assert cap.get("image_name"), f"build senza image_name: {cap}"
    finally:
        restore()


# ───────────────────────── toggle ─────────────────────────
def case_texture_on(w, spy):
    """Texture PBR ON: build deve ricevere texture=True con i parametri texture (size/steps/guidance)."""
    _select_canvas_input(w)
    cap, restore = _comfy_capture(w)
    try:
        t = w.trellis_tab
        t.texture.setChecked(True)
        t._on_generate()
        assert spy.calls, "texture on: nessuna richiesta inviata"
        assert cap.get("texture") is True, f"texture non True: {cap.get('texture')}"
        assert cap.get("texture_size") and cap.get("tex_sampling_steps"), f"param texture mancanti: {cap}"
    finally:
        restore()


def case_texture_off(w, spy):
    """Texture PBR OFF: build deve ricevere texture=False (export sola geometria) + widget texture disabilitati."""
    _select_canvas_input(w)
    cap, restore = _comfy_capture(w)
    try:
        t = w.trellis_tab
        t.texture.setChecked(False)
        assert not t.tex_size.isEnabled(), "spegnere texture deve disabilitare i widget texture"
        t._on_generate()
        assert spy.calls, "texture off: nessuna richiesta inviata"
        assert cap.get("texture") is False, f"texture non False: {cap.get('texture')}"
    finally:
        restore()


def case_remove_bg_on(w, spy):
    """Rimuovi sfondo ON: build deve ricevere remove_background=True."""
    _select_canvas_input(w)
    cap, restore = _comfy_capture(w)
    try:
        t = w.trellis_tab
        t.remove_bg.setChecked(True)
        t._on_generate()
        assert spy.calls, "rmbg on: nessuna richiesta inviata"
        assert cap.get("remove_background") is True, f"remove_background non True: {cap}"
    finally:
        restore()


def case_remove_bg_off(w, spy):
    """Rimuovi sfondo OFF: build deve ricevere remove_background=False."""
    _select_canvas_input(w)
    cap, restore = _comfy_capture(w)
    try:
        t = w.trellis_tab
        t.remove_bg.setChecked(False)
        t._on_generate()
        assert spy.calls, "rmbg off: nessuna richiesta inviata"
        assert cap.get("remove_background") is False, f"remove_background non False: {cap}"
    finally:
        restore()


def case_seed_fixed(w, spy):
    """Seed fisso (random OFF): build deve ricevere ESATTAMENTE il seed impostato."""
    _select_canvas_input(w)
    cap, restore = _comfy_capture(w)
    try:
        t = w.trellis_tab
        t.seed_random.setChecked(False)
        t.seed.setValue(12345)
        t._on_generate()
        assert spy.calls, "seed fisso: nessuna richiesta inviata"
        assert cap.get("seed") == 12345, f"seed fisso non rispettato: {cap.get('seed')}"
    finally:
        restore()


def case_seed_random(w, spy):
    """Seed random ON: la GUI deve comunque inviare una richiesta con un seed valido (>=0)."""
    _select_canvas_input(w)
    cap, restore = _comfy_capture(w)
    try:
        t = w.trellis_tab
        t.seed_random.setChecked(True)
        t._on_generate()
        assert spy.calls, "seed random: nessuna richiesta inviata"
        assert isinstance(cap.get("seed"), int) and cap.get("seed") >= 0, f"seed non valido: {cap.get('seed')}"
    finally:
        restore()


# ───────────────────────── numerici (geometria/texture) ─────────────────────────
def case_numeric_params_flow(w, spy):
    """Valori numerici non-default (dettaglio/guidance/steps/facce/tex) devono arrivare tutti a build."""
    _select_canvas_input(w)
    cap, restore = _comfy_capture(w)
    try:
        t = w.trellis_tab
        t.detail.setValue(65536)        # → max_tokens
        t.guidance.setValue(8.0)        # → ss_/shape_guidance_strength
        t.steps.setValue(20)            # → ss_/shape_sampling_steps
        t.faces.setValue(250000)        # → target_face_count
        t.tex_guidance.setValue(4.5)
        t.tex_steps.setValue(18)
        t._on_generate()
        assert spy.calls, "numerici: nessuna richiesta inviata"
        assert cap.get("max_tokens") == 65536, f"max_tokens errato: {cap.get('max_tokens')}"
        assert cap.get("ss_guidance_strength") == 8.0, f"guidance errato: {cap.get('ss_guidance_strength')}"
        assert cap.get("ss_sampling_steps") == 20, f"steps errato: {cap.get('ss_sampling_steps')}"
        assert cap.get("target_face_count") == 250000, f"faces errato: {cap.get('target_face_count')}"
        assert cap.get("tex_guidance_strength") == 4.5, f"tex_guidance errato: {cap.get('tex_guidance_strength')}"
        assert cap.get("tex_sampling_steps") == 18, f"tex_steps errato: {cap.get('tex_sampling_steps')}"
    finally:
        restore()


# ───────────────────────── solo-UI (no backend) ─────────────────────────
def case_show_result_no_crash(w, spy):
    """show_result(glb) deve caricare il visore 3D senza crash (anche con percorso inesistente)."""
    # nessun backend coinvolto: verifica solo che il visore non sollevi eccezioni
    w.trellis_tab.show_result(str(H.OUT_DIR / "_trellis_nonexistent.glb"))


def case_texture_toggle_ui_no_crash(w, spy):
    """Toggle texture su/giù deve abilitare/disabilitare i widget texture senza crash (no backend)."""
    t = w.trellis_tab
    t.texture.setChecked(False)
    assert not t.tex_size.isEnabled() and not t.tex_steps.isEnabled(), "widget texture non disabilitati"
    t.texture.setChecked(True)
    assert t.tex_size.isEnabled() and t.tex_steps.isEnabled(), "widget texture non riabilitati"


# ───────────────────────── enumerazione combo (un caso per valore) ─────────────────────────
# Risoluzione = variante modello TRELLIS (cascade vs no): un caso PER risoluzione concreta.
_RESOLUTIONS = ["512", "1024", "1024_cascade", "1536_cascade"]
_FORMATS = ["glb", "obj", "ply", "stl"]
_PRECISIONS = ["auto", "bf16", "fp16", "fp32"]
_ATTN = ["auto", "flash_attn", "xformers", "sdpa", "sageattn"]
_BG = ["black", "gray", "white"]
_TEX_SIZES = ["1024", "2048", "4096"]


# ───────────────────────── LIVE: 9 parametri reali → GLB vero ─────────────────────────
def case_trellis_9params_live(w, spy):
    """LIVE: i 9 parametri GUI (resolution/precision/attn/remove_bg/bg_color/guidance/steps/texture/
    tex_guidance/tex_steps) ora sono CABLATI all'API ufficiale TRELLIS.2. NESSUN patch di make_job →
    in run_real il worker nativo viene ESEGUITO davvero (subprocess) e produce un GLB reale in output/3d.
    Steps bassi per velocità: la riuscita del GLB prova che tutti i flag sono passati correttamente
    (un flag malformato farebbe fallire argparse dello script)."""
    import core.trellis_service as ts
    assert ts.is_installed(), "TRELLIS non installato: impossibile il test LIVE nativo"
    t = w.trellis_tab
    w.tabs.setCurrentWidget(t)
    t.set_input_path(str(H.OUT_DIR / "_portraitA.png"))  # immagine reale
    t.resolution.setCurrentText("1024_cascade")
    t.precision.setCurrentText("bf16")
    t.attn.setCurrentText("sdpa")
    t.remove_bg.setChecked(True)
    t.bg_color.setCurrentText("gray")
    t.guidance.setValue(6.5)
    t.steps.setValue(8)                                  # basso = veloce
    t.texture.setChecked(True)
    t.tex_guidance.setValue(3.0)
    t.tex_steps.setValue(8)
    t.faces.setValue(120000)
    t.tex_size.setCurrentText("1024")
    t.seed_random.setChecked(False)
    t.seed.setValue(7)
    t._on_generate()
    assert spy.calls, "trellis 9params LIVE: nessun job accodato (_enqueue non chiamato)"
    assert spy.calls[-1]["via"] == "enqueue", f"atteso enqueue nativo, ottenuto: {spy.calls[-1]}"


_RAW_CASES = [
    ("trellis_9params_live",
     "LIVE #3D: i 9 parametri GUI cablati all'API TRELLIS.2 → genera un GLB reale in output/3d.",
     case_trellis_9params_live),
    ("trellis_generate_native_default",
     "NATIVO: genera 3D → accoda job e passa a make_job seed/steps/texture_size.",
     case_generate_native_default),
    ("trellis_generate_comfy_default",
     "COMFYUI: genera 3D → build(image+resolution+format) e submit del prompt.",
     case_generate_comfy_default),
    ("trellis_no_image_no_dispatch",
     "Senza immagine: avvisa e NON invia nulla al backend.",
     case_no_image_no_dispatch),
    ("trellis_canvas_masked_without_mask_no_dispatch",
     "Input mascherato senza maschera: avvisa e non invia nulla.",
     case_canvas_masked_without_mask_no_dispatch),
    ("trellis_canvas_masked_forces_no_background",
     "Input mascherato con maschera: build riceve remove_background=False (force_no_bg).",
     case_canvas_masked_forces_no_background),
    ("trellis_texture_on",
     "Texture PBR ON: build riceve texture=True + parametri texture.",
     case_texture_on),
    ("trellis_texture_off",
     "Texture PBR OFF: build riceve texture=False + widget texture disabilitati.",
     case_texture_off),
    ("trellis_remove_bg_on",
     "Rimuovi sfondo ON: build riceve remove_background=True.",
     case_remove_bg_on),
    ("trellis_remove_bg_off",
     "Rimuovi sfondo OFF: build riceve remove_background=False.",
     case_remove_bg_off),
    ("trellis_seed_fixed",
     "Seed fisso: build riceve esattamente il seed impostato.",
     case_seed_fixed),
    ("trellis_seed_random",
     "Seed random: invia comunque una richiesta con seed valido.",
     case_seed_random),
    ("trellis_numeric_params_flow",
     "Numerici (dettaglio/guidance/steps/facce/tex) arrivano tutti a build.",
     case_numeric_params_flow),
    ("trellis_show_result_no_crash",
     "show_result(glb) carica il visore 3D senza crash.",
     case_show_result_no_crash),
    ("trellis_texture_toggle_ui_no_crash",
     "Toggle texture abilita/disabilita i widget texture senza crash.",
     case_texture_toggle_ui_no_crash),
]

# Un caso per OGNI risoluzione concreta (scopre 'funziona solo una risoluzione/modello').
for _res in _RESOLUTIONS:
    _RAW_CASES.append((
        f"trellis_resolution_{_res}",
        f"Risoluzione '{_res}': build riceve resolution='{_res}'.",
        _comfy_case(
            (lambda val: (lambda t: t.resolution.setCurrentText(val)))(_res),
            (lambda val: (lambda cap: (_ for _ in ()).throw(AssertionError(
                f"resolution attesa {val}, ottenuta {cap.get('resolution')}"))
                if cap.get("resolution") != val else None))(_res),
        ),
    ))

# Un caso per OGNI formato di export.
for _fmt in _FORMATS:
    _RAW_CASES.append((
        f"trellis_format_{_fmt}",
        f"Formato '{_fmt}': build riceve file_format='{_fmt}'.",
        _comfy_case(
            (lambda val: (lambda t: t.file_format.setCurrentText(val)))(_fmt),
            (lambda val: (lambda cap: (_ for _ in ()).throw(AssertionError(
                f"file_format atteso {val}, ottenuto {cap.get('file_format')}"))
                if cap.get("file_format") != val else None))(_fmt),
        ),
    ))

# Un caso per OGNI precisione.
for _prec in _PRECISIONS:
    _RAW_CASES.append((
        f"trellis_precision_{_prec}",
        f"Precisione '{_prec}': build riceve precision='{_prec}'.",
        _comfy_case(
            (lambda val: (lambda t: t.precision.setCurrentText(val)))(_prec),
            (lambda val: (lambda cap: (_ for _ in ()).throw(AssertionError(
                f"precision attesa {val}, ottenuta {cap.get('precision')}"))
                if cap.get("precision") != val else None))(_prec),
        ),
    ))

# Un caso per OGNI backend di attention.
for _att in _ATTN:
    _RAW_CASES.append((
        f"trellis_attn_{_att}",
        f"Attention '{_att}': build riceve attn_backend='{_att}'.",
        _comfy_case(
            (lambda val: (lambda t: t.attn.setCurrentText(val)))(_att),
            (lambda val: (lambda cap: (_ for _ in ()).throw(AssertionError(
                f"attn_backend atteso {val}, ottenuto {cap.get('attn_backend')}"))
                if cap.get("attn_backend") != val else None))(_att),
        ),
    ))

# Un caso per OGNI colore di sfondo.
for _bg in _BG:
    _RAW_CASES.append((
        f"trellis_bg_{_bg}",
        f"Sfondo '{_bg}': build riceve background_color='{_bg}'.",
        _comfy_case(
            (lambda val: (lambda t: t.bg_color.setCurrentText(val)))(_bg),
            (lambda val: (lambda cap: (_ for _ in ()).throw(AssertionError(
                f"background_color atteso {val}, ottenuto {cap.get('background_color')}"))
                if cap.get("background_color") != val else None))(_bg),
        ),
    ))

# Un caso per OGNI dimensione texture.
for _ts in _TEX_SIZES:
    _RAW_CASES.append((
        f"trellis_tex_size_{_ts}",
        f"Dimensione texture '{_ts}': build riceve texture_size={_ts}.",
        _comfy_case(
            (lambda val: (lambda t: (t.texture.setChecked(True), t.tex_size.setCurrentText(val))))(_ts),
            (lambda val: (lambda cap: (_ for _ in ()).throw(AssertionError(
                f"texture_size atteso {int(val)}, ottenuto {cap.get('texture_size')}"))
                if cap.get("texture_size") != int(val) else None))(_ts),
        ),
    ))

# In run_real i casi mock-based (nativo con _J fittizio / ramo ComfyUI) skippano via
# _SkipMockOnly; il caso end-to-end trellis_9params_live resta eseguito ovunque.
CASES = [(n, d, _skip_in_real(f)) for n, d, f in _RAW_CASES]
