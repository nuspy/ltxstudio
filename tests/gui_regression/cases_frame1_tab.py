"""Casi di regressione GUI-driven per il tab «frame1_tab» (Frame 1 / immagine di partenza).

Ogni caso imposta i campi del tab + invoca l'AZIONE di MainWindow; lo Spy dell'harness cattura la
richiesta GUI→backend SENZA eseguirla (no GPU, no rete). Si verifica no-crash + (dove l'azione DEVE
mandare una richiesta) assert mirati su spy.calls.

Copertura:
  • generate_image           — un caso per OGNI modello concreto (FLUX/Z-Image/Ernie) + opzioni
  • generate_inpaint(inpaint)— un caso per OGNI modello inpaint (hardcoded + dinamico) + opzioni
  • generate_inpaint(outpaint)
  • generate_integrate       — Kontext (nativo) / FLUX.2 / puzzle_lora / guidance
  • generate_recolor         — Kontext (nativo) / FLUX.2 / guardia target mancante
  • generate_upscale         — un caso per OGNI modello upscale + scala + rifinitura
  • generate_compose         — multiref / stitch / FLUX.2 / guardia <2 foto
  • segment_by_text          — detector GroundingDINO(+SAM) / soglia

Backend: data/backend_config.json ha image=native → gli handler instradano al worker NATIVO
(_NativeImageWorker → _enqueue, catturato dallo Spy). I rami FLUX.2/puzzle/refine restano su
ComfyUI (driver.submit, catturato come via='comfy'). I casi nativi forzano native_driver.is_up=True
(l'InferenceServer non gira nei test) così la richiesta viene comunque costruita e catturata.
"""
from __future__ import annotations

import os

import harness as H

# Import statici (sicuri: solo costanti/funzioni pure del tab)
try:
    from ui.tabs.frame1_tab import INPAINT_MODELS, GEN_ARCH, _is_flux2
except Exception:                                            # pragma: no cover
    INPAINT_MODELS, GEN_ARCH = {}, {}
    def _is_flux2(n):  # type: ignore
        n = (n or "").lower()
        return "flux_2" in n or "flux2" in n or "flux 2" in n


# ───────────────────────── inventario reale (filesystem) ─────────────────────────
def _load_inventory():
    """Inventario reale dal filesystem (offline_inventory): rende i casi 'per-modello' aderenti
    a ciò che è davvero installato. Se fallisce, ritorna liste vuote (i casi degradano a no-op)."""
    try:
        from core import offline_inventory as oi
        return (list(oi.models_in("diffusion_models")),
                list(oi.models_in("checkpoints")),
                list(oi.models_in("upscale_models")))
    except Exception:                                        # noqa: BLE001
        return [], [], []


_DIFFUSION, _CHECKPOINTS, _UPSCALE = _load_inventory()


def _gen_models_for_arch(arch_lbl: str):
    """Replica il filtro di Frame1Tab._reload_gen (con NSFW=ON, precisione=Tutte) per enumerare
    a import-time i modelli concreti che il dropdown «Genera immagine» esporrà per l'architettura.
    Il filtro è per ARCHITETTURA DAL FILE (core/arch_filter: header, poi cartella, poi nome), non più per parole
    nel nome: Klein, Fill, Kontext e Redux restano fuori dai t2i FLUX (vivono nei pannelli di modifica)."""
    from core import arch_filter as af
    out: list[str] = [m for m in _DIFFUSION if af.is_t2i_for_arch(arch_lbl, m)]
    if arch_lbl == "FLUX":
        for c in _CHECKPOINTS:
            if af.is_t2i_for_arch("FLUX", c) and af.family(c) != "flux2" and c not in out:
                out.append(c)
    return out


# ───────────────────────── helper comuni ─────────────────────────
def _force_native_up():
    """L'InferenceServer nativo non gira nei test: forziamo is_up=True così gli handler nativi
    costruiscono e ACCODANO la richiesta (catturata dallo Spy via _enqueue)."""
    try:
        from core import native_driver
        native_driver.is_up = lambda: True
    except Exception:                                        # noqa: BLE001
        pass


def _populate_inventory(w):
    """Inietta l'inventario reale nel tab (di norma lo farebbe _refresh_status quando il backend
    è 'up'); nei test va fatto a mano perché il timer di stato non scatta."""
    f = w.frame1_tab
    try:
        f.set_model_inventory(_DIFFUSION, _CHECKPOINTS)
    except Exception:                                        # noqa: BLE001
        pass
    try:
        f.set_upscale_models(_UPSCALE)
    except Exception:                                        # noqa: BLE001
        pass


def _select_combo(combo, name: str) -> bool:
    """Seleziona nel combo la voce con itemData==name (fallback itemText). True se trovata."""
    i = combo.findData(name)
    if i < 0:
        i = combo.findText(name)
    if i < 0:
        return False
    combo.setCurrentIndex(i)
    return True


def _draw_center_mask(cv):
    """Disegna una maschera centrale (rettangolo bianco) sul canvas → inpaint ha un'area."""
    from PySide6.QtGui import QPainter, QColor
    from PySide6.QtCore import QRect
    pm = QPainter(cv.mask_image)
    pm.fillRect(QRect(cv.aspect_w // 4, cv.aspect_h // 4, cv.aspect_w // 2, cv.aspect_h // 2),
                QColor(255, 255, 255))
    pm.end()
    cv._refresh_mask()


def _add_canvas_layer(w):
    """Aggiunge un layer immagine importato (entry non-base) → has_layers()=True per «Integra»."""
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import QGraphicsPixmapItem
    cv = w.frame1_tab.canvas
    path = H.load_canvas_image(w)
    pix = QPixmap(path)
    it = QGraphicsPixmapItem(pix)
    s = min(cv.aspect_w / max(1, pix.width()), cv.aspect_h / max(1, pix.height())) * 0.6
    it.setScale(s)
    it.setPos(cv.aspect_w * 0.2, cv.aspect_h * 0.2)
    cv.scene.addItem(it)
    cv._entries.append({"item": it, "mask": cv._new_mask(), "name": "layer_test.png", "base": False})
    cv._reindex_z()
    return it


def _two_compose_images():
    """Crea (se mancano) due immagini di test su disco per «Componi più foto»."""
    a = str(H.OUT_DIR / "_compose_a.png")
    b = str(H.OUT_DIR / "_compose_b.png")
    if not os.path.exists(a):
        H.make_test_image(a, 512, 512)
    if not os.path.exists(b):
        H.make_test_image(b, 640, 480)
    return a, b


def _setup_gen(w, arch_lbl: str):
    """Prepara la sezione «Genera immagine»: inventario + NSFW (lista piena) + architettura."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")            # mostra TUTTI i modelli (anche NSFW)
    f.gen_arch.setCurrentText(arch_lbl)          # triggera _reload_gen
    f.gen_prompt.setPlainText("a red apple on a wooden table, photorealistic")
    return f


def _assert_image_request(spy, must_model: str | None = None):
    assert spy.calls, "nessuna richiesta inviata al backend"
    p = spy.calls[-1]["params"]
    if must_model is not None:
        got = p.get("model") or ""
        assert os.path.basename(str(got)) == os.path.basename(must_model) or got == must_model, \
            f"modello inviato inatteso: {got!r} (atteso {must_model!r})"
    return p


# ════════════════════════ GENERAZIONE IMMAGINE (per-modello) ════════════════════════
def _make_gen_model_case(arch_lbl: str, model_name: str):
    def case(w, spy):
        f = _setup_gen(w, arch_lbl)
        if not _select_combo(f.gen_model, model_name):
            raise AssertionError(f"modello '{model_name}' assente dal dropdown per arch {arch_lbl} "
                                 f"(items={[f.gen_model.itemText(i) for i in range(f.gen_model.count())]})")
        w.generate_image()
        p = _assert_image_request(spy, must_model=model_name)
        assert p.get("prompt"), f"prompt mancante: {p}"
        assert p.get("arch"), f"arch mancante: {p}"
        assert p.get("width") and p.get("height"), f"dimensioni mancanti: {p}"
    return case


def _gen_model_cases():
    cases = []
    arch_map = {"FLUX": "flux", "Z-Image": "zimage", "Ernie": "ernie"}
    for arch_lbl in GEN_ARCH.keys():
        if arch_lbl in ("Krea 2", "Qwen-Image 2.1"):       # voci di CATALOGO, non file: casi propri
            continue                                        # (cases_qwen21_image, test_qtbot_krea2_edit)
        for idx, m in enumerate(_gen_models_for_arch(arch_lbl)):
            safe = "".join(c if c.isalnum() else "_" for c in os.path.basename(m))[:48]
            name = f"frame1_generate_image_{arch_map.get(arch_lbl, arch_lbl)}__{idx:02d}_{safe}"
            cases.append((name,
                          f"Genera immagine [{arch_lbl}] con il modello '{m}': la GUI deve inviare al "
                          f"backend nativo model={m}, arch, prompt, dimensioni.",
                          _make_gen_model_case(arch_lbl, m)))
    return cases


# ════════════════════════ GENERAZIONE: opzioni / toggle ════════════════════════
def _first_gen_model(f) -> str | None:
    for i in range(f.gen_model.count()):
        d = f.gen_model.itemData(i)
        if d and d != "__auto__":
            f.gen_model.setCurrentIndex(i)
            return d
    return None


def case_gen_auto_select(w, spy):
    """Selezione modello = «⚡ Auto»: la GUI deve risolvere a un modello concreto e inviarlo."""
    f = _setup_gen(w, "FLUX")
    if f.gen_model.count() and f.gen_model.itemData(0) == "__auto__":
        f.gen_model.setCurrentIndex(0)
    if not _first_gen_model(f):            # nessun modello installato → niente da testare
        return
    f.gen_model.setCurrentIndex(0)        # torna su Auto
    if f.gen_model.itemData(0) != "__auto__":
        return
    w.generate_image()
    p = _assert_image_request(spy)
    assert p.get("model") and not str(p["model"]).startswith("("), f"auto non risolto: {p}"


def case_gen_enhance_toggle(w, spy):
    """Toggle «Migliora prompt»: no-crash + richiesta inviata con un modello concreto."""
    f = _setup_gen(w, "FLUX")
    if not _first_gen_model(f):
        return
    f.gen_enhance.setChecked(True)
    w.generate_image()
    _assert_image_request(spy)


def case_gen_seed_fixed(w, spy):
    """Seed FISSO (checkbox random OFF): la GUI deve inviare quel seed esatto."""
    f = _setup_gen(w, "FLUX")
    if not _first_gen_model(f):
        return
    f.gen_seed_rand.setChecked(False)
    f.gen_seed.setValue(12345)
    w.generate_image()
    p = _assert_image_request(spy)
    assert p.get("seed") == 12345, f"seed fisso non propagato: {p.get('seed')}"


def case_gen_cfg_steps(w, spy):
    """CFG e Steps personalizzati: devono arrivare nei params (guidance/steps)."""
    f = _setup_gen(w, "FLUX")
    if not _first_gen_model(f):
        return
    f.gen_steps.setValue(28)
    f.gen_cfg.setValue(4.5)
    w.generate_image()
    p = _assert_image_request(spy)
    assert p.get("steps") == 28, f"steps non propagati: {p.get('steps')}"
    assert abs(float(p.get("guidance", 0)) - 4.5) < 0.01, f"cfg/guidance non propagata: {p.get('guidance')}"


def case_gen_sampler_scheduler(w, spy):
    """Sampler/scheduler espliciti (non-auto): no-crash + richiesta inviata."""
    f = _setup_gen(w, "FLUX")
    if not _first_gen_model(f):
        return
    i = f.gen_sampler.findText("euler")
    if i >= 0:
        f.gen_sampler.setCurrentIndex(i)
    i = f.gen_scheduler.findText("karras")
    if i >= 0:
        f.gen_scheduler.setCurrentIndex(i)
    w.generate_image()
    _assert_image_request(spy)


def case_gen_pulid_face(w, spy):
    """PuLID abilitato con foto volto: la GUI deve inviare pulid_face + pulid_strength."""
    f = _setup_gen(w, "FLUX")
    if not _first_gen_model(f):
        return
    face = H.make_test_image(str(H.OUT_DIR / "_pulid_face.png"), 256, 256)
    f.gen_pulid_enable.setChecked(True)
    f.gen_pulid_face.setText(face)
    f.gen_pulid_strength.setValue(0.7)
    w.generate_image()
    p = _assert_image_request(spy)
    assert p.get("pulid_face") == face, f"pulid_face non propagato: {p.get('pulid_face')}"
    assert abs(float(p.get("pulid_strength", 0)) - 0.7) < 0.01, f"pulid_strength errato: {p.get('pulid_strength')}"


def case_gen_empty_prompt_guard(w, spy):
    """Prompt vuoto: la GUI NON deve inviare nulla (guardia con warning)."""
    f = _setup_gen(w, "FLUX")
    _first_gen_model(f)
    f.gen_prompt.setPlainText("")
    f.prompt.setPlainText("")
    w.generate_image()
    assert not spy.calls, f"prompt vuoto ma richiesta inviata comunque: {spy.calls}"


# ════════════════════════ INPAINT (per-modello) ════════════════════════
def _make_inpaint_case(model_label: str):
    def case(w, spy):
        _force_native_up()
        _populate_inventory(w)
        f = w.frame1_tab
        f.img_nsfw.setCurrentText("NSFW")
        H.load_canvas_image(w)
        cv = f.canvas
        if not _select_combo(f.inp_model, model_label):
            raise AssertionError(f"modello inpaint '{model_label}' assente dal dropdown")
        f.inpaint_prompt.setPlainText("fill the masked area with grass")
        _draw_center_mask(cv)
        assert cv.has_mask(), "maschera non disegnata"
        w.generate_inpaint("inpaint")
        # FLUX.2 Klein = ramo edit-evoluto ancora su ComfyUI/deferred (#14/#44): il dispatch nativo NON è
        # garantito. Per questi modelli basta che non crashi; per i FLUX.1 (nativi) la richiesta DEVE partire.
        _spec = INPAINT_MODELS.get(model_label, {})
        if _is_flux2(_spec.get("unet", "")):
            return
        assert spy.calls, f"inpaint '{model_label}': nessuna richiesta inviata"
        c = spy.calls[-1]
        # ramo NATIVO (FLUX.1 Fill) → params con image_path+mask_path; ramo FLUX.2/checkpoint → ComfyUI
        if c["via"] == "enqueue":
            p = c["params"]
            assert p.get("image_path") and p.get("mask_path"), f"inpaint nativo senza image/mask: {p}"
            assert p.get("arch") == "inpaint", f"arch inpaint atteso: {p.get('arch')}"
    return case


def _inpaint_model_cases():
    cases = []
    for label, spec in INPAINT_MODELS.items():
        if not spec.get("unet") and not spec.get("arch"):
            continue            # salta «Nessuno (solo LTX I2V)»
        safe = "".join(c if c.isalnum() else "_" for c in label)[:48]
        cases.append((f"frame1_inpaint__{safe}",
                      f"Inpaint con modello '{label}': foto+maschera → richiesta con image_path/mask_path.",
                      _make_inpaint_case(label)))
    return cases


def case_inpaint_dynamic_unet(w, spy):
    """Inpaint con un UNET Flux DINAMICO (dall'inventario): copre il ramo _dynamic_unets."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    dyn = getattr(f, "_dynamic_unets", {})
    if not dyn:
        return                  # nessun UNET dinamico installato
    label = next(iter(dyn.keys()))
    H.load_canvas_image(w)
    _select_combo(f.inp_model, label)
    f.inpaint_prompt.setPlainText("repair the area")
    _draw_center_mask(f.canvas)
    w.generate_inpaint("inpaint")
    assert spy.calls, "inpaint UNET dinamico: nessuna richiesta inviata"


def case_inpaint_strength_steps_seed(w, spy):
    """Opzioni inpaint (strength/steps/seed fisso): no-crash + richiesta inviata (ramo nativo Fill)."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    H.load_canvas_image(w)
    # scegli un modello FLUX.1 Fill (ramo nativo) se presente
    target = None
    for label, spec in INPAINT_MODELS.items():
        u = spec.get("unet")
        if u and not _is_flux2(u):
            target = label
            break
    if not target or not _select_combo(f.inp_model, target):
        return
    f.inpaint_prompt.setPlainText("fill")
    f.strength.setValue(60)
    f.inp_steps.setValue(15)
    f.inp_seed_rand.setChecked(False)
    f.inp_seed.setValue(777)
    _draw_center_mask(f.canvas)
    w.generate_inpaint("inpaint")
    assert spy.calls, "inpaint opzioni: nessuna richiesta inviata"
    c = spy.calls[-1]
    if c["via"] == "enqueue":
        assert c["params"].get("seed") == 777, f"seed inpaint non propagato: {c['params'].get('seed')}"


def case_inpaint_no_mask_guard(w, spy):
    """Inpaint senza maschera: la GUI NON deve inviare nulla (guardia 'no mask')."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    H.load_canvas_image(w)
    # modello FLUX.1 (ramo nativo) per arrivare alla guardia maschera
    for label, spec in INPAINT_MODELS.items():
        if spec.get("unet") and not _is_flux2(spec["unet"]):
            _select_combo(f.inp_model, label)
            break
    f.inpaint_prompt.setPlainText("x")
    # nessuna maschera disegnata
    w.generate_inpaint("inpaint")
    assert not spy.calls, f"inpaint senza maschera ma richiesta inviata: {spy.calls}"


def case_inpaint_no_image_guard(w, spy):
    """Inpaint senza foto nel canvas: nessuna richiesta (guardia 'no image')."""
    _force_native_up()
    _populate_inventory(w)
    w.frame1_tab.inpaint_prompt.setPlainText("x")
    w.generate_inpaint("inpaint")
    assert not spy.calls, f"inpaint senza foto ma richiesta inviata: {spy.calls}"


# ════════════════════════ OUTPAINT ════════════════════════
def case_outpaint_aspect_preserved(w, spy):
    """Outpaint: caricando la foto il canvas ne adotta le proporzioni; allargando il frame in modo
    proporzionale (come fa l'utente) le proporzioni restano e nasce l'area da estendere."""
    H.load_canvas_image(w)            # 768x512 → il canvas adotta ~3:2
    cv = w.frame1_tab.canvas
    r0 = cv.aspect_w / cv.aspect_h
    cv.set_aspect(int(cv.aspect_w * 1.4) // 8 * 8, int(cv.aspect_h * 1.4) // 8 * 8)   # allarga il frame
    assert abs(cv.aspect_w / cv.aspect_h - r0) < 0.05, "le proporzioni non sono state mantenute"
    assert cv.has_outpaint_area(), "dopo l'allargamento deve esserci area di outpaint"


def case_outpaint_generate(w, spy):
    """Outpaint con un modello FLUX.1: foto con bordo → richiesta inviata (image_path+mask_path nativo)."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    H.load_canvas_image(w)
    cv = f.canvas
    cv.set_aspect(int(cv.aspect_w * 1.4) // 8 * 8, int(cv.aspect_h * 1.4) // 8 * 8)   # bordo da estendere
    for label, spec in INPAINT_MODELS.items():
        if spec.get("unet") and not _is_flux2(spec["unet"]):
            _select_combo(f.inp_model, label)
            break
    f.outpaint_prompt.setPlainText("extend the scenery naturally")
    assert f.canvas.has_outpaint_area(), "manca area da estendere"
    w.generate_inpaint("outpaint")
    assert spy.calls, "outpaint: nessuna richiesta inviata"
    c = spy.calls[-1]
    if c["via"] == "enqueue":
        p = c["params"]
        assert p.get("image_path") and p.get("mask_path"), f"outpaint nativo senza image/mask: {p}"


# ════════════════════════ INTEGRA ════════════════════════
def case_integrate_kontext_native(w, spy):
    """Integra (Kontext FLUX.1, nativo): foto base + 1 layer → richiesta kontext con image_path."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    H.load_canvas_image(w)
    _add_canvas_layer(w)
    assert f.canvas.has_layers(), "nessun layer aggiunto"
    f.integrate_engine.setCurrentIndex(0)        # Kontext
    f.integrate_puzzle.setChecked(False)
    f.integrate_prompt.setPlainText("integrate the object into the scene")
    w.generate_integrate()
    assert spy.calls, "integrate Kontext: nessuna richiesta inviata"
    c = spy.calls[-1]
    if c["via"] == "enqueue":
        p = c["params"]
        # Integra mask-based: se il canvas esporta la maschera zona-edit → arch=inpaint (FLUX Fill, fonde
        # l'oggetto); altrimenti kontext (preserva). Entrambi validi.
        assert p.get("arch") in ("kontext", "inpaint"), f"arch kontext/inpaint atteso: {p.get('arch')}"
        assert p.get("image_path"), f"image_path mancante: {p}"
        if p.get("arch") == "inpaint":
            assert p.get("mask_path"), f"integra mask-based senza mask_path: {p}"


def case_integrate_guidance_seed(w, spy):
    """Integra: guidance e seed fisso devono propagarsi (ramo nativo Kontext)."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    H.load_canvas_image(w)
    _add_canvas_layer(w)
    f.integrate_engine.setCurrentIndex(0)
    f.integrate_puzzle.setChecked(False)
    f.int_guidance.setValue(35)                  # → 3.5
    f.int_seed_rand.setChecked(False)
    f.int_seed.setValue(4242)
    f.integrate_prompt.setPlainText("blend")
    w.generate_integrate()
    assert spy.calls, "integrate guidance: nessuna richiesta inviata"
    c = spy.calls[-1]
    if c["via"] == "enqueue":
        p = c["params"]
        assert p.get("seed") == 4242, f"seed integrate non propagato: {p.get('seed')}"
        # guidance: ramo kontext usa lo slider (3.5); ramo mask-based (FLUX Fill) usa il regime Fill (~30).
        g = float(p.get("guidance", 0))
        ok = abs(g - 3.5) < 0.01 if p.get("arch") == "kontext" else g >= 20.0
        assert ok, f"guidance integrate errata per arch={p.get('arch')}: {g}"


def case_integrate_pulid_separation(w, spy):
    """I due container PuLID sono SEPARATI: quello in «Genera immagine» agisce solo sulla
    generazione, quello in «Image to Image» solo sull'integra (richiesta utente 22/07)."""
    f = w.frame1_tab
    face = H.make_test_image(str(H.OUT_DIR / "_pulid_int_face.png"), 256, 256)
    # Solo il box dell'Image to Image attivo → get_inpaint_config lo espone, get_gen_config NO.
    f.gen_pulid_enable.setChecked(False)
    f.int_pulid_enable.setChecked(True)
    f.int_pulid_face.setText(face)
    f.int_pulid_strength.setValue(0.65)
    cfg = f.get_inpaint_config()
    assert cfg.get("integrate_pulid_face") == face, f"int PuLID non propagato: {cfg.get('integrate_pulid_face')}"
    assert abs(float(cfg.get("integrate_pulid_strength", 0)) - 0.65) < 0.01, \
        f"int pulid_strength errata: {cfg.get('integrate_pulid_strength')}"
    assert f.get_gen_config().get("pulid_face") is None, "il PuLID dell'integra è finito nella generazione"
    # Viceversa: solo il box di Genera attivo → l'integra NON lo eredita.
    f.int_pulid_enable.setChecked(False)
    f.gen_pulid_enable.setChecked(True)
    f.gen_pulid_face.setText(face)
    assert f.get_inpaint_config().get("integrate_pulid_face") is None, \
        "il PuLID della generazione è finito nell'integra"
    assert f.get_gen_config().get("pulid_face") == face, "gen PuLID non propagato"
    f.gen_pulid_enable.setChecked(False)


def case_integrate_pulid_native(w, spy):
    """Image to Image (Kontext) col SUO box PuLID attivo: pulid_face arriva nella richiesta nativa
    (solo ramo kontext puro: il ramo mask-based/inpaint non applica PuLID, documentato)."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    H.load_canvas_image(w)
    _add_canvas_layer(w)
    face = H.make_test_image(str(H.OUT_DIR / "_pulid_int_face2.png"), 256, 256)
    f.integrate_engine.setCurrentIndex(0)        # Kontext
    f.integrate_puzzle.setChecked(False)
    f.gen_pulid_enable.setChecked(False)         # il box di Genera NON deve contare
    f.int_pulid_enable.setChecked(True)
    f.int_pulid_face.setText(face)
    f.integrate_prompt.setPlainText("integrate the object into the scene")
    w.generate_integrate()
    assert spy.calls, "integrate+PuLID: nessuna richiesta inviata"
    c = spy.calls[-1]
    if c["via"] == "enqueue":
        p = c["params"]
        if p.get("arch") == "kontext":           # ramo kontext puro: PuLID atteso
            assert p.get("pulid_face") == face, f"pulid_face non arrivato al job kontext: {p.get('pulid_face')}"
        else:                                    # ramo mask-based (inpaint): PuLID non applicato lì
            assert p.get("arch") == "inpaint", f"arch inatteso: {p.get('arch')}"
    f.int_pulid_enable.setChecked(False)


def case_ui_footer_and_renames(w, spy):
    """Layout 22/07: footer ANCORATI (Genera Video nel tab Video, Genera immagine nel sotto-tab),
    bottoni-azione con stile «generate», sotto-tab rinominato Modifica Immagine."""
    from PySide6.QtWidgets import QScrollArea

    def _outside_scroll(btn):
        p = btn.parent()
        while p is not None:
            if isinstance(p, QScrollArea):
                return False
            p = p.parent()
        return True
    # Genera Video: nel tab Video, FUORI dallo scroll (ancorato), stile generate, cablato a on_generate
    vb = w.video_tab.generate_btn
    assert _outside_scroll(vb), "Genera Video è dentro lo scroll: non è ancorato"
    assert vb.objectName() == "generate", "Genera Video senza stile #generate"
    assert w.gen_btn is vb, "alias main_window.gen_btn non punta al bottone del tab Video"
    assert w.cancel_btn is w.video_tab.stop_btn, "alias cancel_btn non punta a Interrompi del tab Video"
    # Genera immagine: footer ancorato nel sotto-tab + stile generate
    f = w.frame1_tab
    assert _outside_scroll(f.gen_btn), "Genera immagine è dentro lo scroll: non è ancorato"
    assert f.gen_btn.objectName() == "generate", "Genera immagine senza stile #generate"
    # Rinomina sotto-tab: l'etichetta deve essere quella nuova (tr f1.tab.inpaint = Modifica Immagine)
    from ui.i18n_manager import tr as _tr
    labels = [f.sub.tabText(i) for i in range(f.sub.count())]
    assert _tr("f1.tab.inpaint") in labels, f"sotto-tab Modifica Immagine assente: {labels}"
    assert "Inpaint / Outpaint" not in labels, f"vecchia etichetta ancora presente: {labels}"
    # Il box PuLID dell'Image to Image esiste e si disabilita coi motori non-Kontext
    assert hasattr(f, "int_pulid_enable"), "box PuLID dell'Image to Image mancante"
    f.integrate_engine.setCurrentIndex(2)        # Krea 2 Edit → PuLID OFF
    assert not f._int_pulid_group.isEnabled(), "int PuLID abilitato con motore non-Kontext"
    f.integrate_engine.setCurrentIndex(0)        # Kontext → PuLID ON
    assert f._int_pulid_group.isEnabled(), "int PuLID disabilitato col motore Kontext"


def case_integrate_flux2(w, spy):
    """Integra con motore FLUX.2 (ramo ComfyUI): richiede un modello FLUX.2 selezionato → richiesta inviata."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    H.load_canvas_image(w)
    _add_canvas_layer(w)
    f.integrate_engine.setCurrentIndex(1)        # FLUX.2 → ripopola integrate_model
    if f.integrate_model.count() == 0 or f.integrate_model.currentText().startswith("("):
        return                                   # nessun modello FLUX.2 installato
    f.integrate_model.setCurrentIndex(0)
    f.integrate_prompt.setPlainText("integrate")
    w.generate_integrate()
    assert spy.calls, "integrate FLUX.2: nessuna richiesta inviata (modello selezionato ma niente submit)"


def case_integrate_puzzle_lora_flux1(w, spy):
    """Integra Kontext + puzzle_lora ON: forza il ramo ComfyUI (LoRA FLUX.1) → richiesta inviata."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    H.load_canvas_image(w)
    _add_canvas_layer(w)
    f.integrate_engine.setCurrentIndex(0)        # Kontext (puzzle abilitabile solo qui)
    if not f.integrate_puzzle.isEnabled():
        return
    f.integrate_puzzle.setChecked(True)
    f.integrate_prompt.setPlainText("integrate with puzzle lora")
    w.generate_integrate()
    assert spy.calls, "integrate puzzle_lora: nessuna richiesta inviata"


def case_integrate_no_layer_guard(w, spy):
    """Integra senza layer da fondere: nessuna richiesta (guardia 'no image to integrate')."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    H.load_canvas_image(w)                        # solo base, nessun layer
    f.integrate_prompt.setPlainText("x")
    w.generate_integrate()
    assert not spy.calls, f"integrate senza layer ma richiesta inviata: {spy.calls}"


# ════════════════════════ CAMBIA COLORE (recolor) ════════════════════════
def case_recolor_kontext_native(w, spy):
    """Cambia colore (Kontext nativo): target+soggetto → richiesta kontext con image_path e prompt recolor."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    H.load_canvas_image(w)
    f.color_engine.setCurrentIndex(0)            # Kontext
    f.color_target.setText("#ff0000")
    f.color_subject.setText("the car")
    f.color_steps.setValue(18)
    w.generate_recolor()
    assert spy.calls, "recolor Kontext: nessuna richiesta inviata"
    c = spy.calls[-1]
    if c["via"] == "enqueue":
        p = c["params"]
        assert p.get("arch") == "kontext", f"arch kontext atteso: {p.get('arch')}"
        assert "red" in (p.get("prompt") or "").lower() or "#ff0000" in (p.get("prompt") or "").lower(), \
            f"prompt recolor non contiene il colore: {p.get('prompt')}"


def case_recolor_target_guard(w, spy):
    """Recolor senza colore target: nessuna richiesta (guardia 'color missing')."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    H.load_canvas_image(w)
    f.color_engine.setCurrentIndex(0)
    f.color_target.setText("")
    f.color_subject.setText("the car")
    w.generate_recolor()
    assert not spy.calls, f"recolor senza target ma richiesta inviata: {spy.calls}"


def case_recolor_flux2(w, spy):
    """Recolor con motore FLUX.2 (ramo ComfyUI): serve un modello FLUX.2 (dal tab Integra) → richiesta inviata."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    H.load_canvas_image(w)
    # FLUX.2 recolor legge integrate_model: popolalo selezionando engine Integra=FLUX.2
    f.integrate_engine.setCurrentIndex(1)
    if f.integrate_model.count() == 0 or f.integrate_model.currentText().startswith("("):
        return
    f.integrate_model.setCurrentIndex(0)
    f.color_engine.setCurrentIndex(1)            # FLUX.2
    f.color_target.setText("#00aa55")
    f.color_subject.setText("the shirt")
    w.generate_recolor()
    assert spy.calls, "recolor FLUX.2: nessuna richiesta inviata"


# ════════════════════════ UPSCALE (per-modello) ════════════════════════
def _make_upscale_case(model_name: str):
    def case(w, spy):
        _force_native_up()
        _populate_inventory(w)
        f = w.frame1_tab
        H.load_canvas_image(w)
        if not _select_combo(f.up_model, model_name):
            raise AssertionError(f"modello upscale '{model_name}' assente dal dropdown")
        f.up_mode.setCurrentIndex(0)             # Veloce (ramo nativo ESRGAN)
        w.generate_upscale()
        assert spy.calls, f"upscale '{model_name}': nessuna richiesta inviata"
        c = spy.calls[-1]
        if c["via"] == "enqueue":
            p = c["params"]
            assert p.get("arch") == "upscale", f"arch upscale atteso: {p.get('arch')}"
            assert p.get("image_path"), f"image_path mancante: {p}"
    return case


def _upscale_model_cases():
    cases = []
    for idx, m in enumerate(_UPSCALE):
        safe = "".join(c if c.isalnum() else "_" for c in os.path.basename(m))[:48]
        safe = f"{idx:02d}_{safe}"
        cases.append((f"frame1_upscale__{safe}",
                      f"Upscale (veloce/ESRGAN nativo) con il modello '{m}': richiesta con image_path+arch=upscale.",
                      _make_upscale_case(m)))
    return cases


def case_upscale_scale_factors(w, spy):
    """Fattori di scala 2x/3x/4x: ognuno deve produrre una richiesta (no-crash)."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    H.load_canvas_image(w)
    if not _UPSCALE:
        return
    _select_combo(f.up_model, _UPSCALE[0])
    f.up_mode.setCurrentIndex(0)
    for sf in ("2x", "3x", "4x"):
        i = f.up_scale.findText(sf)
        if i >= 0:
            f.up_scale.setCurrentIndex(i)
        spy.calls.clear()
        w.generate_upscale()
        assert spy.calls, f"upscale {sf}: nessuna richiesta inviata"


def case_upscale_refine_mode(w, spy):
    """Modalità «Rifinisci» con modello FLUX.2: ramo ComfyUI (upscaler.build) → richiesta inviata."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    H.load_canvas_image(w)
    if not _UPSCALE:
        return
    _select_combo(f.up_model, _UPSCALE[0])
    f.up_mode.setCurrentIndex(1)                 # Rifinisci → abilita up_refine_model
    if f.up_refine_model.count() == 0 or f.up_refine_model.currentText().startswith("("):
        return                                   # nessun FLUX.2 per la rifinitura
    f.up_refine_model.setCurrentIndex(0)
    f.up_refine_str.setValue(25)
    f.up_refine_steps.setValue(10)
    w.generate_upscale()
    assert spy.calls, "upscale refine: nessuna richiesta inviata"


def case_upscale_no_model_guard(w, spy):
    """Upscale senza modello selezionato: nessuna richiesta (guardia 'no upscale model')."""
    _force_native_up()
    f = w.frame1_tab
    H.load_canvas_image(w)
    f.up_model.clear()                           # svuota → currentText '' / nessun modello
    w.generate_upscale()
    assert not spy.calls, f"upscale senza modello ma richiesta inviata: {spy.calls}"


# ════════════════════════ COMPONI PIÙ FOTO (compose) ════════════════════════
def case_compose_multiref_kontext(w, spy):
    """Componi (multiref, Kontext nativo): 2 foto → stitch lato Python + richiesta kontext."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    a, b = _two_compose_images()
    f.compose_list.addItem(a)
    f.compose_list.addItem(b)
    f.integrate_engine.setCurrentIndex(0)        # engine compose deriva da integrate_engine → Kontext
    f.compose_method.setCurrentIndex(0)          # multiref
    f.compose_prompt.setPlainText("combine the two references into one scene")
    w.generate_compose()
    assert spy.calls, "compose multiref: nessuna richiesta inviata"
    c = spy.calls[-1]
    if c["via"] == "enqueue":
        assert c["params"].get("arch") == "kontext", f"arch kontext atteso: {c['params'].get('arch')}"


def case_compose_stitch(w, spy):
    """Componi (stitch): 2 foto → richiesta inviata (no-crash, metodo stitch)."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    a, b = _two_compose_images()
    f.compose_list.addItem(a)
    f.compose_list.addItem(b)
    f.integrate_engine.setCurrentIndex(0)
    f.compose_method.setCurrentIndex(1)          # stitch
    f.compose_prompt.setPlainText("place side by side")
    w.generate_compose()
    assert spy.calls, "compose stitch: nessuna richiesta inviata"


def case_compose_flux2(w, spy):
    """Componi con engine FLUX.2 (ramo ComfyUI, multi-reference vero): serve modello FLUX.2 → richiesta inviata."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    f.img_nsfw.setCurrentText("NSFW")
    a, b = _two_compose_images()
    f.compose_list.addItem(a)
    f.compose_list.addItem(b)
    f.integrate_engine.setCurrentIndex(1)        # FLUX.2 → popola integrate_model (usato anche da compose)
    if f.integrate_model.count() == 0 or f.integrate_model.currentText().startswith("("):
        return
    f.integrate_model.setCurrentIndex(0)
    f.compose_method.setCurrentIndex(0)
    f.compose_prompt.setPlainText("merge references")
    w.generate_compose()
    assert spy.calls, "compose FLUX.2: nessuna richiesta inviata"


def case_compose_need_two_guard(w, spy):
    """Componi con UNA sola foto: nessuna richiesta (guardia 'servono almeno 2')."""
    _force_native_up()
    _populate_inventory(w)
    f = w.frame1_tab
    a, _ = _two_compose_images()
    f.compose_list.addItem(a)
    f.compose_prompt.setPlainText("x")
    w.generate_compose()
    assert not spy.calls, f"compose con 1 foto ma richiesta inviata: {spy.calls}"


# ════════════════════════ SELEZIONE DA TESTO (segment) ════════════════════════
def case_segment_text_dino_sam(w, spy):
    """Selezione da testo (GroundingDINO+SAM, nativo): testo → richiesta segment con image_path+text."""
    _force_native_up()
    _populate_inventory(w)
    w.driver.is_up = lambda: True                # _segment_preconds richiede il driver 'up'
    f = w.frame1_tab
    H.load_canvas_image(w)
    i = f.seg_detector.findData("grounding_dino")
    if i >= 0:
        f.seg_detector.setCurrentIndex(i)
    f.seg_text.setText("face")
    f.seg_threshold.setValue(0.50)
    w.segment_by_text()
    assert spy.calls, "segment testo: nessuna richiesta inviata"
    c = spy.calls[-1]
    if c["via"] == "enqueue":
        p = c["params"]
        assert p.get("image_path"), f"segment senza image_path: {p}"
        assert p.get("text") == "face", f"testo segment non propagato: {p.get('text')}"


def case_segment_text_dino_only(w, spy):
    """Selezione da testo con detector GroundingDINO-only + soglia alta: no-crash + richiesta inviata."""
    _force_native_up()
    _populate_inventory(w)
    w.driver.is_up = lambda: True
    f = w.frame1_tab
    H.load_canvas_image(w)
    i = f.seg_detector.findData("grounding_dino_only")
    if i >= 0:
        f.seg_detector.setCurrentIndex(i)
    f.seg_text.setText("dog")
    f.seg_threshold.setValue(0.70)
    w.segment_by_text()
    assert spy.calls, "segment dino-only: nessuna richiesta inviata"


def case_segment_empty_text_guard(w, spy):
    """Selezione da testo senza testo: nessuna richiesta (guardia 'text missing')."""
    _force_native_up()
    _populate_inventory(w)
    w.driver.is_up = lambda: True
    f = w.frame1_tab
    H.load_canvas_image(w)
    f.seg_text.setText("")
    w.segment_by_text()
    assert not spy.calls, f"segment senza testo ma richiesta inviata: {spy.calls}"


# ════════════════════════ REGISTRO CASI ════════════════════════

def case_klein_te_uncensored(w, spy):
    """Inpaint con FLUX.2 Klein 9B: menu «Encoder Klein» visibile, la scelta uncensored arriva come klein_te a ogni
    job Klein (aggancio unico in _NativeImageWorker); con un modello non Klein il menu sparisce e non si invia."""
    from ui.main_window import _NativeImageWorker
    f = w.frame1_tab
    i = next((k for k in range(f.inp_model.count()) if "miraclein" in f.inp_model.itemText(k).lower()), -1)
    if i < 0:
        return                                                    # Klein 9B non installato: niente da verificare
    f.inp_model.setCurrentIndex(i)
    assert not f.inp_klein_te.isHidden(), "menu encoder Klein nascosto con un Klein 9B"
    f.inp_klein_te.setCurrentIndex(f.inp_klein_te.findData("klein9b-te-uncensored"))
    assert _NativeImageWorker({"arch": "flux2_inpaint"}).params.get("klein_te") == "klein9b-te-uncensored"
    assert "klein_te" not in _NativeImageWorker({"arch": "inpaint"}).params, "klein_te inviato a FLUX.1"
    j = next((k for k in range(f.inp_model.count()) if "klein" not in f.inp_model.itemText(k).lower()
              and "flux.2" not in f.inp_model.itemText(k).lower()), -1)
    f.inp_model.setCurrentIndex(j)
    assert f.inp_klein_te.isHidden() and f.klein_te_choice() == ""
    assert "klein_te" not in _NativeImageWorker({"arch": "flux2_inpaint"}).params


CASES = (
    _gen_model_cases()
    + [
        ("frame1_generate_image_auto_select",
         "Genera immagine con modello «⚡ Auto»: la GUI risolve a un modello concreto e lo invia.",
         case_gen_auto_select),
        ("frame1_generate_image_enhance_toggle",
         "Genera immagine con «Migliora prompt» ON: no-crash + richiesta inviata.",
         case_gen_enhance_toggle),
        ("frame1_generate_image_seed_fixed",
         "Genera immagine con seed FISSO: il seed scelto deve arrivare nei params.",
         case_gen_seed_fixed),
        ("frame1_generate_image_cfg_steps",
         "Genera immagine con CFG/Steps custom: guidance e steps propagati.",
         case_gen_cfg_steps),
        ("frame1_generate_image_sampler_scheduler",
         "Genera immagine con sampler/scheduler espliciti (euler/karras): no-crash.",
         case_gen_sampler_scheduler),
        ("frame1_generate_image_pulid_face",
         "Genera immagine con PuLID + foto volto: pulid_face/pulid_strength propagati.",
         case_gen_pulid_face),
        ("frame1_generate_image_empty_prompt_guard",
         "Genera immagine con prompt vuoto: nessuna richiesta (guardia).",
         case_gen_empty_prompt_guard),
    ]
    + _inpaint_model_cases()
    + [
        ("frame1_inpaint_dynamic_unet",
         "Inpaint con UNET Flux dinamico (inventario): copre il ramo _dynamic_unets.",
         case_inpaint_dynamic_unet),
        ("frame1_inpaint_strength_steps_seed",
         "Inpaint con strength/steps/seed fisso: opzioni propagate (ramo nativo Fill).",
         case_inpaint_strength_steps_seed),
        ("frame1_inpaint_no_mask_guard",
         "Inpaint senza maschera: nessuna richiesta (guardia 'no mask').",
         case_inpaint_no_mask_guard),
        ("frame1_inpaint_no_image_guard",
         "Inpaint senza foto: nessuna richiesta (guardia 'no image').",
         case_inpaint_no_image_guard),
        ("frame1_outpaint_aspect_preserved",
         "Outpaint: il canvas mantiene le proporzioni della foto (niente quadrato).",
         case_outpaint_aspect_preserved),
        ("frame1_outpaint_generate",
         "Outpaint con FLUX.1: foto con bordo → richiesta con image_path+mask_path (nativo).",
         case_outpaint_generate),
        ("frame1_integrate_kontext_native",
         "Integra Kontext (nativo): base+layer → richiesta arch=kontext con image_path.",
         case_integrate_kontext_native),
        ("frame1_integrate_guidance_seed",
         "Integra: guidance e seed fisso propagati (ramo nativo Kontext).",
         case_integrate_guidance_seed),
        ("frame1_integrate_pulid_separation",
         "PuLID separati: box di Genera solo su genera, box di Image to Image solo su integra.",
         case_integrate_pulid_separation),
        ("frame1_integrate_pulid_native",
         "Image to Image (Kontext) + PuLID dedicato: pulid_face nella richiesta nativa.",
         case_integrate_pulid_native),
        ("frame1_ui_footer_and_renames",
         "Footer ancorati (Genera Video/immagine), stile bottoni-azione, rinomina Modifica Immagine.",
         case_ui_footer_and_renames),
        ("frame1_integrate_flux2",
         "Integra con motore FLUX.2 (ComfyUI): modello FLUX.2 selezionato → richiesta inviata.",
         case_integrate_flux2),
        ("frame1_integrate_puzzle_lora",
         "Integra Kontext + puzzle_lora ON: ramo ComfyUI FLUX.1 → richiesta inviata.",
         case_integrate_puzzle_lora_flux1),
        ("frame1_integrate_no_layer_guard",
         "Integra senza layer: nessuna richiesta (guardia 'no image to integrate').",
         case_integrate_no_layer_guard),
        ("frame1_recolor_kontext_native",
         "Cambia colore Kontext (nativo): prompt recolor con il colore target + image_path.",
         case_recolor_kontext_native),
        ("frame1_recolor_target_guard",
         "Recolor senza colore target: nessuna richiesta (guardia).",
         case_recolor_target_guard),
        ("frame1_recolor_flux2",
         "Recolor con motore FLUX.2 (ComfyUI): modello FLUX.2 → richiesta inviata.",
         case_recolor_flux2),
    ]
    + _upscale_model_cases()
    + [
        ("frame1_upscale_scale_factors",
         "Upscale 2x/3x/4x: ogni fattore produce una richiesta (no-crash).",
         case_upscale_scale_factors),
        ("frame1_upscale_refine_mode",
         "Upscale «Rifinisci» con FLUX.2 (ComfyUI): richiesta inviata.",
         case_upscale_refine_mode),
        ("frame1_upscale_no_model_guard",
         "Upscale senza modello: nessuna richiesta (guardia).",
         case_upscale_no_model_guard),
        ("frame1_compose_multiref_kontext",
         "Componi multiref (Kontext nativo): 2 foto → stitch+kontext, richiesta inviata.",
         case_compose_multiref_kontext),
        ("frame1_compose_stitch",
         "Componi stitch: 2 foto → richiesta inviata.",
         case_compose_stitch),
        ("frame1_compose_flux2",
         "Componi con engine FLUX.2 (ComfyUI multi-reference): modello FLUX.2 → richiesta inviata.",
         case_compose_flux2),
        ("frame1_compose_need_two_guard",
         "Componi con 1 sola foto: nessuna richiesta (guardia 'servono 2').",
         case_compose_need_two_guard),
        ("frame1_segment_text_dino_sam",
         "Selezione da testo GroundingDINO+SAM (nativo): richiesta segment con image_path+text.",
         case_segment_text_dino_sam),
        ("frame1_segment_text_dino_only",
         "Selezione da testo GroundingDINO-only + soglia 0.70: richiesta inviata.",
         case_segment_text_dino_only),
        ("frame1_segment_empty_text_guard",
         "Selezione da testo senza testo: nessuna richiesta (guardia).",
         case_segment_empty_text_guard),
        ("frame1_klein_te_uncensored",
         "Klein 9B: encoder uncensored scelto in Inpaint arriva come klein_te a ogni job Klein, non altrove.",
         case_klein_te_uncensored),
    ]
)
