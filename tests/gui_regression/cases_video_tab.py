"""Casi di regressione GUI-driven per il tab VIDEO (ui/tabs/video_tab.py).

Ogni caso costruisce la MainWindow (offscreen), imposta i campi del tab Video e invoca l'AZIONE
(`MainWindow.on_generate`, che instrada t2v/i2v al motore nativo LTX-2). Lo Spy cattura la richiesta
GUI→backend (via `_enqueue`/native_driver) SENZA eseguire (no GPU).

Cosa copre:
  • on_generate t2v (solo testo) e i2v (con immagine nel canvas Frame 1)
  • un caso per OGNI modello concreto del dropdown (diffusion / gguf / checkpoint, SFW+NSFW) →
    scopre se "funziona solo un modello" o se un kind rompe get_values()
  • opzioni: risoluzione (preset/custom), fps+durata→num_frames, forza immagine→ref_strength,
    metodo i2v, sampler (steps/cfg/seed fisso), prompt negativo, filtro precisione, NSFW toggle
  • restauro volti auto, face-swap auto, RIFE auto (toggle post-generazione)
  • storyboard multi-scena (→ keyframe), V2V (ReTake / Talk-no-audio guard / Canta=MultiTalk)
  • preset video (applica built-in, salva/elimina personale), preferiti modello
  • guard: prompt vuoto, ReTake senza video sorgente

NOTA QA (finding già emerso in fase di scrittura): il ramo NATIVO video (_generate_video_native /
_NativeLtx2Worker) NON inoltra il modello selezionato nei params — la dropdown modelli video è
ignorata dal backend. Il caso `video_model_selection_reaches_backend` lo asserisce esplicitamente
(fallirà finché il bug è presente). I casi per-modello verificano invece il livello GUI
(get_values riflette il modello) + che la generazione venga comunque dispacciata.

Importabile senza side-effect pesanti: l'enumerazione dei modelli costruisce UNA finestra a
import-time, ma è racchiusa in try/except con fallback.
"""
from __future__ import annotations

import os

import harness as H

OUT = H.OUT_DIR


# ─────────────────────────── helper ───────────────────────────
def _set_prompt(w, text="a golden retriever running on a sunny beach, cinematic"):
    w.video_tab.prompt.setPlainText(text)


def _select_first_concrete(combo) -> dict:
    """Seleziona il primo modello CONCRETO (salta 'Auto'); ritorna il suo itemData."""
    for i in range(combo.count()):
        d = combo.itemData(i) or {}
        if d.get("kind") not in (None, "auto", "unsupported") and d.get("model"):
            combo.setCurrentIndex(i)
            return d
    return combo.currentData() or {}


def _select_model(combo, kind: str, model: str) -> bool:
    for i in range(combo.count()):
        d = combo.itemData(i) or {}
        if d.get("kind") == kind and d.get("model") == model:
            combo.setCurrentIndex(i)
            return True
    return False


def _set_precision(combo, key: str) -> bool:
    for i in range(combo.count()):
        if combo.itemData(i) == key:
            combo.setCurrentIndex(i)
            return True
    return False


def _set_res_preset(combo, name: str) -> bool:
    i = combo.findText(name)
    if i >= 0:
        combo.setCurrentText(name)
        return True
    return False


def _dummy_file(name: str, content: bytes = b"\x00\x00\x00\x00") -> str:
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(content)
    return str(p)


def _last_native_call(spy):
    """Ultima richiesta inoltrata al backend (enqueue worker o native_driver)."""
    for c in reversed(spy.calls):
        if c["via"] in ("enqueue", "native_driver"):
            return c
    return spy.calls[-1] if spy.calls else None


# ─────────────────────────── azioni base: t2v / i2v ───────────────────────────
def case_t2v_generation(w, spy):
    """T2V: con prompt + modello concreto, on_generate deve dispacciare una generazione video
    nativa con prompt/width/height/num_frames/fps coerenti."""
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    w.tabs.setCurrentWidget(w.video_tab)
    w.on_generate()
    assert spy.calls, "T2V: nessuna richiesta inviata da on_generate"
    p = _last_native_call(spy)["params"]
    assert p.get("prompt"), f"prompt mancante nei params: {p}"
    assert p.get("num_frames") and p.get("width") and p.get("height"), f"params video incompleti: {p}"
    assert p.get("fps"), f"fps mancante: {p}"


def case_i2v_generation(w, spy):
    """I2V: con un'immagine nel canvas Frame 1, on_generate deve passare image_path + ref_strength
    nei params nativi (immagine → video)."""
    H.load_canvas_image(w)
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    w.on_generate()
    assert spy.calls, "I2V: nessuna richiesta inviata"
    p = _last_native_call(spy)["params"]
    assert p.get("image_path"), f"I2V senza image_path nei params: {p}"
    assert "ref_strength" in p, f"I2V senza ref_strength: {p}"


def case_t2v_empty_prompt_guard(w, spy):
    """Guard prompt vuoto: on_generate NON deve dispacciare nulla (e non deve crashare)."""
    w.video_tab.prompt.setPlainText("")
    _select_first_concrete(w.video_tab.model)
    w.on_generate()
    assert not spy.calls, f"prompt vuoto ma è stata inviata una richiesta: {spy.calls}"


def case_t2v_negative_prompt(w, spy):
    """Il prompt negativo del tab Video deve arrivare al backend come 'negative'."""
    _set_prompt(w)
    w.video_tab.neg.setPlainText("blurry, watermark, lowres, deformed")
    _select_first_concrete(w.video_tab.model)
    w.on_generate()
    assert spy.calls, "negative: nessuna richiesta"
    p = _last_native_call(spy)["params"]
    assert "blurry" in (p.get("negative") or ""), f"negative non trasmesso: {p.get('negative')!r}"


def case_auto_model_generation(w, spy):
    """Modello = 'Auto': la generazione deve comunque partire (model_kind='auto')."""
    _set_prompt(w)
    w.video_tab.model.setCurrentIndex(0)   # voce Auto
    assert w.video_tab.get_values().get("model") == "auto", "indice 0 non è il modello Auto"
    w.on_generate()
    assert spy.calls, "Auto: nessuna richiesta inviata"


# ─────────────────────────── opzioni: risoluzione ───────────────────────────
def case_resolution_preset_landscape(w, spy):
    """Preset risoluzione 16:9: width/height del preset devono arrivare al backend (orizzontale)."""
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    assert _set_res_preset(w.video_tab.res_preset, "Orizzontale 16:9 (1280x704)"), "preset 16:9 assente"
    assert w.video_tab.width.value() == 1280 and w.video_tab.height.value() == 704, \
        f"preset non applicato: {w.video_tab.width.value()}x{w.video_tab.height.value()}"
    w.on_generate()
    p = _last_native_call(spy)["params"]
    assert p["width"] == 1280 and p["height"] == 704, f"risoluzione errata nei params: {p['width']}x{p['height']}"


def case_resolution_custom(w, spy):
    """Risoluzione personalizzata: spinbox abilitati e valori custom inoltrati al backend."""
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    assert _set_res_preset(w.video_tab.res_preset, "Personalizzato"), "preset Personalizzato assente"
    assert w.video_tab.width.isEnabled() and w.video_tab.height.isEnabled(), \
        "Personalizzato non abilita gli spinbox width/height"
    w.video_tab.width.setValue(896)
    w.video_tab.height.setValue(512)
    w.on_generate()
    p = _last_native_call(spy)["params"]
    assert p["width"] == 896 and p["height"] == 512, f"custom non rispettata: {p['width']}x{p['height']}"


# ─────────────────────────── opzioni: fps / durata / forza immagine ───────────────────────────
def case_fps_duration_to_num_frames(w, spy):
    """fps + durata devono determinare num_frames: nf = round(len*fps)//8*8 + 1."""
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    w.video_tab.fps.setValue(24)
    w.video_tab.length.setValue(2.0)
    w.on_generate()
    p = _last_native_call(spy)["params"]
    expected = max(9, int(round(2.0 * 24)) // 8 * 8 + 1)   # = 49
    assert p["num_frames"] == expected, f"num_frames atteso {expected}, ottenuto {p['num_frames']}"
    assert p["fps"] == 24, f"fps errato: {p['fps']}"


def case_img_strength_ref(w, spy):
    """Forza immagine (slider): in I2V deve diventare ref_strength = valore/100."""
    H.load_canvas_image(w)
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    w.video_tab.img_strength.setValue(80)
    w.on_generate()
    p = _last_native_call(spy)["params"]
    assert abs(float(p.get("ref_strength", -1)) - 0.80) < 0.01, \
        f"ref_strength atteso ~0.80, ottenuto {p.get('ref_strength')}"


def case_i2v_method_inplace(w, spy):
    """Metodo i2v 'inplace': selezionabile, get_values lo riflette, la generazione parte."""
    H.load_canvas_image(w)
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    i = w.video_tab.i2v_method.findData("inplace")
    assert i >= 0, "metodo i2v 'inplace' assente"
    w.video_tab.i2v_method.setCurrentIndex(i)
    assert w.video_tab.get_values().get("i2v_method") == "inplace", "i2v_method non riflesso in get_values"
    w.on_generate()
    assert spy.calls, "i2v inplace: nessuna richiesta"


# ─────────────────────────── opzioni: sampler / seed ───────────────────────────
def case_sampler_steps_cfg(w, spy):
    """Steps/CFG impostati dall'utente devono arrivare al backend (steps/guidance)."""
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    w.video_tab.sampler_seed.steps.setValue(12)
    w.video_tab.sampler_seed.cfg.setValue(2.5)
    w.on_generate()
    p = _last_native_call(spy)["params"]
    assert p.get("steps") == 12, f"steps non trasmessi: {p.get('steps')}"
    assert abs(float(p.get("guidance", -1)) - 2.5) < 0.01, f"guidance non trasmessa: {p.get('guidance')}"


def case_fixed_seed(w, spy):
    """Seed fisso: il valore dello spinbox deve arrivare invariato al backend."""
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    w.video_tab.sampler_seed.fixed_seed.setChecked(True)
    w.video_tab.sampler_seed.seed.setValue(424242)
    assert w.video_tab.wants_fixed_seed(), "fixed_seed non attivo"
    w.on_generate()
    p = _last_native_call(spy)["params"]
    assert p.get("seed") == 424242, f"seed fisso non rispettato: {p.get('seed')}"


# ─────────────────────────── opzioni: precisione / NSFW ───────────────────────────
def case_precision_filter_gguf(w, spy):
    """Filtro precisione 'gguf': il dropdown si ripopola senza crash mantenendo modelli concreti."""
    if not _set_precision(w.video_tab.precision, "gguf"):
        return                                  # selettore senza voce gguf → niente da testare
    # il refilter è già scattato; verifica che resti almeno un modello concreto selezionabile
    concretes = [w.video_tab.model.itemData(i) for i in range(w.video_tab.model.count())
                 if (w.video_tab.model.itemData(i) or {}).get("kind") not in (None, "auto")]
    assert w.video_tab.model.count() >= 1, "dropdown vuoto dopo filtro precisione"
    # con un concreto presente, la generazione deve comunque partire
    if concretes:
        _set_prompt(w)
        _select_first_concrete(w.video_tab.model)
        w.on_generate()
        assert spy.calls, "filtro gguf: generazione non dispacciata"


def case_nsfw_toggle_reveals_eros(w, spy):
    """Passando a NSFW il dropdown deve esporre modelli NSFW (es. *eros*) nascosti in SFW."""
    w.video_tab.nsfw.setCurrentText("SFW")
    sfw_models = {(w.video_tab.model.itemData(i) or {}).get("model")
                  for i in range(w.video_tab.model.count())}
    w.video_tab.nsfw.setCurrentText("NSFW")
    nsfw_models = {(w.video_tab.model.itemData(i) or {}).get("model")
                   for i in range(w.video_tab.model.count())}
    extra = nsfw_models - sfw_models
    assert extra, "NSFW non ha aggiunto alcun modello rispetto a SFW"
    # almeno un modello NSFW concreto generabile
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    w.on_generate()
    assert spy.calls, "NSFW: generazione non dispacciata"


# ─────────────────────────── post-processing toggles ───────────────────────────
def case_face_restore_auto_toggle(w, spy):
    """Restauro volti auto attivo: get_face_restore_auto() ritorna config e la generazione parte."""
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    w.video_tab.fr_auto.setChecked(True)
    cfg = w.video_tab.get_face_restore_auto()
    assert cfg and "model" in cfg and "fidelity" in cfg, f"config restauro volti incompleta: {cfg}"
    w.on_generate()
    assert spy.calls, "restauro volti: generazione non dispacciata"


def case_face_swap_auto_toggle(w, spy):
    """Face-swap auto + foto volto: get_face_swap_auto() ritorna {face} e la generazione parte."""
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    face = H.make_test_image(str(OUT / "_vid_faceswap_src.png"))
    w.video_tab.fs_auto.setChecked(True)
    w.video_tab.fs_face.setText(face)
    cfg = w.video_tab.get_face_swap_auto()
    assert cfg and cfg.get("face") == face, f"config face-swap errata: {cfg}"
    w.on_generate()
    assert spy.calls, "face-swap: generazione non dispacciata"


def case_rife_auto_toggle(w, spy):
    """RIFE auto x4: get_rife_auto() ritorna factor=4 e la generazione parte."""
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    w.video_tab.rife_auto.setChecked(True)
    w.video_tab.rife_factor.setCurrentText("x4")
    cfg = w.video_tab.get_rife_auto()
    assert cfg and cfg.get("factor") == 4, f"config RIFE errata: {cfg}"
    w.on_generate()
    assert spy.calls, "RIFE: generazione non dispacciata"


# ─────────────────────────── storyboard multi-scena ───────────────────────────
def case_storyboard_two_scenes(w, spy):
    """Storyboard con 2 scene-keyframe: on_generate deve trasformarle in 'keyframes' (a tempo)."""
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    sb = w.video_tab.storyboard_panel
    sb.enable.setChecked(True)
    img = H.make_test_image(str(OUT / "_vid_sb_scene.png"))
    rows = sb._rows[:2]
    assert len(rows) >= 2, "lo storyboard non ha 2 righe di partenza"
    for i, r in enumerate(rows):
        r.image_path = img
        r.second.setValue(float(i))
    assert len(sb.get_storyboard()) == 2, "get_storyboard non ritorna 2 scene"
    w.on_generate()
    assert spy.calls, "storyboard: nessuna richiesta"
    p = _last_native_call(spy)["params"]
    assert p.get("keyframes") and len(p["keyframes"]) == 2, f"keyframes non costruiti: {p.get('keyframes')}"


# ─────────────────────────── V2V: ReTake / Talk / Canta ───────────────────────────
def case_v2v_retake_no_source_guard(w, spy):
    """ReTake senza video sorgente: deve fermarsi con avviso (no crash, niente richiesta)."""
    _set_prompt(w)
    w.video_tab.v2v_panel.rb_retake.setChecked(True)
    assert w.video_tab.v2v_panel.validation_errors(), "ReTake senza video dovrebbe avere errori di validazione"
    w.on_generate()
    assert not spy.calls, f"ReTake senza sorgente non doveva dispacciare nulla: {spy.calls}"


def case_v2v_retake_native(w, spy):
    """ReTake con video sorgente (stato simulato): se l'upload riesce, deve partire la pipeline
    ReTake nativa (params con retake/video_path)."""
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    panel = w.video_tab.v2v_panel
    panel.rb_retake.setChecked(True)
    src = _dummy_file("_vid_retake_src.mp4")
    panel._video_path = src
    panel._duration_s = 10.0
    panel._reset_sliders()             # imposta start/end validi entro [3, dur-3]
    panel.rb_audio_regen.setChecked(True)
    try:
        w.on_generate()
    except Exception:
        # ambiente senza la input-dir ComfyUI o ffprobe: il guard può fermare; non deve crashare il test
        raise
    if spy.calls:
        p = _last_native_call(spy)["params"]
        assert p.get("retake") or p.get("video_path"), f"ReTake nativo senza marcatori retake: {p}"


def case_v2v_talk_no_audio_guard(w, spy):
    """Talk (lip-sync) senza audio reale: MultiTalk deve fermarsi con avviso (no crash)."""
    _set_prompt(w)
    w.video_tab.v2v_panel.rb_talk.setChecked(True)
    w.on_generate()
    assert not spy.calls, f"Talk senza audio non doveva dispacciare: {spy.calls}"


def case_v2v_sing_multitalk(w, spy):
    """Canta (MultiTalk): con foto (Frame 1) + canzone, on_generate deve dispacciare a2v/lip-sync
    nativo con audio_path."""
    H.load_canvas_image(w)
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    panel = w.video_tab.v2v_panel
    panel.rb_sing.setChecked(True)
    song = _dummy_file("_vid_sing_song.wav", b"RIFF0000WAVEfmt ")
    panel._sing_audio_path = song
    w.on_generate()
    assert spy.calls, "Canta: nessuna richiesta (MultiTalk non dispacciato)"
    p = _last_native_call(spy)["params"]
    assert p.get("audio_path") == song, f"audio canzone non trasmesso: {p.get('audio_path')}"


# ─────────────────────────── preset video / preferiti ───────────────────────────
def case_preset_apply_builtin(w, spy):
    """Applicare un preset video built-in deve aggiornare le impostazioni (steps) e la generazione
    deve usare quei valori."""
    _set_prompt(w)
    _select_first_concrete(w.video_tab.model)
    pc = w.video_tab.preset_combo
    target_idx = None
    target_steps = None
    for i in range(pc.count()):
        d = pc.itemData(i)
        if isinstance(d, dict) and d.get("steps") and not d.get("_custom"):
            target_idx, target_steps = i, d["steps"]
            break
    if target_idx is None:
        return                                  # nessun preset built-in con steps
    pc.setCurrentIndex(target_idx)              # → _on_preset_apply
    assert w.video_tab.sampler_seed.steps.value() == target_steps, "preset non ha applicato gli steps"
    w.on_generate()
    p = _last_native_call(spy)["params"]
    assert p.get("steps") == target_steps, f"steps del preset non nei params: {p.get('steps')}"


def case_preset_save_and_delete(w, spy):
    """Salvataggio + eliminazione di un preset personale (dialog patchati): nessun crash, lo stato
    di user_prefs resta pulito."""
    from core import user_prefs
    name = "test"                                # QInputDialog.getText patchato → ('test', True)
    pre = set(user_prefs.custom_presets().keys())
    w.video_tab._save_current_preset()
    assert name in user_prefs.custom_presets(), "preset personale non salvato"
    # seleziona il preset salvato e cancellalo (QMessageBox.question patchato → Yes)
    i = w.video_tab.preset_combo.findText(f"⭐ {name}")
    assert i >= 0, "preset salvato non comparso nel combo"
    w.video_tab.preset_combo.setCurrentIndex(i)
    w.video_tab._delete_selected_preset()
    # ripristina lo stato originale comunque
    if name not in pre and name in user_prefs.custom_presets():
        user_prefs.delete_preset(name)
    assert name not in user_prefs.custom_presets(), "preset personale non eliminato"


def case_toggle_favorite_model(w, spy):
    """Stella preferiti modello: toggle on→off senza crash, lo stato user_prefs viene ripristinato."""
    from core import user_prefs
    d = _select_first_concrete(w.video_tab.model)
    model = d.get("model")
    assert model, "nessun modello concreto da marcare preferito"
    was_fav = user_prefs.is_fav_model(model)
    w.video_tab._toggle_fav_model_current()
    assert user_prefs.is_fav_model(model) != was_fav, "toggle preferito non ha cambiato lo stato"
    w.video_tab._toggle_fav_model_current()      # ripristina
    assert user_prefs.is_fav_model(model) == was_fav, "stato preferito non ripristinato"


# ─────────────────────────── FINDING: il modello non raggiunge il backend ───────────────────────────
def case_model_selection_reaches_backend(w, spy):
    """CONTRATTO: il modello scelto nella dropdown DEVE essere inoltrato al backend.
    (Attualmente il ramo video nativo NON lo passa → questo caso scopre il difetto.)"""
    _set_prompt(w)
    d = _select_first_concrete(w.video_tab.model)
    model = d.get("model", "")
    w.on_generate()
    assert spy.calls, "nessuna richiesta inviata"
    p = _last_native_call(spy)["params"]
    sent = " ".join(str(p.get(k, "")) for k in ("model", "model_gguf", "model_standard", "checkpoint"))
    assert model and model in sent, (
        "Il modello selezionato NON è stato inoltrato al backend: la dropdown modelli video "
        f"sembra ignorata (selezionato={model!r}, params senza campo modello: {sorted(p.keys())})")


# ─────────────────────────── enumerazione per-modello (un caso per modello concreto) ───────────────────────────
def _make_model_case(kind: str, model: str, requires_nsfw: bool):
    def _case(w, spy):
        if requires_nsfw:
            w.video_tab.nsfw.setCurrentText("NSFW")
        assert _select_model(w.video_tab.model, kind, model), \
            f"modello non trovato nel dropdown: kind={kind} model={model}"
        _set_prompt(w)
        gv = w.video_tab.get_values()
        if kind in ("diffusion", "gguf"):
            assert gv.get("model") == model, (
                f"get_values non riflette il modello selezionato ({kind}): "
                f"atteso {model!r}, ottenuto {gv.get('model')!r}")
        else:
            # kind 'checkpoint' non è gestito da get_values() → ricade su 'auto' (finding documentato)
            assert gv.get("model") == "auto", (
                f"kind {kind}: comportamento get_values cambiato (atteso fallback 'auto', "
                f"ottenuto {gv.get('model')!r})")
        w.on_generate()
        assert spy.calls, f"modello {model} ({kind}): generazione non dispacciata"
    return _case


def _enumerate_model_specs():
    """Costruisce UNA finestra a import-time per elencare i modelli concreti del dropdown (SFW+NSFW).
    Robusto: in caso di errore ritorna lista vuota (fallback ai soli casi generici)."""
    specs: list[tuple[str, str, bool]] = []
    try:
        w = H.build_window()
        v = w.video_tab
        v.nsfw.setCurrentText("SFW")
        sfw = {((v.model.itemData(i) or {}).get("kind"), (v.model.itemData(i) or {}).get("model"))
               for i in range(v.model.count())}
        v.nsfw.setCurrentText("NSFW")
        seen = set()
        for i in range(v.model.count()):
            d = v.model.itemData(i) or {}
            kind, model = d.get("kind"), d.get("model")
            if kind in (None, "auto", "unsupported") or not model:
                continue
            key = (kind, model)
            if key in seen:
                continue
            seen.add(key)
            specs.append((kind, model, key not in sfw))
    except Exception:                            # noqa: BLE001
        return []
    return specs


def _safe_id(model: str) -> str:
    base = os.path.basename(str(model).replace("\\", "/"))
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in base)[:48]


# ─────────────────────────── registro casi ───────────────────────────
CASES = [
    ("video_t2v_generation", "T2V: on_generate dispaccia un video nativo con prompt/dimensioni/num_frames/fps.",
     case_t2v_generation),
    ("video_i2v_generation", "I2V: con immagine nel canvas passa image_path + ref_strength al backend.",
     case_i2v_generation),
    ("video_t2v_empty_prompt_guard", "Prompt vuoto: on_generate si ferma senza inviare richieste.",
     case_t2v_empty_prompt_guard),
    ("video_t2v_negative_prompt", "Prompt negativo: trasmesso al backend come 'negative'.",
     case_t2v_negative_prompt),
    ("video_auto_model_generation", "Modello 'Auto': la generazione parte comunque (model_kind=auto).",
     case_auto_model_generation),
    ("video_resolution_preset_landscape", "Preset 16:9: width/height (1280x704) inoltrati al backend.",
     case_resolution_preset_landscape),
    ("video_resolution_custom", "Risoluzione personalizzata: spinbox abilitati e valori custom inoltrati.",
     case_resolution_custom),
    ("video_fps_duration_num_frames", "fps+durata → num_frames = round(len*fps)//8*8+1.",
     case_fps_duration_to_num_frames),
    ("video_img_strength_ref", "Forza immagine (slider) → ref_strength in I2V.",
     case_img_strength_ref),
    ("video_i2v_method_inplace", "Metodo i2v 'inplace': selezionabile e riflesso in get_values.",
     case_i2v_method_inplace),
    ("video_sampler_steps_cfg", "Steps/CFG utente → steps/guidance nei params.",
     case_sampler_steps_cfg),
    ("video_fixed_seed", "Seed fisso: valore inoltrato invariato al backend.",
     case_fixed_seed),
    ("video_precision_filter_gguf", "Filtro precisione 'gguf': ripopola il dropdown senza crash.",
     case_precision_filter_gguf),
    ("video_nsfw_toggle_reveals_models", "Toggle NSFW: espone modelli NSFW nascosti in SFW.",
     case_nsfw_toggle_reveals_eros),
    ("video_face_restore_auto", "Restauro volti auto: config valida + generazione dispacciata.",
     case_face_restore_auto_toggle),
    ("video_face_swap_auto", "Face-swap auto + foto: config {face} valida + generazione dispacciata.",
     case_face_swap_auto_toggle),
    ("video_rife_auto", "RIFE auto x4: config factor=4 + generazione dispacciata.",
     case_rife_auto_toggle),
    ("video_storyboard_two_scenes", "Storyboard 2 scene → params 'keyframes' a tempo.",
     case_storyboard_two_scenes),
    ("video_v2v_retake_no_source_guard", "ReTake senza video: guard, nessuna richiesta, no crash.",
     case_v2v_retake_no_source_guard),
    ("video_v2v_retake_native", "ReTake con sorgente: pipeline ReTake nativa (retake/video_path).",
     case_v2v_retake_native),
    ("video_v2v_talk_no_audio_guard", "Talk senza audio: MultiTalk si ferma con avviso, no crash.",
     case_v2v_talk_no_audio_guard),
    ("video_v2v_sing_multitalk", "Canta: foto+canzone → a2v/lip-sync nativo con audio_path.",
     case_v2v_sing_multitalk),
    ("video_preset_apply_builtin", "Preset video built-in: applica gli steps e li usa nella generazione.",
     case_preset_apply_builtin),
    ("video_preset_save_and_delete", "Preset personale: salva ed elimina senza crash (stato pulito).",
     case_preset_save_and_delete),
    ("video_toggle_favorite_model", "Stella preferiti modello: toggle on/off con ripristino stato.",
     case_toggle_favorite_model),
    ("video_model_selection_reaches_backend",
     "CONTRATTO: il modello selezionato deve raggiungere il backend (scopre la dropdown ignorata).",
     case_model_selection_reaches_backend),
]

def case_h3_custom_dit(w, spy):
    """DiT MiniMax H3 personalizzati (finetune 18+): solo in NSFW, e la scelta arriva come h3_dit."""
    t = w.video_tab
    t.nsfw.setCurrentText("SFW")
    t.video_arch.setCurrentIndex(t.video_arch.findData("minimax_h3"))
    keys = lambda: [(t.model.itemData(i) or {}).get("h3_dit") for i in range(t.model.count())]
    assert not any(keys()), f"DiT 18+ visibili in SFW: {keys()}"
    t.nsfw.setCurrentText("NSFW")
    i = next((i for i in range(t.model.count())
              if (t.model.itemData(i) or {}).get("h3_dit") == "mmh3-dit-10erosmax-b6"), -1)
    assert i >= 0, f"10Eros-Max assente in NSFW: {keys()}"
    t.model.setCurrentIndex(i)
    v = t.get_values()
    assert v.get("model_kind") == "minimax_h3" and v.get("h3_dit") == "mmh3-dit-10erosmax-b6", v
    t.nsfw.setCurrentText("SFW")
    t.video_arch.setCurrentIndex(t.video_arch.findData("ltx"))


CASES.append(("video_h3_custom_dit", "DiT MiniMax H3 personalizzati (18+): visibili solo in NSFW, arrivano come h3_dit.",
              case_h3_custom_dit))


def case_sfw_hides_registry_adult(w, spy):
    """SFW: nessun checkpoint marcato 18+ nel registro pesi nel dropdown, anche senza keyword nel nome
    (PinkCherry, Stubelius)."""
    from core import weights_catalog as wc
    t = w.video_tab
    t.nsfw.setCurrentText("NSFW")
    t.nsfw.setCurrentText("SFW")
    models = [(t.model.itemData(i) or {}).get("model") or "" for i in range(t.model.count())]
    bad = [m for m in models if m and wc.is_adult_file(m)]
    assert not bad, f"modelli 18+ del registro visibili in SFW: {bad}"


def case_ltx25_te_uncensored(w, spy):
    """Text encoder LTX-2.5 alternativo: riga visibile solo con un transformer 2.5 in NSFW; la scelta arriva al worker
    come ltx_te (con ltx_release 2.5); con un modello 2.3 o in SFW la riga sparisce e torna l'ufficiale."""
    t = w.video_tab
    t.video_arch.setCurrentIndex(t.video_arch.findData("ltx"))
    t.nsfw.setCurrentText("NSFW")
    m25 = "LTX\\ltx2.5-Stubelius_remix_v1_distilled_int8_convrot.safetensors"
    t.model.addItem("Stubelius 2.5 (test)", {"kind": "diffusion", "model": m25, "key": "test:25"})
    t.model.setCurrentIndex(t.model.count() - 1)
    assert not t.ltx_te.isHidden(), "riga encoder LTX-2.5 assente con un transformer 2.5 in NSFW"
    t.ltx_te.setCurrentIndex(t.ltx_te.findData("ltx25-te-uncensored"))
    assert t.get_values()["ltx_te"] == "ltx25-te-uncensored"
    _set_prompt(w)
    w.on_generate()
    c = _last_native_call(spy)
    assert c, "generazione non dispacciata"
    p = c["params"]
    assert p.get("ltx_release") == "2.5" and p.get("ltx_te") == "ltx25-te-uncensored", p
    t.nsfw.setCurrentText("SFW")
    assert t.ltx_te.isHidden() and t.get_values()["ltx_te"] == "", "encoder 18+ ancora attivo in SFW"
    t.nsfw.setCurrentText("NSFW")
    _select_model(t.model, "diffusion", "LTX\\ltx-2.3-22b-dev-fp8.safetensors")
    assert t.ltx_te.isHidden(), "riga encoder 2.5 visibile con un modello 2.3"


CASES.append(("video_ltx25_te_uncensored",
              "LTX 2.5: encoder Gemma 4 uncensored solo con transformer 2.5 in NSFW, arriva al worker come ltx_te.",
              case_ltx25_te_uncensored))
CASES.append(("video_sfw_hides_registry_adult",
              "SFW: i checkpoint 18+ del registro pesi restano nascosti anche senza keyword nel nome.",
              case_sfw_hides_registry_adult))

# Un caso per OGNI modello concreto del dropdown (scopre 'funziona solo un modello' / kind rotti).
for _kind, _model, _nsfw in _enumerate_model_specs():
    CASES.append((
        f"video_model_{_kind}_{_safe_id(_model)}",
        f"Modello {_kind} '{_model}'{' (NSFW)' if _nsfw else ''}: selezionabile, get_values coerente, "
        f"generazione dispacciata.",
        _make_model_case(_kind, _model, _nsfw),
    ))
