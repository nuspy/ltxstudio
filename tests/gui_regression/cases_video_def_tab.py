"""Casi di regressione GUI-driven per il tab "Definizione video" (ui/tabs/video_def_tab.py,
montato come sotto-tab di "Migliora qualità" → w.video_def_tab).

Copre le DUE azioni che il tab manda al backend tramite MainWindow:
  • request_enhance  → MainWindow.generate_video_enhance()  (super-risoluzione video)
       engine: seedvr2 / esrgan / flashvsr  + risoluzione/batch/blocks/scala + opz. RIFE
  • request_face_restore → MainWindow.generate_face_restore() (restauro volti per-frame)
       modello (codeformer/GFPGAN) + detector + fidelity

Backend ATTUALE (data/backend_config.json): video=native, face=native → i due handler
instradano a `_enhance_video_native` / `_face_restore_native`, che fanno `self._enqueue(worker)`.
Lo Spy dell'harness cattura l'enqueue come {label, module, params}:
  • esrgan      → _NativeImageWorker(module="video_upscale", params={video,model,scale,seed})  → ASSERT mirati
  • video_face  → _NativeImageWorker(module="video_face",   params={image_path,model,detection,fidelity,seed}) → ASSERT mirati
  • seedvr2/flashvsr → worker dedicato SENZA .module/.params → si verifica solo che l'enqueue parta
    (module=None) dopo aver neutralizzato il preflight (richiede pesi/venv assenti in CI).

Obiettivo: scoprire regressioni del tipo "funziona solo un motore" o "funziona solo un modello"
generando UN caso per OGNI engine, per OGNI modello upscale ESRGAN concreto, per OGNI modello
SeedVR2, per OGNI modello/detector di restauro volti, più i toggle/guardie principali.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import harness as H

# ── liste modelli concrete (statiche → enumerabili a import-time senza rete/GPU) ──
try:
    from core.video_enhance_pipeline import SEEDVR2_MODELS
    _SEEDVR2_KEYS = list(SEEDVR2_MODELS.keys())
except Exception:                                          # noqa: BLE001
    _SEEDVR2_KEYS = []
try:
    from core import face_restore_pipeline as _fr
    _FR_MODELS = list(_fr.FACERESTORE_MODELS)
    _FR_DETECTORS = list(_fr.FACE_DETECTORS)
except Exception:                                          # noqa: BLE001
    _FR_MODELS, _FR_DETECTORS = [], []

# I modelli ESRGAN provengono dallo scan disco (set_upscale_models): li enumero costruendo una
# finestra usa-e-getta a import-time. Se fallisce, i casi per-modello vengono saltati e resta
# comunque il caso "loop" che itera la combo a runtime.
_ESRGAN_NAMES: list[str] = []
try:
    _w0 = H.build_window()
    t0 = _w0.video_def_tab
    _ESRGAN_NAMES = [t0.esrgan_model.itemText(i) for i in range(t0.esrgan_model.count())
                     if not t0.esrgan_model.itemText(i).startswith("(")]
    del _w0, t0
except Exception:                                         # noqa: BLE001
    _ESRGAN_NAMES = []


# ── helper precondizioni ─────────────────────────────────────────────────────────
def _make_test_video(name: str = "_vdef_test_input.mp4") -> str:
    """Crea un file segnaposto che ESISTE (i due handler validano solo os.path.exists del sorgente,
    non il contenuto del container). Sufficiente per far partire l'enqueue verso il worker."""
    p = H.OUT_DIR / name
    if not p.exists():
        # header mp4 minimale (ftyp) — basta che il file esista e non sia vuoto
        p.write_bytes(bytes.fromhex("0000001866747970697336320000000069736f6d6d703432"))
    return str(p)


def _set_engine(tab, idx: int):
    tab.engine.setCurrentIndex(idx)
    tab._on_engine()


def _select(combo, text: str) -> bool:
    i = combo.findText(text)
    if i >= 0:
        combo.setCurrentIndex(i)
        return True
    return False


# ════════════════════════ ENGINE: super-risoluzione (generate_video_enhance) ════════════════════════
def case_enhance_engine_esrgan(w, spy):
    """ESRGAN per-fotogramma: con un video sorgente la GUI deve enqueuare module='video_upscale'
    con params video + scale (target_scale) + model (esrgan_model)."""
    src = _make_test_video()
    t = w.video_def_tab
    t.set_video(src); _set_engine(t, 1)
    t.scale.setValue(2.0)
    w.generate_video_enhance()
    assert spy.calls, "ESRGAN: nessun enqueue (generate_video_enhance non ha instradato al nativo)"
    c = spy.calls[-1]
    assert c["module"] == "video_upscale", f"module inatteso per ESRGAN: {c}"
    p = c["params"]
    assert p.get("image_path") == src, f"image_path (video sorgente) mancante/errato: {p}"
    assert p.get("target_scale"), f"target_scale mancante: {p}"


def case_enhance_engine_seedvr2(w, spy):
    """SeedVR2 (AI VSR): neutralizzato il preflight (pesi/venv assenti in CI), la GUI deve
    enqueuare il worker SeedVR2 (worker dedicato → module=None, si verifica solo che parta)."""
    src = _make_test_video()
    t = w.video_def_tab
    t.set_video(src); _set_engine(t, 0)
    from core import seedvr2_service as sv
    orig = sv.preflight
    sv.preflight = lambda: []
    try:
        w.generate_video_enhance()
    finally:
        sv.preflight = orig
    assert spy.calls, "SeedVR2: nessun enqueue dopo preflight ok (engine non instradato)"


def case_enhance_engine_flashvsr(w, spy):
    """FlashVSR x4: neutralizzato il preflight, la GUI deve enqueuare il worker FlashVSR."""
    src = _make_test_video()
    t = w.video_def_tab
    t.set_video(src); _set_engine(t, 2)
    from core import flashvsr_service as fv
    orig = fv.preflight
    fv.preflight = lambda: []
    try:
        w.generate_video_enhance()
    finally:
        fv.preflight = orig
    assert spy.calls, "FlashVSR: nessun enqueue dopo preflight ok (engine non instradato)"


def case_enhance_engine_flashvsrplus(w, spy):
    """FlashVSR+ (Turbo/Base): neutralizzato il preflight, la GUI deve enqueuare il worker FlashVSR+ con la
    variante e la compressione temporale scelte (get_config['fvp']) e il nome file con la variante."""
    src = _make_test_video()
    t = w.video_def_tab
    t.set_video(src); _set_engine(t, 4)
    t.fvp_mode.setCurrentIndex(t.fvp_mode.findData("base"))
    t.fvp_motion.setCurrentIndex(t.fvp_motion.findData(2))
    assert t.get_config()["fvp"] == {"mode": "base", "temporal_compression": 2}, t.get_config()["fvp"]
    from core import flashvsrplus_service as fvp
    seen = {}
    orig = fvp.preflight
    fvp.preflight = lambda mode="turbo": seen.setdefault("mode", mode) and []
    try:
        w.generate_video_enhance()
    finally:
        fvp.preflight = orig
    assert seen.get("mode") == "base", f"preflight non ha ricevuto la variante scelta: {seen}"
    assert spy.calls, "FlashVSR+: nessun enqueue dopo preflight ok (engine non instradato)"
    wk = getattr(w, "_fvp_worker", None)
    assert wk is not None and wk.mode == "base" and wk.temporal_compression == 2, "worker FlashVSR+ senza i parametri GUI"
    assert "flashvsrplus_base_" in wk.out, f"nome file senza variante: {wk.out}"


def case_enhance_flashvsrplus_controls(w, spy):
    """_on_engine: i controlli FlashVSR+ (variante, movimento) sono attivi SOLO con quel motore; la scala vale."""
    t = w.video_def_tab
    _set_engine(t, 4)
    assert t.fvp_mode.isEnabled() and t.fvp_motion.isEnabled() and t.scale.isEnabled()
    assert not t.seedvr2_model.isEnabled() and not t.esrgan_model.isEnabled()
    _set_engine(t, 0)
    assert not t.fvp_mode.isEnabled() and not t.fvp_motion.isEnabled()
    assert [t.fvp_mode.itemData(i) for i in range(t.fvp_mode.count())] == ["turbo", "base"]
    assert [t.fvp_motion.itemData(i) for i in range(t.fvp_motion.count())] == [4, 2, 1]


def case_enhance_no_video_guard(w, spy):
    """Senza video sorgente generate_video_enhance NON deve enqueuare nulla (solo warning)."""
    t = w.video_def_tab
    t._video = None; _set_engine(t, 1)
    w.generate_video_enhance()
    assert not spy.calls, f"enhance senza video non doveva inviare richieste: {spy.calls}"


def case_enhance_engine_enable_toggle(w, spy):
    """_on_engine: SeedVR2 abilita seedvr2_model/resolution/blocks/batch e disabilita esrgan_model/scale;
    ESRGAN fa l'opposto. Regressione se i controlli restano abilitati/disabilitati a sproposito."""
    t = w.video_def_tab
    _set_engine(t, 0)
    assert t.seedvr2_model.isEnabled() and t.resolution.isEnabled(), "SeedVR2: controlli SeedVR2 disabilitati"
    assert not t.esrgan_model.isEnabled() and not t.scale.isEnabled(), "SeedVR2: controlli ESRGAN restano abilitati"
    _set_engine(t, 1)
    assert t.esrgan_model.isEnabled() and t.scale.isEnabled(), "ESRGAN: controlli ESRGAN disabilitati"
    assert not t.seedvr2_model.isEnabled(), "ESRGAN: seedvr2_model resta abilitato"


def case_enhance_config_resolution_batch(w, spy):
    """Risoluzione/batch/blocks impostati nella GUI devono comparire in get_config (param SeedVR2)."""
    t = w.video_def_tab
    _set_engine(t, 0)
    t.resolution.setValue(2160); t.batch.setValue(9); t.blocks.setValue(12)
    cfg = t.get_config()
    assert cfg["resolution"] == 2160 and cfg["batch_size"] == 9 and cfg["blocks_to_swap"] == 12, \
        f"config SeedVR2 non riflette la GUI: {cfg}"


def case_enhance_config_rife_toggle(w, spy):
    """RIFE: checkbox + moltiplicatore (indice+2) + checkpoint devono fluire in get_config."""
    t = w.video_def_tab
    t.rife.setChecked(True); t.rife_mult.setCurrentIndex(2); t.rife_ckpt.setCurrentIndex(0)
    cfg = t.get_config()
    assert cfg["rife"] is True, f"rife non attivato: {cfg}"
    assert cfg["rife_multiplier"] == 4, f"moltiplicatore RIFE atteso 4 (idx2+2): {cfg}"
    assert cfg["rife_ckpt"] == t.rife_ckpt.currentText(), f"ckpt RIFE non propagato: {cfg}"


def case_enhance_config_engine_mapping(w, spy):
    """get_config deve mappare l'indice combo → chiave engine ('seedvr2'/'esrgan'/'flashvsr')."""
    t = w.video_def_tab
    want = {0: "seedvr2", 1: "esrgan", 2: "flashvsr", 3: "h3_upsampler", 4: "flashvsr_plus"}
    for idx, key in want.items():
        _set_engine(t, idx)
        assert t.get_config()["engine"] == key, f"engine idx {idx} → atteso {key}, got {t.get_config()['engine']}"


# ─── factory: UN caso per OGNI modello upscale ESRGAN concreto ──────────────────────
def _make_esrgan_model_case(model_name: str):
    def case(w, spy):
        src = _make_test_video()
        t = w.video_def_tab
        t.set_video(src); _set_engine(t, 1)
        assert _select(t.esrgan_model, model_name), f"modello ESRGAN non in combo: {model_name}"
        w.generate_video_enhance()
        assert spy.calls, f"ESRGAN [{model_name}]: nessun enqueue"
        c = spy.calls[-1]
        assert c["module"] == "video_upscale", f"module inatteso: {c}"
        assert c["params"].get("model") == model_name, \
            f"il modello selezionato non è arrivato al backend: atteso {model_name}, got {c['params'].get('model')}"
    return case


def case_enhance_esrgan_all_models_loop(w, spy):
    """Fallback se l'enumerazione a import-time è fallita: itera a runtime TUTTI i modelli ESRGAN
    della combo e verifica che ciascuno produca un enqueue video_upscale con quel modello.
    Raccoglie i falliti così si scopre 'funziona solo un modello'."""
    src = _make_test_video()
    t = w.video_def_tab
    t.set_video(src); _set_engine(t, 1)
    names = [t.esrgan_model.itemText(i) for i in range(t.esrgan_model.count())
             if not t.esrgan_model.itemText(i).startswith("(")]
    if not names:
        return  # nessun modello upscale installato: niente da iterare (no-crash)
    failed = []
    for n in names:
        t.esrgan_model.setCurrentIndex(t.esrgan_model.findText(n))
        before = len(spy.calls)
        try:
            w.generate_video_enhance()
            c = spy.calls[-1]
            if len(spy.calls) == before or c["module"] != "video_upscale" or c["params"].get("model") != n:
                failed.append(n)
        except Exception as e:                            # noqa: BLE001
            failed.append(f"{n}({type(e).__name__})")
    assert not failed, f"modelli ESRGAN che NON instradano correttamente: {failed}"


# ─── factory: UN caso per OGNI modello SeedVR2 ─────────────────────────────────────
def _make_seedvr2_model_case(model_key: str):
    def case(w, spy):
        src = _make_test_video()
        t = w.video_def_tab
        t.set_video(src); _set_engine(t, 0)
        assert _select(t.seedvr2_model, model_key), f"modello SeedVR2 non in combo: {model_key}"
        cfg = t.get_config()
        assert cfg.get("seedvr2_model"), f"mapping SeedVR2 vuoto per {model_key}: {cfg}"
        from core import seedvr2_service as sv
        orig = sv.preflight
        sv.preflight = lambda: []
        try:
            w.generate_video_enhance()
        finally:
            sv.preflight = orig
        assert spy.calls, f"SeedVR2 [{model_key}]: nessun enqueue"
    return case


# ════════════════════════ RESTAURO VOLTI (generate_face_restore) ════════════════════════
def case_face_restore_basic(w, spy):
    """Restauro volti: con video sorgente la GUI deve enqueuare module='video_face' con
    image_path + model + detection + fidelity."""
    src = _make_test_video()
    t = w.video_def_tab
    t.set_video(src)
    w.generate_face_restore()
    assert spy.calls, "face restore: nessun enqueue (handler non ha instradato al nativo)"
    c = spy.calls[-1]
    assert c["module"] == "video_face", f"module inatteso per face restore: {c}"
    p = c["params"]
    assert p.get("image_path") == src, f"sorgente mancante: {p}"
    assert p.get("model") and p.get("detection") is not None, f"model/detection mancanti: {p}"


def case_face_restore_no_video_guard(w, spy):
    """Senza video generate_face_restore NON deve enqueuare nulla."""
    t = w.video_def_tab
    t._video = None
    w.generate_face_restore()
    assert not spy.calls, f"face restore senza video non doveva inviare richieste: {spy.calls}"


def case_face_restore_fidelity(w, spy):
    """Lo slider fidelity (0..100) deve arrivare al backend come 0..1."""
    src = _make_test_video()
    t = w.video_def_tab
    t.set_video(src); t.fr_fidelity.setValue(30)
    w.generate_face_restore()
    assert spy.calls, "face restore: nessun enqueue"
    p = spy.calls[-1]["params"]
    assert abs(float(p.get("fidelity", -1)) - 0.30) < 1e-6, f"fidelity 30→0.30 non propagata: {p}"


def _make_fr_model_case(model_name: str):
    def case(w, spy):
        src = _make_test_video()
        t = w.video_def_tab
        t.set_video(src)
        assert _select(t.fr_model, model_name), f"modello restauro non in combo: {model_name}"
        w.generate_face_restore()
        assert spy.calls, f"face restore [{model_name}]: nessun enqueue"
        c = spy.calls[-1]
        assert c["module"] == "video_face", f"module inatteso: {c}"
        assert c["params"].get("model") == model_name, \
            f"modello volti selezionato non arrivato al backend: atteso {model_name}, got {c['params'].get('model')}"
    return case


def _make_fr_detector_case(det_name: str):
    def case(w, spy):
        src = _make_test_video()
        t = w.video_def_tab
        t.set_video(src)
        assert _select(t.fr_detection, det_name), f"detector non in combo: {det_name}"
        w.generate_face_restore()
        assert spy.calls, f"face restore detector [{det_name}]: nessun enqueue"
        c = spy.calls[-1]
        assert c["params"].get("detection") == det_name, \
            f"detector selezionato non arrivato al backend: atteso {det_name}, got {c['params'].get('detection')}"
    return case


# ── LIVE: auto-chain RIFE + FaceRestore dopo super-risoluzione ────────────────────
def case_rife_facerestore_chain_live(w, spy):
    """LIVE: ESRGAN (veloce) + RIFE (fps) + chain_face ON su un video REALE. In run_real il worker
    di upscale viene eseguito; il callback done, con chain_face, enqueua a sua volta il restauro volti
    (catturato ed eseguito anch'esso da RealSpy) → valida l'auto-chain RIFE→FaceRestore end-to-end."""
    src = _make_test_video()                             # _vdef_test_input.mp4 ora è un video REALE
    t = w.video_def_tab
    t.set_video(src)
    _set_engine(t, 1)                                    # ESRGAN = più veloce per il chain
    if t.esrgan_model.count() == 0:
        raise AssertionError("nessun modello ESRGAN disponibile (scan disco vuoto)")
    t.esrgan_model.setCurrentIndex(0)
    t.scale.setValue(2.0)
    t.rife.setChecked(True)
    t.rife_mult.setCurrentText("2×")
    t.chain_face.setChecked(True)
    w.generate_video_enhance()
    assert spy.calls, "enhance+rife+chain: nessun enqueue"
    assert spy.calls[0]["module"] == "video_upscale", f"module iniziale inatteso: {spy.calls[0]}"


# ════════════════════════ assemblaggio CASES ════════════════════════
CASES = [
    ("videodef_rife_facerestore_chain_live",
     "LIVE: ESRGAN + RIFE + chain_face → auto-chain super-risoluzione→RIFE→restauro volti.",
     case_rife_facerestore_chain_live),
    ("videodef_enhance_engine_esrgan",
     "ESRGAN: la GUI enqueua module=video_upscale con video+scale+model.", case_enhance_engine_esrgan),
    ("videodef_enhance_engine_seedvr2",
     "SeedVR2: con preflight ok la GUI enqueua il worker SeedVR2.", case_enhance_engine_seedvr2),
    ("videodef_enhance_engine_flashvsr",
     "FlashVSR: con preflight ok la GUI enqueua il worker FlashVSR.", case_enhance_engine_flashvsr),
    ("videodef_enhance_engine_flashvsrplus",
     "FlashVSR+: variante e movimento arrivano al worker, preflight con la variante scelta.",
     case_enhance_engine_flashvsrplus),
    ("videodef_enhance_flashvsrplus_controls",
     "FlashVSR+: i controlli variante/movimento si attivano solo con quel motore.",
     case_enhance_flashvsrplus_controls),
    ("videodef_enhance_no_video_guard",
     "Enhance senza video: nessuna richiesta inviata (solo warning, no-crash).", case_enhance_no_video_guard),
    ("videodef_enhance_engine_enable_toggle",
     "_on_engine abilita/disabilita i controlli giusti per SeedVR2 vs ESRGAN.", case_enhance_engine_enable_toggle),
    ("videodef_enhance_config_resolution_batch",
     "Risoluzione/batch/blocks dalla GUI fluiscono in get_config (SeedVR2).", case_enhance_config_resolution_batch),
    ("videodef_enhance_config_rife_toggle",
     "RIFE: checkbox + moltiplicatore (idx+2) + ckpt fluiscono in get_config.", case_enhance_config_rife_toggle),
    ("videodef_enhance_config_engine_mapping",
     "get_config mappa indice combo → chiave engine seedvr2/esrgan/flashvsr.", case_enhance_config_engine_mapping),
    ("videodef_enhance_esrgan_all_models_loop",
     "Itera TUTTI i modelli ESRGAN a runtime: scopre se 'funziona solo un modello'.",
     case_enhance_esrgan_all_models_loop),
    ("videodef_face_restore_basic",
     "Restauro volti: enqueua module=video_face con image_path+model+detection+fidelity.",
     case_face_restore_basic),
    ("videodef_face_restore_no_video_guard",
     "Restauro volti senza video: nessuna richiesta (solo warning).", case_face_restore_no_video_guard),
    ("videodef_face_restore_fidelity",
     "Slider fidelity 0..100 → backend 0..1 (30→0.30).", case_face_restore_fidelity),
]

# per-modello ESRGAN (UN caso ciascuno) — enumerati a import-time
for _name in _ESRGAN_NAMES:
    _slug = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in _name)
    CASES.append((f"videodef_enhance_esrgan_model__{_slug}",
                  f"ESRGAN modello '{_name}': la selezione arriva al backend (video_upscale).",
                  _make_esrgan_model_case(_name)))

# per-modello SeedVR2 (UN caso ciascuno)
for _name in _SEEDVR2_KEYS:
    _slug = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in _name)
    CASES.append((f"videodef_enhance_seedvr2_model__{_slug}",
                  f"SeedVR2 modello '{_name}': get_config risolve il file e l'enqueue parte.",
                  _make_seedvr2_model_case(_name)))

# per-modello restauro volti (UN caso ciascuno)
for _name in _FR_MODELS:
    _slug = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in _name)
    CASES.append((f"videodef_face_restore_model__{_slug}",
                  f"Restauro volti modello '{_name}': la selezione arriva al backend (video_face).",
                  _make_fr_model_case(_name)))

# per-detector restauro volti (UN caso ciascuno)
for _name in _FR_DETECTORS:
    _slug = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in _name)
    CASES.append((f"videodef_face_restore_detector__{_slug}",
                  f"Restauro volti detector '{_name}': la selezione arriva al backend.",
                  _make_fr_detector_case(_name)))
