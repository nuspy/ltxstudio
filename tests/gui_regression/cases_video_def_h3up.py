"""Casi GUI del motore «MiniMax H3 Video Upsampler» nel tab Definizione video (ui/tabs/video_def_tab.py,
w.video_def_tab). Funzione = upscale video, arch = MiniMax H3.

Verificano, dalla GUI vera (widget reali + generate_video_enhance, backend intercettato dallo Spy):
  • il motore compare, mostra le SUE opzioni e solo quelle (progressive disclosure);
  • la scelta dell'utente (modello H3 Ref2VA, megapixel, forza, guida, budget) arriva intatta al worker;
  • pesi mancanti → avviso + «Apri tab Pesi modelli», NESSUN download e nessun job;
  • arch+funzione: il dropdown offre SOLO i DiT Ref2VA (non FL2VA), la LoRA di guida e l'upscaler non compaiono
    né nel pannello LoRA né nei menu modello del tab Video, né fra i modelli ESRGAN.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import harness as H


def _video() -> str:
    p = H.OUT_DIR / "_h3up_test_input.mp4"
    if not p.exists():
        p.write_bytes(bytes.fromhex("0000001866747970697336320000000069736f6d6d703432"))
    return str(p)


def _engine(t, idx=3):
    t.engine.setCurrentIndex(idx)
    t._on_engine()


def _patch(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    return lambda: setattr(obj, name, old)


def case_h3up_engine_disclosure(w, spy):
    """Il motore H3 mostra le sue opzioni; gli altri motori no. get_config → engine 'h3_upsampler'."""
    t = w.video_def_tab
    assert t.engine.count() == 4 and t.engine.itemText(3), "manca il quarto motore nel combo"
    _engine(t, 3)
    assert not t.h3up_box.isHidden(), "opzioni H3 nascoste col motore H3"
    cfg = t.get_config()
    assert cfg["engine"] == "h3_upsampler" and cfg["h3up"], cfg
    for k in ("h3_profile", "megapixels", "denoise", "guide", "lora_strength", "tile_tokens"):
        assert k in cfg["h3up"], f"manca {k} in h3up: {cfg['h3up']}"
    for idx in (0, 1, 2):
        _engine(t, idx)
        assert t.h3up_box.isHidden(), f"opzioni H3 visibili con il motore {idx}"
        assert t.get_config()["h3up"] is None
    _engine(t, 0)


def case_h3up_routes_with_user_choices(w, spy):
    """generate_video_enhance con il motore H3: enqueua il worker H3 con ESATTAMENTE le scelte dell'utente."""
    from core import h3_upsample_service as hs
    t = w.video_def_tab
    src = _video()
    t.set_video(src)
    _engine(t, 3)
    assert t.h3up_model.count() >= 1, "nessun modello H3 Ref2VA nel menu"
    t.h3up_model.setCurrentIndex(t.h3up_model.count() - 1)
    prof = t.h3up_model.currentData()
    t.h3up_mp.setValue(1.5)
    t.h3up_denoise.setValue(0.2)
    t.h3up_guide.setChecked(False)
    t.h3up_lora.setValue(0.8)
    t.h3up_tiles.setValue(50000)
    t.h3up_te.setCurrentIndex(1)
    undo = _patch(hs, "preflight", lambda p: [])             # pesi/venv assenti in CI: qui si verifica l'istruzione
    try:
        w.generate_video_enhance()
    finally:
        undo()
    assert spy.calls, "H3 Upsampler: nessun enqueue (motore non instradato)"
    c = spy.calls[-1]
    p = c["params"]
    assert "H3" in c["label"], c
    assert p["video"] == src and p["h3_profile"] == prof, p
    assert abs(p["megapixels"] - 1.5) < 1e-9 and abs(p["denoise"] - 0.2) < 1e-9, p
    assert p["guide"] is False and abs(p["lora_strength"] - 0.8) < 1e-9 and p["tile_tokens"] == 50000, p
    assert p.get("h3_te") == "mmh3-te-fp8", p


def case_h3up_no_auto_model_choice(w, spy):
    """Nessun profilo scelto al posto dell'utente: con il menu senza voci il job NON parte e dice perché."""
    from core import h3_upsample_service as hs
    t = w.video_def_tab
    t.set_video(_video())
    _engine(t, 3)
    t.h3up_model.clear()                                     # utente senza alcun modello nel menu
    undo = _patch(hs, "preflight", hs.preflight)             # preflight VERO: deve rifiutare (profilo mancante)
    try:
        w.generate_video_enhance()
    finally:
        undo()
    assert not spy.calls, f"senza modello H3 scelto non doveva partire nulla: {spy.calls}"
    t._refilter_h3up_models()


def case_h3up_missing_weights_no_download(w, spy):
    """Pesi mancanti: dialog con «Apri tab Pesi modelli» (segnale emesso), nessun job, nessun download."""
    from core import ensure_weights as ew
    from core import h3_upsample_spec as U
    t = w.video_def_tab
    t.set_video(_video())
    _engine(t, 3)
    t.h3up_model.setCurrentIndex(0)
    fake = [{"id": U.UPSCALER_ID, "approx_gb": 0.69}, {"id": U.GUIDE_LORA_ID, "approx_gb": 0.62}]
    seen = {"open": 0, "text": ""}
    t.request_open_weights_tab.connect(lambda: seen.__setitem__("open", seen["open"] + 1))

    def ask(text):
        seen["text"] = text
        return "weights"
    undo = [_patch(U, "missing_weights", lambda p: fake), _patch(t, "_ask_missing_h3up", ask),
            _patch(ew, "offer_download_gui", lambda *a, **k: (_ for _ in ()).throw(AssertionError("download tentato")))]
    try:
        t._on_run()
    finally:
        for u in undo:
            u()
    assert seen["open"] == 1, "«Apri tab Pesi modelli» non emesso"
    assert U.UPSCALER_ID in seen["text"] and "GB" in seen["text"], seen["text"]
    assert not spy.calls, f"con pesi mancanti non doveva partire nessun job: {spy.calls}"
    # con «Annulla» nessun segnale
    seen["open"] = 0
    undo = [_patch(U, "missing_weights", lambda p: fake), _patch(t, "_ask_missing_h3up", lambda text: "cancel")]
    try:
        t._on_run()
    finally:
        for u in undo:
            u()
    assert seen["open"] == 0 and not spy.calls


def case_h3up_main_window_opens_weights_tab(w, spy):
    """Il segnale del tab è collegato: porta davvero al tab «Pesi modelli» della finestra."""
    w.video_def_tab.request_open_weights_tab.emit()
    assert w.tabs.currentWidget() is w.weights_tab, "il segnale non apre il tab Pesi modelli"


def case_h3up_arch_function_filters(w, spy):
    """Arch+funzione dichiarate nel catalogo e lette dai filtri: il menu H3 offre SOLO i DiT Ref2VA; upscaler e LoRA
    di guida (arch minimax_h3, use upscale_video) non compaiono né fra le LoRA, né nei menu del tab Video, né fra gli
    ESRGAN."""
    from core import h3_upsample_spec as U
    from core import lora_catalog as lc
    from core import minimax_h3_spec as S
    from core import weights_catalog as wc
    by = {e["id"]: e for e in wc.load()}
    for i in (U.UPSCALER_ID, U.GUIDE_LORA_ID):
        e = by[i]
        assert e["arch"] == U.ARCH and e["category"] == "video" and "upscale_video" in e["use"], e
        assert U.CAP in e["used_by"], e
    # 1) il menu del tab contiene solo i DiT Ref2VA dei profili
    t = w.video_def_tab
    ids = {U.dit_id(t.h3up_model.itemData(i)) for i in range(t.h3up_model.count())}
    assert ids <= {r for (_f, r, _t) in S.PROFILES.values()}, ids
    assert not (ids & {f for (f, _r, _t) in S.PROFILES.values()}), "FL2VA nel menu dell'Upsampler"
    # 2) pannello LoRA: con il file di guida PRESENTE su disco non compare per nessuna arch
    with tempfile.TemporaryDirectory() as d:
        e = by[U.GUIDE_LORA_ID]
        dest = Path(wc.dest_for(e, d))
        dest.mkdir(parents=True, exist_ok=True)
        (dest / e["filename"]).write_bytes(b"x")
        old = os.environ.get("AI_MODELS_DIR")
        os.environ["AI_MODELS_DIR"] = d
        try:
            for arch in ("MiniMax-H3", "LTX", "Flux", "Krea2", "Qwen21"):
                names = [x.filename for x in lc._registry_loras(arch)]
                assert e["filename"] not in names, f"LoRA di guida nel pannello LoRA {arch}: {names}"
        finally:
            if old is None:
                os.environ.pop("AI_MODELS_DIR", None)
            else:
                os.environ["AI_MODELS_DIR"] = old
    # 3) menu modello del tab Video con arch H3: nessuna voce dell'upscaler/LoRA di guida
    vt = w.video_tab
    keys = []
    for i in range(vt.model.count()):
        dta = vt.model.itemData(i)
        keys.append(str(dta))
    assert not any(U.UPSCALER_ID in k or U.GUIDE_LORA_ID in k or "h3upscale" in k for k in keys), keys
    # 4) ESRGAN: l'upscaler latente H3 non è un modello ESRGAN
    esr = [t.esrgan_model.itemText(i) for i in range(t.esrgan_model.count())]
    assert not any("latent_upscaler" in n for n in esr), esr


CASES = [
    ("videodef_h3up_engine_disclosure",
     "Motore H3 Upsampler: opzioni visibili solo con quel motore; get_config → h3_upsampler.", case_h3up_engine_disclosure),
    ("videodef_h3up_routes_with_user_choices",
     "H3 Upsampler: modello/MP/denoise/guida/LoRA/tile scelti nella GUI arrivano al worker.",
     case_h3up_routes_with_user_choices),
    ("videodef_h3up_no_auto_model_choice",
     "H3 Upsampler: senza modello H3 scelto non parte nulla (nessuna scelta automatica).", case_h3up_no_auto_model_choice),
    ("videodef_h3up_missing_weights_no_download",
     "H3 Upsampler: pesi mancanti → avviso con «Apri tab Pesi modelli», nessun job, nessun download.",
     case_h3up_missing_weights_no_download),
    ("videodef_h3up_main_window_opens_weights_tab",
     "H3 Upsampler: il segnale del tab apre il tab Pesi modelli della finestra.",
     case_h3up_main_window_opens_weights_tab),
    ("videodef_h3up_arch_function_filters",
     "H3 Upsampler: arch minimax_h3 + funzione upscale_video; compare SOLO nel suo tab e fra i DiT Ref2VA.",
     case_h3up_arch_function_filters),
]
