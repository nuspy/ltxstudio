"""Casi GUI (mock) del FILTRO PER ARCHITETTURA di immagini e video (task 6, 17/10).

Inventario SINTETICO: file safetensors con il solo header (modelspec/chiavi) in una cartella temporanea, così il test
non dipende dai modelli dell'utente e riproduce le cause trovate sul PC:
  - «PornFlux_20» (FLUX.1) passava per FLUX.2 (sottostringa «flux_2» nel nome);
  - un FLUX.2 Klein senza «flux2» nel nome spariva dal motore FLUX.2;
  - i Krea 2 finivano fra gli «UNET Flux» dell'Inpaint (parola «krea» nel nome);
  - un checkpoint Z-Image senza «z-image» nel nome non compariva fra gli Z-Image;
  - encoder e VAE di «Genera immagine» mostravano l'intero inventario con ogni architettura;
  - un Qwen21 etichettato «LTX» dal catalogo esterno e i file `_aux_` finivano nel menu video LTX;
  - cambiando architettura il combo sceglieva da solo il primo modello della nuova.
"""
from __future__ import annotations

import json
import os
import struct
import tempfile
from pathlib import Path

import harness as H

F16 = "F16"


def _write_st(path: Path, tensors: dict, meta: dict | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    head = {k: {"dtype": F16, "shape": list(sh), "data_offsets": [0, 0]} for k, sh in tensors.items()}
    if meta:
        head["__metadata__"] = meta
    raw = json.dumps(head).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)


FLUX1 = {"double_blocks.0.img_attn.qkv.weight": (9216, 3072), "single_blocks.0.linear1.weight": (21504, 3072),
         "vector_in.in_layer.weight": (3072, 768), "img_in.weight": (3072, 64)}
FLUX2_DEV = {"double_blocks.0.img_attn.qkv.weight": (9216, 6144), "single_blocks.0.linear1.weight": (21504, 6144),
             "guidance_in.in_layer.weight": (6144, 256), "img_in.weight": (6144, 128),
             "txt_in.weight": (6144, 15360)}
FLUX2_KLEIN = {"double_blocks.0.img_attn.qkv.weight": (9216, 4096), "single_blocks.0.linear1.weight": (21504, 4096),
               "img_in.weight": (4096, 128)}
FLUX1_FILL = dict(FLUX1, **{"img_in.weight": (3072, 384)})
NUNCHAKU = {"single_transformer_blocks.0.norm.linear.qweight": (64, 64), "img_in.weight": (3072, 64)}
OPAQUE = {"blocks.0.attn.wq.weight": (64, 64)}                 # convertito «-native»: l'header non dice nulla

# nome relativo → (tensori, metadata)
DIFFUSION = {
    "Flux\\flux1_dev_model.safetensors": (FLUX1, None),
    "Flux\\PornFlux_20_trap.safetensors": (FLUX1, None),            # FLUX.1 con «flux_2» nel nome
    "Flux\\klein_senza_nome_flux.safetensors": (FLUX2_KLEIN, None),  # FLUX.2 Klein: header, non nome
    "Flux\\flux2_dev_named.safetensors": (FLUX2_DEV, None),
    "Flux\\Flux_Kontext_X.safetensors": (FLUX1, None),
    "Flux\\FillX.safetensors": (FLUX1_FILL, None),                 # FLUX.1 Fill (384 canali)
    "Flux\\nunchaku\\svdq-fp4_r32-flux.1-dev.safetensors": (NUNCHAKU, None),   # SVDQuant: nessun inpaint
    "Krea2\\KreaX_native.safetensors": (OPAQUE, None),
    "Qwen21\\NoctFake_native.safetensors": (OPAQUE, None),
    "miofavorito.safetensors": (OPAQUE, {"modelspec.architecture": "z-image"}),   # Z-Image dal solo header
    "Z-Image\\zi_in_cartella.safetensors": (OPAQUE, None),
    "Ernie\\ernie_x.safetensors": (OPAQUE, None),
    "LTX\\ltx_ok.safetensors": (OPAQUE, {"modelspec.architecture": "Lightricks/ltx-2"}),
    "LTX\\_ltx23_aux_components.safetensors": (OPAQUE, {"modelspec.architecture": "Lightricks/ltx-2"}),
    "MiniMax\\mm_h3_fl2va.safetensors": (OPAQUE, None),
}
TEXT_ENCODERS = ["t5xxl_fp16.safetensors", "ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors",
                 "Flux\\mistral_3_small_flux2_fp4_mixed.safetensors", "Flux\\qwen_3_8b_fp8mixed.safetensors",
                 "Qwen\\qwen_3_4b.safetensors", "Ernie\\ministral-3-3b.safetensors",
                 "Krea2\\qwen3vl_4b_uncensored-fp8-native.safetensors", "Qwen21\\qwen3vl_8b_nvfp4-native.safetensors",
                 "gemma_3_12B_it.safetensors", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"]
VAES = ["Flux\\ae.safetensors", "Flux\\flux2-vae.safetensors", "Qwen21\\qwen_image_2.1_vae_bf16.safetensors",
        "LTX23_video_vae_bf16.safetensors", "Wan\\Wan2_1_VAE_bf16.safetensors"]


class Fixture:
    """Cartella modelli sintetica + patch di MODELS_ROOT/cache del detector (ripristinate in close())."""

    def __init__(self, w):
        from core import arch_filter, model_detect, offline_inventory
        self.tmp = tempfile.TemporaryDirectory(prefix="archf_")
        self.root = Path(self.tmp.name)
        for rel, (tens, meta) in DIFFUSION.items():
            _write_st(self.root / "diffusion_models" / rel.replace("\\", os.sep), tens, meta)
        self._saved = (offline_inventory.MODELS_ROOT, model_detect._MODELS_ROOT, model_detect.CACHE_PATH,
                       dict(model_detect._MEM))
        offline_inventory.MODELS_ROOT = self.root
        model_detect._MODELS_ROOT = self.root
        model_detect.CACHE_PATH = self.root / "model_detect_test.json"       # MAI data/model_detect.json
        model_detect._MEM.update({"cache": None, "mtime": None})
        arch_filter.clear_memo()
        self.names = list(DIFFUSION)
        w._inventory_set = True                      # il timer di stato non ripopola con l'inventario vero
        w.frame1_tab.set_model_inventory(list(self.names), [])
        w.frame1_tab.set_gen_inventory(text_encoders=list(TEXT_ENCODERS), vaes=list(VAES))
        w.video_tab.set_model_inventory(list(self.names), [], [], vram_gb=32)

    def close(self):
        from core import arch_filter, model_detect, offline_inventory
        offline_inventory.MODELS_ROOT, model_detect._MODELS_ROOT, model_detect.CACHE_PATH, mem = self._saved
        model_detect._MEM.clear(); model_detect._MEM.update(mem)
        arch_filter.clear_memo()
        self.tmp.cleanup()


def _texts(combo, skip_auto=True):
    out = []
    for i in range(combo.count()):
        d = combo.itemData(i)
        if skip_auto and (d == "__auto__" or d is None):
            continue
        out.append(combo.itemText(i))
    return out


def _base(names):
    return sorted(n.split("\\")[-1] for n in names)


def case_arch_gen_models_by_header(w, spy):
    fx = Fixture(w)
    try:
        f = w.frame1_tab
        f.img_nsfw.setCurrentText("SFW")
        f.gen_arch.setCurrentText("FLUX")
        got = _base(_texts(f.gen_model))
        want = sorted(["flux1_dev_model.safetensors", "PornFlux_20_trap.safetensors", "flux2_dev_named.safetensors",
                       "svdq-fp4_r32-flux.1-dev.safetensors"])
        assert got == want, f"FLUX: {got} != {want}"
        f.gen_arch.setCurrentText("Z-Image")
        got = _base(_texts(f.gen_model))
        want = sorted(["miofavorito.safetensors", "zi_in_cartella.safetensors"])
        assert got == want, f"Z-Image (anche solo da header): {got} != {want}"
        f.gen_arch.setCurrentText("Ernie")
        got = _base(_texts(f.gen_model))
        assert got == ["ernie_x.safetensors"], f"Ernie: {got}"
    finally:
        fx.close()


def case_arch_edit_engines_by_header(w, spy):
    fx = Fixture(w)
    try:
        f = w.frame1_tab
        f.img_nsfw.setCurrentText("NSFW")
        f.integrate_engine.setCurrentIndex(0)            # Kontext (FLUX.1)
        f._reload_flux_combos()
        got = _base(_texts(f.integrate_model, skip_auto=False))
        assert got == ["Flux_Kontext_X.safetensors"], f"Kontext: {got}"
        f.integrate_engine.setCurrentIndex(1)            # FLUX.2
        f._reload_flux_combos()
        got = _base(_texts(f.integrate_model, skip_auto=False))
        want = ["klein_senza_nome_flux.safetensors"]
        assert got == want, f"FLUX.2 (header, non nome; solo Klein: flux2_kontext non esiste per il Dev): {got} != {want}"
        assert "PornFlux_20_trap.safetensors" not in got, "un FLUX.1 con «Flux_20» nel nome non è FLUX.2"
        refine = _base(_texts(f.up_refine_model, skip_auto=False))
        assert refine == sorted(["klein_senza_nome_flux.safetensors", "flux2_dev_named.safetensors"]), \
            f"rifinitura upscale (FLUX.2 caricabile): {refine}"
        from core.gen_pipeline import is_flux2
        assert not is_flux2("PornFlux_20_trap.safetensors") and is_flux2("FLUX_2_Dev.safetensors"), "is_flux2 del dispatch"
    finally:
        fx.close()


def case_arch_inpaint_dynamic_by_header(w, spy):
    """Il menu Inpaint mostra solo ciò che il backend sa fare (header): Fill (384 canali), FLUX.1 base (64), Klein
    (128 canali senza guidance_in). Fuori: FLUX.2 Dev, Nunchaku, Krea 2, Kontext, checkpoint (ComfyUI). Ogni voce
    prende passi e guidance del PROPRIO modello (non quelli del modello scelto prima)."""
    fx = Fixture(w)
    try:
        f = w.frame1_tab
        f._refresh_inpaint_checkpoints(["flux_aaa.safetensors", "Redux_X.safetensors"])   # checkpoint: mai in menu
        labels = [f.inp_model.itemText(i) for i in range(f.inp_model.count())]
        dyn = [l for l in labels if "UNET Flux" in l]
        joined = " | ".join(dyn)
        assert not [l for l in labels if "checkpoint Flux" in l], f"checkpoint (ComfyUI) nel menu Inpaint: {labels}"
        for bad in ("KreaX_native", "Flux_Kontext_X", "flux2_dev_named", "svdq-fp4", "PornFlux_20_trap_NO"):
            assert bad not in joined, f"{bad} non sa fare inpaint ma è nel menu: {dyn}"
        for ok in ("flux1_dev_model", "PornFlux_20_trap", "klein_senza_nome_flux", "FillX"):
            assert ok in joined, f"{ok} manca dall'Inpaint: {dyn}"
        # default del proprio modello, scelti in sequenza (nessuna eredità dal precedente)
        want = {"FillX  · UNET Flux": (40, 30.0), "flux1_dev_model  · UNET Flux": (28, 3.5),
                "klein_senza_nome_flux  · UNET Flux": (12, 4.0), "PornFlux_20_trap  · UNET Flux": (28, 3.5)}
        for lbl in ("FillX  · UNET Flux", "klein_senza_nome_flux  · UNET Flux", "flux1_dev_model  · UNET Flux",
                    "FillX  · UNET Flux", "PornFlux_20_trap  · UNET Flux"):
            i = f.inp_model.findText(lbl)
            assert i >= 0, (lbl, labels)
            f.inp_model.setCurrentIndex(i)
            got = (f.inp_steps.value(), round(f.inp_guidance.value(), 2))
            assert got == want[lbl], f"{lbl}: passi/guidance {got} != {want[lbl]}"
        f.inp_model.setCurrentIndex(f.inp_model.findText("klein_senza_nome_flux  · UNET Flux"))
        assert f.gen_lora_tab is not None and f._dynamic_kind["klein_senza_nome_flux  · UNET Flux"] == "klein"
    finally:
        fx.close()


def case_arch_components_by_arch(w, spy):
    fx = Fixture(w)
    try:
        f = w.frame1_tab
        f.img_nsfw.setCurrentText("SFW")

        def enc():
            return sorted(f.gen_encoder.itemText(i).split("\\")[-1] for i in range(1, f.gen_encoder.count()))

        def vae():
            return sorted(f.gen_vae.itemText(i).split("\\")[-1] for i in range(1, f.gen_vae.count()))
        f.gen_arch.setCurrentText("Z-Image")
        assert enc() == ["qwen_3_4b.safetensors"], enc()
        assert vae() == ["ae.safetensors"], vae()
        f.gen_arch.setCurrentText("Ernie")
        assert enc() == ["ministral-3-3b.safetensors"], enc()
        assert vae() == ["flux2-vae.safetensors"], vae()
        f.gen_arch.setCurrentText("Krea 2")
        assert enc() == ["qwen3vl_4b_uncensored-fp8-native.safetensors"], enc()
        assert vae() == [], vae()
        f.gen_arch.setCurrentText("Qwen-Image 2.1")
        assert enc() == ["qwen3vl_8b_nvfp4-native.safetensors"], enc()
        assert vae() == ["qwen_image_2.1_vae_bf16.safetensors"], vae()
        f.gen_arch.setCurrentText("FLUX")
        f.gen_model.setCurrentIndex(f.gen_model.findText("Flux\\flux1_dev_model.safetensors"))
        assert enc() == sorted(["t5xxl_fp16.safetensors",
                                "ViT-L-14-TEXT-detail-improved-hiT-GmP-TE-only-HF.safetensors"]), enc()
        assert vae() == ["ae.safetensors"], vae()
        f.gen_model.setCurrentIndex(f.gen_model.findText("Flux\\flux2_dev_named.safetensors"))
        assert enc() == sorted(["mistral_3_small_flux2_fp4_mixed.safetensors"]), enc()
        assert vae() == ["flux2-vae.safetensors"], vae()
    finally:
        fx.close()


def case_arch_switch_never_picks_other_model(w, spy):
    """Il modello concreto scelto non appartiene alla nuova architettura → segnaposto, mai un altro modello."""
    from ui import model_menu as mm
    fx = Fixture(w)
    try:
        f = w.frame1_tab
        f.img_nsfw.setCurrentText("SFW")
        f.gen_arch.setCurrentText("FLUX")
        f.gen_model.setCurrentIndex(f.gen_model.findText("Flux\\flux1_dev_model.safetensors"))
        f.gen_arch.setCurrentText("Z-Image")
        assert mm.is_placeholder(f.gen_model), f"scelta FLUX → Z-Image: {f.gen_model.currentText()!r}"
        assert f._gen_model_name() == "", "il segnaposto non è un nome modello"
        assert mm.blocked(w, f.gen_model), "la generazione deve fermarsi sul segnaposto"
        f.gen_model.setCurrentIndex(f.gen_model.findText("Z-Image\\zi_in_cartella.safetensors"))
        f.gen_arch.setCurrentText("Krea 2")
        assert mm.is_placeholder(f.gen_model), f"Z-Image → Krea 2: {f.gen_model.currentText()!r}"
        f.gen_model.setCurrentIndex(f.gen_model.findData("krea2_turbo"))
        f.gen_arch.setCurrentText("Qwen-Image 2.1")
        assert mm.is_placeholder(f.gen_model), f"Krea 2 → Qwen: {f.gen_model.currentText()!r}"
        # «Auto» è un automatismo dichiarato: resta Auto dove esiste
        f.gen_arch.setCurrentText("FLUX")
        assert f.gen_model.currentData() == "__auto__" or mm.is_placeholder(f.gen_model)
    finally:
        fx.close()


def case_arch_video_ltx_menu_by_header(w, spy):
    from ui import model_menu as mm
    fx = Fixture(w)
    try:
        v = w.video_tab
        v._mcat.arch_of = lambda n: "LTX" if "NoctFake" in n else ""      # il catalogo esterno lo etichetta «LTX»
        v.nsfw.setCurrentText("NSFW")
        v.video_arch.setCurrentIndex(v.video_arch.findData("ltx"))
        v._refilter_models()
        diff = [v.model.itemData(i) or {} for i in range(v.model.count())]
        shown = sorted(d["model"].split("\\")[-1] for d in diff if d.get("kind") == "diffusion")
        assert shown == ["ltx_ok.safetensors"], f"menu LTX: {shown}"
        # scelta LTX concreta → cambio architettura → segnaposto
        v.model.setCurrentIndex(next(i for i in range(v.model.count())
                                     if (v.model.itemData(i) or {}).get("kind") == "diffusion"))
        v.video_arch.setCurrentIndex(v.video_arch.findData("lingbot"))
        assert mm.is_placeholder(v.model), f"LTX → LingBot: {v.model.currentText()!r}"
        assert v.model_is_placeholder() and mm.blocked(w, v.model)
        v.video_arch.setCurrentIndex(v.video_arch.findData("ltx"))
    finally:
        fx.close()


def case_arch_inpaint_unet_disappears_placeholder(w, spy):
    """Voce dinamica dell'Inpaint scelta e poi sparita dall'inventario → segnaposto, nessuna richiesta al backend."""
    from ui import model_menu as mm
    from core import arch_filter as af
    f = w.frame1_tab
    saved = af.inpaint_kind
    af.inpaint_kind = lambda rel: "base" if str(rel).startswith("flux_") else None
    try:
        f._models = ["flux_aaa.safetensors", "flux_bbb.safetensors"]
        f._refresh_inpaint_checkpoints([])
        j = f.inp_model.findText("flux_bbb  · UNET Flux")
        assert j >= 0, [f.inp_model.itemText(i) for i in range(f.inp_model.count())]
        f.inp_model.setCurrentIndex(j)
        f._models = ["flux_aaa.safetensors"]
        f._refresh_inpaint_checkpoints([])
        assert mm.is_placeholder(f.inp_model) and f.inp_model.currentIndex() == 0, f.inp_model.currentText()
        before = len(spy.calls)
        w.generate_inpaint("inpaint")
        assert len(spy.calls) == before, "Inpaint (inventario): con il segnaposto non deve partire nulla"
    finally:
        af.inpaint_kind = saved
        f._models = []


def case_arch_klein_default_steps(w, spy):
    """Klein e' distillato: il DEFAULT dei passi e' 12 (21 = foto invariata, live 17/10) in Inpaint, Integra e Colore;
    un valore scelto a mano dall'utente non viene toccato."""
    f = w.frame1_tab
    assert f.int_steps.value() == 20 and f.color_steps.value() == 20
    f.integrate_engine.setCurrentIndex(1)                      # FLUX.2 (Klein)
    assert f.int_steps.value() == 12, f.int_steps.value()
    f.integrate_engine.setCurrentIndex(0)                      # Kontext: torna 20
    assert f.int_steps.value() == 20
    f.integrate_engine.setCurrentIndex(1)
    f.int_steps.setValue(30)                                   # scelta dell'utente
    f.integrate_engine.setCurrentIndex(0)
    assert f.int_steps.value() == 30, "il valore scelto a mano non va cambiato"
    f.color_engine.setCurrentIndex(1)
    assert f.color_steps.value() == 12, f.color_steps.value()
    f.color_steps.setValue(18)
    f.color_engine.setCurrentIndex(0)
    assert f.color_steps.value() == 18
    # inpaint: voce Klein hardcoded = 12 passi
    f.inp_model.setCurrentIndex(f.inp_model.findText("miraclein_flux2klein_nsfw_v2  (FLUX.2 Klein, edit)"))
    assert f.inp_steps.value() == 12


def case_arch_child_error_reaches_dialog(w, spy):
    """L'errore chiaro del backend (FLUX.2 Dev non ha una pipeline di inpaint) arriva intero al messaggio della GUI."""
    from PySide6.QtWidgets import QMessageBox
    shown = []
    old = QMessageBox.critical
    QMessageBox.critical = staticmethod(lambda *a, **k: shown.append(a[2] if len(a) > 2 else "") or QMessageBox.Ok)
    msg = "«flux2-dev-nvfp4-mixed.safetensors» è FLUX.2 Dev: non ha una pipeline di inpaint/edit. Serve una variante Klein."
    try:
        w._on_native_inpaint_done({"status": "error", "error": msg})
    finally:
        QMessageBox.critical = old
    assert shown and "FLUX.2 Dev" in shown[0] and "Klein" in shown[0] and "Log:" not in shown[0], shown


def case_arch_function_menus_overlap(w, spy):
    """Un modello che sa fare PIU' funzioni compare in TUTTI i menu di quelle funzioni; uno che ne sa fare una sola solo
    nel suo. FLUX.1 base: Crea + Inpaint. Qwen 2.1: Crea + Inpaint + Integra + Colore. Krea 2 Turbo/Raw: Crea + Integra
    (motore Krea 2 Edit). Kontext: solo Integra. Klein e Fill: solo Modifica (backend: edit/inpaint richiedono una
    foto). FLUX.2 Dev: solo Crea."""
    fx = Fixture(w)
    try:
        f = w.frame1_tab
        f.img_nsfw.setCurrentText("SFW")
        f.gen_arch.setCurrentText("FLUX")
        gen = " | ".join(_texts(f.gen_model))
        inp = " | ".join(f.inp_model.itemText(i) for i in range(f.inp_model.count()))
        f.integrate_engine.setCurrentIndex(0)
        kontext = " | ".join(_texts(f.integrate_model, skip_auto=False))
        f.integrate_engine.setCurrentIndex(1)
        flux2 = " | ".join(_texts(f.integrate_model, skip_auto=False))
        assert "flux1_dev_model" in gen and "flux1_dev_model" in inp, "FLUX.1 base deve stare in Crea E in Inpaint"
        assert "Flux_Kontext_X" in kontext and "Flux_Kontext_X" not in gen and "Flux_Kontext_X" not in inp
        assert "klein_senza_nome_flux" in flux2 and "klein_senza_nome_flux" in inp and "klein_senza_nome_flux" not in gen
        assert "FillX" in inp and "FillX" not in gen, "Fill: solo Inpaint"
        assert "flux2_dev_named" in gen and "flux2_dev_named" not in inp and "flux2_dev_named" not in flux2, \
            "FLUX.2 Dev: solo Crea (nessuna pipeline di edit/inpaint)"
        # Qwen-Image 2.1: Crea + Inpaint + Integra + Colore
        f.gen_arch.setCurrentText("Qwen-Image 2.1")
        qgen = {f.gen_model.itemData(i) for i in range(f.gen_model.count())} - {None}
        f.integrate_engine.setCurrentIndex(3)
        qint = {f.integrate_model.itemData(i) for i in range(f.integrate_model.count())} - {None}
        f.color_engine.setCurrentIndex(3)
        qcol = {f.color_model.itemData(i) for i in range(f.color_model.count())} - {None}
        qinp = set(getattr(f, "_qwen21_inp", {}).values())
        assert qgen and qgen == qint == qcol == qinp, f"Qwen non uguale nei menu: {qgen} {qint} {qcol} {qinp}"
        # Krea 2: Turbo/Raw in Crea e nel motore Krea 2 Edit di Integra
        f.gen_arch.setCurrentText("Krea 2")
        kgen = {f.gen_model.itemData(i) for i in range(f.gen_model.count())}
        f.integrate_engine.setCurrentIndex(2)
        kint = {(f.integrate_model.itemData(i) or {}).get("krea2_variant") for i in range(f.integrate_model.count())}
        assert {"krea2_turbo", "krea2_raw"} <= kgen and kint == {"turbo", "raw"}, (kgen, kint)
    finally:
        fx.close()


def case_arch_inpaint_arch_selector(w, spy):
    """Selettore «Architettura» nel tab Modifica (Inpaint): filtra il menu; la scelta che non appartiene alla nuova
    architettura diventa «scegli un modello», mai un altro modello."""
    from ui import model_menu as mm
    fx = Fixture(w)
    try:
        f = w.frame1_tab
        f.img_nsfw.setCurrentText("SFW")
        view = f.inp_model.view()

        def visible():
            return [f.inp_model.itemText(i) for i in range(f.inp_model.count()) if not view.isRowHidden(i)]
        f.inp_arch.setCurrentIndex(f.inp_arch.findData(""))
        all_items = visible()
        assert any("Qwen-Image 2.1" in t for t in all_items) and any("UNET Flux" in t for t in all_items)
        f.inp_model.setCurrentIndex(f.inp_model.findText("flux1_dev_model  · UNET Flux"))
        f.inp_arch.setCurrentIndex(f.inp_arch.findData("Qwen21"))
        vis = visible()
        assert any("Qwen-Image 2.1" in t for t in vis) and not any("UNET Flux" in t for t in vis), vis
        assert mm.is_placeholder(f.inp_model), f"scelta FLUX.1 con filtro Qwen: {f.inp_model.currentText()!r}"
        assert mm.blocked(w, f.inp_model)
        f.inp_arch.setCurrentIndex(f.inp_arch.findData("Flux2"))
        vis = visible()
        assert any("klein_senza_nome_flux" in t for t in vis) and not any("flux1_dev_model" in t or "Qwen" in t for t in vis), vis
        f.inp_arch.setCurrentIndex(f.inp_arch.findData("Flux"))
        vis = visible()
        assert any("flux1_dev_model" in t for t in vis) and not any("klein_senza" in t or "Qwen" in t for t in vis), vis
        f.inp_arch.setCurrentIndex(f.inp_arch.findData(""))
        assert len(visible()) >= len(all_items), "«Tutte le architetture» deve ripristinare il menu"
    finally:
        fx.close()


def case_arch_adult_badge_menus(w, spy):
    """Icona «vietato ai minori di 18» sui modelli NSFW dei menu Crea e Modifica: icona + tooltip testuale, il testo
    della voce (nome file usato dai job) NON cambia; i modelli non 18+ non hanno icona."""
    from PySide6.QtCore import Qt
    fx = Fixture(w)
    try:
        _write_st(fx.root / "diffusion_models" / "Flux" / "xxx_nsfw_flux1.safetensors", FLUX1)
        f = w.frame1_tab
        f.set_model_inventory(list(fx.names) + ["Flux\\xxx_nsfw_flux1.safetensors"], [])
        f.img_nsfw.setCurrentText("NSFW")
        f.gen_arch.setCurrentText("FLUX")

        def row(combo, needle):
            i = next(i for i in range(combo.count()) if needle in combo.itemText(i))
            return i, not combo.itemIcon(i).isNull(), str(combo.itemData(i, Qt.ToolTipRole) or "")
        i, icon, tip = row(f.gen_model, "xxx_nsfw_flux1")
        assert icon and "18" in tip, f"Crea: modello NSFW senza icona/tooltip: {icon} {tip!r}"
        assert f.gen_model.itemText(i).endswith("xxx_nsfw_flux1.safetensors") and "🔞" not in f.gen_model.itemText(i), \
            "il testo della voce (nome file) non deve cambiare"
        assert not row(f.gen_model, "flux1_dev_model")[1], "modello non 18+ con icona"
        i, icon, tip = row(f.inp_model, "xxx_nsfw_flux1")
        assert icon and "18" in tip, f"Inpaint: modello NSFW senza icona/tooltip: {icon} {tip!r}"
        assert not row(f.inp_model, "flux1_dev_model")[1]
        f.gen_arch.setCurrentText("Qwen-Image 2.1")                 # voci di catalogo: sensitivity «adult»
        adult_ids = [f.gen_model.itemData(i) for i in range(f.gen_model.count())
                     if not f.gen_model.itemIcon(i).isNull()]
        assert any(str(c).startswith("qwen21-uc") or "noctq" in str(c) for c in adult_ids), adult_ids
        i = f.gen_model.findData("qwen21-dit-nvfp4")
        assert i >= 0 and f.gen_model.itemIcon(i).isNull(), "Qwen base (non 18+) con icona"
    finally:
        fx.close()


CASES = [
    ("arch_gen_models_by_header", "Genera immagine: FLUX/Z-Image/Ernie mostrano i file della propria famiglia (header), "
     "non quelli con una parola nel nome.", case_arch_gen_models_by_header),
    ("arch_edit_engines_by_header", "Integra/Colore/rifinitura: Kontext e FLUX.2 dal contenuto del file; «PornFlux_20» "
     "non è FLUX.2, un Klein senza «flux2» nel nome sì.", case_arch_edit_engines_by_header),
    ("arch_inpaint_dynamic_by_header", "Inpaint/Outpaint: niente Krea 2 né Kontext fra gli UNET Flux.",
     case_arch_inpaint_dynamic_by_header),
    ("arch_components_by_arch", "Encoder e VAE di Genera immagine filtrati per architettura (e per FLUX.1/FLUX.2 "
     "del file scelto).", case_arch_components_by_arch),
    ("arch_switch_never_picks_other_model", "Cambio architettura: il modello scelto non vale più → segnaposto, "
     "generazione bloccata; Auto resta Auto.", case_arch_switch_never_picks_other_model),
    ("arch_video_ltx_menu_by_header", "Video: il menu LTX mostra solo transformer LTX (header), non Qwen21 "
     "etichettato LTX né i file _aux_; cambio arch → segnaposto.", case_arch_video_ltx_menu_by_header),
    ("arch_inpaint_unet_disappears_placeholder", "Voce dinamica dell'Inpaint sparita dall'inventario: segnaposto e "
     "nessuna richiesta.", case_arch_inpaint_unet_disappears_placeholder),
    ("arch_klein_default_steps", "Klein: default 12 passi in Inpaint/Integra/Colore; il valore scelto a mano resta.",
     case_arch_klein_default_steps),
    ("arch_child_error_reaches_dialog", "L'errore chiaro del backend arriva intero al dialog della GUI.",
     case_arch_child_error_reaches_dialog),
    ("arch_function_menus_overlap", "Un modello che fa piu' funzioni compare in tutti i loro menu (FLUX.1 base, Qwen, "
     "Krea 2); Kontext/Klein/Fill solo Modifica; FLUX.2 Dev solo Crea.", case_arch_function_menus_overlap),
    ("arch_inpaint_arch_selector", "Selettore Architettura in Modifica (Inpaint): filtra il menu, segnaposto se la scelta "
     "non appartiene.", case_arch_inpaint_arch_selector),
    ("arch_adult_badge_menus", "Icona 18+ (con tooltip) sui modelli NSFW di Crea e Modifica; testo invariato.",
     case_arch_adult_badge_menus),
]
