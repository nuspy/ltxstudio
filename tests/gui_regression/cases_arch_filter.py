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
OPAQUE = {"blocks.0.attn.wq.weight": (64, 64)}                 # convertito «-native»: l'header non dice nulla

# nome relativo → (tensori, metadata)
DIFFUSION = {
    "Flux\\flux1_dev_model.safetensors": (FLUX1, None),
    "Flux\\PornFlux_20_trap.safetensors": (FLUX1, None),            # FLUX.1 con «flux_2» nel nome
    "Flux\\klein_senza_nome_flux.safetensors": (FLUX2_KLEIN, None),  # FLUX.2 Klein: header, non nome
    "Flux\\flux2_dev_named.safetensors": (FLUX2_DEV, None),
    "Flux\\Flux_Kontext_X.safetensors": (FLUX1, None),
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
        want = sorted(["flux1_dev_model.safetensors", "PornFlux_20_trap.safetensors", "flux2_dev_named.safetensors"])
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
        want = sorted(["klein_senza_nome_flux.safetensors", "flux2_dev_named.safetensors"])
        assert got == want, f"FLUX.2 (header, non nome): {got} != {want}"
        assert "PornFlux_20_trap.safetensors" not in got, "un FLUX.1 con «Flux_20» nel nome non è FLUX.2"
        refine = _base(_texts(f.up_refine_model, skip_auto=False))
        assert refine == want, f"rifinitura upscale (FLUX.2): {refine}"
        from core.gen_pipeline import is_flux2
        assert not is_flux2("PornFlux_20_trap.safetensors") and is_flux2("FLUX_2_Dev.safetensors"), "is_flux2 del dispatch"
    finally:
        fx.close()


def case_arch_inpaint_dynamic_by_header(w, spy):
    fx = Fixture(w)
    try:
        f = w.frame1_tab
        labels = [f.inp_model.itemText(i) for i in range(f.inp_model.count())]
        dyn = [l for l in labels if "UNET Flux" in l]
        joined = " | ".join(dyn)
        assert "KreaX_native" not in joined, f"Krea 2 fra gli UNET Flux dell'Inpaint: {dyn}"
        assert "Flux_Kontext_X" not in joined, f"Kontext fra gli UNET Flux dell'Inpaint: {dyn}"
        for ok in ("flux1_dev_model", "PornFlux_20_trap", "flux2_dev_named", "klein_senza_nome_flux"):
            assert ok in joined, f"{ok} (header FLUX) manca dall'Inpaint: {dyn}"
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
]
