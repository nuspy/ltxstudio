"""Casi GUI: i checkpoint 18+ dei menu modello FLUX.1 / Z-Image compaiono SOLO in modalità NSFW.

Il selettore SFW/NSFW del tab immagine è `frame1_tab.img_nsfw`; il menu modello di generazione è `gen_model`
(popolato da `_reload_gen`, che filtra con `_mcat.is_nsfw`). Nessuna generazione/GPU: solo widget.
"""
from __future__ import annotations

import harness as H

# (arch GUI, sottostringhe del checkpoint 18+ nel nome file)
_FLUX = ("FLUX", ["fuxcapacity", "getphat"])
_ZIMG = ("Z-Image", ["biglovez", "zimageturbonsfw"])


def _items(f1):
    return [f1.gen_model.itemText(i) for i in range(f1.gen_model.count())]


def _check(w, arch, keys):
    f1 = w.frame1_tab
    if not f1._models:        # l'inventario arriva da un timer di stato: lo carico dalla stessa sorgente dell'app (filesystem)
        from core import offline_inventory as oi
        f1.set_model_inventory(oi.models_in("diffusion_models"), oi.models_in("checkpoints"))
    f1.gen_arch.setCurrentText(arch)
    assert f1.gen_arch.currentText() == arch, f"arch {arch} non selezionabile"
    f1.img_nsfw.setCurrentText("NSFW")
    nsfw_items = _items(f1)
    f1.img_nsfw.setCurrentText("SFW")
    sfw_items = _items(f1)
    f1.img_nsfw.setCurrentText("NSFW")
    for k in keys:
        in_nsfw = [n for n in nsfw_items if k in n.lower()]
        in_sfw = [n for n in sfw_items if k in n.lower()]
        assert in_nsfw, f"[{arch}] '{k}' assente dal menu in NSFW: {nsfw_items}"
        assert not in_sfw, f"[{arch}] '{k}' visibile in SFW: {in_sfw}"


def case_flux_18plus_only_nsfw(w, spy):
    _check(w, *_FLUX)


def case_zimage_18plus_only_nsfw(w, spy):
    _check(w, *_ZIMG)


CASES = [
    ("image_menu_flux_18plus_only_nsfw",
     "Menu modello FLUX.1: fuxCapacity e getphat compaiono solo in NSFW, non in SFW.", case_flux_18plus_only_nsfw),
    ("image_menu_zimage_18plus_only_nsfw",
     "Menu modello Z-Image: Big Love Z ZT1 e Z-Image NSFW 2602 compaiono solo in NSFW, non in SFW.",
     case_zimage_18plus_only_nsfw),
]
