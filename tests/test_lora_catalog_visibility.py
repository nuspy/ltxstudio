"""Visibilità delle LoRA tenute nei pannelli (core/lora_catalog) — nessuna GPU, nessun Qt.

Controlla sui file REALI di E:\\AI\\Models (si salta se non ci sono): le LoRA Flux/Wan stanno nel pannello giusto,
niente voci fantasma, FLUX.1-Turbo-Alpha compare tra le LoRA FLUX, le LoRA 18+ sono marcate NSFW.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import lora_catalog as lc                                    # noqa: E402

ROOT = Path(r"E:\AI\Models")
pytestmark = pytest.mark.skipif(not (ROOT / "LoRA").is_dir() or not lc.CATALOG_PATH.exists(),
                                reason="pesi/dizionario non presenti")


def _names(arch: str) -> dict[str, lc.LoraEntry]:
    return {e.filename.lower(): e for e in lc.LoraCatalog(compatible_arch=arch).loras()}


def test_no_ghost_entries():
    for arch in ("LTX", "Flux", "Wan21", "Wan22", "Krea2", "Qwen21", "MiniMax-H3"):
        for e in lc.LoraCatalog(compatible_arch=arch).loras():
            assert os.path.isfile(e.path), f"{arch}: voce fantasma {e.path}"


def test_flux_folder_fully_listed():
    shown = {os.path.normcase(e.path) for e in lc.LoraCatalog(compatible_arch="Flux").loras()}
    for f in (ROOT / "LoRA" / "Flux").glob("*.safetensors"):
        assert os.path.normcase(str(f)) in shown, f"LoRA FLUX non mostrata: {f.name}"


def test_flux_turbo_alpha_is_a_flux_lora():
    f = ROOT / "Checkpoints" / "DreamO" / "diffusion_pytorch_model.safetensors"
    g = ROOT / "LoRA" / "Flux" / "FLUX.1-Turbo-Alpha.safetensors"
    if not (f.is_file() or g.is_file()):
        pytest.skip("FLUX.1-Turbo-Alpha assente")
    hit = [e for e in lc.LoraCatalog(compatible_arch="Flux").loras() if "Turbo-Alpha" in e.name]
    assert len(hit) == 1
    assert os.path.isfile(hit[0].comfy_name) or os.path.isfile(ROOT / "LoRA" / hit[0].comfy_name)
    assert not any("Turbo-Alpha" in e.name for e in lc.LoraCatalog(compatible_arch="Wan21").loras())


def test_wan_split_21_vs_22():
    w21, w22 = _names("Wan21"), _names("Wan22")
    for n in ("wan_cumshot_i2v.safetensors", "phantom_wan_14b_fusionx_lora.safetensors"):
        if (ROOT / "LoRA" / "Wan" / n).exists() or n in w21:
            assert n in w21 and n not in w22
    seko = [n for n in w22 if "seko" in n]
    for n in seko:
        assert n not in w21
    for d in (ROOT / "LoRA" / "Wan2.2").glob("*.safetensors"):
        assert d.name.lower() in w22 and d.name.lower() not in w21


def test_wan_moved_from_root_are_visible_and_adult():
    w21 = _names("Wan21")
    for n in ("deepthroat_epoch_80_(loras).safetensors", "fingering_i2v_e248_(loras).safetensors",
              "grinding_pussyjob_e130_(loras).safetensors", "t2v-pov_bj-14b_(loras).safetensors"):
        if (ROOT / "LoRA" / "Wan" / n).exists():
            assert n in w21, n
            assert w21[n].nsfw, n


def test_adult_override_marks_nsfw():
    flux = _names("Flux")
    for n in ("pov_fist.safetensors", "tan-lines-01.safetensors", "vmorflux.safetensors"):
        if n in flux:
            assert flux[n].nsfw, n
    sfw_visible = lc.LoraCatalog(compatible_arch="Flux").loras(allow_nsfw=False)
    assert all(not e.nsfw for e in sfw_visible)
