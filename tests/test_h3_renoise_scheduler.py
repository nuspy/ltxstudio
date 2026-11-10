"""Scheduler «renoise» di Pika H3 Flex (6 passi): formule del riferimento a secco, composizione con l'hold degli edit, scelta della classe nel worker."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT))
import h3_renoise_scheduler as renoise  # noqa: E402

VENV_PY = r"E:\AI\Tools\diffusers_next_venv\Scripts\python.exe"
KNOTS = (1000, 750, 500, 250, 100, 40, 0)


def test_shifted_sigmas_match_the_reference_numbers():
    video = renoise.shifted_sigmas(KNOTS, 12.0)
    audio = renoise.shifted_sigmas(KNOTS, 3.0)
    assert [round(v, 5) for v in video] == [1.0, 0.97297, 0.92308, 0.8, 0.57143, 0.33333, 0.0]       # sigma' = 12s/(1+11s), s = nodo/1000
    assert [round(v, 5) for v in audio] == [1.0, 0.9, 0.75, 0.5, 0.25, 0.11111, 0.0]                # shift audio 3


def test_renoise_update_is_x0_plus_sigma_v_then_fresh_noise_and_the_last_step_returns_x0():
    sample, velocity, noise = torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)
    sigma, sigma_next = torch.tensor(0.8), torch.tensor(0.5714286)
    out = renoise.renoise_update(sample, velocity, sigma, sigma_next, noise)
    x0 = sample + sigma * velocity
    assert torch.allclose(out, (1 - sigma_next) * x0 + sigma_next * noise, atol=1e-6)
    last = renoise.renoise_update(sample, velocity, torch.tensor(0.3333333), torch.tensor(0.0), noise)
    assert torch.allclose(last, sample + torch.tensor(0.3333333) * velocity, atol=1e-6)               # sigma_next = 0: niente rumore
    half = renoise.renoise_update(sample.bfloat16(), velocity.bfloat16(), sigma, sigma_next, noise)
    assert half.dtype == torch.bfloat16                                                              # il tipo del campione si conserva, il calcolo e in float32


def test_worker_selects_the_scheduler_class_and_default_table_is_empty():
    from core import minimax_h3_spec as spec
    assert spec.TURBO_SAMPLER == {} or all(v["method"] == "renoise" for v in spec.TURBO_SAMPLER.values())
    source = (ROOT / "workers" / "minimax_h3_infer.py").read_text(encoding="utf-8")
    assert "video_class, audio_class = scheduler_classes(turbo, seed)" in source and "make_hold_scheduler_class(video_class)" in source


@pytest.mark.skipif(not os.path.isfile(VENV_PY), reason="serve il venv di H3")
def test_scheduler_against_the_reference_formulas_with_the_real_base_class():
    result = subprocess.run([VENV_PY, str(ROOT / "tests" / "h3_renoise_scheduler_check.py")], capture_output=True, text=True, timeout=600)
    assert "RENOISE OK" in result.stdout, result.stdout[-800:] + result.stderr[-1000:]


def test_keyframe_conditions_are_rounded_to_bf16_only_for_fl2va_with_a_turbo_that_declares_it(monkeypatch, capsys):
    worker = pytest.importorskip("minimax_h3_infer")
    from core import minimax_h3_spec as spec
    monkeypatch.setitem(spec.TURBO_SAMPLER, "pika_probe", {"method": "renoise", "knots": KNOTS, "keyframe_bf16": True})

    class State:
        def __init__(self, latents):
            self.values = {"condition_latents": latents}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

    latent = torch.randn(1, 24, 1, 4, 4)
    state = State([latent.clone()])
    worker.round_keyframe_conditions(state, "pika_probe", "fl2va")
    rounded = state.values["condition_latents"][0]
    assert rounded.dtype == torch.float32 and torch.equal(rounded, latent.to(torch.bfloat16).to(torch.float32)) and not torch.equal(rounded, latent)
    assert "bf16_conditioning_v1" in capsys.readouterr().out
    for turbo, mode in (("pika_probe", "ref2va"), ("v4", "fl2va"), ("", "fl2va")):                  # Ref2VA, turbo normali: niente
        untouched = State([latent.clone()])
        worker.round_keyframe_conditions(untouched, turbo, mode)
        assert torch.equal(untouched.values["condition_latents"][0], latent)
    worker.round_keyframe_conditions(State([]), "pika_probe", "fl2va")                              # senza keyframe: nessun errore
    source = (ROOT / "workers" / "minimax_h3_infer.py").read_text(encoding="utf-8")
    assert "round_keyframe_conditions(state, turbo, mode)" in source


def test_pika6_is_registered_in_spec_catalog_and_menus():
    import json
    from core import minimax_h3_spec as spec
    assert spec.TURBO["pika6"] == ({"fl2va": "mmh3-turbo-pika6-fl2v", "ref2va": "mmh3-turbo-pika6-ref2v"}, 6, 12.0)
    assert spec.TURBO_SAMPLER["pika6"]["knots"] == KNOTS and spec.TURBO_SAMPLER["pika6"]["keyframe_bf16"] is True
    ids = {task: spec.weight_ids({"h3_profile": "int8", "h3_turbo": "pika6", "references": [{"kind": "image"}] if task == "ref2va" else []})["turbo"] for task in ("ref2va", "fl2va")}
    assert ids == {"ref2va": "mmh3-turbo-pika6-ref2v", "fl2va": "mmh3-turbo-pika6-fl2v"}
    catalog = {e["id"]: e for e in json.loads((ROOT / "data" / "weights_catalog.json").read_text(encoding="utf-8"))["weights"]}
    for entry_id in ids.values():
        entry = catalog[entry_id]
        assert entry["convert"] == "mmh3_lora_diffusers" and entry["keep_source"] and len(entry["sha256"]) == 64 and entry["filename"].endswith("-native.safetensors")
    assert catalog["mmh3-turbo-pika6-ref2v"]["remote_filename"] != catalog["mmh3-turbo-pika6-fl2v"]["remote_filename"]        # nomi remoti diversi: non si sovrappongono nello stesso cartella
    for language in ("it", "en", "ar", "zh"):
        table = json.loads((ROOT / "ui" / "i18n" / f"{language}.json").read_text(encoding="utf-8"))
        assert table["vedit.h3.turbo.pika6"] and table["vid.h3.turbo.pika6"]


def test_forced_steps_for_fused_turbo_and_runkey(monkeypatch, capsys):
    worker = pytest.importorskip("minimax_h3_infer")
    fused = {"nfe": 4, "video_shift": 12.0}
    monkeypatch.delenv("H3_NFE", raising=False)
    assert worker.fused_turbo_steps(fused) == 4
    monkeypatch.setenv("H3_NFE", "8")
    assert worker.fused_turbo_steps(fused) == 8 and "H3_NFE=8" in capsys.readouterr().out
    edit = (ROOT / "workers" / "h3_edit_run.py").read_text(encoding="utf-8")
    assert '(("numerics", numerics_options()),)' in edit                               # i clip a 4 e a 8 passi non si scambiano nei checkpoint
