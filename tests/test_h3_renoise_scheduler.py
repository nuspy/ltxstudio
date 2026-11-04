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
