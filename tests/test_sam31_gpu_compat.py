"""SAM 3.1 su GPU (turno 36 del 04/11): il torch Windows del venv non ha flash attention e il decoder di Meta chiede la SOLA FLASH_ATTENTION («No available kernel»);
l'immagine su cuda senza autocast bf16 dava «mat1 and mat2 must have the same dtype, but got BFloat16 and Float». I due job (immagine e video) correggono entrambe le cose."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

JOBS = Path(__file__).resolve().parents[1] / "workers" / "sam3_jobs"
torch = pytest.importorskip("torch")


def _load(name):
    spec = importlib.util.spec_from_file_location("job_" + name, JOBS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("job", ["sam31_image_text", "sam31_video_text"])
def test_decoder_sdpa_backends_are_widened_beyond_flash_only(job, monkeypatch):
    module = _load(job)
    seen = []
    import torch.nn.attention as attention
    monkeypatch.setattr(attention, "sdpa_kernel", lambda backends, *a, **k: seen.append(list(backends)) or "contesto")
    decoder = types.ModuleType("sam3.model.decoder")
    decoder.sdpa_kernel = None
    monkeypatch.setitem(sys.modules, "sam3", types.ModuleType("sam3"))
    monkeypatch.setitem(sys.modules, "sam3.model", types.ModuleType("sam3.model"))
    monkeypatch.setitem(sys.modules, "sam3.model.decoder", decoder)
    sys.modules["sam3"].model = sys.modules["sam3.model"]
    sys.modules["sam3.model"].decoder = decoder
    module.patch_sdpa_backends()
    flash_only = [attention.SDPBackend.FLASH_ATTENTION]
    assert decoder.sdpa_kernel(flash_only) == "contesto"
    assert seen and attention.SDPBackend.FLASH_ATTENTION in seen[0] and attention.SDPBackend.EFFICIENT_ATTENTION in seen[0] and attention.SDPBackend.MATH in seen[0]


def test_image_job_runs_under_bf16_autocast_on_cuda_only():
    module = _load("sam31_image_text")
    import contextlib
    assert isinstance(module.gpu_autocast("cpu"), contextlib.nullcontext)
    cuda_context = module.gpu_autocast("cuda")
    assert isinstance(cuda_context, torch.autocast) and cuda_context.fast_dtype == torch.bfloat16
