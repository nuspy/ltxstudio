"""Decode del VAE H3 con pesi fp16 (come l'autocast del pipeline e come ComfyUI); fp32 con H3_VAE_DECODE_DTYPE=fp32."""
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT))
worker = pytest.importorskip("minimax_h3_infer")


def test_decode_dtype_default_fp16_and_env_override(monkeypatch):
    monkeypatch.delenv("H3_VAE_DECODE_DTYPE", raising=False)
    assert worker.decode_dtype() is torch.float16
    monkeypatch.setenv("H3_VAE_DECODE_DTYPE", "fp32")
    assert worker.decode_dtype() is torch.float32


def test_decode_loads_the_vae_with_the_decode_dtype_and_frees_it(monkeypatch):
    seen = {}

    def fake_load_vae(settings, encoder_only=False, device="cuda", dtype=None):
        seen["dtype"], seen["encoder_only"] = dtype, encoder_only
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(worker, "load_vae", fake_load_vae)
    monkeypatch.setattr(worker, "load_audio_vae", lambda settings: torch.nn.Linear(2, 2))
    monkeypatch.setattr(worker, "_gc", lambda: None)
    monkeypatch.delenv("H3_VAE_DECODE_DTYPE", raising=False)
    result = worker._decode_and_free(None, [], {}, {}, stage_fn=lambda blocks, group, comps: (lambda state, output: {"videos": "ok"}))
    assert result == {"videos": "ok"} and seen == {"dtype": torch.float16, "encoder_only": False}              # il decode usa il VAE intero, ma fp16


def test_the_pipeline_decodes_under_fp16_autocast_so_fp16_weights_change_nothing():
    source = Path(r"E:\AI\Tools\diffusers_next_venv\Lib\site-packages\diffusers\modular_pipelines\minimax_h3\decoders.py")
    if not source.is_file():
        pytest.skip("serve il venv di H3")
    assert "torch.autocast(device_type=device.type, dtype=torch.float16" in source.read_text(encoding="utf-8")
