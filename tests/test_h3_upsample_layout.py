"""Verifiche del nucleo H3 Upsampler che richiedono diffusers (layout contro il pipeline ufficiale, VAE a finestre
contro vae.encode/decode, ri-disegno a tile senza cuciture, DiT minuscolo): girano nel venv diffusers_next con
tests/h3_upsample_diffusers_check.py (CPU, pesi casuali minuscoli). Qui le lancia pytest, se il venv esiste."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = Path(r"E:\AI\Tools\diffusers_next_venv\Scripts\python.exe")


@pytest.mark.skipif(not VENV_PY.is_file(), reason="venv diffusers_next assente")
def test_nucleo_h3_upsampler_nel_venv_diffusers():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONIOENCODING="utf-8")
    for k in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(k, None)
    r = subprocess.run([str(VENV_PY), str(ROOT / "tests" / "h3_upsample_diffusers_check.py")], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env, timeout=1200)
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == 0, out[-4000:]
    assert "verifiche passate" in out and " FAIL " not in out
