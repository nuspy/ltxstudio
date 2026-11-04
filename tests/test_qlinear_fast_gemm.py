"""GEMM veloci di comfy_kitchen nei Linear quantizzati (04/11): int8 vero acceso di default, FP4 spento; ogni scarto e scritto nel log, mai ingoiato."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT))
import _qlinear as q  # noqa: E402

VENV_PY = r"E:\AI\Tools\diffusers_next_venv\Scripts\python.exe"


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    q._LOGGED_ONCE.clear()
    for gemm in (q.INT8_GEMM, q.NVFP4_GEMM):
        monkeypatch.setattr(gemm, "reason", None)
        monkeypatch.setattr(gemm, "verified", False)
    for name in ("H3_INT8_GEMM", "H3_NVFP4_GEMM"):
        monkeypatch.delenv(name, raising=False)


def test_default_switches_and_reasons_are_logged(capsys, monkeypatch):
    cpu = torch.device("cpu")
    assert q.INT8_GEMM.usable(cpu) is False and "serve CUDA" in q.INT8_GEMM.reason or "comfy_kitchen non importabile" in q.INT8_GEMM.reason
    assert "GEMM int8 non usato" in capsys.readouterr().out                             # il motivo e nel log
    assert q.NVFP4_GEMM.usable(cpu) is False and "spento di default" in q.NVFP4_GEMM.reason and "H3_NVFP4_GEMM=1" in q.NVFP4_GEMM.reason
    monkeypatch.setattr(q.INT8_GEMM, "reason", None)
    monkeypatch.setenv("H3_INT8_GEMM", "0")
    assert q.INT8_GEMM.usable(cpu) is False and q.INT8_GEMM.reason == "H3_INT8_GEMM=0"


def test_verify_accepts_close_results_and_rejects_far_ones_with_a_log_line(capsys):
    reference = torch.randn(8, 16)
    assert q.INT8_GEMM.verify(reference * 1.01, reference, "prova") is True and q.INT8_GEMM.verified
    assert "GEMM int8 attivo" in capsys.readouterr().out
    q.INT8_GEMM.verified = False
    assert q.INT8_GEMM.verify(torch.randn(8, 16), reference, "prova") is False
    out = capsys.readouterr().out
    assert "GEMM int8 non usato" in out and "errore relativo" in out and q.INT8_GEMM.reason.startswith("errore relativo")
    assert q.INT8_GEMM.verify(torch.full((8, 16), float("nan")), reference, "prova") is False          # NaN: scartato


def test_kernel_failure_falls_back_to_the_reference_and_is_logged_once(capsys, monkeypatch):
    class Broken:
        @staticmethod
        def int8_linear(*args, **kwargs):
            raise RuntimeError("kernel assente per questa GPU")
    monkeypatch.setitem(sys.modules, "comfy_kitchen", Broken)
    layer = q.RowInt8Linear(32, 16, False, 0)
    layer.weight = torch.randint(-10, 10, (16, 32), dtype=torch.int8)
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    assert layer._forward_gemm(x, torch.bfloat16) is None
    assert layer._forward_gemm(x, torch.bfloat16) is None
    out = capsys.readouterr().out
    assert out.count("GEMM int8 non usato") == 1 and "kernel assente per questa GPU" in out and "RowInt8Linear._forward_gemm" in out    # una riga sola, con funzione e messaggio


def test_forward_uses_the_reference_when_the_fast_path_is_unusable():
    layer = q.RowInt8Linear(32, 16, False, 0)
    layer.weight = torch.randint(-10, 10, (16, 32), dtype=torch.int8)
    layer.scale = torch.full((16, 1), 0.05)
    x = torch.randn(4, 32, dtype=torch.bfloat16)
    out = layer(x)                                              # su CPU il percorso veloce non e usabile: nessuna eccezione, risultato del riferimento
    assert torch.equal(out, layer._forward_dequant(x, torch.bfloat16).to(x.dtype))


def test_w4a8_fallback_is_logged_not_silent():
    source = (ROOT / "workers" / "_qlinear.py").read_text(encoding="utf-8")
    assert "log_once(f\"GEMM W4A8 non usato" in source and "except Exception:  # noqa: BLE001  (nessun backend capace" not in source


@pytest.mark.skipif(not os.path.isfile(VENV_PY), reason="serve il venv di H3 (comfy_kitchen)")
def test_int8_gemm_matches_the_dequantized_reference_on_cpu():
    result = subprocess.run([VENV_PY, str(ROOT / "tests" / "qlinear_fast_gemm_check.py")], capture_output=True, text=True, timeout=600)
    assert "GEMM INT8 OK" in result.stdout, result.stdout[-800:] + result.stderr[-800:]
