r"""Verifica su GPU dei percorsi veloci di comfy_kitchen con le FORME VERE del DiT H3 (turno di misura, GPU nostra): errore e tempo contro il riferimento per
 · GEMM int8 (RowInt8Linear qkv e MLP, con ConvRot),
 · GEMM FP4 (Nvfp4Linear, solo Blackwell),
 · attenzione int8 contro SDPA (56 teste x 128, sequenza lunga).
NON si lancia con la GPU occupata: serve ~2 GB di VRAM. Stampa una tabella; esce con codice 1 se un percorso attivo supera la soglia.
    E:\AI\Tools\diffusers_next_venv\Scripts\python.exe tests\h3_gpu_fast_paths_check.py"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT))
import _qlinear  # noqa: E402
import h3_int8_attention  # noqa: E402

DEVICE = torch.device("cuda")
TOKENS = 4096
HIDDEN = 5376


def timed(function, repeats: int = 5):
    function()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        result = function()
    torch.cuda.synchronize()
    return result, (time.time() - start) / repeats * 1000.0


def relative_error(fast: torch.Tensor, reference: torch.Tensor) -> float:
    return float((fast.float() - reference.float()).norm() / reference.float().norm())


def check_int8_linear(name: str, out_features: int, in_features: int) -> float:
    layer = _qlinear.RowInt8Linear(in_features, out_features, False, 256).to(DEVICE)
    layer.weight = torch.randint(-127, 128, (out_features, in_features), dtype=torch.int8, device=DEVICE)
    layer.scale = (torch.rand(out_features, 1, device=DEVICE) * 0.01 + 0.002)
    x = torch.randn(TOKENS, in_features, dtype=torch.bfloat16, device=DEVICE)
    reference, reference_ms = timed(lambda: layer._forward_dequant(x, torch.bfloat16))
    fast, fast_ms = timed(lambda: layer._forward_gemm(x, torch.bfloat16))
    error = relative_error(fast, reference)
    print(f"int8 {name:12s} {out_features}x{in_features}: errore {error:.4f}, riferimento {reference_ms:.1f} ms, GEMM {fast_ms:.1f} ms ({reference_ms / fast_ms:.2f}x)")
    return error


def check_attention() -> float:
    q, k, v = (torch.randn(1, 8192, 56, 128, dtype=torch.bfloat16, device=DEVICE) for _ in range(3))
    reference, reference_ms = timed(lambda: h3_int8_attention.reference_attention(q, k, v), 3)
    h3_int8_attention.INT8_ATTENTION.reason = ""
    fast, fast_ms = timed(lambda: h3_int8_attention.int8_attention_backend(q, k, v), 3)
    error = relative_error(fast, reference)
    state = "attiva" if h3_int8_attention.INT8_ATTENTION.verified else "scartata: " + str(h3_int8_attention.INT8_ATTENTION.reason)
    print(f"attenzione int8 8192x56x128 ({state}): errore {error:.4f}, SDPA {reference_ms:.1f} ms, int8 {fast_ms:.1f} ms ({reference_ms / fast_ms:.2f}x)")
    return error if h3_int8_attention.INT8_ATTENTION.verified else 0.0


def check_fp4() -> None:
    capability = torch.cuda.get_device_capability(DEVICE)
    if capability < _qlinear.NVFP4_GEMM.min_capability:
        print(f"FP4: GPU SM {capability[0]}.{capability[1]} sotto il minimo, non provato")
        return
    import comfy_kitchen as ck
    layer = _qlinear.Nvfp4Linear(HIDDEN, 21504, False).to(DEVICE)
    weights = torch.randn(21504, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    scale_2 = weights.abs().amax().float() / (448.0 * 6.0)
    layer.weight, layer.weight_scale = ck.quantize_nvfp4(weights, scale_2)
    layer.weight_scale_2 = scale_2
    x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    reference, reference_ms = timed(lambda: layer._forward_dequant(x))
    _qlinear.NVFP4_GEMM.reason, _qlinear.NVFP4_GEMM.verified = "", False
    fast, fast_ms = timed(lambda: layer._forward_gemm(x))
    if fast is None:
        print(f"FP4: percorso scartato ({_qlinear.NVFP4_GEMM.reason})")
        return
    print(f"FP4 qkv {21504}x{HIDDEN}: errore {relative_error(fast, reference):.4f}, riferimento {reference_ms:.1f} ms, GEMM {fast_ms:.1f} ms ({reference_ms / fast_ms:.2f}x)")


def main() -> int:
    torch.manual_seed(5)
    errors = [check_int8_linear("qkv", 21504, HIDDEN), check_int8_linear("mlp gate_up", 28672, HIDDEN), check_int8_linear("mlp out", HIDDEN, 14336), check_attention()]
    check_fp4()
    worst = max(errors)
    print(f"peggior errore relativo: {worst:.4f}")
    return 0 if worst <= 0.1 else 1


if __name__ == "__main__":
    sys.exit(main())
