"""Verifica su CPU (venv diffusers con comfy_kitchen) del GEMM int8 vero di RowInt8Linear contro il riferimento dequantizzato: differenza massima e relativa, con e senza
ConvRot, 2D e 3D, con bias; il caso con zero point resta sul riferimento.
Si lancia:  E:\\AI\\Tools\\diffusers_next_venv\\Scripts\\python.exe tests\\qlinear_fast_gemm_check.py   (stampa «GEMM INT8 OK»)."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT))
import _qlinear  # noqa: E402


def make_layer(rows_out: int, columns_in: int, convrot: int, bias: bool) -> "_qlinear.RowInt8Linear":
    layer = _qlinear.RowInt8Linear(columns_in, rows_out, bias, convrot)
    layer.weight = torch.randint(-127, 128, (rows_out, columns_in), dtype=torch.int8)
    layer.scale = torch.rand(rows_out, 1) * 0.01 + 0.002
    if bias:
        layer.bias = torch.nn.Parameter(torch.randn(rows_out, dtype=torch.bfloat16))
    return layer


def compare(layer, x: torch.Tensor):
    _qlinear.INT8_GEMM.verified = False
    _qlinear.INT8_GEMM.reason = ""
    reference = layer._forward_dequant(x, torch.bfloat16).float()
    fast = layer._forward_gemm(x, torch.bfloat16)
    assert fast is not None, _qlinear.INT8_GEMM.reason
    fast = fast.float()
    relative = float((fast - reference).norm() / reference.norm())
    return relative, float((fast - reference).abs().max()), float(reference.abs().max())


def main() -> int:
    torch.manual_seed(3)
    worst = 0.0
    for convrot, bias, shape in ((256, False, (64, 512)), (256, True, (48, 768)), (0, False, (32, 512)), (256, False, (2, 17, 512))):
        layer = make_layer(256, 512 if shape[-1] == 512 else 768, convrot, bias)
        x = torch.randn(*shape, dtype=torch.bfloat16)
        relative, max_abs, scale = compare(layer, x)
        print(f"convrot {convrot} bias {bias} forma {shape}: errore relativo {relative:.4f}, differenza massima {max_abs:.3f} su valori fino a {scale:.1f}")
        assert relative < 0.03, relative
        worst = max(worst, relative)
    layer = make_layer(128, 512, 256, False)
    layer.zero_point = torch.ones(128, 1)
    assert not layer._has_no_zero_point()                                    # con zero point il GEMM di comfy_kitchen non vale: riferimento
    print(f"peggior errore relativo {worst:.4f}")
    print("GEMM INT8 OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
