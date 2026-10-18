"""Krea 2 Edit — canale semantica (grounded encode): niente rete, niente GPU (task 16b).

Il 18/10 l'edit Krea 2 ha prodotto una scena NUOVA (due donne in studio) invece di modificare la foto: il processor
Qwen3-VL veniva cercato sull'hub (HF_HUB_OFFLINE=1 nel worker -> fallisce) e l'encode degradava in silenzio a SOLO TESTO.
Questi test girano col python del venv Krea (lingbot_venv311: diffusers con Krea2 + transformers 5) e si saltano altrove:
    E:\\AI\\Tools\\lingbot_venv311\\Scripts\\python.exe tests\\test_krea2_edit_grounding.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SNAP = Path(r"E:\AI\Models\diffusers\Krea2\Krea-2-Turbo")


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        print("SKIP:", msg)
        return True


def _pipe():
    try:
        from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline  # noqa: F401
        from transformers import AutoTokenizer
    except Exception as e:                                     # noqa: BLE001
        return None, f"diffusers Krea2/transformers non disponibili: {e}"
    if not (SNAP / "tokenizer").is_dir():
        return None, "snapshot Krea 2 assente"
    from workers.krea2_edit_pipeline import Krea2EditPipeline
    p = Krea2EditPipeline.__new__(Krea2EditPipeline)           # senza pesi: bastano tokenizer e processor
    p.tokenizer = AutoTokenizer.from_pretrained(str(SNAP / "tokenizer"))
    return p, ""


def test_processor_builds_offline_and_matches_reference():
    p, why = _pipe()
    if p is None:
        return _skip(why)
    import numpy as np
    from PIL import Image
    proc = p._get_processor()
    assert proc is not None, getattr(p, "_edit_processor_error", "?")
    im = Image.fromarray((np.random.rand(768, 768, 3) * 255).astype("uint8"))
    txt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>hello<|im_end|>\n"
    out = proc(text=[txt], images=[im], padding=True, return_tensors="pt")
    n_pad = int((out["input_ids"] == proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum())
    t, h, w = [int(x) for x in out["image_grid_thw"][0]]
    assert n_pad == t * h * w // 4, (n_pad, (t, h, w))         # merge 2x2: un token ogni 4 patch
    assert "mm_token_type_ids" in out                          # richiesto da transformers >= 5 per l'mRoPE
    ref = Path(r"E:\AI\Models\configs\Qwen-Image-2.1\processor")
    if (ref / "preprocessor_config.json").is_file():           # stesso risultato del processor ufficiale
        from transformers import AutoProcessor
        r = AutoProcessor.from_pretrained(str(ref))(text=[txt], images=[im], padding=True, return_tensors="pt")
        assert (r["input_ids"] == out["input_ids"]).all() and (r["image_grid_thw"] == out["image_grid_thw"]).all()
        assert abs(r["pixel_values"].float() - out["pixel_values"].float()).max() < 1e-4


def test_ground_drop_is_the_template_prefix_not_34():
    p, why = _pipe()
    if p is None:
        return _skip(why)
    from workers.krea2_edit_pipeline import _GROUND_SYS
    drop = p._ground_drop()
    assert drop == len(p.tokenizer(_GROUND_SYS).input_ids) == 64, drop     # 34 e' il prefisso del t2i, non di questo template
    ids = p.tokenizer(_GROUND_SYS + "<|vision_start|><|image_pad|><|vision_end|>x").input_ids
    assert p.tokenizer.convert_ids_to_tokens(ids[drop]) == "<|vision_start|>"   # dopo lo scarto resta solo blocco visione + istruzione


def test_no_silent_text_only_fallback():
    p, why = _pipe()
    if p is None:
        return _skip(why)
    import torch
    from PIL import Image
    p._edit_processor_tried, p._edit_processor = True, None
    p._edit_processor_error = "test"
    os.environ.pop("KREA2_EDIT_ALLOW_TEXT_ONLY", None)
    try:
        p.grounded_text_hidden_states("x", [Image.new("RGB", (64, 64))], 768, device=torch.device("cpu"))
    except RuntimeError as e:
        assert "processor Qwen3-VL" in str(e)
    else:
        raise AssertionError("l'edit deve fermarsi, non ripiegare a solo-testo")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("OK", name)
