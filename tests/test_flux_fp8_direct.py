"""FLUX.1 fp8 caricato DIRETTO (meta+assign) = stessi pesi del vecchio percorso (from_single_file bf16 poi layerwise fp8), su un mini-checkpoint ricavato
da getphat (2 double + 2 single block + embedder): ogni parametro identico per dtype, forma e byte."""
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
import torch

GETPHAT = Path(r"E:\AI\Models\diffusion_models\Flux\getphat Flux Reality v30 Hardcore FP8.safetensors")
CFG = Path(r"E:\AI\Models\diffusers\_te\flux_transformer")


def _fp(model):
    out = {}
    for n, p in list(model.named_parameters()) + list(model.named_buffers()):
        t = p.detach().contiguous()
        out[n] = (str(t.dtype), tuple(t.shape), hashlib.sha1(t.view(torch.uint8).numpy().tobytes()).hexdigest())
    return out


@pytest.mark.skipif(not GETPHAT.exists() or not CFG.exists(), reason="servono getphat e la config FLUX locale")
def test_fp8_direct_equals_old_path(tmp_path, monkeypatch, capsys):
    from safetensors import safe_open
    from safetensors.torch import save_file
    from inference_server.backends import image_flux as F
    keep = {}
    with safe_open(str(GETPHAT), framework="pt") as h:
        for k in h.keys():
            if k.startswith(("double_blocks.", "single_blocks.")):
                if int(k.split(".")[1]) > 1:
                    continue
            keep[k] = h.get_tensor(k)
    mini = tmp_path / "mini_fp8.safetensors"
    save_file(keep, str(mini))
    te = tmp_path / "_te"
    cfg_dir = te / "flux_transformer"
    shutil.copytree(CFG, cfg_dir)
    c = json.load(open(cfg_dir / "config.json"))
    c["num_layers"], c["num_single_layers"] = 2, 2
    json.dump(c, open(cfg_dir / "config.json", "w"))
    monkeypatch.setattr(F, "TE_DIR", str(te))
    monkeypatch.setenv("NATIVE_FLUX1_FP8", "1")
    monkeypatch.setenv("NATIVE_FLUX_FP8_DIRECT", "0")
    old, _ = F._load_flux_tr(str(mini), "flux")
    fo = _fp(old)
    del old
    capsys.readouterr()
    monkeypatch.setenv("NATIVE_FLUX_FP8_DIRECT", "1")
    new, _ = F._load_flux_tr(str(mini), "flux")
    out = capsys.readouterr().out
    assert "fp8 DIRETTO" in out and "non riuscito" not in out, out[-500:]            # il nuovo percorso e stato davvero usato (niente ripiego silenzioso)
    fn = _fp(new)
    assert set(fo) == set(fn)
    diff = [n for n in fo if fo[n] != fn[n]]
    assert not diff, f"{len(diff)} parametri diversi, es. {diff[:3]}: {fo[diff[0]]} vs {fn[diff[0]]}" if diff else ""
    assert any(v[0] == "torch.float8_e4m3fn" for v in fn.values()) and any(v[0] == "torch.bfloat16" for v in fn.values())
