"""MagicQuill V2 nativo: le LoRA di controllo REALI (380 tensori) e la puzzle (992) vengono consumate per intero dal
loader, senza chiavi orfane e con le shape giuste. CPU pura, nessun checkpoint FLUX.

Eseguire con il venv immagini (diffusers 0.38):  E:\\AI\\Tools\\inference_server_venv\\Scripts\\python.exe -m pytest
tests\\test_mq2_loader.py -q   (serve pytest nel path: vedi report). Prima controllare core.resources.commit_free_gb()>=3.5."""
import os
import re

import pytest
import torch

from core.resources import commit_free_gb
from core import optional_components as _oc  # noqa: E402
import pytest as _pt  # noqa: E402
if _oc.load_magicquill() is None:                # componente opzionale (optional/magicquill o cartella installata)
    _pt.skip(_oc.last_error(), allow_module_level=True)
from ltx_magicquill.mq2 import puzzle
from ltx_magicquill.mq2.control_lora import ControlLoRA

MQ_DIR = r"E:\AI\Models\LoRA\MagicQuill"
MODES = ("local", "removal", "edge", "color")


def _f(name):
    return os.path.join(MQ_DIR, f"{name}_lora.safetensors")


@pytest.fixture(autouse=True)
def _risorse():
    if commit_free_gb() < 3.5:
        pytest.skip("commit libero < 3.5 GB")


@pytest.mark.skipif(not os.path.isfile(_f("local")), reason="LoRA MagicQuill non presenti")
@pytest.mark.parametrize("mode", MODES)
def test_control_lora_reale_consumata_per_intero(mode):
    lo = ControlLoRA.from_file(_f(mode))                  # solleva se orfane/mancanti/shape errate
    assert lo.n_tensors() == 380
    assert lo.rank == 128 and lo.dim == 3072
    d = [k for k in lo.layers if k[0] == "d"]
    s = [k for k in lo.layers if k[0] == "s"]
    assert len(d) == 19 * 4 and len(s) == 38 * 3          # q,k,v,proj nei doppi; q,k,v nei singoli
    assert not [k for k in s if k[2] == "proj"]
    down, up = lo.layers[("d", 18, "proj")]
    assert tuple(down.shape) == (128, 3072) and tuple(up.shape) == (3072, 128)
    assert down.dtype == torch.bfloat16
    assert lo.scale == 1.0                                # network_alpha == rank


def _tiny(n_double=1, n_single=1, dim=8, rank=2):
    sd = {}
    for i in range(n_double):
        for p in ("q", "k", "v", "proj"):
            sd[f"transformer_blocks.{i}.attn.processor.{p}_loras.0.down.weight"] = torch.zeros(rank, dim)
            sd[f"transformer_blocks.{i}.attn.processor.{p}_loras.0.up.weight"] = torch.zeros(dim, rank)
    for i in range(n_single):
        for p in ("q", "k", "v"):
            sd[f"single_transformer_blocks.{i}.attn.processor.{p}_loras.0.down.weight"] = torch.zeros(rank, dim)
            sd[f"single_transformer_blocks.{i}.attn.processor.{p}_loras.0.up.weight"] = torch.zeros(dim, rank)
    return sd


def test_loader_sintetico_ok_e_errori():
    kw = dict(n_double=1, n_single=1, dim=8)
    assert ControlLoRA.from_state_dict(_tiny(), **kw).n_tensors() == 1 * 4 * 2 + 1 * 3 * 2
    sd = _tiny()
    sd["transformer_blocks.0.attn.processor.z_loras.0.down.weight"] = torch.zeros(2, 8)      # orfana
    with pytest.raises(ValueError, match="orfane"):
        ControlLoRA.from_state_dict(sd, **kw)
    sd = _tiny()
    del sd["single_transformer_blocks.0.attn.processor.v_loras.0.up.weight"]                 # mancante
    with pytest.raises(ValueError, match="mancanti"):
        ControlLoRA.from_state_dict(sd, **kw)
    sd = _tiny()
    sd["transformer_blocks.0.attn.processor.q_loras.0.down.weight"] = torch.zeros(2, 9)      # shape errata
    with pytest.raises(ValueError, match="shape"):
        ControlLoRA.from_state_dict(sd, **kw)
    sd = _tiny()
    sd["transformer_blocks.0.attn.processor.q_loras.1.down.weight"] = torch.zeros(2, 8)      # 2a LoRA nel file
    with pytest.raises(ValueError, match="1 LoRA per file"):
        ControlLoRA.from_state_dict(sd, **kw)
    sd = _tiny()
    sd["single_transformer_blocks.0.attn.processor.proj_loras.0.down.weight"] = torch.zeros(2, 8)
    with pytest.raises(ValueError, match="orfane"):                                           # proj nei singoli
        ControlLoRA.from_state_dict(sd, **kw)


@pytest.mark.skipif(not os.path.isfile(_f("puzzle")), reason="LoRA puzzle non presente")
def test_puzzle_conversione_chiavi_e_peft_load():
    from safetensors.torch import load_file
    raw = load_file(_f("puzzle"), device="cpu")
    assert len(raw) == 992
    sd = puzzle.convert_state_dict(raw)
    assert len(sd) == 992 and all(k.startswith("transformer.") for k in sd)
    assert sum(k.endswith(".lora_A.weight") for k in sd) == 496 == sum(k.endswith(".lora_B.weight") for k in sd)
    assert not any(".lora.down." in k or ".lora.up." in k for k in sd)
    assert {v.shape[0] for k, v in sd.items() if k.endswith(".lora_A.weight")} == {32}     # rank 32

    # carica nel loader PEFT diffusers su un transformer a larghezza REALE (3072) ma 1+1 blocchi: tutte le chiavi del
    # blocco 0 e di norm_out/proj_out devono essere consumate (nessuna orfana)
    from diffusers import FluxTransformer2DModel
    from diffusers.loaders import FluxLoraLoaderMixin
    if commit_free_gb() < 6:
        pytest.skip("commit libero < 6 GB per il transformer 1+1 a 3072")

    def keep(k):
        m = re.search(r"(?:single_)?transformer_blocks\.(\d+)\.", k)
        return m is None or int(m.group(1)) == 0
    sub = {k: v for k, v in sd.items() if keep(k)}
    tr = FluxTransformer2DModel(num_layers=1, num_single_layers=1, guidance_embeds=True).to(torch.bfloat16)
    FluxLoraLoaderMixin.load_lora_into_transformer(sub, network_alphas=None, transformer=tr, adapter_name=puzzle.ADAPTER)
    lora_a = {n: p for n, p in tr.named_parameters() if "lora_A" in n}
    lora_b = {n: p for n, p in tr.named_parameters() if "lora_B" in n}
    assert len(lora_a) == len(lora_b) == len(sub) // 2
    assert tr.peft_config[puzzle.ADAPTER].r == 32
    p = tr.transformer_blocks[0].attn.to_q
    assert torch.equal(p.lora_A[puzzle.ADAPTER].weight, sd["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"])
    assert torch.equal(p.lora_B[puzzle.ADAPTER].weight, sd["transformer.transformer_blocks.0.attn.to_q.lora_B.weight"])
    del tr
