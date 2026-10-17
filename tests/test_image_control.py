"""inference_server/backends/image_control: caricamento ControlNet (chiavi vere, tensori meta), convenzioni Alimama,
Redux e un giro completo CPU con pipeline minuscole (nessuna GPU, nessun checkpoint FLUX vero)."""
from __future__ import annotations

import json
import os
import struct

import pytest

torch = pytest.importorskip("torch")
from PIL import Image  # noqa: E402

from inference_server.backends import image_control as IC  # noqa: E402

ROOT = r"E:\AI\Models"
FILES = {
    "union": IC.CN_FLUX_UNION, "inpaint": IC.CN_FLUX_INPAINT, "zimage": IC.CN_ZIMAGE, "redux": IC.REDUX_FILE,
    "siglip": IC.SIGLIP_FILE,
}


def _meta_state(path, prefix=""):
    """{chiave: tensore meta} dall'header del file (nessun peso letto)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    dt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    return {k: torch.empty(v["shape"], dtype=dt.get(v["dtype"], torch.float32), device="meta")
            for k, v in h.items() if k.startswith(prefix)}


def _need_file(key):
    if not os.path.isfile(FILES[key]):
        pytest.skip(f"file assente: {FILES[key]}")


def test_flux_union_keys_match_diffusers(monkeypatch):
    _need_file("union")
    import safetensors.torch as st
    monkeypatch.setattr(st, "load_file", lambda p, *a, **k: _meta_state(p))
    cn = IC.load_flux_controlnet(IC.CN_FLUX_UNION)
    assert cn.config.num_layers == 6 and cn.config.num_single_layers == 0 and cn.config.guidance_embeds
    assert getattr(cn, "extra_channels", 0) == 0


def test_flux_alimama_has_68_channel_embedder(monkeypatch):
    _need_file("inpaint")
    import safetensors.torch as st
    monkeypatch.setattr(st, "load_file", lambda p, *a, **k: _meta_state(p))
    cn = IC.load_flux_controlnet(IC.CN_FLUX_INPAINT)
    assert cn.controlnet_x_embedder.in_features == 68 and cn.extra_channels == 4
    assert type(cn).__name__ == "AlimamaFluxControlNet"


def test_zimage_union21_keys_match_diffusers(monkeypatch):
    _need_file("zimage")
    import safetensors.torch as st
    monkeypatch.setattr(st, "load_file", lambda p, *a, **k: _meta_state(p))
    cn = IC.load_zimage_controlnet(IC.CN_ZIMAGE)
    c = cn.config
    assert c.control_in_dim == 33 and len(c.control_layers_places) == 15 and c.add_control_noise_refiner == "control_noise_refiner"


def test_redux_and_siglip_keys(monkeypatch):
    _need_file("redux")
    import safetensors.torch as st
    monkeypatch.setattr(st, "load_file", lambda p, *a, **k: _meta_state(p))
    enc = IC.load_redux_embedder(IC.REDUX_FILE)
    assert enc.redux_up.in_features == 1152 and enc.redux_down.out_features == 4096
    _need_file("siglip")
    # SigLIP: strict=True sulla torre visiva dal file completo (letto solo l'header → meta)
    import safetensors

    class _F:
        def __init__(self, p, *a, **k): self.sd = _meta_state(p)
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def keys(self): return self.sd.keys()
        def get_tensor(self, k): return self.sd[k]
    monkeypatch.setattr(safetensors, "safe_open", _F)
    v = IC.load_siglip_vision(IC.SIGLIP_FILE)
    assert v.config.hidden_size == 1152 and v.config.num_hidden_layers == 27


def test_alimama_polarity_and_pack():
    img = Image.new("RGB", (64, 32), (200, 100, 50))
    mk = Image.new("L", (64, 32), 0)
    mk.paste(255, (0, 0, 32, 32))                          # metà sinistra = da rigenerare
    ctl, packed = IC.alimama_inputs(img, mk, 64, 32)
    import numpy as np
    a = np.asarray(ctl)
    assert (a[:, :32] == 0).all() and (a[:, 32:] == (200, 100, 50)).all()      # area da rifare NERA
    assert packed.shape == (1, (32 // 16) * (64 // 16), 4)                     # H/8·W/8 latenti, impacchettati 2x2
    # token di sinistra = 0 (da rigenerare), di destra = 1 (da tenere): 1 − maschera
    p = packed[0].view(2, 4, 4)
    assert p[:, :2].sum() == 0 and p[:, 2:].min() == 1


def test_pack_matches_flux_pipeline():
    from diffusers import FluxPipeline
    x = torch.randn(1, 5, 8, 12)
    assert torch.equal(IC._pack_latents(x), FluxPipeline._pack_latents(x, 1, 5, 8, 12))


# ── giro completo CPU con componenti minuscoli ─────────────────────────────────────────────────────────
def _tiny_flux_pipe():
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel
    torch.manual_seed(0)
    tr = FluxTransformer2DModel(patch_size=1, in_channels=64, num_layers=1, num_single_layers=1, attention_head_dim=16,
                                num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=16,
                                guidance_embeds=True, axes_dims_rope=(4, 6, 6))
    vae = AutoencoderKL(in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 4,
                        up_block_types=("UpDecoderBlock2D",) * 4, block_out_channels=(8, 8, 8, 8), layers_per_block=1,
                        latent_channels=16, norm_num_groups=4, sample_size=64, shift_factor=0.1, scaling_factor=0.3)
    return FluxPipeline(scheduler=FlowMatchEulerDiscreteScheduler(), vae=vae, text_encoder=None, tokenizer=None,
                        text_encoder_2=None, tokenizer_2=None, transformer=tr)


def _tiny_cn(extra=0):
    from diffusers import FluxControlNetModel
    cn = FluxControlNetModel(patch_size=1, in_channels=64, num_layers=1, num_single_layers=0, attention_head_dim=16,
                             num_attention_heads=2, joint_attention_dim=32, pooled_projection_dim=16,
                             guidance_embeds=True, axes_dims_rope=[4, 6, 6])
    if extra:
        cn.controlnet_x_embedder = torch.nn.Linear(64 + extra, cn.controlnet_x_embedder.out_features)
        cn.__class__ = IC._alimama_class()
    return cn


def _embeds(pipe_kw, hidden=32, pooled=16):
    pipe_kw.pop("prompt", None)
    pipe_kw["prompt_embeds"] = torch.randn(1, 8, hidden)
    pipe_kw["pooled_prompt_embeds"] = torch.randn(1, pooled)


def test_flux_cn_cpu_round_trip(tmp_path):
    pipe = _tiny_flux_pipe()
    cn = _tiny_cn()
    src = tmp_path / "g.png"
    Image.effect_noise((96, 64), 60).convert("RGB").save(src)
    params = {"mode": "flux_cn", "model": "x", "prompt": "p", "width": 64, "height": 64, "steps": 2, "guidance": 3.5,
              "seed": 1, "control_image": str(src), "control_preprocess": "canny", "gen_device": "cpu"}
    cpipe, kw, keep = IC.build_job(pipe, params, cn)
    assert type(cpipe).__name__ == "FluxControlNetPipeline" and cpipe.transformer is pipe.transformer
    assert kw["controlnet_conditioning_scale"] == 0.7 and kw["control_guidance_end"] == 0.8      # canny (model card)
    _embeds(kw)
    out = cpipe(**kw).images[0]
    assert out.size == (64, 64)


def test_flux_alimama_cpu_round_trip(tmp_path):
    pipe = _tiny_flux_pipe()
    cn = _tiny_cn(extra=4)
    img = Image.effect_noise((64, 64), 80).convert("RGB")
    img.save(tmp_path / "i.png")
    mk = Image.new("L", (64, 64), 0)
    mk.paste(255, (16, 16, 48, 48))
    mk.save(tmp_path / "m.png")
    params = {"mode": "flux_cn_inpaint", "model": "x", "prompt": "p", "steps": 2, "seed": 2, "gen_device": "cpu",
              "image_path": str(tmp_path / "i.png"), "mask_path": str(tmp_path / "m.png")}
    cpipe, kw, keep = IC.build_job(pipe, params, cn)
    assert kw["controlnet_conditioning_scale"] == 0.9 and keep is not None
    _embeds(kw)
    out = cpipe(**kw).images[0]
    res = IC._finish(out, params, str(tmp_path / "o"), "flux_cn_inpaint", keep)
    assert res["status"] == "success"
    # fuori maschera il risultato è l'originale (PSNR infinito sugli angoli)
    import numpy as np
    gi = Image.open(res["output"][0]).convert("RGB")
    got = np.asarray(gi).astype(int)
    ref = np.asarray(img.resize(gi.size, Image.LANCZOS)).astype(int)      # il backend lavora a >=256 px
    assert np.abs(got[:20, :20] - ref[:20, :20]).max() <= 2


def test_redux_embeds_shapes():
    class V(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1))
        def forward(self, pixel_values):
            class O: last_hidden_state = torch.randn(1, 729, 1152)
            assert pixel_values.shape == (1, 3, 384, 384)
            return O

    class E(torch.nn.Module):
        def forward(self, h):
            class O: image_embeds = torch.randn(1, 729, 4096)
            return O

    imgs = [Image.new("RGB", (100, 50)), Image.new("RGB", (30, 90))]
    emb = IC.redux_embeds(V(), E(), imgs, torch.device("cpu"), 0.5)
    assert emb.shape == (1, 729 * 2, 4096)
    pe, pooled = IC.redux_prompt_embeds(None, emb, "", torch.device("cpu"))
    assert pe.shape == (1, 512 + 729 * 2, 4096) and pooled.shape == (1, 768) and float(pe[:, :512].abs().sum()) == 0
