"""FLUX.3 Image Klein (task 13): worker su CPU con modelli MINUSCOLI (nessuna GPU, nessun peso vero).

Coperto: DiT (forma, flow map, riferimenti, denoise), chiavi/forme esatte contro gli header VERI di bf16 e fp8r, fp8 per riga,
LoRA (normalizzazione delle chiavi native e diffusers, fusione), preparazione dei riferimenti, VAE (patchify + BatchNorm,
andata e ritorno), controllo dei pesi mancanti, lettore GGUF su un file sintetico (parser, Q8_0, trasformazioni inverse di llama.cpp).
La parità matematica con diffusers.Flux3Transformer2DModel è in tests/dev/flux3_dit_parity.py (venv con diffusers ≥ 15/10)."""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
os.environ["FLUX3_DEVICE"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))
sys.path.insert(0, str(ROOT / "tests"))

import flux3_dit as D  # noqa: E402
import flux3_infer as W  # noqa: E402
from core import flux3_spec as S  # noqa: E402

DATA = ROOT / "tests" / "data"


def _hdr(name):
    return json.loads((DATA / f"flux3_header_{name}.json").read_text("utf-8"))["tensors"]


def tiny_cfg(flow=True):
    return D.Config(hidden=64, heads=4, depth=2, depth_single=3, axes_dim=(4, 4, 4, 4),
                    in_channels={"image": 128, "image_cond": 128, "text": 96}, flow_map=flow)


def _ids(h, w, scale=1.0, l=0.0):
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    z = torch.zeros_like(rows)
    ids = torch.stack([z, rows, cols, z], -1).reshape(-1, 4).float()
    ids[:, 1:3] *= scale
    ids[:, 3] = l
    return ids


def _tiny_model(flow=True, seed=0, std=0.05):
    torch.manual_seed(seed)
    m = D.DiT(tiny_cfg(flow))
    for p in m.parameters():
        torch.nn.init.normal_(p, std=std)
    return m.float().eval()


def _packed(txt, img, cond=None, dtype=torch.float32):
    p = D.make_packed([[(txt, torch.stack([torch.zeros(len(txt))] * 3 + [torch.arange(len(txt)).float()], -1)),
                        cond, img]], "cpu")
    for m in D.MODES:
        p.data[m] = p.data[m].to(dtype)
        p.timesteps[m] = p.timesteps[m].to(dtype)
    return p


# ── DiT ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_chiavi_e_forme_esatte_contro_gli_header_ufficiali():
    for name, n_fp8 in (("bf16", 0), ("fp8r", 234)):
        h = _hdr(name)
        m = D.build_meta()
        assert D.swap_fp8_linears(m, h.keys()) == n_fp8
        exp = m.state_dict()
        assert set(exp) == set(h), (name, sorted(set(exp) ^ set(h))[:5])
        assert all(list(exp[k].shape) == h[k]["shape"] for k in exp)
        assert sum(p.numel() for p in D.build_meta().parameters()) == 5_548_681_984       # = manifest ufficiale
    assert D.is_flow_map_keys(_hdr("fp8r")) and not D.is_flow_map_keys({k: v for k, v in _hdr("bf16").items()
                                                                       if "ref_time_in" not in k})


def test_forward_forma_e_riferimenti_cambiano_l_uscita():
    m = _tiny_model()
    img, txt, cond = torch.randn(12, 128), torch.randn(7, 96), torch.randn(6, 128)
    with torch.no_grad():
        a = m(_packed(txt, (img, _ids(3, 4))), timesteps_ref=torch.tensor([0.5]))
        b = m(_packed(txt, (img, _ids(3, 4)), (cond, _ids(2, 3, 0.5, -1))), timesteps_ref=torch.tensor([0.5]))
    assert a.shape == (12, 128) and b.shape == (12, 128)
    assert (a - b).abs().max() > 1e-4                                                 # il riferimento conta
    with pytest.raises(ValueError):
        m(_packed(txt, (img, _ids(3, 4))))                                           # flow map richiede timesteps_ref
    base = _tiny_model(flow=False)
    with pytest.raises(ValueError):
        base(_packed(txt, (img, _ids(3, 4))), timesteps_ref=torch.tensor([0.5]))


def test_riferimento_ids_posizione_e_indice():
    """Le posizioni RoPE dei riferimenti sono ×0,5 e l'asse l distingue il riferimento (-(1+i)): ids diversi → uscita diversa."""
    m = _tiny_model(std=0.4)                                 # pesi non troppo piccoli: l'attenzione deve contare
    img, txt, cond = torch.randn(12, 128), torch.randn(7, 96), torch.randn(6, 128)
    with torch.no_grad():
        a = m(_packed(txt, (img, _ids(3, 4)), (cond, _ids(2, 3, 0.5, -1))), timesteps_ref=torch.tensor([0.5]))
        b = m(_packed(txt, (img, _ids(3, 4)), (cond, _ids(2, 3, 0.5, -2))), timesteps_ref=torch.tensor([0.5]))
    assert (a - b).abs().max() > 1e-5


def test_schedule_e_denoise_flow_map():
    s = D.schedule(16, S.SHIFT_ALPHA)
    assert len(s) == 17 and abs(float(s[0]) - 1.0) < 1e-6 and abs(float(s[-1])) < 1e-6
    assert all(float(s[i]) > float(s[i + 1]) for i in range(16))
    alpha1 = D.schedule(4, 1.0)
    assert torch.allclose(alpha1, torch.linspace(1, 0, 5))                           # alpha = 1 → lineare
    m = _tiny_model()
    img, txt = torch.randn(12, 128), torch.randn(7, 96)
    steps = []
    out = D.denoise_flow_map(m, _packed(txt, (img, _ids(3, 4))), D.schedule(4, S.SHIFT_ALPHA),
                             on_step=lambda i, n: steps.append((i, n)))
    assert out.shape == (12, 128) and steps == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert torch.isfinite(out).all()


def test_cfg_base_due_campioni():
    m = _tiny_model(flow=False)
    img, txt, nul = torch.randn(12, 128), torch.randn(7, 96), torch.randn(7, 96)
    def ids(n):
        return torch.stack([torch.zeros(n)] * 3 + [torch.arange(n).float()], -1)
    p = D.make_packed([[(nul, ids(7)), None, (img, _ids(3, 4))], [(txt, ids(7)), None, (img.clone(), _ids(3, 4))]], "cpu")
    for k in D.MODES:
        p.data[k] = p.data[k].float(); p.timesteps[k] = p.timesteps[k].float()
    out = D.denoise_cfg(m, p, D.schedule(2, 6.93), 3.5)
    assert out.shape == (12, 128) and torch.isfinite(out).all()


def test_load_state_controlla_le_chiavi():
    m = D.build_meta(tiny_cfg())
    st = {k: torch.zeros(v.shape) for k, v in _tiny_model().state_dict().items()}
    st.pop("txt_in.weight")
    with pytest.raises(KeyError, match="mancanti"):
        D.load_state(m, dict(st), "cpu")
    st2 = {k: torch.zeros(v.shape) for k, v in _tiny_model().state_dict().items()}
    st2["extra.weight"] = torch.zeros(1)
    with pytest.raises(KeyError, match="inattese"):
        D.load_state(D.build_meta(tiny_cfg()), st2, "cpu")
    ok = D.load_state(D.build_meta(tiny_cfg()), {k: torch.zeros(v.shape) for k, v in _tiny_model().state_dict().items()}, "cpu")
    assert next(ok.parameters()).dtype == torch.bfloat16 and ok._n_fp8 == 0


def test_fp8_per_riga_quantizzazione():
    if D.FP8 is None:
        pytest.skip("torch senza float8")
    w = torch.randn(64, 128) * 0.02
    q, s = D.quantize_rows(w)
    assert q.dtype == D.FP8 and s.shape == (64,)
    back = q.float() * s[:, None]
    assert ((back - w).abs().max() / w.abs().max()) < 0.07                              # E4M3: ~6% al massimo per riga
    # il file ufficiale segue la stessa convenzione: weight.float() * weight_scale[:, None]
    m = D.build_meta(tiny_cfg())
    names = [n for n, _ in m.named_modules() if n.endswith("q_proj")][:2]
    keys = {n + ".weight_scale" for n in names}
    assert D.swap_fp8_linears(m, keys) == 2
    assert isinstance(m.get_submodule(names[0]), D.FP8Linear)


# ── LoRA ─────────────────────────────────────────────────────────────────────────────────────────────────
def test_lora_chiavi_native_e_diffusers_stessi_moduli():
    A, B = torch.randn(4, 64), torch.randn(64, 4)
    native = {"diffusion_model.content_mode_blocks.image.0.q_proj.lora_A.weight": A,
              "diffusion_model.content_mode_blocks.image.0.q_proj.lora_B.weight": B,
              "diffusion_model.single_blocks.1.mlp_out.lora_A.weight": A,
              "diffusion_model.single_blocks.1.mlp_out.lora_B.weight": B,
              "diffusion_model.txt_mode_blocks.0.attn_out.alpha": torch.tensor(8.0)}
    diff = {"transformer.mode_blocks.image.0.attn.to_q.lora_A.weight": A,
            "transformer.mode_blocks.image.0.attn.to_q.lora_B.weight": B,
            "transformer.single_transformer_blocks.1.ff.linear_out.lora_A.weight": A,
            "transformer.single_transformer_blocks.1.ff.linear_out.lora_B.weight": B}
    pn, pd = W.lora_pairs(native), W.lora_pairs(diff)
    assert set(pn) == set(pd) == {"content_mode_blocks.image.0.q_proj", "single_blocks.1.mlp_out"}
    alpha = {"x.lora_down.weight": A, "x.lora_up.weight": B, "x.alpha": torch.tensor(2.0)}
    assert W.lora_pairs(alpha)["x"][2] == 2.0


def test_merge_lora_nel_peso():
    m = D.build_meta(tiny_cfg())
    st = {k: torch.zeros(v.shape) for k, v in _tiny_model().state_dict().items()}
    m = D.load_state(m, st, "cpu")
    A, B = torch.randn(4, 64), torch.randn(64, 4)
    sd = {"diffusion_model.content_mode_blocks.image.0.q_proj.lora_A.weight": A,
          "diffusion_model.content_mode_blocks.image.0.q_proj.lora_B.weight": B,
          "diffusion_model.nonesiste.lora_A.weight": A, "diffusion_model.nonesiste.lora_B.weight": B}
    n = W.merge_lora(m, sd, 0.5)
    assert n == 1
    w = m.content_mode_blocks["image"][0].q_proj.weight.float()
    assert torch.allclose(w, (B @ A) * 0.5, atol=0.05, rtol=0.02)                       # bf16
    assert W.merge_lora(m, {"foo.lora_A.weight": A, "foo.lora_B.weight": B}, 1.0) == 0   # nessun modulo → il job rifiuta


# ── riferimenti ──────────────────────────────────────────────────────────────────────────────────────────
def test_prepare_reference_singolo_padding_e_ritaglio():
    from PIL import Image
    im = Image.new("RGB", (100, 70), (200, 30, 30))                    # lati non multipli di 16 → canvas a 16 con padding
    out, box = W.prepare_reference(im)
    assert out.size[0] % 16 == 0 and out.size[1] % 16 == 0
    assert box is not None and (box[2] - box[0], box[3] - box[1]) == (100, 70)          # il riquadro dell'originale
    big, _ = W.prepare_reference(Image.new("RGB", (8000, 6000)))
    assert big.size[0] * big.size[1] <= S.FIRST_REF_MAX_PIXELS
    ok, none = W.prepare_reference(Image.new("RGB", (64, 64)))
    assert none is None and ok.size == (64, 64)
    rgba = W.prepare_reference(Image.new("RGBA", (96, 96), (255, 0, 0, 0)))[0]
    assert rgba.mode == "RGB"


def test_prepare_multi_reference_un_megapixel_griglia_16():
    from PIL import Image
    out = W.prepare_multi_reference(Image.new("RGB", (3000, 2000)))
    assert out.size[0] * out.size[1] <= S.MULTI_REF_MAX_PIXELS and out.size[0] % 16 == 0 and out.size[1] % 16 == 0
    assert W.prepare_multi_reference(Image.new("RGB", (100, 100))).size == (96, 96)


# ── VAE ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_vae_patchify_e_decode_andata_e_ritorno():
    from diffusers import AutoencoderKLFlux2
    cfg = dict(S.VAE_CONFIG, block_out_channels=[16, 16, 16, 16], norm_num_groups=4)
    torch.manual_seed(0)
    vae = AutoencoderKLFlux2.from_config(cfg).float().eval()
    from PIL import Image
    import numpy as np
    img = Image.fromarray((np.random.RandomState(0).rand(64, 96, 3) * 255).astype("uint8"))
    toks, ids = W.vae_encode(vae, img)
    assert toks.shape == (4 * 6, 128) and ids.shape == (24, 4) and ids[:, 1].max() == 3 and ids[:, 2].max() == 5
    out = W.vae_decode(vae, toks, ids)
    assert out.size == (96, 64) and out.mode == "RGB"                                    # la forma torna quella dell'ingresso


# ── pesi mancanti ────────────────────────────────────────────────────────────────────────────────────────
def test_resolve_pesi_mancanti_errore_chiaro(monkeypatch):
    from core import weights_catalog as wc
    monkeypatch.setattr(wc, "is_present", lambda e, base=None: False)
    with pytest.raises(RuntimeError) as e:
        W.resolve({}, need_pup=True)
    msg = str(e.value)
    assert "Pesi FLUX.3 mancanti" in msg and S.DEFAULT_DIT in msg and S.TE in msg and S.PUP_DEFAULT in msg
    assert "tab «Pesi modelli»" in msg
    with pytest.raises(RuntimeError, match="sconosciuto"):
        W.resolve({"flux3_dit": "flux3-boh"}, need_pup=False)


def test_job_rifiuta_riquadri_non_validi_prima_di_caricare(tmp_path):
    from PIL import Image
    ref = tmp_path / "r.png"
    Image.new("RGB", (64, 64)).save(ref)
    with pytest.raises(RuntimeError, match="Riquadri non validi"):
        W.run({"prompt": "x", "boxes": [{"id": "1x", "desc": "", "bbox": [0, 0, 1, 1]}], "output": str(tmp_path / "o.png")})
    with pytest.raises(RuntimeError, match="al massimo 10"):
        W.run({"prompt": "x", "ref_images": [str(ref)] * 11})
    with pytest.raises(RuntimeError, match="serve un prompt"):
        W.run({"prompt": " "})
    with pytest.raises(RuntimeError, match="manca l'immagine"):
        W.run({"prompt": "x", "mask_path": str(ref)})


def test_paste_back_maschera_binarizzata(tmp_path):
    from PIL import Image
    import numpy as np
    base = Image.new("RGB", (32, 32), (0, 0, 0))
    gen = Image.new("RGB", (32, 32), (255, 255, 255))
    m = np.zeros((32, 32), dtype="uint8")
    m[:, :16] = 150                                                       # pennellata sotto 255 (visto dal vivo 17/10)
    mp = tmp_path / "m.png"
    Image.fromarray(m).save(mp)
    out = np.asarray(W._paste_back(gen, base, str(mp), 0))
    assert (out[:, :16] == 255).all() and (out[:, 16:] == 0).all()        # 150 > 127 → rifatto al 100 %, non al 60 %


# ── lettore GGUF su file sintetico ───────────────────────────────────────────────────────────────────────
NV, NK, HD = 4, 2, 8                       # teste V, teste K, dimensione di testa (ssm); key_dim = NK*HD = 16
HID = 24


from flux3_gguf_writer import gguf_write as _gguf_write  # noqa: E402


def _inv_perm():
    import numpy as np
    perm = np.arange(NV).reshape(NV // NK, NK).T.reshape(-1)             # HF i ← GGUF perm[i]
    return perm, np.argsort(perm)


def test_gguf_lettore_sintetico_trasformazioni_inverse(tmp_path):
    import numpy as np
    import flux3_gguf as G
    rng = np.random.RandomState(1)
    perm, inv = _inv_perm()
    key_dim = NK * HD
    hf = {                                                  # tensori «HF» attesi (layer 0 lineare, layer 1 piena)
        "in_proj_qkv": rng.randn(2 * key_dim + NV * HD, HID), "in_proj_z": rng.randn(NV * HD, HID),
        "in_proj_a": rng.randn(NV, HID), "in_proj_b": rng.randn(NV, HID), "A_log": rng.randn(NV),
        "dt_bias": rng.randn(NV), "conv": rng.randn(2 * key_dim + NV * HD, 1, 4), "snorm": rng.randn(HD),
        "out_proj": rng.randn(HID, NV * HD), "in_norm": rng.randn(HID) * 0.1, "q_norm": rng.randn(16) * 0.1,
        "q": rng.randn(32, HID), "embed": rng.randn(40, HID), "head": rng.randn(40, HID), "fnorm": rng.randn(HID) * 0.1,
    }
    f32 = lambda a: np.asarray(a, dtype=np.float32)

    def hperm(x, axis=0):                                    # HF → ggml: testa GGUF j = inv… (gguf[perm[i]] = hf[i])
        shape = list(x.shape)
        shape[axis:axis + 1] = [NV, x.shape[axis] // NV]
        y = np.take(x.reshape(shape), inv, axis=axis)
        return y.reshape(x.shape)

    qkv = hf["in_proj_qkv"].copy()
    qkv[2 * key_dim:] = hperm(qkv[2 * key_dim:])
    conv = hf["conv"][:, 0, :].copy()
    conv[2 * key_dim:] = hperm(conv[2 * key_dim:])
    tens = [
        ("token_embd.weight", f32(hf["embed"]), 0), ("output.weight", f32(hf["head"]), 0),
        ("output_norm.weight", f32(hf["fnorm"] + 1), 0),
        ("blk.0.attn_norm.weight", f32(hf["in_norm"] + 1), 0),
        ("blk.0.attn_qkv.weight", f32(qkv), 0), ("blk.0.attn_gate.weight", f32(hperm(hf["in_proj_z"])), 0),
        ("blk.0.ssm_alpha.weight", f32(hperm(hf["in_proj_a"])), 0), ("blk.0.ssm_beta.weight", f32(hperm(hf["in_proj_b"])), 0),
        ("blk.0.ssm_a", f32(hperm(-np.exp(hf["A_log"]))), 0), ("blk.0.ssm_dt.bias", f32(hperm(hf["dt_bias"])), 0),
        ("blk.0.ssm_conv1d.weight", f32(conv), 0), ("blk.0.ssm_norm.weight", f32(hf["snorm"]), 0),
        ("blk.0.ssm_out.weight", f32(hperm(hf["out_proj"], axis=1)), 0),
        ("blk.1.attn_q.weight", f32(hf["q"]), 8 if False else 0), ("blk.1.attn_q_norm.weight", f32(hf["q_norm"] + 1), 0),
        ("blk.2.attn_norm.weight", f32(np.ones(HID)), 0),                     # blocco MTP: non si legge
    ]
    # un tensore Q8_0 vero (multiplo di 32) per provare la dequantizzazione
    wq = rng.randn(32, 64) * 0.05
    tens.append(("blk.1.ffn_up.weight", f32(wq), 8))
    kv = {"general.architecture": "qwen35", "qwen35.block_count": 3, "qwen35.ssm.time_step_rank": NV,
          "qwen35.ssm.group_count": NK, "qwen35.ssm.state_size": HD, "qwen35.nextn_predict_layers": 1}
    p = tmp_path / "t.gguf"
    _gguf_write(p, kv, tens)
    st = G.read_state(str(p), "cpu")
    L0, L1 = "model.language_model.layers.0.", "model.language_model.layers.1."
    exp = {"model.language_model.embed_tokens.weight": hf["embed"], "lm_head.weight": hf["head"],
           "model.language_model.norm.weight": hf["fnorm"], L0 + "input_layernorm.weight": hf["in_norm"],
           L0 + "linear_attn.in_proj_qkv.weight": hf["in_proj_qkv"], L0 + "linear_attn.in_proj_z.weight": hf["in_proj_z"],
           L0 + "linear_attn.in_proj_a.weight": hf["in_proj_a"], L0 + "linear_attn.in_proj_b.weight": hf["in_proj_b"],
           L0 + "linear_attn.A_log": hf["A_log"], L0 + "linear_attn.dt_bias": hf["dt_bias"],
           L0 + "linear_attn.conv1d.weight": hf["conv"], L0 + "linear_attn.norm.weight": hf["snorm"],
           L0 + "linear_attn.out_proj.weight": hf["out_proj"], L1 + "self_attn.q_proj.weight": hf["q"],
           L1 + "self_attn.q_norm.weight": hf["q_norm"], L1 + "mlp.up_proj.weight": wq}
    assert set(st) == set(exp)                                                           # MTP escluso, nessun tensore in più
    for k, v in exp.items():
        got = st[k].float().numpy()
        assert got.shape == v.shape, (k, got.shape, v.shape)
        tol = 2e-2 if "up_proj" in k else 5e-2                                           # Q8_0 / bf16
        assert np.abs(got - v).max() < tol * max(1.0, np.abs(v).max()), (k, np.abs(got - v).max())
    assert all(t.dtype == torch.bfloat16 for t in st.values())


def test_gguf_rifiuta_architettura_e_tipi(tmp_path):
    import numpy as np
    import flux3_gguf as G
    p = tmp_path / "a.gguf"
    _gguf_write(p, {"general.architecture": "llama", "qwen35.block_count": 1}, [("token_embd.weight", np.zeros((4, 4), "float32"), 0)])
    with pytest.raises(RuntimeError, match="qwen35"):
        G.read_state(str(p), "cpu")
    (tmp_path / "x.gguf").write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(RuntimeError, match="non è un file GGUF"):
        G.read_state(str(tmp_path / "x.gguf"), "cpu")
    kv, tens, base = G.read_header(str(p))
    assert kv["general.architecture"] == "llama" and tens[0][0] == "token_embd.weight" and base % 32 == 0


# ── text encoder: ufficiale (shard bf16) e Krea2 uncensored (file singolo fp8, nomi senza «model.») ──────────────────────
def _te_hdr(name):
    d = json.loads((DATA / name).read_text("utf-8"))
    return d.get("tensors", d)


def test_te_krea2_rimappatura_chiavi_su_header_veri():
    """Tutti i 398 tensori del modello di linguaggio del TE Krea2 (header VERO) diventano chiavi e forme del TE ufficiale FLUX.3."""
    krea, off = _te_hdr("krea2_te_header.json"), _te_hdr("flux3_te_header.json")
    mapped = {W.remap_te_key(k): v for k, v in krea.items() if W.remap_te_key(k)}
    mapped = {k: v for k, v in mapped.items() if not k.endswith(".weight_scale")}
    ref = {k: v for k, v in off.items() if not k.startswith("model.visual.")}
    assert set(mapped) == set(ref), (sorted(set(mapped) ^ set(ref))[:6])
    assert all(mapped[k]["shape"] == ref[k]["shape"] for k in ref)
    assert len(ref) == 398
    assert W.remap_te_key("lm_head.weight") is None and W.remap_te_key("visual.blocks.0.attn.qkv.weight") is None
    assert W.remap_te_key("model.visual.x") is None and W.remap_te_key("model.language_model.norm.weight") == "model.language_model.norm.weight"
    assert sum(1 for v in krea.values() if v["dtype"] == "F8_E4M3") == 252                   # matrici fp8 con scala per riga


def _tiny_te_cfg(tmp_path):
    from transformers import AutoConfig
    cfg = {"architectures": ["Qwen3VLForConditionalGeneration"], "model_type": "qwen3_vl", "image_token_id": 151655,
           "video_token_id": 151656, "vision_start_token_id": 151652, "vision_end_token_id": 151653, "tie_word_embeddings": True,
           "text_config": {"model_type": "qwen3_vl_text", "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
                           "num_key_value_heads": 2, "head_dim": 16, "num_hidden_layers": 3, "vocab_size": 300, "rms_norm_eps": 1e-6,
                           "rope_theta": 5000000, "tie_word_embeddings": True, "max_position_embeddings": 4096,
                           "rope_scaling": {"mrope_interleaved": True, "mrope_section": [2, 3, 3], "rope_type": "default"}},
           "vision_config": {"model_type": "qwen3_vl", "depth": 1, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                             "out_hidden_size": 64, "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                             "num_position_embeddings": 16, "deepstack_visual_indexes": [0]}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return AutoConfig.from_pretrained(str(tmp_path))


def test_te_caricamento_krea2_fp8_per_riga_vs_ufficiale_bf16(tmp_path):
    """Stesso modello minuscolo: shard bf16 «ufficiale» e file singolo «Krea2» (fp8 per riga, nomi `language_model.`, con torre
    visiva e scale): gli stati nascosti coincidono entro l'errore di fp8. Nessun modello vero in RAM."""
    if D.FP8 is None:
        pytest.skip("torch senza float8")
    from safetensors.torch import save_file
    from transformers import Qwen3VLForConditionalGeneration
    cfg = _tiny_te_cfg(tmp_path)
    torch.manual_seed(0)
    ref = Qwen3VLForConditionalGeneration._from_config(cfg, dtype=torch.bfloat16)
    sd = {k: v.detach().clone().contiguous() for k, v in ref.state_dict().items() if k != "lm_head.weight"}
    off = tmp_path / "off"
    off.mkdir()
    (off / "config.json").write_text((tmp_path / "config.json").read_text())
    save_file(sd, str(off / "model.safetensors"))
    krea = {}
    for k, v in sd.items():                                              # formato Krea2: fp8 per riga sui Linear del linguaggio
        nk = k[len("model."):] if k.startswith("model.") else k
        if k.startswith("model.language_model.layers") and k.endswith("_proj.weight"):
            q, s = D.quantize_rows(v.float())
            krea[nk] = q
            krea[nk.replace(".weight", ".weight_scale")] = s.reshape(-1, 1).float()
        else:
            krea[nk] = v
    single = tmp_path / "krea.safetensors"
    save_file(krea, str(single))
    a = W.build_te_model(str(off), str(tmp_path))
    b = W.build_te_model(str(single), str(tmp_path))
    from _qlinear import RowFp8Linear
    assert sum(isinstance(m, RowFp8Linear) for m in b.modules()) == 3 * 7                  # q,k,v,o,gate,up,down × 3 strati
    ids = torch.randint(0, 300, (1, 12))
    pos = torch.arange(12)[None]
    outs = []
    for m in (a, b):
        with torch.no_grad():
            o = m(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos[None].expand(4, -1, -1),
                  output_hidden_states=True, use_cache=False)
        outs.append(torch.stack([o.hidden_states[i] for i in (1, 2, 3)]).float())
    err = (outs[0] - outs[1]).abs().max() / outs[0].abs().max()
    assert err < 0.2, float(err)                       # fp8 per riga su pesi casuali: scarto relativo contenuto, non identità
    with torch.no_grad():
        o = ref.eval()(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos[None].expand(4, -1, -1),
                       output_hidden_states=True, use_cache=False)
    refh = torch.stack([o.hidden_states[i] for i in (1, 2, 3)]).float()
    assert (outs[0] - refh).abs().max() < 1e-2          # il caricamento «ufficiale» è esatto


def test_te_caricamento_file_singolo_bf16_stile_krea2_turbo(tmp_path):
    """Il TE di Krea 2 Turbo: file singolo bf16 con nomi `language_model.…` (e `visual.…`), senza scale. Stessi stati nascosti
    dello shard «ufficiale» (bit a bit: nessuna quantizzazione) e nessun Linear fp8."""
    from safetensors.torch import save_file
    from transformers import Qwen3VLForConditionalGeneration
    cfg = _tiny_te_cfg(tmp_path)
    torch.manual_seed(1)
    ref = Qwen3VLForConditionalGeneration._from_config(cfg, dtype=torch.bfloat16)
    sd = {k: v.detach().clone().contiguous() for k, v in ref.state_dict().items() if k != "lm_head.weight"}
    off = tmp_path / "off"
    off.mkdir()
    (off / "config.json").write_text((tmp_path / "config.json").read_text())
    save_file(sd, str(off / "model.safetensors"))
    turbo = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in sd.items()}
    assert any(k.startswith("visual.") for k in turbo) and any(k.startswith("language_model.") for k in turbo)
    single = tmp_path / "turbo.safetensors"
    save_file(turbo, str(single))
    a = W.build_te_model(str(off), str(tmp_path))
    b = W.build_te_model(str(single), str(tmp_path))
    from _qlinear import RowFp8Linear
    assert not any(isinstance(m, RowFp8Linear) for m in b.modules())
    ids = torch.randint(0, 300, (1, 12))
    pos = torch.arange(12)[None]
    outs = []
    for m in (a, b):
        with torch.no_grad():
            o = m(input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=pos[None].expand(4, -1, -1),
                  output_hidden_states=True, use_cache=False)
        outs.append(torch.stack([o.hidden_states[i] for i in (1, 2, 3)]).float())
    assert torch.equal(outs[0], outs[1])


def test_te_scelta_krea2_preferito_se_presente():
    T = S.TE_KREA2_TURBO
    assert S.te_id("", lambda i: i == S.TE_KREA2) == S.TE_KREA2                              # automatico: Krea2 uncensored se su disco
    assert S.te_id("", lambda i: False) == S.TE and S.te_id("", None) == S.TE
    assert S.te_id("official", lambda i: True) == S.TE and S.te_id("krea2", lambda i: False) == S.TE_KREA2
    assert S.TE_CHOICES == ("", "krea2", "official")
    # «ufficiale» riusato da Krea 2 Turbo: solo se flux3-te non c'è (nessun download in più)
    assert S.te_id("official", lambda i: i == T) == T
    assert S.te_id("", lambda i: i == T) == T                                                # automatico senza uncensored: l'ufficiale riusato
    assert S.te_id("official", lambda i: i in (T, S.TE_KREA2)) == T                          # scelta esplicita ufficiale: mai l'uncensored
    assert S.te_id("official", lambda i: i in (T, S.TE)) == S.TE                             # lo snapshot scaricato ha la precedenza
    assert S.te_id("", lambda i: i in (T, S.TE_KREA2)) == S.TE_KREA2
    assert S.te_id("official", lambda i: False) == S.TE                                      # niente di niente: da scaricare flux3-te


def test_te_krea2_turbo_header_vero_uguale_all_ufficiale():
    """Il TE nello snapshot di Krea 2 Turbo (header VERO, bf16 a file singolo) ha gli stessi tensori e forme dell'ufficiale FLUX.3
    a meno del prefisso `model.`; la torre visiva e lm_head non servono."""
    turbo, off = _te_hdr("krea2_turbo_te_header.json"), _te_hdr("flux3_te_header.json")
    assert len(turbo) == 713 and {v["dtype"] for v in turbo.values()} == {"BF16"}
    mapped = {W.remap_te_key(k): v for k, v in turbo.items() if W.remap_te_key(k)}
    ref = {k: v for k, v in off.items() if not k.startswith("model.visual.")}
    assert set(mapped) == set(ref) and len(ref) == 398, sorted(set(mapped) ^ set(ref))[:6]
    assert all(mapped[k]["shape"] == ref[k]["shape"] and mapped[k]["dtype"] == ref[k]["dtype"] for k in ref)


def test_resolve_te_krea2_usa_configs_dello_snapshot(monkeypatch):
    from core import weights_catalog as wc
    present = {S.DEFAULT_DIT, S.TE_KREA2, S.TE_CONFIGS, S.VAE}
    monkeypatch.setattr(wc, "is_present", lambda e, base=None: e["id"] in present)
    out = W.resolve({"flux3_te": ""}, need_pup=False)
    assert out["te_id"] == S.TE_KREA2 and out["te_src"].endswith("qwen3vl_4b_uncensored-fp8-native.safetensors")
    assert "flux-3-image-klein" in out["te_cfg"].replace("\\", "/")
    present.add(S.TE)
    assert W.resolve({"flux3_te": "official"}, need_pup=False)["te_id"] == S.TE            # scelta esplicita vince
    present.discard(S.TE_KREA2)
    present.discard(S.TE_CONFIGS)
    assert W.resolve({"flux3_te": ""}, need_pup=False)["te_id"] == S.TE                     # automatico senza Krea2: ufficiale
    present.discard(S.TE)
    with pytest.raises(RuntimeError, match=S.TE):
        W.resolve({"flux3_te": ""}, need_pup=False)


def test_resolve_te_ufficiale_riusato_da_krea2_turbo(monkeypatch, tmp_path):
    from core import model_paths as mp
    from core import weights_catalog as wc
    present = {S.DEFAULT_DIT, S.TE_CONFIGS, S.VAE, S.TE_KREA2_TURBO}
    monkeypatch.setattr(wc, "is_present", lambda e, base=None: e["id"] in present)
    snap = tmp_path / "Krea-2-Turbo"
    monkeypatch.setattr(mp, "p", lambda i, sub="": str(snap) if i == S.TE_KREA2_TURBO else str(tmp_path / i))
    with pytest.raises(RuntimeError, match=S.TE):                                     # snapshot senza il file del TE: non conta
        W.resolve({"flux3_te": "official"}, need_pup=False)
    (snap / "text_encoder").mkdir(parents=True)
    (snap / "text_encoder" / "model.safetensors").write_bytes(b"x")
    out = W.resolve({"flux3_te": "official"}, need_pup=False)
    assert out["te_id"] == S.TE_KREA2_TURBO and out["te_src"].replace("\\", "/").endswith("Krea-2-Turbo/text_encoder/model.safetensors")
    assert "flux3-te-configs" in out["te_cfg"].replace("\\", "/")                      # tokenizer/config dallo snapshot di configurazione
    assert W.resolve({"flux3_te": ""}, need_pup=False)["te_id"] == S.TE_KREA2_TURBO     # automatico senza uncensored: lo stesso


def test_fp8_linear_ripiego_dequantizzato_senza_kernel():
    """Senza `_scaled_mm` per riga (torch 2.9 su Blackwell, CPU) FP8Linear dequantizza il layer e usa F.linear: stesso risultato di un
    Linear bf16 con il peso dequantizzato."""
    if D.FP8 is None:
        pytest.skip("torch senza float8")
    torch.manual_seed(0)
    w = torch.randn(48, 64) * 0.05
    q, s = D.quantize_rows(w)
    lin = D.FP8Linear(64, 48)
    lin.weight, lin.weight_scale = q, s
    x = torch.randn(2, 5, 64)
    y = lin(x)
    ref = torch.nn.functional.linear(x.to(torch.bfloat16), (q.float() * s[:, None]).to(torch.bfloat16))
    assert y.shape == (2, 5, 48) and y.dtype == torch.bfloat16 and torch.equal(y, ref)
    assert D.FP8Linear.rowwise_ok in (True, False)
    assert lin(x.new_zeros(0, 64)).shape == (0, 48)
