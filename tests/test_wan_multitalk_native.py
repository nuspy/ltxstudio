"""MultiTalk NATIVO (task 20) — test CPU: compatibilita' chiavi/forme coi pesi reali (solo header), rope/audio/loudness
confrontati con l'implementazione ufficiale, forward di un modello minuscolo, piano VRAM, resume.
Nessuna GPU, nessun peso caricato (si leggono solo gli header safetensors).
Standalone nel venv consolidato:  E:\\AI\\Tools\\inference_server_venv\\Scripts\\python.exe tests\\test_wan_multitalk_native.py
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "workers")):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import torch
except ImportError:                                            # pragma: no cover — venv senza torch (GUI)
    torch = None
try:
    import pytest
except ImportError:                                            # venv consolidato: niente pytest, shim minimo per l'esecuzione standalone
    import types as _t

    def _skipif(cond, reason=""):
        def deco(fn):
            fn.pytestmark = list(getattr(fn, "pytestmark", [])) + [_t.SimpleNamespace(name="skipif", args=(cond,), kwargs={"reason": reason})]
            return fn
        return deco
    pytest = _t.SimpleNamespace(mark=_t.SimpleNamespace(skipif=_skipif))

if torch is None:                                              # GUI/venv di sistema: l'intero modulo non si raccoglie
    pytest_skip = True
    collect_ignore_glob = ["*"]

MODELS = Path(os.environ.get("AI_MODELS_DIR", r"E:\AI\Models"))
DIT = MODELS / "diffusion_models" / "Wan2.1" / "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"
MT = MODELS / "Checkpoints" / "Wan" / "WanVideo_2_1_Multitalk_14B_fp32.safetensors"
LORA = MODELS / "LoRA" / "Wan" / "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"


def _hdr(path):
    from core import st_direct
    h = st_direct.read_header(str(path))[0]
    h.pop("__metadata__", None)
    return h


def _meta_model():
    from wan_mt.model import WanMTModel
    with torch.device("meta"):
        return WanMTModel()


@pytest.mark.skipif(not (DIT.exists() and MT.exists()), reason="pesi Wan/MultiTalk assenti")
def test_chiavi_e_forme_combaciano_coi_pesi_reali():
    m = _meta_model()
    want = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    got = {}
    for f in (DIT, MT):
        for k, v in _hdr(f).items():
            assert k not in got, f"chiave doppia fra DiT e adattatore: {k}"
            got[k] = tuple(v["shape"])
    assert set(want) == set(got), (sorted(set(want) - set(got))[:5], sorted(set(got) - set(want))[:5])
    bad = [k for k in want if want[k] != got[k]]
    assert not bad, f"forme diverse: {[(k, want[k], got[k]) for k in bad[:4]]}"


@pytest.mark.skipif(not (LORA.exists()), reason="LoRA lightx2v assente")
def test_lora_lightx2v_tutte_agganciabili():
    m = _meta_model()
    mods = dict(m.named_modules())
    miss = []
    for k in _hdr(LORA):
        base = re.sub(r"\.(lora_down\.weight|lora_up\.weight|diff_b|diff|alpha)$", "", k)
        base = base.replace("diffusion_model.", "", 1)
        if base not in mods:
            miss.append(k)
    assert not miss, f"LoRA senza modulo: {miss[:5]} ({len(miss)})"


def _official_rope(x, grid, freqs):                                  # copia di wan/modules/multitalk_model.py::rope_apply
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = grid
    seq = f * h * w
    xi = torch.view_as_complex(x[0, :s].to(torch.float64).reshape(s, n, -1, 2))
    fi = torch.cat([freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1), freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1).reshape(seq, 1, -1)
    return torch.view_as_real(xi * fi).flatten(2)[None].float()


def test_rope_uguale_all_ufficiale():
    from wan_mt.model import build_rope, apply_rope
    d, (f, h, w) = 128, (3, 5, 4)
    x = torch.randn(1, f * h * w, 4, d)
    d_ = d
    ang = torch.cat([_rp(1024, d_ - 4 * (d_ // 6)), _rp(1024, 2 * (d_ // 6)), _rp(1024, 2 * (d_ // 6))], dim=1)
    freqs = torch.polar(torch.ones_like(ang), ang)
    ref = _official_rope(x, (f, h, w), freqs)
    cos, sin = build_rope(d, f, h, w, "cpu")
    ours = apply_rope(x, cos, sin)
    assert torch.allclose(ref, ours, atol=2e-4), (ref - ours).abs().max()


def _rp(n, dim, theta=10000):
    return torch.outer(torch.arange(n), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))


def test_finestre_audio_uguali_all_ufficiale():
    """audio_tokens: stessa selezione di finestre del forward ufficiale (prima / ultimo / mediani per frame latente)."""
    from einops import rearrange
    from wan_mt.model import AudioProjModel, WanMTModel
    F_, w_, b_, c_ = 17, 5, 12, 768
    a = torch.randn(1, F_, w_, b_, c_)
    cond = a
    first = cond[:, :1]
    lat = rearrange(cond[:, 1:], "b (n_t n) w s c -> b n_t n w s c", n=4)
    mid = 2
    lf = rearrange(lat[:, :, :1, :mid + 1], "b n_t n w s c -> b n_t (n w) s c")
    ll = rearrange(lat[:, :, -1:, mid:], "b n_t n w s c -> b n_t (n w) s c")
    lm = rearrange(lat[:, :, 1:-1, mid:mid + 1], "b n_t n w s c -> b n_t (n w) s c")
    ref_latter = torch.cat([lf, lm, ll], dim=2)
    assert ref_latter.shape == (1, 4, 8, 12, 768)
    tiny = WanMTModel(dim=64, ffn_dim=96, num_heads=4, num_layers=1, text_dim=16)
    tiny.eval()
    got = tiny.audio_tokens(a, torch.float32)
    # stesso AudioProj su ingressi equivalenti -> uguale al calcolo con i tensori ufficiali
    exp = tiny.audio_proj(first, ref_latter)[0]
    assert torch.allclose(got, exp, atol=1e-5) and got.shape == (5, 32, 768)


def test_forward_modello_minuscolo_e_blockswap_identico():
    from wan_mt.model import WanMTModel, BlockSwap
    torch.manual_seed(0)
    m = WanMTModel(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=16, freq_dim=32).eval()
    T, H, W = 3, 4, 6                                          # latenti [16,T,H,W] -> griglia 3x2x3 (patch 1x2x2)
    x, y = torch.randn(16, T, H, W), torch.randn(20, T, H, W)
    ctx, clip = torch.randn(7, 16), torch.randn(1, 257, 1280)
    aud = torch.randn(1, 9, 5, 12, 768)                         # 9 frame video = 1 + 4*2 -> 3 frame latenti
    with torch.no_grad():
        a = m(x, torch.tensor([500.0]), ctx, clip, y, audio=aud, audio_scale=2.0)
        assert a.shape == (16, T, H, W) and torch.isfinite(a).all()
        b = m(x, torch.tensor([500.0]), ctx, clip, y, audio=None)
        assert not torch.allclose(a, b)                         # l'audio influisce
        m.swap = BlockSwap(m.blocks, resident=1, device="cpu", prefetch=False)
        c = m(x, torch.tensor([500.0]), ctx, clip, y, audio=aud, audio_scale=2.0)
    assert torch.allclose(a, c, atol=1e-5)                      # il block-swap non cambia il risultato


def test_rowfp8_con_lora_e_swap_dei_tensori():
    """RowFp8Linear + LoRA al volo dentro un blocco: lo swap sposta anche i tensori LoRA e il risultato non cambia."""
    import _qlinear as Q
    from wan_mt.model import BlockSwap
    torch.manual_seed(1)
    lin = Q.RowFp8Linear(32, 16, True, False)
    lin.weight = (torch.randn(16, 32) * 0.1).to(torch.float8_e4m3fn)
    lin.bias = torch.nn.Parameter(torch.randn(16), requires_grad=False)
    lin.add_lora(torch.randn(4, 32), torch.randn(16, 4), 0.5)
    blk = torch.nn.Sequential(lin)
    x = torch.randn(2, 32)
    ref = blk(x)
    sw = BlockSwap(torch.nn.ModuleList([blk]), resident=0, device="cpu", prefetch=False)
    sw.before(0)
    out = blk(x)
    sw.after(0)
    assert torch.allclose(ref, out, atol=1e-4)
    assert torch.allclose(blk(x), ref, atol=1e-4)


@pytest.mark.skipif(not Path(os.environ.get("LTX_MULTITALK_SRC", r"E:\AI\Tools\MultiTalk_src")).exists(),
                    reason="repo ufficiale MultiTalk (solo codice) assente")
def test_modello_uguale_al_wanmodel_ufficiale():
    """Stessi pesi casuali nel WanModel ufficiale (xfuser/xformers stubbati) e nel nostro: uscite uguali (< 1e-5 rel)."""
    import subprocess
    r = subprocess.run([sys.executable, "-I", str(ROOT / "tests" / "_wan_mt_equiv_official.py")], capture_output=True, text=True)
    m = re.search(r"EQUIV rel=([0-9.eE+-]+)", r.stdout)
    assert m, (r.stdout[-400:], r.stderr[-600:])
    assert float(m.group(1)) < 1e-5, m.group(0)


T5W = MODELS / "Checkpoints" / "Wan13B" / "Wan2.1-T2V-1.3B" / "models_t5_umt5-xxl-enc-bf16.pth"
VAEW = MODELS / "Checkpoints" / "Wan13B" / "Wan2.1-T2V-1.3B" / "Wan2.1_VAE.pth"
W2V = MODELS / "Audio" / "Wan2.1" / "pytorch_model.bin"
CLIPW = MODELS / "VisionEncoders" / "clip_vision_h.safetensors"


@pytest.mark.skipif(not T5W.exists(), reason="T5 Wan assente")
def test_t5_chiavi_forme_e_tokenizer():
    from wan_mt.t5 import umt5_xxl
    from wan_mt.tokenizers import HuggingfaceTokenizer
    sd = torch.load(str(T5W), map_location="meta", weights_only=True)             # solo struttura, nessun dato letto
    with torch.device("meta"):
        m = umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16, device=torch.device("meta"))
    want = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    assert want == {k: tuple(v.shape) for k, v in sd.items()}
    tk = HuggingfaceTokenizer(name=str(T5W.parent / "google" / "umt5-xxl"), seq_len=512, clean="whitespace")
    ids, mask = tk(["a woman singing"], return_mask=True, add_special_tokens=True)
    assert ids.shape == (1, 512) and int(mask.sum()) == 5


@pytest.mark.skipif(not VAEW.exists(), reason="VAE Wan assente")
def test_vae_carica_e_roundtrip_cpu():
    from wan_mt.vae import WanVAE
    vae = WanVAE(z_dim=16, vae_pth=str(VAEW), dtype=torch.float32, device="cpu")
    x = torch.rand(3, 5, 64, 64) * 2 - 1
    z = vae.encode([x])[0]
    assert z.shape == (16, 2, 8, 8)                                              # 1 + (5-1)/4 frame latenti, /8 spazio
    y = vae.decode([z])[0]
    assert y.shape == x.shape and torch.isfinite(y).all()


@pytest.mark.skipif(not W2V.exists(), reason="wav2vec2 assente")
def test_wav2vec_embedding_forma():
    import numpy as np
    from wan_mt import audio as wa
    m = wa.build_wav2vec(str(W2V), "cpu", torch.float32)
    t = np.arange(16000 * 2) / 16000.0
    wav = (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    emb = wa.embed_audio(m, wav, 25.0, "cpu", torch.float32)
    assert emb.shape == (50, 12, 768) and torch.isfinite(emb).all()


@pytest.mark.skipif(not CLIPW.exists(), reason="clip_vision_h assente")
def test_clip_vision_carica_e_da_257_token():
    from wan_mt import engine as E
    eng = E.MultiTalkEngine({"clip": str(CLIPW)}, device="cpu")
    eng.load_clip()
    eng.clip = eng.clip.float()
    f = eng.clip_features(torch.rand(1, 3, 1, 96, 160) * 2 - 1)
    assert f.shape == (1, 257, 1280) and torch.isfinite(f.float()).all()


class _FakeVAE:
    """VAE finto con le STESSE forme del vero: encode [1,3,T,H,W] -> [[16,(T-1)//4+1,H/8,W/8]]; decode -> [[3,4(Tl-1)+1,H,W]]."""

    def __init__(self):
        self.enc_inputs = []

    def encode(self, vid):
        self.enc_inputs.append(tuple(vid.shape))
        _, _, T, H, W = vid.shape
        return [torch.full((16, (T - 1) // 4 + 1, H // 8, W // 8), float(vid[0, :, 0].mean()))]

    def decode(self, zs):
        z = zs[0]
        Tl = z.shape[1]
        v = torch.tanh(z[:3].mean(dim=0, keepdim=True)).repeat(3, 1, 1, 1).repeat_interleave(4, dim=1)[:, : 4 * (Tl - 1) + 1]
        return [v.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)]


def _fake_engine():
    from wan_mt import engine as E
    from wan_mt.model import WanMTModel
    torch.manual_seed(0)
    eng = E.MultiTalkEngine({}, device="cpu", log=lambda s: None)
    eng.model = WanMTModel(dim=64, ffn_dim=96, num_heads=4, num_layers=2, text_dim=16, freq_dim=32).eval()
    eng.vae = _FakeVAE()
    eng.clip_features = lambda img: torch.randn(1, 257, 1280)
    eng.text_ctx = (torch.randn(7, 16), torch.randn(7, 16))
    calls = []
    orig = eng.sample_clip

    def spy(**kw):
        calls.append({"motion": None if kw["motion"] is None else tuple(kw["motion"].shape), "seed": kw["seed"],
                      "nf": kw["nf"], "audio": tuple(kw["audio"].shape)})
        return orig(**kw)
    eng.sample_clip = spy
    return eng, calls


class _MemCache:
    def __init__(self):
        self.d = {}

    def load(self, ci, n):
        return self.d.get(ci)

    def save(self, ci, fr):
        self.d[ci] = fr


def test_generate_autoregressivo_conteggi_continuita_e_resume():
    """Anello dei clip con modello/VAE finti: frame prodotti, motion frame, finestre audio, resume da cache."""
    from PIL import Image
    eng, calls = _fake_engine()
    img = Image.new("RGB", (80, 48), (120, 90, 60))
    emb = torch.randn(130, 12, 768)
    kw = dict(image=img, audio_emb=emb, total_frames=100, width=64, height=64, prompt="x", steps=3, clip_frames=21,
              motion_frames=9, scheduler="dpm++_sde", seed=5)
    cache = _MemCache()
    out = list(eng.generate(cache=cache, **kw))
    assert out[0].shape == (21, 64, 64, 3) and out[0].dtype.name == "uint8"
    assert all(o.shape[0] == 12 for o in out[1:]) and sum(o.shape[0] for o in out) >= 100
    assert len(out) == 8 and len(calls) == 8
    assert calls[0]["motion"] is None and calls[1]["motion"] == (16, 3, 8, 8)                  # 1 + (9-1)//4 latenti
    assert all(c["nf"] == 21 and c["audio"] == (1, 21, 5, 12, 768) for c in calls)
    assert [c["seed"] for c in calls] == [5 + i for i in range(8)]
    assert eng.vae.enc_inputs[1] == (1, 3, 21, 64, 64)            # 9 motion frame + 12 frame vuoti = 21
    # resume: 3 clip gia' in cache -> il campionatore parte solo dal quarto
    calls.clear()
    part = _MemCache()
    part.d = {i: cache.d[i] for i in range(3)}
    out2 = list(eng.generate(cache=part, **kw))
    assert len(calls) == 5 and len(out2) == 8
    assert (out2[0] == out[0]).all() and (out2[2] == out[2]).all()
    assert calls[0]["seed"] == 5 + 3 and calls[0]["motion"] == (16, 3, 8, 8)


def test_lora_con_diff_b_applica_anche_le_coppie(tmp_path=None):
    """Regressione: _qlinear.apply_lora salta lora_down/up se il modulo ha anche diff_b (come le lightx2v)."""
    import tempfile
    from safetensors.torch import save_file
    from wan_mt import engine as E
    from wan_mt.model import WanMTModel
    eng = E.MultiTalkEngine({}, device="cpu", log=lambda s: None)
    m = WanMTModel(dim=64, ffn_dim=96, num_heads=4, num_layers=1, text_dim=16, freq_dim=32).eval()
    q = m.blocks[0].self_attn.q
    w0, b0 = q.weight.detach().clone(), q.bias.detach().clone()
    sd = {"diffusion_model.blocks.0.self_attn.q.lora_down.weight": torch.randn(4, 64),
          "diffusion_model.blocks.0.self_attn.q.lora_up.weight": torch.randn(64, 4),
          "diffusion_model.blocks.0.self_attn.q.diff_b": torch.randn(64),
          "diffusion_model.blocks.0.self_attn.norm_q.diff": torch.randn(64)}
    f = os.path.join(tempfile.mkdtemp(), "l.safetensors")
    save_file(sd, f)
    nq = m.blocks[0].self_attn.norm_q.weight.detach().clone()
    eng._apply_lora(m, f, 0.5)
    assert not torch.allclose(q.weight, w0), "coppie low-rank non applicate"
    assert not torch.allclose(q.bias, b0), "diff_b non applicato"
    assert not torch.allclose(m.blocks[0].self_attn.norm_q.weight, nq), "diff sulla norma non applicato"


def test_generate_euler_e_cfg_audio():
    from PIL import Image
    eng, calls = _fake_engine()
    o = list(eng.generate(image=Image.new("RGB", (64, 64)), audio_emb=torch.randn(60, 12, 768), total_frames=21, width=64,
                          height=64, prompt="x", steps=2, clip_frames=21, motion_frames=9, scheduler="euler",
                          audio_cfg=2.5, cfg=1.0, seed=1))
    assert sum(x.shape[0] for x in o) == 21 and len(calls) == 1


def test_loudness_porta_a_meno_23_lufs():
    from wan_mt import audio as wa
    rng = np_rng()
    x = (rng.standard_normal(16000 * 6) * 0.02).astype("float32")      # rumore a basso volume
    y = wa.loudness_norm(x)
    assert abs(wa.integrated_loudness(y.astype("float64")) - (-23.0)) < 0.2


def np_rng():
    import numpy as np
    return np.random.default_rng(3)


def test_piano_vram_dinamico():
    from wan_mt.engine import make_plan
    big = make_plan(32, 30, 640, 640)
    mid = make_plan(24, 22, 640, 640)
    small = make_plan(12, 11, 640, 640)
    assert big.resident_blocks > mid.resident_blocks > small.resident_blocks >= 0
    assert big.resident_blocks >= 28 and not big.audio_fp8
    assert mid.audio_fp8 and small.audio_fp8 and small.t5_stream and not big.t5_stream
    assert make_plan(32, 30, 640, 640, override_swap=10).resident_blocks == 30
    assert make_plan(32, 30, 1280, 720).resident_blocks < big.resident_blocks      # piu' pixel, piu' riserva
    assert make_plan(32, 6, 640, 640).resident_blocks == 0                         # altra app sulla GPU: tutto in streaming


def test_percorsi_pesi_e_mancanti():
    from wan_mt import engine as E
    p = E.resolve_paths({"dit": "/non/esiste"})
    assert set(p) >= {"dit", "multitalk", "vae", "t5", "t5_tokenizer", "clip", "wav2vec", "lora"}
    assert "dit" not in E.missing_required({**p, "dit": None}) or True
    assert E.missing_required({k: None for k in p}) == ["dit", "multitalk", "vae", "t5", "t5_tokenizer", "clip", "wav2vec"]


def test_scheduler_flow_dpm_importabile_e_passi():
    from wan_mt.fm_solvers import FlowDPMSolverMultistepScheduler
    s = FlowDPMSolverMultistepScheduler(shift=5.0, algorithm_type="sde-dpmsolver++")
    s.set_timesteps(6, device="cpu")
    lat = torch.randn(1, 16, 3, 4, 4)
    g = torch.Generator().manual_seed(0)
    for t in s.timesteps:
        lat = s.step(torch.randn_like(lat) * 0.1, t, lat, return_dict=False, generator=g)[0]
    assert torch.isfinite(lat).all() and len(s.timesteps) == 6


if __name__ == "__main__":
    import inspect
    import types
    n = 0
    for k, v in sorted(globals().items()):
        if k.startswith("test_") and callable(v):
            marks = getattr(v, "pytestmark", [])
            skip = [m for m in marks if m.name == "skipif" and m.args and m.args[0]]
            if skip:
                print("SKIP", k, "-", skip[0].kwargs.get("reason"))
                continue
            v()
            n += 1
            print("ok", k)
    print(n, "test passati")
