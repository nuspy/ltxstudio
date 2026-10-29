"""Confronto numerico WanMTModel (nostro) vs WanModel ufficiale MeiGen-AI/MultiTalk, pesi casuali minuscoli, CPU.
Eseguito come SUBPROCESS da test_wan_multitalk_native.py (stubba xfuser/xformers: non va importato nello stesso processo).
Serve il repo ufficiale clonato in E:\AI\Tools\MultiTalk_src (solo codice). Stampa `EQUIV rel=<diff relativa>`."""
import sys, types, importlib, os
import torch, torch.nn.functional as F
import diffusers, transformers

OFF = os.environ.get("LTX_MULTITALK_SRC", r"E:\AI\Tools\MultiTalk_src")
sys.path.insert(0, r"E:\ai\ltxstudio\.claude\worktrees\task20\workers")
sys.path.insert(0, r"E:\ai\ltxstudio\.claude\worktrees\task20")

def stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m

dummy = lambda *a, **k: 0
stub("xfuser"); stub("xfuser.core"); stub("xfuser.core.distributed",
     get_sequence_parallel_rank=dummy, get_sequence_parallel_world_size=lambda: 1, get_sp_group=dummy,
     init_distributed_environment=dummy, initialize_model_parallel=dummy, get_world_group=dummy)
try:
    import skimage
except Exception:
    stub("skimage", color=types.SimpleNamespace())
def mea(q, k, v, attn_bias=None, op=None):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
stub("xformers"); stub("xformers.ops", memory_efficient_attention=mea, fmha=types.SimpleNamespace(attn_bias=types.SimpleNamespace(BlockDiagonalMask=None)))
sys.modules["xformers"].ops = sys.modules["xformers.ops"]

for pk, sub in (("wan", ""), ("wan.modules", "modules"), ("wan.utils", "utils")):
    m = types.ModuleType(pk); m.__path__ = [os.path.join(OFF, "wan", sub) if sub else os.path.join(OFF, "wan")]; sys.modules[pk] = m
sys.path.insert(0, OFF)
mt = importlib.import_module("wan.modules.multitalk_model")
mt.USE_SAGEATTN = False
att = importlib.import_module("wan.modules.attention")
def flash_sdpa(q, k, v, k_lens=None, window_size=(-1, -1), **kw):
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
mt.flash_attention = flash_sdpa
mt.get_attn_map_with_target = lambda *a, **k: None

cfg = dict(model_type="i2v", in_dim=36, dim=64, ffn_dim=96, freq_dim=32, text_dim=16, out_dim=16, num_heads=4, num_layers=2, weight_init=False)
off = mt.WanModel(**cfg)
for m in off.modules():
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, std=0.05); torch.nn.init.normal_(m.bias, std=0.02)
for p in off.parameters():
    if p.dim() == 1 and p.abs().sum() == 0: torch.nn.init.normal_(p, std=0.02)
torch.nn.init.normal_(off.head.head.weight, std=0.05)
off.eval().disable_teacache()

from wan_mt.model import WanMTModel
mine = WanMTModel(dim=64, ffn_dim=96, num_heads=4, num_layers=2, text_dim=16, freq_dim=32).eval()
sd = {k: v for k, v in off.state_dict().items()}
miss, unexp = mine.load_state_dict(sd, strict=False)
print("missing", miss[:3], "unexpected", unexp[:3])

torch.manual_seed(0)
T, H, W = 5, 6, 8
x, y = torch.randn(16, T, H, W), torch.randn(20, T, H, W)
ctx, clip = torch.randn(9, 16), torch.randn(1, 257, 1280)
aud = torch.randn(1, 17, 5, 12, 768)        # 17 frame video -> 5 latenti
t = torch.tensor([637.0])
seq_len = T * (H // 2) * (W // 2)
with torch.no_grad():
    ref = off([x.clone()], t=t, context=[ctx], seq_len=seq_len, clip_fea=clip, y=[y], audio=aud, ref_target_masks=torch.ones(3, H, W))[0]
    ours = mine(x, t, ctx, clip, y, audio=aud, audio_scale=1.0)
print("ref", tuple(ref.shape), "ours", tuple(ours.shape))
d = (ref - ours).abs()
print("EQUIV rel=%.3e" % (d.max() / ref.abs().max()).item())
