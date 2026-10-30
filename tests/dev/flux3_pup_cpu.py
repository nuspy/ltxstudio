"""Smoke CPU dell'upsampler FLUX.3 (workers/flux3_pup) con un Qwen3.5 MINUSCOLO a pesi casuali e i json/tokenizer/system prompt VERI
del repo ufficiale (SOLO json e template, ~20 MB, nessun peso). Prova il caricamento HF e quello GGUF (file sintetico generato dal
modello minuscolo con le stesse trasformazioni di llama.cpp), la costruzione dei messaggi t2i/i2i/bbox, il presence penalty e la
riconciliazione dei riquadri. Nessuna GPU. Venv:  E:\\AI\\Tools\\diffusers_next_venv\\Scripts\\python.exe -I tests\\dev\\flux3_pup_cpu.py
"""
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

os.environ["FLUX3_DEVICE"] = "cpu"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "workers")); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForImageTextToText

import flux3_pup as P
from core import flux3_caption as C
from core import flux3_spec as S

REPO = "https://huggingface.co/black-forest-labs/flux-3-image-klein-pup/resolve/main/"
FILES = ("config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json", "system_prompts.json",
         "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "chat_template.jinja")
cache = Path(os.environ.get("FLUX3_PUP_CACHE", tempfile.gettempdir() + "/flux3_pup_cfg"))
cache.mkdir(parents=True, exist_ok=True)
for f in FILES:
    if not (cache / f).exists():
        (cache / f).write_bytes(urllib.request.urlopen(REPO + f).read())
tmp = Path(tempfile.mkdtemp(prefix="f3pup_"))
d = tmp / "pup"
d.mkdir()
for f in FILES:
    shutil.copy(cache / f, d / f)
cfg = json.loads((cache / "config.json").read_text())
NV, NK, HD = 4, 2, 8
tc = cfg["text_config"]
tc.update(hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
          num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"], linear_num_key_heads=NK,
          linear_num_value_heads=NV, linear_key_head_dim=HD, linear_value_head_dim=HD, mtp_num_hidden_layers=0,
          rope_parameters={"mrope_interleaved": True, "mrope_section": [1, 1, 2], "partial_rotary_factor": 0.25,
                           "rope_theta": 10000, "rope_type": "default"})
cfg["vision_config"].update(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32)
(d / "config.json").write_text(json.dumps(cfg))
torch.manual_seed(0)
model = AutoModelForImageTextToText.from_config(AutoConfig.from_pretrained(str(d)), dtype=torch.bfloat16)
sd = {k: v.contiguous() for k, v in model.state_dict().items()}
print("chiavi del modello minuscolo (lingua):", sum(1 for k in sd if "visual" not in k))
save_file(sd, str(d / "model.safetensors"))
S.PUP_MAX_NEW_TOKENS = 12                                     # il modello è casuale: basta provare il giro

img = tmp / "r.png"
Image.fromarray((np.random.RandomState(0).rand(96, 128, 3) * 255).astype("uint8")).save(img)
rows = C.to_rows([{"id": "obj_1", "desc": "a balloon", "bbox": [0.1, 0.1, 0.5, 0.5]}])

model_, proc, prompts = P.load(str(d / "model.safetensors"), str(d))
for label, args in (("t2i", ("a red balloon", [], None)), ("i2i", ("make it night", [Image.open(img)], None)),
                    ("t2i_bbox", ("a poster", [], rows))):
    out = P.upsample(model_, proc, prompts, *args)
    print(f"{label:9s} -> {out[:70]!r}")
    assert isinstance(out, str) and out
print("HF OK")

# GGUF sintetico dallo stesso modello (trasformazioni DIRETTE di llama.cpp) → stesse chiavi/valori letti da flux3_gguf
import flux3_gguf_writer as TW                                  # noqa: E402  (writer GGUF dei test)
perm = np.arange(NV).reshape(NV // NK, NK).T.reshape(-1)
inv = np.argsort(perm)
key_dim = NK * HD


def hperm(x, axis=0):
    shape = list(x.shape); shape[axis:axis + 1] = [NV, x.shape[axis] // NV]
    return np.take(x.reshape(shape), inv, axis=axis).reshape(x.shape)


f32 = lambda t: t.float().numpy()
lm = "model.language_model."
tens = [("token_embd.weight", f32(sd[lm + "embed_tokens.weight"]), 0), ("output.weight", f32(sd["lm_head.weight"]), 0),
        ("output_norm.weight", f32(sd[lm + "norm.weight"]) + 1, 0)]
for i in range(4):
    L = f"{lm}layers.{i}."
    g = f"blk.{i}."
    tens += [(g + "attn_norm.weight", f32(sd[L + "input_layernorm.weight"]) + 1, 0),
             (g + "post_attention_norm.weight", f32(sd[L + "post_attention_layernorm.weight"]) + 1, 0),
             (g + "ffn_gate.weight", f32(sd[L + "mlp.gate_proj.weight"]), 0), (g + "ffn_up.weight", f32(sd[L + "mlp.up_proj.weight"]), 0),
             (g + "ffn_down.weight", f32(sd[L + "mlp.down_proj.weight"]), 0)]
    if i < 3:
        qkv = f32(sd[L + "linear_attn.in_proj_qkv.weight"]).copy(); qkv[2 * key_dim:] = hperm(qkv[2 * key_dim:])
        conv = f32(sd[L + "linear_attn.conv1d.weight"])[:, 0, :].copy(); conv[2 * key_dim:] = hperm(conv[2 * key_dim:])
        tens += [(g + "attn_qkv.weight", qkv, 0), (g + "attn_gate.weight", hperm(f32(sd[L + "linear_attn.in_proj_z.weight"])), 0),
                 (g + "ssm_alpha.weight", hperm(f32(sd[L + "linear_attn.in_proj_a.weight"])), 0),
                 (g + "ssm_beta.weight", hperm(f32(sd[L + "linear_attn.in_proj_b.weight"])), 0),
                 (g + "ssm_a", hperm(-np.exp(f32(sd[L + "linear_attn.A_log"]))), 0),
                 (g + "ssm_dt.bias", hperm(f32(sd[L + "linear_attn.dt_bias"])), 0), (g + "ssm_conv1d.weight", conv, 0),
                 (g + "ssm_norm.weight", f32(sd[L + "linear_attn.norm.weight"]), 0),
                 (g + "ssm_out.weight", hperm(f32(sd[L + "linear_attn.out_proj.weight"]), axis=1), 0)]
    else:
        tens += [(g + "attn_q.weight", f32(sd[L + "self_attn.q_proj.weight"]), 0), (g + "attn_k.weight", f32(sd[L + "self_attn.k_proj.weight"]), 0),
                 (g + "attn_v.weight", f32(sd[L + "self_attn.v_proj.weight"]), 0), (g + "attn_output.weight", f32(sd[L + "self_attn.o_proj.weight"]), 0),
                 (g + "attn_q_norm.weight", f32(sd[L + "self_attn.q_norm.weight"]) + 1, 0),
                 (g + "attn_k_norm.weight", f32(sd[L + "self_attn.k_norm.weight"]) + 1, 0)]
kv = {"general.architecture": "qwen35", "qwen35.block_count": 4, "qwen35.nextn_predict_layers": 0,
      "qwen35.ssm.time_step_rank": NV, "qwen35.ssm.group_count": NK, "qwen35.ssm.state_size": HD}
TW.gguf_write(tmp / "pup.gguf", kv, tens)
import flux3_gguf as G
st = G.read_state(str(tmp / "pup.gguf"), "cpu")
lang = {k: v for k, v in sd.items() if "visual" not in k}
assert set(st) == set(lang), sorted(set(st) ^ set(lang))[:6]
worst = max(float((st[k].float() - lang[k].float()).abs().max()) for k in lang)
print("GGUF -> HF: tutti i", len(lang), "tensori, scarto massimo", worst)
assert worst < 0.02
model2, proc2, prompts2 = P.load(str(tmp / "pup.gguf"), str(d))
out = P.upsample(model2, proc2, prompts2, "a red balloon", [Image.open(img)], None)   # senza visione: i riferimenti vengono scartati, t2i
print("GGUF generazione OK:", repr(out[:50]))
print("PUP CPU OK")
shutil.rmtree(tmp, ignore_errors=True)
