"""Smoke END-TO-END su CPU del worker FLUX.3 (workers/flux3_infer.run) con modelli MINUSCOLI a pesi casuali.

Nessuna GPU e nessun peso vero: text encoder Qwen3-VL ridotto (tokenizer e chat template VERI scaricati dal mirror ungated
ml-resources, ~11 MB), DiT e VAE minuscoli. Verifica l'intera sequenza (caption → TE → riferimenti → VAE → DiT → decodifica →
PNG) per t2i, edit con 1 riferimento (crop del padding), multi-riferimento, riquadri di layout e di edit, inpaint con maschera.
Va eseguito nel venv del worker:  E:\\AI\\Tools\\diffusers_next_venv\\Scripts\\python.exe -I tests\\dev\\flux3_cpu_e2e.py
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "workers"))
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoConfig, Qwen3VLForConditionalGeneration

import flux3_dit as D
import flux3_infer as W
from core import flux3_spec as S
import core.resources as R

R.require = lambda **k: None                       # nessuna lettura della GPU: il test non la usa
R.start_commit_guard = lambda *a, **k: None

MIRROR = "https://huggingface.co/ml-resources/flux-3-image-klein/resolve/main/text_encoder/"
FILES = ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "chat_template.json",
         "preprocessor_config.json", "video_preprocessor_config.json", "config.json", "generation_config.json")
tmp = Path(tempfile.mkdtemp(prefix="f3e2e_"))
cache = Path(os.environ.get("FLUX3_TOK_CACHE", tmp / "tok"))
cache.mkdir(parents=True, exist_ok=True)
for f in FILES:
    if not (cache / f).exists():
        (cache / f).write_bytes(urllib.request.urlopen(MIRROR + f).read())
te_dir = tmp / "te"
te_dir.mkdir()
for f in FILES:
    shutil.copy(cache / f, te_dir / f)
cfg = json.loads((cache / "config.json").read_text())
cfg["text_config"].update(hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                          num_hidden_layers=33, rope_scaling={"mrope_interleaved": True, "mrope_section": [1, 1, 2],
                                                              "rope_type": "default"})
cfg["vision_config"].update(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32,
                            deepstack_visual_indexes=[0])
(te_dir / "config.json").write_text(json.dumps(cfg))
torch.manual_seed(0)
te = Qwen3VLForConditionalGeneration._from_config(AutoConfig.from_pretrained(str(te_dir)), dtype=torch.bfloat16)
save_file({k: v.contiguous() for k, v in te.state_dict().items() if k != "lm_head.weight"}, str(te_dir / "model.safetensors"))

TINY = D.Config(hidden=64, heads=4, depth=2, depth_single=3, axes_dim=(4, 4, 4, 4),
                in_channels={"image": 128, "image_cond": 128, "text": 8 * 32}, flow_map=True)


def tiny_dit(_path):
    torch.manual_seed(1)
    m = D.DiT(TINY)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.2)
    m = m.to(torch.bfloat16).eval()
    m._n_fp8 = 0
    return m


def tiny_vae(_path):
    from diffusers import AutoencoderKLFlux2
    torch.manual_seed(2)
    return AutoencoderKLFlux2.from_config(dict(S.VAE_CONFIG, block_out_channels=[16, 16, 16, 16], norm_num_groups=4)).float().eval()


W.resolve = lambda p, need_pup: {"dit": "-", "te": str(te_dir), "te_dir": str(te_dir), "vae": "-"}
W.load_dit = tiny_dit
W.load_vae = tiny_vae
imgs = []
for i, (w, h, c) in enumerate(((100, 70, (200, 30, 30)), (96, 64, (30, 200, 30)), (80, 80, (30, 30, 200)))):
    p = tmp / f"ref{i}.png"
    Image.new("RGB", (w, h), c).save(p)
    imgs.append(str(p))
mask = tmp / "mask.png"
m = np.zeros((70, 100), "uint8"); m[:, :50] = 255
Image.fromarray(m).save(mask)


def go(name, **p):
    out = tmp / f"{name}.png"
    p.setdefault("steps", 4); p.setdefault("seed", 7); p.setdefault("flux3_pup", False)
    r = W.run(dict(p, output=str(out)))
    im = Image.open(r)
    print(f"{name:12s} -> {im.size} {im.mode}")
    return im


a = go("t2i", prompt="a red balloon over the sea", width=128, height=96)
assert a.size == (128, 96)
b = go("t2i_seed", prompt="a red balloon over the sea", width=128, height=96)
assert (np.asarray(a) == np.asarray(b)).all()                                         # stesso seed → stesso risultato
c = go("t2i_other", prompt="a red balloon over the sea", width=128, height=96, seed=8)
assert not (np.asarray(a) == np.asarray(c)).all()
e1 = go("edit1", prompt="make it night", ref_images=[imgs[0]])
assert e1.size == (100, 70)                                                           # il padding del canvas è ritagliato via
e2 = go("edit1_wh", prompt="make it night", ref_images=[imgs[0]], width=128, height=96)
assert e2.size == (128, 96)
e3 = go("multi", prompt="put the second image next to the first image", ref_images=imgs, width=96, height=64)
assert e3.size == (96, 64)
l1 = go("layout", prompt="a poster with <b1>", width=128, height=96,
        boxes=[{"id": "b1", "desc": "a red balloon", "bbox": [0.1, 0.1, 0.6, 0.5]}])
ed = go("editbox", prompt="add a hat", ref_images=[imgs[0], imgs[1]], width=96, height=64,
        boxes=[{"id": "hat_1", "desc": "a straw hat", "mode": "new", "tgt_bbox": [0.0, 0.2, 0.3, 0.6]},
               {"id": "x_1", "desc": "", "mode": "remove", "ref": 2, "src_bbox": [0.2, 0.2, 0.5, 0.5]}])
inp = go("inpaint", prompt="paint it blue", init_image=imgs[0], mask_path=str(mask))
assert inp.size == (100, 70)
arr = np.asarray(inp).astype(int); orig = np.asarray(Image.open(imgs[0]).convert("RGB")).astype(int)
assert np.abs(arr[:, 60:] - orig[:, 60:]).max() == 0                                  # fuori maschera = originale al pixel
assert np.abs(arr[:, :40] - orig[:, :40]).max() > 0                                   # dentro maschera = rigenerato
# errori chiari
for bad, msg in (({"prompt": "x", "boxes": [{"id": "1x", "desc": "d", "bbox": [0, 0, 1, 1]}]}, "Riquadri non validi"),
                 ({"prompt": "x", "ref_images": imgs * 4}, "al massimo")):
    try:
        W.run(dict(bad, flux3_pup=False))
    except RuntimeError as ex:
        assert msg in str(ex), ex
    else:
        raise AssertionError(msg)
print("E2E CPU OK")
shutil.rmtree(tmp, ignore_errors=True)
