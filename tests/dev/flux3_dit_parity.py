"""Parità del DiT nativo (workers/flux3_dit.py) con diffusers.Flux3Transformer2DModel — modello MINUSCOLO su CPU.

Va eseguito nel venv con diffusers ≥ 15/10 (E:\AI\Tools\flux3_venv): stessi pesi casuali (chiavi native → conversione ufficiale
di diffusers `convert_flux3_transformer_checkpoint_to_diffusers`), stessi ingressi, confronto delle uscite. Nessuna GPU.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "workers"))
import flux3_dit as D                                                    # noqa: E402
from diffusers import Flux3Transformer2DModel                            # noqa: E402
from diffusers.loaders.single_file_utils import convert_flux3_transformer_checkpoint_to_diffusers as conv  # noqa: E402

torch.manual_seed(0)
HID, HEADS, HD = 64, 4, 16
cfg = D.Config(hidden=HID, heads=HEADS, depth=2, depth_single=3, axes_dim=(4, 4, 4, 4),
               in_channels={"image": 128, "image_cond": 128, "text": 96}, flow_map=True)
mine = D.DiT(cfg)
for p in mine.parameters():
    torch.nn.init.normal_(p, std=0.05)
mine = mine.float().eval()
state = {k: v.clone() for k, v in mine.state_dict().items()}
ref = Flux3Transformer2DModel(in_channels=128, cond_in_channels=128, num_layers=2, num_single_layers=3,
                              attention_head_dim=HD, num_attention_heads=HEADS, joint_attention_dim=96,
                              axes_dims_rope=(4, 4, 4, 4), flow_map=True).float().eval()
missing, unexpected = ref.load_state_dict(conv(dict(state)), strict=False)
print("diffusers: mancanti", missing, "inattesi", unexpected)

Ht, Wt = 3, 4
n_img, n_txt, n_cond = Ht * Wt, 7, 6
img = torch.randn(n_img, 128)
txt = torch.randn(n_txt, 96)
cond = torch.randn(n_cond, 128)
rows, cols = torch.meshgrid(torch.arange(Ht), torch.arange(Wt), indexing="ij")
z = torch.zeros_like(rows)
img_ids = torch.stack([z, rows, cols, z], -1).reshape(-1, 4).float()
img_ids_cond = img_ids[:n_cond].clone(); img_ids_cond[:, 1:3] *= 0.5; img_ids_cond[:, 3] = -1
txt_ids = torch.stack([torch.zeros(n_txt)] * 3 + [torch.arange(n_txt).float()], -1)
t, tgt = 0.8, 0.55

packed = D.make_packed([[(txt, txt_ids), (cond, img_ids_cond), (img, img_ids)]], "cpu")
for m in D.MODES:
    packed.data[m] = packed.data[m].float(); packed.timesteps[m] = packed.timesteps[m].float()
packed.timesteps["image"].fill_(t)
with torch.no_grad():
    out_mine = mine(packed, timesteps_ref=torch.tensor([tgt]))
    out_ref = ref(hidden_states=img[None], encoder_hidden_states=txt[None], timestep=torch.tensor([t]),
                  img_ids=img_ids, txt_ids=txt_ids, image_cond_hidden_states=cond[None],
                  image_cond_ids=img_ids_cond, target_timestep=torch.tensor([tgt]), return_dict=False)[0][0]
err = (out_mine - out_ref).abs().max().item()
print("max |diff| =", err, " (scala uscita", out_ref.abs().mean().item(), ")")
assert err < 2e-4, err
# senza riferimenti
packed = D.make_packed([[(txt, txt_ids), None, (img, img_ids)]], "cpu")
for m in D.MODES:
    packed.data[m] = packed.data[m].float(); packed.timesteps[m] = packed.timesteps[m].float()
packed.timesteps["image"].fill_(t)
with torch.no_grad():
    a = mine(packed, timesteps_ref=torch.tensor([tgt]))
    b = ref(hidden_states=img[None], encoder_hidden_states=txt[None], timestep=torch.tensor([t]), img_ids=img_ids,
            txt_ids=txt_ids, target_timestep=torch.tensor([tgt]), return_dict=False)[0][0]
err2 = (a - b).abs().max().item()
print("senza riferimenti: max |diff| =", err2)
assert err2 < 2e-4, err2
print("PARITA OK")
