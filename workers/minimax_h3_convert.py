"""MiniMax H3: pesi single-file ComfyUI → classi diffusers (transformer pruned, VAE video, text encoder).

Tutte le mappature sono VERIFICATE sui pesi pubblicati (letture HTTP range, 26/09/2026):
  · DiT pruned ComfyUI (bf16/fp8/nvfp4) ≡ `multimodalart/MiniMax-H3-Pruned` (diffusers) tensore per tensore
    (maxdiff 0.0 su table/proj/adaln; folded_bias fp16 vs fp32);
  · `qkv_proj` = [q; k; v] (coseno 1.0 con to_q/to_k/to_v ufficiali);
  · `mlp.fc1` = [gate; up] — metà INVERTITE rispetto a `ff.net.0.proj` di diffusers ([up; gate]);
    per non spezzare i blocchi nvfp4 (scale swizzled) il fc1 resta FUSO e il MLP diventa FusedGateUpSwiGLU
    (silu(prima metà) * seconda metà), che è esattamente la semantica ComfyUI;
  · il qkv resta fuso anche lui (`attn.to_qkv` + `fused_projections=True`, percorso nativo del processor).
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn

# ── transformer ────────────────────────────────────────────────────────────────────────────────
_DIT_RULES = [
    (r"^adaln_t_table$", "time_embedder.table"),
    (r"^video_patch_proj\.", "proj_in."),
    (r"^audio_patch_proj\.", "audio_proj_in."),
    (r"^condition_proj\.", "context_embedder."),
    (r"^final_layer\.video_out\.", "proj_out."),
    (r"^final_layer\.audio_out\.", "audio_proj_out."),
    (r"^final_layer\.norm\.", "norm_out.norm."),
    (r"^final_layer\.adaln_proj\.linear\.bias$", "norm_out.folded_bias"),
    (r"^final_layer\.adaln_proj\.linear\.", "norm_out.linear."),
    (r"^blocks\.(\d+)\.adaln_proj\.linear\.bias$", r"transformer_blocks.\1.adaln_proj.folded_bias"),
    (r"^blocks\.(\d+)\.", r"transformer_blocks.\1."),
    (r"^token_refiner\.blocks\.(\d+)\.", r"token_refiner.refiner_blocks.\1."),
    (r"\.attn\.q_norm\.", ".attn.norm_q."),
    (r"\.attn\.k_norm\.", ".attn.norm_k."),
    (r"\.attn\.qkv_proj\.", ".attn.to_qkv."),
    (r"\.attn\.out_proj\.", ".attn.to_out.0."),
    (r"\.mlp\.fc1\.", ".ff.gate_up."),
    (r"\.mlp\.fc2\.", ".ff.out."),
]
DIT_SKIP = ("rope.inv_freq",)            # buffer non persistente, ricalcolato dal modello


def dit_key(k: str) -> str:
    for pat, rep in _DIT_RULES:
        k = re.sub(pat, rep, k)
    return k


def fuse_dit_projections(model: nn.Module) -> None:
    """Su meta-device: qkv fuso nelle attenzioni e MLP gate/up fuso (vedi docstring del modulo)."""
    from _qlinear import FusedGateUpSwiGLU
    for name, mod in list(model.named_modules()):
        if hasattr(mod, "to_q") and hasattr(mod, "to_k") and hasattr(mod, "to_v") and hasattr(mod, "to_out"):
            q = mod.to_q
            with torch.device("meta"):
                mod.to_qkv = nn.Linear(q.in_features, 3 * q.out_features, bias=q.bias is not None)
            del mod.to_q, mod.to_k, mod.to_v
            mod.fused_projections = True
    for name, mod in list(model.named_modules()):
        ff = getattr(mod, "ff", None)
        if ff is None or not hasattr(ff, "net"):
            continue
        proj, out = ff.net[0].proj, ff.net[2]
        with torch.device("meta"):
            gu = nn.Linear(proj.in_features, proj.out_features, bias=proj.bias is not None)
        mod.ff = FusedGateUpSwiGLU(gu, out)


def dit_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if k in DIT_SKIP:
            continue
        nk = dit_key(k)
        if nk.endswith("folded_bias"):
            v = v.float()
        out[nk] = v
    return out


# ── VAE video (ComfyUI fp16 → AutoencoderKLMiniMaxH3) ──────────────────────────────────────────
VAE_SKIP = ("decoder.mask_token", "latents_mean", "latents_std")   # mean/std stanno nel config ufficiale


def vae_video_key(k: str, shape) -> list:
    """→ [(chiave_diffusers, (r0, r1) | None)]: le righe da prendere quando il tensore va diviso."""
    if k in VAE_SKIP:
        return []
    k = re.sub(r"^encoder\.down\.(\d+)\.block\.(\d+)\.nin_shortcut\.", r"encoder.down_blocks.\1.resnets.\2.conv_shortcut.", k)
    k = re.sub(r"^encoder\.down\.(\d+)\.block\.(\d+)\.", r"encoder.down_blocks.\1.resnets.\2.", k)
    k = re.sub(r"^encoder\.down\.(\d+)\.downsample\.", r"encoder.down_blocks.\1.downsamplers.0.", k)
    k = k.replace("decoder.x_embedder.", "decoder.proj_in.")
    k = k.replace(".attn.to_out.", ".attn.to_out.0.")
    k = k.replace(".ff.w2.", ".ff.net.2.")
    m = re.match(r"^(.*\.attn)\.to_qkv\.(weight|bias)$", k)
    if m:   # righe intercalate PER TESTA: [heads][q,k,v][head_dim] (VERIFICATO: coseno 1.0 riga per riga)
        return [(f"{m.group(1)}.to_{x}.{m.group(2)}", ("head3", i)) for i, x in enumerate("qkv")]
    if ".ff.w1." in k:
        return [(k.replace(".ff.w1.", ".ff.net.0.proj."), VAE_FF_ORDER)]
    return [(k, None)]


# `ff.w1` → `ff.net.0.proj`: metà invertite ([gate; up] → [up; gate]), VERIFICATO (coseno 1.0).
VAE_FF_ORDER = "swap"
VAE_HEAD_DIM = 64                        # decoder_attention_head_dim del config ufficiale


def vae_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        for nk, sl in vae_video_key(k, list(v.shape)):
            if isinstance(sl, tuple) and sl[0] == "head3":
                heads = v.shape[0] // (3 * VAE_HEAD_DIM)
                out[nk] = v.reshape(heads, 3, VAE_HEAD_DIM, *v.shape[1:])[:, sl[1]].reshape(-1, *v.shape[1:])
            elif sl == "swap":
                h = v.shape[0] // 2
                out[nk] = torch.cat([v[h:], v[:h]], 0)
            elif sl is not None:
                out[nk] = v[sl[0]:sl[1]]
            else:
                out[nk] = v
    return out


# ── text encoder Qwen3-VL-32B (ComfyUI: 50 layer, nomi «model.layers» / «visual.») ────────────────
def te_key(k: str) -> str:
    if k.startswith("model.layers.") or k.startswith("model.embed_tokens.") or k.startswith("model.norm."):
        return "model.language_model." + k[len("model."):]
    if k.startswith("visual."):
        return "model." + k
    return k


class PassthroughDecoderLayer(nn.Module):
    """51° layer vuoto: il pipeline legge `hidden_states[50]` e rifiuta uno stack di soli 50 layer (l'ultimo
    hidden state sarebbe post-norm). Col layer passante hidden_states[50] è l'uscita VERA del 50° layer."""

    def __init__(self, attention_type: str = "full_attention"):
        super().__init__()
        self.attention_type = attention_type

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states
