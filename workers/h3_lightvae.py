"""LightVAE (lightx2v/LightVAE, ModelTC, Apache-2.0) sul VAE video di MiniMax H3.

Stessa interfaccia latente del VAE originale (nessun cambio a DiT o scheduler), architettura potata:
  · decoder Lite: 12 dei 36 blocchi transformer del decoder ViT (`kept_layers` nei metadata), rinumerati 0..11 →
    basta `decoder_num_layers=12` nella config diffusers. Misurato su RTX 5090 (clip 5 s 544p): decode 21.8 s →
    7.3 s, picco VRAM 11.8 → 5.8 GB;
  · encoder Lite: 6 dei 12 blocchi residui (`kept_residual_indices`); quelli potati diventano la sola scorciatoia
    (conv 1×1 se il blocco cambia canali, altrimenti identità), come `ResidualShortcut` del codice ufficiale.
I checkpoint (non «-comfy») sono già nei nomi di AutoencoderKLMiniMaxH3: si sostituiscono le chiavi `decoder.` /
`encoder.` del VAE originale, il resto (quant_conv, statistiche dei latenti) resta quello ufficiale.
"""
from __future__ import annotations

import json

import torch.nn as nn


def architecture(path: str) -> dict:
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        return json.loads((f.metadata() or {}).get("architecture") or "{}")


def decoder_layers(path: str) -> int:
    kept = architecture(path).get("kept_layers")
    if not kept:
        raise ValueError(f"{path}: non è un decoder LightVAE H3 (manca kept_layers)")
    return len(kept)


def kept_residuals(path: str) -> list[int]:
    kept = architecture(path).get("kept_residual_indices")
    if kept is None:
        raise ValueError(f"{path}: non è un encoder LightVAE H3 (manca kept_residual_indices)")
    return list(kept)


class ResidualShortcut(nn.Module):
    """Blocco residuo potato: resta solo la proiezione della scorciatoia (se c'era)."""

    def __init__(self, block):
        super().__init__()
        self.conv_shortcut = block.conv_shortcut

    def forward(self, hidden_states):
        return hidden_states if self.conv_shortcut is None else self.conv_shortcut(hidden_states)


def prune_encoder(vae, kept: list[int]) -> int:
    """Sostituisce i blocchi residui dell'encoder non in `kept` (indice globale nell'ordine dei down_blocks).
    Ritorna quanti ne ha potati."""
    depth = sum(len(stage.resnets) for stage in vae.encoder.down_blocks)
    if kept != sorted(set(kept)) or any(i < 0 or i >= depth for i in kept):
        raise ValueError(f"kept_residual_indices non valido per un encoder da {depth} blocchi: {kept}")
    index = pruned = 0
    for stage in vae.encoder.down_blocks:
        for local, block in enumerate(stage.resnets):
            if index not in kept:
                stage.resnets[local] = ResidualShortcut(block)
                pruned += 1
            index += 1
    return pruned


# chiavi che un file Lite porta per il suo componente (come load_h3_component del codice ufficiale); il resto
# del file (es. latents_mean/latents_std, che nel VAE diffusers sono config e non tensori) si ignora
_PARTS = {"dec": ("decoder.", "post_quant_conv."), "enc": ("encoder.", "quant_conv.")}


def merge_state(base: dict, dec: dict | None = None, enc: dict | None = None) -> dict:
    """State dict del VAE originale con decoder e/o encoder sostituiti da quelli Lite."""
    out = dict(base)
    for part, sd in (("dec", dec), ("enc", enc)):
        if sd is None:
            continue
        pre = _PARTS[part]
        mine = {k: v for k, v in sd.items() if k.startswith(pre)}
        if not any(k.startswith(pre[0]) for k in mine):
            raise ValueError(f"file LightVAE senza chiavi {pre[0]}* (componente sbagliato?)")
        for k in [k for k in out if k.startswith(pre[0]) or (k.startswith(pre[1]) and k in mine)]:
            del out[k]
        out.update(mine)
    return out
