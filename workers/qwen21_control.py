"""ControlNet-Union «Fun» di Qwen-Image 2.1 — port NATIVO del ramo di controllo di VideoX-Fun.

Riferimento: aigc-apps/VideoX-Fun `videox_fun/models/qwenimage21_transformer2d_control.py` e
`pipeline/pipeline_qwenimage21_control.py` (alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union).
Design VACE: una catena PARALLELA di 16 blocchi (uguali a quelli del DiT, `QwenImage21TransformerBlock`) legge il
condizionamento di controllo e produce un «suggerimento» per ogni blocco pari del DiT (0, 2, …, 30), sommato
all'uscita di quel blocco × `control_context_scale`. Il primo blocco di controllo fonde il flusso del DiT tramite
`before_proj`; ogni blocco emette il suo suggerimento con `after_proj`.

Condizionamento (129 canali, allineato ai token dell'immagine finale): latenti VAE dell'immagine di controllo (64) |
maschera di inpaint (1) | latenti dell'immagine mascherata (64). Per il controllo puro gli ultimi 65 canali sono zero.
Come nella pipeline ufficiale: niente cache KV (il controllo cambia a ogni passo) e niente immagini di riferimento
(il condizionamento si allinea alla sola immagine finale). Il DiT resta quello di diffusers: si sostituisce solo la
`forward` dell'istanza con una copia che aggiunge il ramo di controllo.
"""
from __future__ import annotations

import math
import types

CONTROL_LAYERS = tuple(range(0, 32, 2))
CONTROL_IN_DIM = 129


def _log(m: str) -> None:
    print(f"[qwen21-cn] {m}", flush=True)


def build_adapter(transformer):
    """Ramo di controllo vuoto (meta) con la geometria del DiT."""
    import torch.nn as nn
    from accelerate import init_empty_weights
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock
    c = transformer.config
    dim = c.num_attention_heads * c.attention_head_dim

    class ControlBlock(QwenImage21TransformerBlock):
        def __init__(self, first: bool):
            super().__init__(dim, c.num_attention_heads, c.attention_head_dim, c.mlp_ratio, c.eps)
            if first:
                self.before_proj = nn.Linear(dim, dim)
            self.after_proj = nn.Linear(dim, dim)

    with init_empty_weights():
        adapter = nn.Module()
        adapter.control_blocks = nn.ModuleList([ControlBlock(i == 0) for i in range(len(CONTROL_LAYERS))])
        adapter.control_img_in = nn.Linear(CONTROL_IN_DIM, dim)
    return adapter


def attach(transformer, weights: str):
    """Carica il ramo di controllo (file nativo INT8 ConvRot) e aggancia la forward con controllo al DiT."""
    import _qlinear as Q
    adapter = build_adapter(transformer)
    sd = Q.quantize_from_state_dict(adapter, Q.read_safetensors(weights, "cuda"))
    miss, unex = Q.load_into(adapter, sd)
    if miss or unex:
        raise RuntimeError(f"ControlNet: chiavi mancanti {miss[:6]} / inattese {unex[:6]}")
    del sd
    adapter = adapter.to("cuda").eval()
    transformer._ctrl_adapter = adapter
    transformer._ctrl = None                       # (contesto impacchettato, scala) impostati per la generazione
    transformer.forward = types.MethodType(_forward_with_control, transformer)
    _log(f"ramo di controllo caricato ({len(adapter.control_blocks)} blocchi)")
    return adapter


def set_context(transformer, context, scale: float) -> None:
    transformer._ctrl = (context, float(scale))


def build_context(pipe, control_img, width: int, height: int, generator):
    """Immagine di controllo (già alla forma finale) → condizionamento impacchettato (1, lh·lw, 129)."""
    import torch
    dev, dt = "cuda", torch.bfloat16
    x = pipe.image_processor.preprocess(control_img.convert("RGB"), height=height, width=width)
    x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)          # il VAE di 2.1 legge RGBA: alfa opaco
    ctrl = pipe._encode_vae_image(x.unsqueeze(2).to(dev, dt), generator)        # (1, 64, 1, lh, lw)
    _, ch, _, lh, lw = ctrl.shape
    mask_latent = torch.zeros(1, 1, 1, lh, lw, device=dev, dtype=dt)            # 1 - maschera (nessun inpaint)
    inpaint_latent = torch.zeros_like(ctrl)                                     # nessuna immagine mascherata
    ctx = torch.cat([ctrl, mask_latent, inpaint_latent], dim=1)                 # (1, 129, 1, lh, lw)
    return pipe._pack_latents(ctx, 1, ctx.shape[1], lh, lw).to(dev, dt)


def preprocess(path: str, kind: str, width: int, height: int, depth_dir: str = ""):
    """Foto → mappa di controllo alla forma finale (ritaglio centrato, mai stirata).
    kind: canny | lineart | gray | depth | none (l'immagine è già una mappa: pose, scribble, HED…)."""
    import numpy as np
    from PIL import Image, ImageOps
    img = ImageOps.fit(Image.open(path).convert("RGB"), (width, height), Image.LANCZOS)
    if kind in ("", "none"):
        return img
    if kind == "gray":
        return img.convert("L").convert("RGB")
    import cv2
    a = np.asarray(img)
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    if kind == "canny":
        e = cv2.Canny(gray, 100, 200)
        return Image.fromarray(e).convert("RGB")
    if kind == "lineart":                      # contorni bianchi su nero, come i preprocessori lineart
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        e = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 6)
        return Image.fromarray(e).convert("RGB")
    if kind == "depth":
        import torch
        from transformers import pipeline as hf_pipeline
        if not depth_dir:
            raise RuntimeError("Profondità: manca DepthAnything-V2 (voce depth-anything-v2 del tab Pesi)")
        est = hf_pipeline(task="depth-estimation", model=depth_dir, device=0, dtype=torch.float16)
        d = np.asarray(est(img)["depth"], dtype="float32")
        del est
        torch.cuda.empty_cache()
        d = (d - d.min()) / (np.ptp(d) + 1e-6) * 255.0
        return Image.fromarray(d.astype("uint8")).resize((width, height)).convert("RGB")
    raise ValueError(f"preprocessore di controllo sconosciuto: {kind}")


def _forward_with_control(self, hidden_states, encoder_hidden_states, timestep, img_shapes, img_mask,
                          encoder_hidden_states_mask=None, attention_kwargs=None, kv_cache=None,
                          kv_cache_mode=None, return_dict=True):
    """Copia della forward di diffusers QwenImage21Transformer2DModel + ramo di controllo (VideoX-Fun).
    Senza contesto impostato (`self._ctrl` None) si comporta esattamente come l'originale."""
    import torch
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.models.transformers.transformer_qwenimage21 import (
        _IMG_TOKENS_PER_SLOT, _qwenimage21_prefix_segments)
    ctrl = getattr(self, "_ctrl", None)
    if ctrl is not None and kv_cache is not None:
        raise RuntimeError("ControlNet: la cache KV va disattivata (use_kv_cache=False)")
    batch_size = hidden_states.shape[0]

    hidden_states = self.img_in(hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)
    repeats = torch.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]
    image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
    target_tokens = math.prod(img_shapes[0][-1])
    joint = torch.cat([encoder_hidden_states,
                       encoder_hidden_states.new_zeros(batch_size, target_tokens // 4, encoder_hidden_states.shape[2])],
                      dim=1)
    joint = joint.repeat_interleave(repeats, dim=1)
    joint[:, image_pad_mask] = hidden_states

    control_joint = None
    if ctrl is not None:
        context, scale = ctrl
        if int(image_pad_mask.sum()) != context.shape[1]:
            raise RuntimeError("ControlNet: non si combina con le immagini di riferimento (solo testo→immagine "
                               "e img2img)")
        feats = self._ctrl_adapter.control_img_in(context.to(hidden_states.dtype))
        control_joint = torch.zeros_like(joint)
        control_joint[:, image_pad_mask] = feats

    rotary_emb = self.pos_embed(img_shapes[0], image_pad_mask, device=hidden_states.device)
    image_ids, target_token_mask = self.build_token_metadata(image_pad_mask, img_shapes[0])
    timestep = timestep.to(hidden_states.dtype)
    if self.config.causal_condition:
        timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
        modulation_mask = target_token_mask
    else:
        modulation_mask = None
    temb = self.time_text_embed(timestep, hidden_states)
    modulation = self.modulation(temb)

    joint_key_valid = None
    if encoder_hidden_states_mask is not None:
        joint_key_valid = torch.ones(batch_size, image_pad_mask.shape[0], dtype=torch.bool,
                                     device=hidden_states.device)
        text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
        vlm_text_positions = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
        joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[:, vlm_text_positions]
    prefix_len = int((~target_token_mask).sum())

    if kv_cache_mode == "cached":
        joint = joint[:, prefix_len:]
        rotary_emb = rotary_emb[prefix_len:]
        modulation_mask = modulation_mask[prefix_len:]
        attention_mask = None if joint_key_valid is None else joint_key_valid[:, None, None, :]
        cache_write_slice = None
        block_segments, block_key_valid = None, None
    else:
        attention_mask = None                     # processore standard: struttura causale a segmenti
        block_segments = _qwenimage21_prefix_segments(image_ids, prefix_len)
        cache_write_slice = slice(0, prefix_len) if kv_cache_mode == "extract" else None
        block_key_valid = joint_key_valid

    hints = None
    if control_joint is not None:
        kw = dict(modulation=modulation, rotary_emb=rotary_emb, attention_mask=attention_mask,
                  target_token_mask=modulation_mask, layer_cache=None, kv_cache_mode=None, cache_write_slice=None,
                  segments=block_segments, key_valid=block_key_valid)
        hints = []
        c = None
        for i, blk in enumerate(self._ctrl_adapter.control_blocks):
            if i == 0:                             # il primo blocco fonde il flusso del DiT nel controllo
                c = blk.before_proj(control_joint) + joint
            c = blk(c, **kw)                       # stesso blocco del DiT (forward ereditata)
            hints.append(blk.after_proj(c))        # suggerimento per il blocco CONTROL_LAYERS[i]

    for index_block, block in enumerate(self.transformer_blocks):
        layer_cache = kv_cache.get_layer(index_block) if kv_cache is not None else None
        joint = block(hidden_states=joint, modulation=modulation, rotary_emb=rotary_emb,
                      attention_mask=attention_mask, target_token_mask=modulation_mask, layer_cache=layer_cache,
                      kv_cache_mode=kv_cache_mode, cache_write_slice=cache_write_slice, segments=block_segments,
                      key_valid=block_key_valid)
        if hints is not None and index_block in CONTROL_LAYERS:
            joint = joint + hints[CONTROL_LAYERS.index(index_block)] * ctrl[1]

    joint = self.norm_out(joint, temb, modulation_mask)
    output = self.proj_out(joint)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
