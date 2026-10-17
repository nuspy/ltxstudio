"""Aggancio di MagicQuill V2 alla `FluxKontextPipeline` diffusers 0.38 già costruita da image_flux.

Derivato da MagicQuillV2 `src/pipeline_flux_kontext_control.py` (CC BY-NC 4.0, (c) Zichen Liu et al.). Vedi
`inference_server/backends/mq2/__init__.py` per attribuzione e avvertenza di licenza (NON commerciale).

Invece di copiare l'intera pipeline upstream (basata su un diffusers più vecchio) si usa la `__call__` STANDARD di
FluxKontextPipeline e si sostituisce, per la durata del job, solo `prepare_latents`: restituisce i latenti Kontext
con in CODA i latenti della condizione (e gli id di posizione relativi, marcati 2 nel primo canale come upstream).
Il forward standard del transformer li embedda; i blocchi (attention.py) li riconoscono dalla coda.
"""
from __future__ import annotations

import contextlib

import torch

from .attention import MQState, patch_transformer


def cond_latent_ids(latent_h: int, latent_w: int, cond_size: int, device, dtype):
    """Id di posizione dei token-condizione: griglia (cond_size/16)^2, passo scalato come upstream
    (`resize_position_encoding`: posizione = indice * (altezza_latente / (2*g))), primo canale = 2."""
    g = cond_size // 16
    scale_h = latent_h / (2 * g)
    scale_w = latent_w / (2 * g)
    ids = torch.zeros(g, g, 3, device=device, dtype=dtype)
    ids[..., 0] = 2
    ids[..., 1] = ids[..., 1] + torch.arange(g, device=device, dtype=dtype)[:, None] * scale_h
    ids[..., 2] = ids[..., 2] + torch.arange(g, device=device, dtype=dtype)[None, :] * scale_w
    return ids.reshape(g * g, 3)


def make_prepare_latents(pipe, st: MQState):
    """`prepare_latents` che aggiunge la condizione. Stessa firma posizionale della stock (diffusers 0.38)."""
    stock = type(pipe).prepare_latents

    def prepare_latents(image, batch_size, num_channels_latents, height, width, dtype, device,
                        generator=None, latents=None):
        lat, img_lat, lat_ids, img_ids = stock(pipe, image, batch_size, num_channels_latents, height, width, dtype,
                                               device, generator, latents)
        vsf = pipe.vae_scale_factor
        h_l = 2 * (int(height) // (vsf * 2))
        w_l = 2 * (int(width) // (vsf * 2))
        cond_lat, cond_ids = [], []
        for ctl in st.controls:
            cs = int(ctl.cond_size)
            t = pipe.image_processor.preprocess(ctl.image, height=cs, width=cs).to(device=device, dtype=dtype)
            z = pipe._encode_vae_image(image=t, generator=generator)
            if z.shape[0] < batch_size and batch_size % z.shape[0] == 0:
                z = torch.cat([z] * (batch_size // z.shape[0]), dim=0)
            cond_lat.append(pipe._pack_latents(z, batch_size, num_channels_latents, z.shape[2], z.shape[3]))
            cond_ids.append(cond_latent_ids(h_l, w_l, cs, device, dtype))
        cl = torch.cat(cond_lat, dim=1)
        ci = torch.cat(cond_ids, dim=0)
        st.n_base = lat.shape[1] + (img_lat.shape[1] if img_lat is not None else 0)
        if img_lat is None:
            return lat, cl, lat_ids, ci
        return lat, torch.cat([img_lat, cl], dim=1), lat_ids, torch.cat([img_ids, ci], dim=0)

    return prepare_latents


@contextlib.contextmanager
def control_session(pipe, controls: list):
    """Durante il `with`: blocchi del transformer + prepare_latents della pipe sostituiti; all'uscita (anche su
    errore) TUTTO ripristinato e banca K/V svuotata (il transformer è CONDIVISO fra i job)."""
    st = MQState(controls)
    had = "prepare_latents" in pipe.__dict__
    old = pipe.__dict__.get("prepare_latents")
    try:
        with patch_transformer(pipe.transformer, st):
            pipe.__dict__["prepare_latents"] = make_prepare_latents(pipe, st)
            yield st
    finally:
        if had:
            pipe.__dict__["prepare_latents"] = old
        else:
            pipe.__dict__.pop("prepare_latents", None)
