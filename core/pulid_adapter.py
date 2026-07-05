"""Face adapter PuLID-FLUX nativo: imprime l'identità di un volto (da foto) sull'immagine FLUX.

Usa il repo PuLID (E:\\AI\\Tools\\PuLID) come libreria per: face encoder (InsightFace antelopev2 +
EVA-CLIP) + pulid_encoder (IDFormer) → id_embedding; e i 20 moduli pulid_ca (PerceiverAttentionCA,
dim=3072 kv_dim=2048). L'integrazione col FluxTransformer2DModel di diffusers è fatta QUI via hook
sui blocchi (il repo userebbe il suo FLUX): `hidden += strength * pulid_ca[idx](id_emb, hidden)`
ogni `double_interval` transformer_blocks e ogni `single_interval` single_transformer_blocks.
"""
from __future__ import annotations

import os
import sys

PULID_REPO = r"E:\AI\Tools\PuLID"
PULID_WEIGHTS = r"E:\AI\Models\pulid\pulid_flux_v0.9.1.safetensors"
INSIGHTFACE_ROOT = r"E:\AI\Models\insightface"   # contiene models/antelopev2/

_CACHE: dict = {}


def available() -> bool:
    if not os.path.exists(PULID_WEIGHTS):
        return False
    try:
        import facexlib  # noqa: F401
        import insightface  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _ensure_path():
    for p in (PULID_REPO, os.path.join(PULID_REPO, "eva_clip")):
        if p not in sys.path:
            sys.path.insert(0, p)


def load(transformer, device="cuda", dtype=None):
    """Istanzia PuLIDPipeline (crea pulid_ca + face encoder + pulid_encoder) e carica i pesi."""
    import torch
    if dtype is None:
        dtype = torch.bfloat16
    key = id(transformer)
    if key in _CACHE:
        return _CACHE[key]
    _ensure_path()
    cwd = os.getcwd()
    os.chdir(PULID_REPO)                                  # path relativi del repo (models/antelopev2)
    try:
        # antelopev2 già scaricato: linka E:\AI\Models\insightface\models → ./models se assente
        if not os.path.exists(os.path.join(PULID_REPO, "models", "antelopev2")):
            os.makedirs(os.path.join(PULID_REPO, "models"), exist_ok=True)
            src = os.path.join(INSIGHTFACE_ROOT, "models", "antelopev2")
            dst = os.path.join(PULID_REPO, "models", "antelopev2")
            if os.path.exists(src) and not os.path.exists(dst):
                import shutil
                shutil.copytree(src, dst)
        from pulid.pipeline_flux import PuLIDPipeline
        pulid = PuLIDPipeline(transformer, device=device, weight_dtype=dtype)
        pulid.load_pretrain(PULID_WEIGHTS, version="v0.9.1")
    finally:
        os.chdir(cwd)
    _CACHE[key] = pulid
    return pulid


def extract_id(pulid, face_image_path):
    """id_embedding dal volto (antelopev2 + eva-clip + pulid_encoder). Accetta UNA foto o una LISTA:
    con più foto dello stesso volto ne fa la MEDIA degli embedding → identità più robusta/stabile."""
    import cv2
    import torch
    paths = list(face_image_path) if isinstance(face_image_path, (list, tuple)) else [face_image_path]
    embs = []
    for p in paths:
        if not p:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        e, _ = pulid.get_id_embedding(img, cal_uncond=False)
        embs.append(e)
    if not embs:
        raise FileNotFoundError(str(face_image_path))
    return embs[0] if len(embs) == 1 else torch.stack(embs, dim=0).mean(dim=0)


def attach(transformer, pulid, id_emb, strength: float = 0.8, start_step: int = 0):
    """Hook sui blocchi del FluxTransformer2DModel diffusers: replica il forward di PuLID-FLUX
    (flux/model.py) — `hid = hid + strength * pulid_ca[ca_idx](id, hid)` ogni `double_interval`
    double-block e ogni `single_interval` single-block, con ca_idx CONDIVISO double→single e
    AZZERATO a ogni forward del transformer (ogni step di denoising).
    start_step: inietta l'ID SOLO dai step >= start_step (0 = da subito) — così la composizione/posa
    si forma prima e l'identità entra dopo (migliore aderenza al prompt), come il parametro ufficiale."""
    ca = pulid.pulid_ca
    di = int(getattr(transformer, "pulid_double_interval", 2) or 2)
    si = int(getattr(transformer, "pulid_single_interval", 4) or 4)
    handles = []
    idx = [0]
    step = [0]

    # CRITICO: azzera il contatore ca_idx all'inizio di OGNI forward del transformer (per-step),
    # altrimenti accumula tra gli step e dopo ~2 step esaurisce i pulid_ca → effetto nullo.
    def _reset(_m, _inp):
        idx[0] = 0
        step[0] += 1                                          # conta gli step (1 forward/step in distilled)
    handles.append(transformer.register_forward_pre_hook(_reset))

    def _ca_hook(_m, _inp, out):
        # FluxTransformerBlock / FluxSingleTransformerBlock → (encoder_hidden_states, hidden_states)
        if idx[0] >= len(ca) or (step[0] - 1) < start_step:  # gating start_step (salta i primi step)
            return out
        enc, hid = out if isinstance(out, tuple) else (None, out)
        hid = hid + strength * ca[idx[0]](id_emb, hid)
        idx[0] += 1
        return (enc, hid) if enc is not None else hid

    # double-block (transformer_blocks) — eseguiti per primi nel forward → ca_idx 0..N
    for i, blk in enumerate(getattr(transformer, "transformer_blocks", [])):
        if i % di == 0:
            handles.append(blk.register_forward_hook(_ca_hook))
    # single-block (single_transformer_blocks) — eseguiti dopo → ca_idx prosegue (come l'originale)
    for i, blk in enumerate(getattr(transformer, "single_transformer_blocks", [])):
        if i % si == 0:
            handles.append(blk.register_forward_hook(_ca_hook))
    return handles


def detach(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:  # noqa: BLE001
            pass
