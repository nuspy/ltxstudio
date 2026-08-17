"""MSR — Multiple Subject Reference per LTX (riferimenti multipli SENZA posizione temporale).

Le LoRA `LiconStudio/LTX-2.{3,5}-Multiple-Subject-Reference` codificano fino a 5 immagini di
riferimento come token visivi nello STESSO spazio latente del video target. Non basta caricare
la LoRA: ogni riferimento va (a) codificato col VAE video, (b) sommato a uno **slot embedding**
appreso — una Fourier-MLP che vive DENTRO il file della LoRA, sotto `reference_slot_embedding.*`
— e (c) appeso al latente a una **posizione temporale NEGATIVA** distinta. È così che il modello
li distingue e li recupera con la self-attention, senza che occupino frame del video.

Ricetta replicata dal plugin di riferimento ComfyUI-LTX2.5-MSR (liconstudio), studiato prima di
implementare come da prassi. La pipeline nativa regge gli indici negativi senza modifiche:
`VideoConditionByKeyframeIndex.apply_to` fa `positions[:, 0, ...] += frame_idx` e concatena i
token come clean-latent — nessun clamp, nessun controllo di bounds.

⚠ Da validare LIVE: la ricetta è corretta sulla carta ma non è mai stata eseguita qui.

Le due LoRA NON sono intercambiabili: ciascuna è addestrata sulla propria architettura
(`core.ltx_release.msr_lora` sceglie il file dalla release).
"""
from __future__ import annotations

import os

SLOT_PREFIX = "reference_slot_embedding."
MAX_REFERENCES = 5


def slot_state_dict(lora_path: str) -> dict:
    """Pesi della Fourier-MLP estratti dal file della LoRA (chiavi senza il prefisso).
    Vuoto se la LoRA non è una MSR: il chiamante deve dirlo chiaro, non tirare a indovinare."""
    from safetensors import safe_open
    out: dict = {}
    with safe_open(lora_path, framework="pt") as h:
        for k in h.keys():
            i = k.find(SLOT_PREFIX)
            if i >= 0:
                out[k[i + len(SLOT_PREFIX):]] = h.get_tensor(k)
    return out


def slot_embedding(state: dict, slot_id: int, device, dtype):
    """Embedding dello slot `slot_id`: Fourier features (sin/cos) → Linear+SiLU → Linear.
    Identica al plugin: il valore dello slot è normalizzato /16 prima delle frequenze."""
    import torch

    def t(key):                                              # tensore fp32 sul device, None se assente
        v = state.get(key)
        return None if v is None else v.to(device=device, dtype=torch.float32)

    freq = t("frequencies")
    scaled = torch.tensor(float(slot_id), device=device, dtype=torch.float32) / 16.0
    phases = scaled * freq
    feats = torch.cat((scaled.reshape(1), torch.sin(phases), torch.cos(phases)))
    hidden = torch.nn.functional.silu(
        torch.nn.functional.linear(feats, t("net.0.weight"), t("net.0.bias")))
    emb = torch.nn.functional.linear(hidden, t("net.2.weight"), t("net.2.bias"))
    return emb.to(dtype=dtype)


def frame_offsets(num_slots: int) -> list[int]:
    """Posizioni temporali dei riferimenti: slot i → -(num_slots - i), cioè -N … -1.
    Negative = fuori dalla finestra del video, così non rubano frame alla generazione."""
    return [-(num_slots - i) for i in range(num_slots)]


def references_to_inputs(references: list[dict], strength: float = 1.0, crf: int = 33) -> list:
    """`references` del pannello GUI → ImageConditioningInput con frame_idx NEGATIVI."""
    from ltx_pipelines.utils.args import ImageConditioningInput
    refs = [r for r in references if r.get("kind", "image") == "image"][:MAX_REFERENCES]
    offs = frame_offsets(len(refs))
    return [ImageConditioningInput(r["path"], off, float(r.get("strength", strength)), crf)
            for r, off in zip(refs, offs)]


def install(lora_path: str, *, reference_frames: int = 1, log=print) -> bool:
    """Innesta lo slot embedding nella costruzione dei conditioning della pipeline.

    `combined_image_conditionings` (usata da TI2VidTwoStagesPipeline per entrambi gli stage)
    codifica le immagini e costruisce i ConditioningItem: la si avvolge per intercettare le voci
    a frame NEGATIVO — i riferimenti — e sommare al loro latente l'embedding dello slot. Le altre
    immagini (i2v, keyframe) passano invariate all'originale.

    Ritorna False se la LoRA non contiene gli slot embedding (non è una MSR)."""
    state = slot_state_dict(lora_path)
    if "frequencies" not in state:
        log(f"[msr] {os.path.basename(lora_path)} non contiene '{SLOT_PREFIX}*': "
            f"non è una LoRA MSR → riferimenti NON applicati")
        return False

    import ltx_pipelines.ti2vid_two_stages as _t2
    from ltx_core.conditioning import VideoConditionByKeyframeIndex
    from ltx_pipelines.utils.media_io.decode import load_image_and_preprocess

    if getattr(_t2, "_msr_patched", False):                  # idempotente (2 stage, stessa run)
        _t2._msr_state = state
        return True
    original = _t2.combined_image_conditionings

    def patched(images, height, width, video_encoder, dtype, device, color_space=None):
        normal = [im for im in images if im.frame_idx >= 0]
        refs = [im for im in images if im.frame_idx < 0]
        out = list(original(normal, height, width, video_encoder, dtype, device, color_space)) \
            if normal else []
        # slot id = ordine di inserimento (offset -N … -1 → 0 … N-1), come nel plugin
        n = len(refs)
        for im in sorted(refs, key=lambda x: x.frame_idx):
            slot = n + im.frame_idx                          # -N→0, -1→N-1
            image = load_image_and_preprocess(
                image_path=im.path, height=height, width=width, dtype=dtype, device=device,
                crf=im.crf if im.crf is not None else 33, color_space=color_space)
            if reference_frames > 1:                          # il plugin ripete il riferimento su N frame
                image = _repeat_frames(image, reference_frames)
            lat = video_encoder(image)
            emb = slot_embedding(_t2._msr_state, slot, lat.device, lat.dtype)
            lat = lat + emb.view(1, -1, 1, 1, 1)
            out.append(VideoConditionByKeyframeIndex(
                keyframes=lat, frame_idx=im.frame_idx, strength=im.strength))
        return out

    _t2.combined_image_conditionings = patched
    _t2._msr_patched = True
    _t2._msr_state = state
    log(f"[msr] slot embedding attivo ({len(state)} tensori da {os.path.basename(lora_path)})")
    return True


def _repeat_frames(image, n: int):
    """Ripete l'immagine su `n` frame lungo l'asse temporale (5-D: B,C,F,H,W; 4-D: B,C,H,W)."""
    if image.dim() == 5:
        return image.repeat_interleave(n, dim=2) if image.shape[2] == 1 else image
    return image.unsqueeze(2).repeat_interleave(n, dim=2)
