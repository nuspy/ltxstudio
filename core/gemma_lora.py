"""LoRA «abliterata» sull'encoder Gemma-3 12B di LTX-2.3: fusione sul ramo TESTO al caricamento dell'encoder.

La LoRA (rank 64, chiavi `text_encoders.transformer.{model,vision_model}.…lora_down/lora_up`, senza alpha → scala 1)
tocca 48 layer × 7 Linear + embed_tokens del ramo testo (337 pesi, tutti verificati contro gli header del Gemma
`gemma-3-ltx`: 337/337 esistono, shape coerenti) e il vision tower, che QUI si ignora (l'encoder di LTX non lo usa).

Meccanismo: ltx-core sa già fondere LoRA nel builder dell'encoder (`builder.with_loras`, regola bf16); basta una
mappa chiavi LoRA→modello. Si agganciano entrambi i builder del `PromptEncoder` (normale e streaming). L'embedding
duplicato in `lm_head` (serve a generate()) riceve la stessa LoRA, come in Gemma (pesi legati).

Cache degli embedding: l'impronta della LoRA entra nel materiale della chiave (`key_material`) SOLO quando c'è,
così le chiavi esistenti non cambiano. Solo release 2.3 (Gemma 3): sulla 2.5 (Gemma 4) la LoRA non è applicabile.
"""
from __future__ import annotations

import os

LORA_CATALOG_ID = "gemma3-abliterated-lora"
LORA_FILENAME = "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors"
_SRC_PREFIX = "text_encoders.transformer.model."            # ramo testo (il vision_model NON passa il filtro)
_DST_PREFIX = "model.model.language_model."                 # nomi del modello dopo GEMMA_LLM_KEY_OPS


def lora_sd_ops():
    """SDOps (ltx-core) che traduce le chiavi della LoRA nei nomi del modello e duplica l'embedding su lm_head."""
    from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
    ops = (SDOps("GEMMA3_ABLIT_LORA")
           .with_matching(prefix=_SRC_PREFIX)
           .with_replacement(_SRC_PREFIX, _DST_PREFIX)
           .with_replacement(".lora_down.weight", ".lora_A.weight")
           .with_replacement(".lora_up.weight", ".lora_B.weight"))
    for ab in ("lora_A", "lora_B"):
        emb = f"{_DST_PREFIX}embed_tokens.{ab}"
        ops = ops.with_kv_operation(
            operation=lambda key, value, _ab=ab: [
                KeyValueOperationResult(key, value),
                KeyValueOperationResult(f"model.lm_head.{_ab}.weight", value)],
            key_prefix=emb)
    return ops


def resolve_lora_path(lora_id: str, release: str) -> str:
    """Percorso del file dall'id di catalogo. Errore CHIARO (mai ripiego silenzioso sull'encoder standard)."""
    if release != "2.3":
        raise RuntimeError(f"La LoRA Gemma abliterata vale solo per LTX 2.3 (Gemma 3), non per la {release}.")
    from core import weights_catalog as wc
    e = {x["id"]: x for x in wc.load()}.get(lora_id)
    if not e or not e.get("filename"):
        raise RuntimeError(f"LoRA encoder «{lora_id}» sconosciuta.")
    path = os.path.join(str(wc.dest_for(e)), e["filename"])
    if not os.path.isfile(path):
        raise RuntimeError(f"LoRA encoder «{lora_id}» non trovata su disco ({path}): collocala lì o aprila dal tab "
                           "«Pesi modelli». Non uso l'encoder standard al suo posto.")
    return path


def attach(pipe, lora_path: str, strength: float = 1.0) -> int:
    """Aggancia la LoRA ai builder dell'encoder di OGNI `PromptEncoder` della pipeline. Ritorna quanti ne ha toccati
    (0 = nessun PromptEncoder trovato: chi chiama deve trattarlo come errore, non come «forse standard»)."""
    from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
    from ltx_pipelines.utils.blocks import PromptEncoder
    lora = LoraPathStrengthAndSDOps(lora_path, float(strength), lora_sd_ops())
    n = 0
    for pe in _prompt_encoders(pipe, PromptEncoder):
        for attr in ("_text_encoder_builder", "_streaming_text_encoder_builder"):
            b = getattr(pe, attr, None)
            if b is None or not hasattr(b, "with_loras"):
                continue
            if any(getattr(l, "path", None) == lora_path for l in getattr(b, "loras", ())):
                continue                                           # già agganciata (pipeline riusata)
            setattr(pe, attr, b.with_loras((*getattr(b, "loras", ()), lora)))
        n += 1
    return n


def _prompt_encoders(obj, cls, depth: int = 2):
    """PromptEncoder della pipeline: `pipe.prompt_encoder` (tutte le pipeline ltx-pipelines) e, per i wrapper,
    degli oggetti-attributo fino a 2 livelli. Niente visita del grafo intero (modelli/tensori)."""
    if isinstance(obj, cls):
        yield obj
        return
    if depth <= 0:
        return
    for v in list(getattr(obj, "__dict__", {}).values()):
        if isinstance(v, cls):
            yield v
        elif depth > 1 and type(v).__module__.split(".")[0] in ("ltx_pipelines", "workers", "__main__"):
            yield from _prompt_encoders(v, cls, depth - 1)


def key_fragment(text_encoder_builder, file_fp) -> list | None:
    """Per la cache embedding: impronta delle LoRA fuse sull'encoder (lista) o None se non ce ne sono."""
    loras = tuple(getattr(text_encoder_builder, "loras", ()) or ())
    if not loras:
        return None
    return [[file_fp(l.path), float(l.strength)] for l in loras]
