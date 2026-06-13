"""Risoluzione dell'encoder testo corretto per un modello FLUX.2.

FLUX.2 ha PIÙ varianti, ognuna con un TEXT ENCODER diverso → quindi una dimensione di `txt_in`
diversa nel modello. Passare l'encoder sbagliato fa crashare il sampler con
`mat1 and mat2 shapes cannot be multiplied` (la conditioning ha N feature, il modello ne vuole M).

Mappa (in-features di `txt_in` = dim della conditioning prodotta dall'encoder):
- FLUX.2 [dev]/[pro]   → Mistral-3 24B   (flux2_te: 3 layer × 5120 = 15360)
- FLUX.2 Klein Qwen3-8B → Qwen3-8B       (klein_te: 3 × 4096 = 12288)
- FLUX.2 Klein Qwen3-4B → Qwen3-4B       (klein_te: 3 × 2560 = 7680)

`txt_in.weight` ha shape [out, in]: leggendo SOLO l'header safetensors (senza caricare i pesi)
ricaviamo `in` e quindi l'encoder esatto. Per i GGUF (header non standard) si ricade su
un'euristica sul nome. Il `type` del CLIPLoader resta "flux2" per tutte le varianti: ComfyUI
sceglie la classe encoder dal CONTENUTO del file, non dal type.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

# Root dei diffusion_models (da extra_model_paths.yaml di Comfy_Nexus).
DIFFUSION_DIRS = [
    r"E:\AI\Models\diffusion_models",
    r"E:\AI\Models\Inpainting",
    r"E:\AI\Models\Checkpoints",
]

# in-features di txt_in → keyword per trovare l'encoder nell'inventario text_encoders + etichetta.
INDIM_KEYWORDS: dict[int, tuple[str, ...]] = {
    15360: ("mistral_3_small_flux2", "mistral_3", "mistral"),
    12288: ("qwen_3_8b", "qwen3_8b", "qwen_3_8", "qwen3-8b"),
    7680: ("qwen_3_4b", "qwen3_4b", "qwen_3_4", "qwen3-4b"),
}
INDIM_LABEL: dict[int, str] = {15360: "Mistral-3 24B", 12288: "Qwen3-8B", 7680: "Qwen3-4B"}
# nome-file atteso per il download (hint nel messaggio d'errore)
INDIM_FILE_HINT: dict[int, str] = {
    15360: "mistral_3_small_flux2.safetensors", 12288: "qwen_3_8b.safetensors",
    7680: "qwen_3_4b.safetensors",
}

DEFAULT_ENCODER = "Flux\\mistral_3_small_flux2.safetensors"


def _resolve_model_path(name: str | None) -> Path | None:
    if not name:
        return None
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p
    for root in DIFFUSION_DIRS:
        cand = Path(root) / name
        if cand.exists():
            return cand
    bn = Path(name).name                      # ultimo tentativo: cerca per basename
    for root in DIFFUSION_DIRS:
        rp = Path(root)
        if rp.exists():
            for c in rp.rglob(bn):
                return c
    return None


def txt_in_indim(model_name: str | None) -> int | None:
    """In-features di `txt_in` del modello FLUX.2 (None se non leggibile, es. GGUF)."""
    path = _resolve_model_path(model_name)
    if not path or path.suffix.lower() != ".safetensors":
        return None
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
    except Exception:
        return None
    for k, v in head.items():
        if k.lower().endswith("txt_in.weight"):
            shp = v.get("shape") if isinstance(v, dict) else None
            if isinstance(shp, list) and len(shp) == 2:
                return int(shp[1])
    return None


def resolve_encoder(model_name: str | None, text_encoders: list[str] | None,
                    default: str = DEFAULT_ENCODER) -> tuple[str | None, str | None]:
    """Ritorna (encoder_file, errore).

    - encoder_file: nome-file da passare al CLIPLoader (type "flux2").
    - errore: None se ok; stringa esplicativa se l'encoder richiesto NON è installato
      (in tal caso encoder_file è None → il chiamante deve interrompere con un messaggio chiaro).
    """
    indim = txt_in_indim(model_name)
    if indim is None:
        # GGUF / header non leggibile: euristica sul nome.
        nl = (model_name or "").lower()
        if "klein" in nl:
            indim = 12288 if "8b" in nl else 7680
        else:
            return default, None              # dev/pro → Mistral (default)
    kws = INDIM_KEYWORDS.get(indim)
    if not kws:
        return default, None
    inv = list(text_encoders or [])
    for kw in kws:
        for e in inv:
            if kw in e.lower().replace("\\", "/"):
                return e, None
    label = INDIM_LABEL.get(indim, str(indim))
    fname = INDIM_FILE_HINT.get(indim, "<encoder>.safetensors")
    return None, (
        f"Il modello FLUX.2 selezionato richiede l'encoder testo {label} (txt_in={indim}), "
        f"non installato in ComfyUI.\n"
        f"  → Scarica '{fname}' (Comfy-Org/FLUX.2 su HuggingFace) e copialo in "
        f"E:\\AI\\Models\\TextEncoders\\Flux\\, poi riavvia ComfyUI e ricarica l'inventario.\n"
        f"  → In alternativa usa subito un modello FLUX.2 con encoder già presente — "
        f"es. FLUX.2 [dev] (usa Mistral, installato) — oppure i modelli FLUX Fill."
    )
