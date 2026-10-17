"""LoRA 'puzzle' di MagicQuill V2 (modalità foreground = integrazione di un oggetto, senza immagine di controllo).

Dal progetto MagicQuillV2 (CC BY-NC 4.0, (c) Zichen Liu et al.): vedi `NOTICE.md` (cartella del componente) per
attribuzione e avvertenza di licenza (NON commerciale). `edit.py` upstream la carica con un monkey-patch di
`load_lora_weights` che converte il formato MISTO del file (2 stili di chiave) e aggiunge il prefisso `transformer.`.
Qui la stessa conversione è fatta in memoria e la LoRA entra nel normale PEFT/diffusers (adapter 'mq_puzzle').

Stili di chiave nel file (992 tensori, rank 32, bf16, nessun alpha -> scala 1):
    <modulo>.lora.down.weight / <modulo>.lora.up.weight      (to_q/to_k/to_v/to_out.0 dei blocchi: stile 'lora.down')
    <modulo>.lora_A.weight   / <modulo>.lora_B.weight        (add_*_proj, to_add_out, ff, norm, proj_*, norm_out)
"""
from __future__ import annotations

import os

ADAPTER = "mq_puzzle"


def convert_state_dict(sd: dict) -> dict:
    """Chiavi -> formato PEFT diffusers: `transformer.<modulo>.lora_A.weight` / `.lora_B.weight`."""
    out = {}
    for k, v in sd.items():
        if k.endswith(".lora.down.weight"):
            nk = k[: -len(".lora.down.weight")] + ".lora_A.weight"
        elif k.endswith(".lora.up.weight"):
            nk = k[: -len(".lora.up.weight")] + ".lora_B.weight"
        elif k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"):
            nk = k
        else:
            raise ValueError(f"puzzle LoRA: chiave inattesa '{k}'")
        if not nk.startswith("transformer."):
            nk = "transformer." + nk
        out[nk] = v
    return out


def load_state_dict(path: str) -> dict:
    from safetensors.torch import load_file
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return convert_state_dict(load_file(path, device="cpu"))


def attach(pipe, sd: dict, user_adapters: list[str], user_weights: list[float], weight: float = 1.0) -> None:
    """Carica la puzzle LoRA come adapter 'mq_puzzle' e attiva (insieme alle LoRA utente già caricate)."""
    pipe.load_lora_weights(sd, adapter_name=ADAPTER)
    pipe.set_adapters(list(user_adapters) + [ADAPTER], adapter_weights=list(user_weights) + [float(weight)])


def detach(pipe) -> None:
    """Rimuove la puzzle LoRA dal transformer CONDIVISO (best-effort, non solleva)."""
    try:
        pipe.delete_adapters(ADAPTER)
        return
    except Exception:                                   # noqa: BLE001
        pass
    try:
        pipe.unload_lora_weights()
    except Exception:                                   # noqa: BLE001
        pass
