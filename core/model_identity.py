"""Identità di RESIDENZA di un modello immagine — UNA sola definizione per server, runner e registro GPU.

Problema (task 12, 17/10/2026): la chiave di residenza era `arch:basename(model)` con `arch` = il COMPITO
(flux = crea, inpaint = modifica con maschera, kontext = modifica per istruzione). Lo stesso file di pesi
usato in crea e in modifica produceva due chiavi diverse → il server uccideva il figlio persistente (VRAM e
RAM del parcheggio perse) e ricaricava tutto da SSD, anche se le due pipeline condividono transformer, VAE e
text-encoder (ComponentStore: la seconda pipeline si costruisce con gli stessi oggetti modulo).

Regola: la chiave dice QUALI PESI sono residenti, non quale compito li sta usando.

  FLUX.1  flux / kontext / inpaint          → famiglia "flux1"  (stessi componenti nello store)
  Klein   flux2_kontext / flux2_inpaint     → famiglia "flux2k" (stesso transformer + stesso encoder Qwen3)
  tutto il resto (flux2, zimage, ernie, upscale, segment, video…) → l'arch com'è (nessun cambiamento).

Solo stdlib: importabile dal server (che non importa torch) e dal client GUI.
"""
from __future__ import annotations

# arch (compito) → famiglia di pesi condivisi. Aggiungere QUI un arch nuovo che riusa i pesi di un altro.
_FAMILY = {
    "flux": "flux1", "kontext": "flux1", "inpaint": "flux1",
    "flux2_kontext": "flux2k", "flux2_inpaint": "flux2k",
}


def arch_family(arch: str) -> str:
    """Famiglia di pesi di un arch immagine (arch sconosciuto/non immagine → l'arch stesso, minuscolo)."""
    a = str(arch or "").strip().lower()
    return _FAMILY.get(a, a)


def _basename(path: str) -> str:
    # indipendente dalla piattaforma: la GUI manda sia "Flux\\x.safetensors" sia "E:\\AI\\...\\x.safetensors"
    p = str(path or "").replace("\\", "/").rstrip("/")
    return p.rsplit("/", 1)[-1].strip().lower()


def residency_key(arch: str, model: str = "", gguf: str = "") -> str:
    """Chiave di residenza `famiglia:basename` (es. `flux1:flux_dev.safetensors`).
    Con NATIVE_RESIDENCY_BY_TASK=1 torna a quella storica per-compito (`arch:basename`): rete di sicurezza
    per il rollback in una riga."""
    import os
    a = str(arch or "").strip().lower()
    fam = a if os.environ.get("NATIVE_RESIDENCY_BY_TASK") == "1" else arch_family(a)
    return f"{fam}:{_basename(model or gguf) or '-'}"

