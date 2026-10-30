"""MagicQuill V2 — port NATIVO (diffusers 0.38) delle LoRA di controllo su FLUX.1 Kontext.

CREDITI
    Idea e algoritmo (cache KV delle condizioni, LoRA di controllo, preprocessing e prompt delle modalità) di MagicQuill V2
    (https://github.com/zliucz/MagicQuillV2, Zichen Liu et al., arXiv:2512.03046); il meccanismo di attenzione discende da
    EasyControl (Apache-2.0). Codice scritto da zero per diffusers 0.38. Attribuzioni complete:
    `docs/THIRD_PARTY_NOTICES_MAGICQUILL.md`.

COME FUNZIONA (riassunto, vedi i singoli moduli)
    * Le 4 LoRA di controllo (local/removal/edge/color) NON sono LoRA diffusers: sono pesi q/k/v(+proj) applicati SOLO
      ai token di una immagine-condizione 512x512 (1024 token) che viaggia in coda alla sequenza immagine.
    * La condizione attraversa i blocchi con modulazione a timestep 0; il suo K/V (per blocco) viene scritto una volta
      al primo passaggio e riletto ai successivi; l'attenzione immagine->condizione riceve un bias log(gamma).
    * Si riusa il transformer diffusers GIÀ caricato: nessuna copia dei pesi; si sostituisce temporaneamente il solo
      forward dei blocchi (ripristinato a fine job).
"""
from .control_lora import ControlLoRA, ControlSpec  # noqa: F401
