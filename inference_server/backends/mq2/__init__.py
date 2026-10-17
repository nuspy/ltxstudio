"""MagicQuill V2 — port NATIVO (diffusers 0.38) delle LoRA di controllo su FLUX.1 Kontext.

ATTRIBUZIONE E LICENZA
    Logica di attenzione/cache KV, formato delle LoRA di controllo, preprocessing delle modalità e prompt derivano da
    MagicQuill V2 (https://github.com/zliucz/MagicQuillV2), (c) Zichen Liu, Yue Yu, Hao Ouyang, Qiuyu Wang,
    Shuailei Ma, Ka Leong Cheng, Wen Wang, Qingyan Bai, Yuxuan Zhang, Yanhong Zeng, Yixuan Li, Xing Zhu, Yujun Shen,
    Qifeng Chen ("MagicQuill V2: Precise and Interactive Image Editing with Layered Visual Cues", 2025). Il codice di
    riferimento è distribuito con licenza Creative Commons CC BY-NC 4.0 (NON COMMERCIALE; il README upstream riporta
    però anche un badge CC BY-SA 4.0: l'incongruenza va chiarita). Il processore di attenzione con LoRA condizionali
    discende a sua volta da EasyControl. Questo è un RIFACIMENTO adattato a diffusers 0.38, non una copia del codice.
    ATTENZIONE: il prodotto LTX Studio è venduto: l'uso di questi moduli e dei pesi MagicQuill in un'edizione
    commerciale richiede una decisione di licenza dell'utente (vedi report del task 9).

COME FUNZIONA (riassunto, vedi i singoli moduli)
    * Le 4 LoRA di controllo (local/removal/edge/color) NON sono LoRA diffusers: sono pesi q/k/v(+proj) applicati SOLO
      ai token di una immagine-condizione 512x512 (1024 token) che viaggia in coda alla sequenza immagine.
    * La condizione attraversa i blocchi con modulazione a timestep 0; il suo K/V (per blocco) viene scritto una volta
      al primo passaggio e riletto ai successivi; l'attenzione immagine->condizione riceve un bias log(gamma).
    * Si riusa il transformer diffusers GIÀ caricato: nessuna copia dei pesi; si sostituisce temporaneamente il solo
      forward dei blocchi (ripristinato a fine job).
"""
from .control_lora import ControlLoRA, ControlSpec  # noqa: F401
