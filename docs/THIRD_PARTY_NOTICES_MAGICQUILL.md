# MagicQuill V2 — attribuzioni

La funzione «MagicQuill V2» (tab Immagini > Inpaint, `inference_server/backends/magicquill_v2.py` e
`inference_server/backends/mq2/`) è un'implementazione NATIVA (diffusers 0.38, nessun ComfyUI) dell'idea e dell'algoritmo di
**MagicQuill V2** — (c) Zichen Liu, Yue Yu, Hao Ouyang, Qiuyu Wang, Shuailei Ma, Ka Leong Cheng, Wen Wang, Qingyan Bai,
Yuxuan Zhang, Yanhong Zeng, Yixuan Li, Xing Zhu, Yujun Shen, Qifeng Chen, «MagicQuill V2: Precise and Interactive Image
Editing with Layered Visual Cues» (arXiv:2512.03046, 2025), https://github.com/zliucz/MagicQuillV2.

- Il codice di questo progetto è scritto da zero per diffusers 0.38: nessuna riga copiata dal repository di MagicQuill V2
  (verificato confrontando i sorgenti: nessun blocco di 4 o più righe identiche). Segue la ricetta dell'originale: condizione
  512×512 in coda alla sequenza immagine, LoRA q/k/v(+proj) applicate solo ai suoi token, cache K/V scritta al primo passo,
  bias log(gamma) sull'attenzione verso la condizione, preprocessing e prompt delle modalità.
- Il meccanismo di attenzione con LoRA condizionali discende da [EasyControl](https://github.com/Xiaojiu-z/EasyControl)
  (Apache-2.0).
- Il repository originale riporta la licenza CC BY-NC 4.0 (uso non commerciale).

## Pesi (LoRA `color/edge/local/removal/puzzle_lora.safetensors`)
Autori: gli stessi, https://huggingface.co/LiuZichen/MagicQuillV2-models — licenza **CC BY-NC-SA 4.0** (non commerciale,
condivisione alle stesse condizioni). Nessun peso è distribuito con l'app né inserito nell'installer: li scarica l'utente dal
tab «Pesi modelli», dopo aver letto l'avviso di licenza e confermato.

## ControlNet usati dalle funzioni «Controllo struttura» e «ControlNet Inpainting»
Pesi scaricati dall'utente (non distribuiti con l'app): Shakker-Labs `FLUX.1-dev-ControlNet-Union-Pro-2.0` e
`alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta` (licenza FLUX.1 [dev] non commerciale), `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`
(Apache-2.0), FLUX.1 Redux dev (FLUX.1 [dev] non commerciale) con SigLIP so400m (Apache-2.0). La convenzione di maschera del
ControlNet di AlimamaCreative è ricostruita dalla scheda del modello e dalle shape del file, senza il loro codice (il
repository GitHub non dichiara una licenza di codice).
