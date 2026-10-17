# Componente MagicQuill V2 — NOTICE di licenza

**Uso NON COMMERCIALE.** Questo componente (cartella `optional/magicquill/`, pacchetto scaricabile `mq2-code`) è
distribuito separatamente dall'applicazione e NON fa parte di nessuna build/installer: lo installa l'utente.

## Origine e attribuzione
Il codice di `magicquill_v2.py` e `mq2/*.py` è un RIFACIMENTO, adattato a diffusers 0.38, di **MagicQuill V2**
(https://github.com/zliucz/MagicQuillV2), (c) Zichen Liu, Yue Yu, Hao Ouyang, Qiuyu Wang, Shuailei Ma, Ka Leong Cheng,
Wen Wang, Qingyan Bai, Yuxuan Zhang, Yanhong Zeng, Yixuan Li, Xing Zhu, Yujun Shen, Qifeng Chen
("MagicQuill V2: Precise and Interactive Image Editing with Layered Visual Cues", arXiv:2512.03046, 2025).
Licenza del codice upstream: **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** —
testo in `LICENSE-upstream-CC-BY-NC-4.0.txt`. Il README upstream mostra anche un badge CC BY-SA 4.0; il file LICENSE del
repository è CC BY-NC 4.0.

Il meccanismo di attenzione con LoRA condizionali e cache K/V discende da **EasyControl**
(https://github.com/Xiaojiu-z/EasyControl, Apache-2.0 — testo in `LICENSE-EasyControl-Apache-2.0.txt`).

Non è una copia letterale (verificato contro il sorgente upstream: nessun blocco di 4 o più righe identiche), ma segue la
struttura dell'algoritmo upstream: va trattato come OPERA DERIVATA e sottoposto alla stessa licenza CC BY-NC 4.0.

## Pesi
Le LoRA `color/edge/local/removal/puzzle_lora.safetensors` (https://huggingface.co/LiuZichen/MagicQuillV2-models) sono
distribuite dagli autori con licenza **CC BY-NC-SA 4.0** (non commerciale, condivisione alle stesse condizioni).
Non sono incluse qui: si scaricano dal tab «Pesi modelli» dopo conferma di licenza.

## Condizioni d'uso in breve
- Solo uso personale/non commerciale; attribuzione agli autori; indica le modifiche (adattamento a diffusers 0.38,
  riscrittura in numpy/cv2, integrazione in LTX Studio).
- Redistribuzione: solo non commerciale, con questo NOTICE e i testi di licenza.
