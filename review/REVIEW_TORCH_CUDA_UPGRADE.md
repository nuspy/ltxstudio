# Valutazione — upgrade Torch 2.9 + CUDA 13 (Blackwell sm_120) — #103
_Data: 2026-07-10 · valutazione STATICA (nessun test GPU: processi pesanti in esecuzione). Versioni misurate
leggendo `torch/version.py` di ogni venv (nessun import torch)._

## Stato REALE della flotta venv (misurato)
| venv | ruolo | torch | CUDA |
|---|---|---|---|
| `Comfy_Nexus.bak/ComfyUI/venv` | **app/GUI** | **2.8.0** | cu128 |
| `Tools/ltx2_venv` | **video LTX-2** | **2.8.0** | cu128 |
| `Tools/inference_server_venv` | **immagini/backend** | 2.11.0 | cu128 |
| `Tools/CosyVoice/venv` | voce | 2.9.1 | **cu130** |
| `Tools/F5-TTS`, `TRELLIS2`, `SeedVR2`, `FlashVSR`, `ACE-Step`, `heartlib` | tool | 2.11.0 | cu128 |

Osservazione chiave: la flotta è **eterogenea** (torch 2.8 / 2.9.1 / 2.11.0; CUDA 12.8 e 13.0). I due venv
**core** (GUI + video) sono sui **più vecchi** (2.8.0), i tool periferici già su 2.11.0.

## Verdetto: l'obiettivo del task è in gran parte SUPERATO
1. **Blackwell sm_120 è GIÀ supportato.** CUDA 12.8 (cu128) include l'architettura `sm_120` della RTX 5090 →
   l'app gira già su Blackwell con i venv attuali. **Non serve CUDA 13 per far girare il 5090.**
2. **Torch ≥ 2.9 è già presente** in 8 venv su 10 (molti su 2.11.0). L'"upgrade a 2.9" riguarderebbe solo i
   due core rimasti a 2.8.0 — e 2.8.0+cu128 è comunque Blackwell-capable.
3. Quindi il problema REALE non è "salire a 2.9+CUDA13", ma l'**INCOERENZA** tra venv, che complica: le wheel
   precompilate condivise (#75), il consolidamento venv (#61) e la riproducibilità dei build.

## CUDA 13 (cu130): conviene? — NO, non ora
- **Beneficio marginale** su Blackwell: cu128 espone già tutte le feature sm_120 usate (fp8 scaled_mm, nvfp4,
  matmul). CUDA 13 non sblocca nulla di necessario per i modelli attuali.
- **Rischio ALTO di rottura degli acceleratori**: SageAttention, FlashAttention, xFormers, spandrel, onnxruntime-gpu
  e le wheel precompilate SECourses sono buildate per **cu128**. Passando a cu130 andrebbero **ricompilate**
  (tempo di build lungo, dipendenze, sm_120) — proprio ciò che #75 (wheel precompilate) vuole evitare.
- Un solo venv (CosyVoice) è già su cu130 e funziona **isolato** → conferma che cu130 va bene per singoli tool,
  ma imporlo a tutta la flotta è costo>beneficio.

## Raccomandazione
1. **Standardizzare i CORE su UNA versione cu128** (non cu130): allineare `ComfyUI/venv` (app) e `ltx2_venv`
   (video), oggi a 2.8.0, alla stessa release degli altri core. Candidata: **torch 2.9.1+cu128** (LTS-like,
   compatibile con gli acceleratori cu128 già installati) oppure 2.11.0+cu128 SE testato con SageAttention/
   FlashAttn sul 5090. La scelta va **validata live** (giro di generazione immagine+video) — rimandata (no GPU).
2. **NON adottare CUDA 13** in modo generalizzato: tenerlo solo dove un tool lo richiede (CosyVoice), isolato.
3. I venv periferici (2.11.0+cu128) restano come sono finché #61 non li consolida.
4. Coordinare con **#75** (wheel precompilate per arch): fissare cu128 come target unico delle wheel
   (flash_attn/xformers/sageattention/cumesh) → un solo set di binari per tutta la flotta core.

## Passi (quando la macchina sarà libera per il live-test)
- Pin `torch==<scelta>+cu128` in `ltx2_venv` e `ComfyUI/venv`; reinstallare gli acceleratori cu128
  corrispondenti; giro di smoke t2i + t2v + a2v + upscale per confermare (no regressioni fp8/nvfp4/sage).
- Aggiornare `docs/precompiled` / installer con la matrice unica cu128.
- **Non** toccare cu130 se non per necessità puntuali documentate.

**Conclusione:** l'upgrade "2.9+CUDA13" come formulato è superato dai fatti (Blackwell già supportato in cu128,
torch già ≥2.9 quasi ovunque). L'azione utile è **omogeneizzare i core su cu128**, non salire a CUDA 13. Costo
basso, rischio basso; il vero guadagno è coerenza/riproducibilità, non prestazioni.
