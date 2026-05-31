# Studio del path modelli EROS (NSFW) — gruppo "LTX2.3 MODEL [10EROS]"

(Dallo screenshot MODELS + grafo. Nel golden catturato il gruppo è **bypassed**, quindi questi
nodi NON sono attivi nell'export attuale; ecco come andrebbero usati/migliorati per l'NSFW.)

## Cosa contiene il gruppo EROS
- **`6402` UNETLoader** = `ltx2310eros_v1.safetensors` (il modello NSFW dedicato).
- **Load LoRA** (nel gruppo 10EROS): `ltx-2.3-22b-distilled-lora-**dynamic**_fro09_avg_rank_105…`
  - strength **1.00** → `Set_EROS_SAMPLING_MODEL` (modello per il sampler)
  - strength **0.58** → `Set_EROS_UPSCALER_MODEL` (modello per la passata di upscale)
- Selezione STANDARD vs EROS via **`Any Switch` 5752** (any_01=STANDARD 5751, any_02=EROS 6402).

## Differenza chiave rispetto allo STANDARD
Lo STANDARD usa la distilled‑LoRA `…distilled-lora-**1.1**_fro90_ceil72_condsafe`. L'EROS usa invece
la **distilled‑LoRA dinamica** `…distilled-lora-**dynamic**_fro09_avg_rank_105`, a **due forze
diverse** per sampler (1.0) e upscaler (0.58). Quindi non basta scambiare l'UNET: l'EROS ha la sua
catena di distilled‑LoRA dedicata.

## Come lo gestisce oggi LTX Studio (e perché è migliorabile)
Per NSFW, il preset (`nsfw_presets` in `bindings.json`) imposta `5751.unet_name → ltx2310eros_v1`
e `collapse_model_selector` tiene il loader base. **Risultato**: si usa il modello EROS **ma con la
distilled‑LoRA dello STANDARD** (1.1_condsafe), non con la `dynamic` a 1.0/0.58 del gruppo EROS.

## Miglioramento proposto (fedele al workflow)
Quando `nsfw_mode == "nsfw"`, costruire la catena modello come il gruppo EROS:
1. UNET = `ltx2310eros_v1.safetensors` (6402).
2. + LoRA `ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105…` a **strength 1.0** per il sampler
   (e **0.58** per l'eventuale passata di upscale/HR).
3. Niente IC‑control / CelebVHQ (come per lo STANDARD: condizionali, default off).

## CABLATO (2026-05) ✅
Aggiunto in `core/bindings.json` il param `distilled_lora` → `[6428.lora_name, 6429.lora_name]` e nel
preset `nsfw_presets.nsfw`: `distilled_lora = "LTX-2\ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors"`.
Risultato verificato: `nsfw_mode=nsfw` → UNET `ltx2310eros_v1` + distilled‑LoRA **dynamic** (entrambi gli
stadi 6428/6429); `sfw` → invariato (distilled‑LoRA `1.1_condsafe`). Le forze per stadio (6428 sampler,
6429=0.58 refine) restano quelle del template. IC‑control/CelebVHQ restano condizionali (off di default).
