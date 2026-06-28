# FLUX.2 dev sotto 32GB (e meno) — piano d'azione

Ricerca 28/06 (Reddit + SECourses discussion #357 + apatero + unsloth). Per il task #44.
**Conferma chiave: la nostra strada (NVFP4) è quella giusta.** Il nostro `flux2_te.py` già fa dequant
nvfp4 e `flux2_transformer.py` fp8-forward + block-swap → resta da debuggare il forward (output rumore).

## Numeri su RTX 5090 (32GB) — quale formato

| Formato | VRAM | Tempo (40 step) | Qualità vs BF16 |
|---|---|---|---|
| **NVFP4 mixed** | **14 GB** | **5.7 s** | quasi identica ("almost no diff") ← MIGLIORE 5090 |
| FP8 Scaled (mixed) | ~16-18 GB | veloce (> GGUF Q8) | quasi identica — **universale** |
| GGUF Q8_0 | 14-16 GB | 7.97 s/it | 98-99% |
| BF16 (full) | 26 GB | ~14 s | baseline |

NVFP4 è 2× più veloce di BF16 e usa metà VRAM. FP8 Scaled = miglior compromesso universale.

## Scendere sotto 24/16/12/8 GB — GGUF (repo: `unsloth/FLUX.2-dev-GGUF`)

| Quant | Size | VRAM | Qualità | GPU target |
|---|---|---|---|---|
| Q8_0 | ~12GB | 14-16GB | 98-99% | 16GB |
| Q6_K | ~9GB | 11-13GB | ~95% | 12GB |
| Q5_K_M | ~8GB | 9-11GB | ~90% | 8GB (sweet spot) |
| Q4_K_M | ~6GB | 7-9GB | 75-85% | 8GB / 6GB |

GGUF dequant on-the-fly: **abbiamo GIÀ il codice** (`ltx-core` GGUF dequant del motore video, task #40) → riusabile per FLUX.2.

## Tecniche di risparmio VRAM (stackano, moltiplicative)
1. **Text encoder Mistral offload** dopo l'encoding → libera 2-4GB (encode → sposta TE su CPU → poi DiT).
2. **VAE offload** (su CPU tranne durante decode).
3. **Block-swap** del transformer (già implementato in flux2_transformer.enable_blockswap).
4. `--cache-none` / `--disable-smart-memory` equivalenti: scaricare i modelli tra una gen e l'altra.
5. **CUDA 13 + Flash/Sage Attention/xFormers** (wheel precompilati #75) → guadagni, specie su GGUF.

## Piano implementativo (#44)
1. **Debug forward NVFP4** (priorità): output rumore = dequant/forward del TE Mistral o del DiT fp8.
   Verificare scale/zero-point nvfp4 (formato comfy_quant) e l'ordine dequant×scale nel forward.
2. **Aggiungere path GGUF** (Q8/Q6/Q5/Q4) via il dequant esistente (#40) → copre 8-24GB.
3. **TE Mistral offload** post-encoding + VAE offload (libera 2-4GB).
4. **Auto-select formato per VRAM** (profili #74): 5090/32GB→NVFP4; 16GB→Q8/FP8; 12GB→Q6; 8GB→Q5_K_M.
5. Pesi: NVFP4 (BFL/comfy) già scaricati; GGUF da `unsloth/FLUX.2-dev-GGUF`.

Fonti: SECourses discussion #357, apatero low-vram guide, unsloth/FLUX.2-dev-GGUF, botmonster.
