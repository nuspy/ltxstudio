# Audit ACCELERATORI — lato IMMAGINI (2026-07-07)

Ambiente verificato: torch 2.11.0+cu128, diffusers 0.38, flash_sdp_enabled=True; venv: sageattention ✓,
xformers 0.0.35 ✓, triton 3.7.1 ✓, flash_attn pacchetto ✗ (inutile: SDPA-flash nativo su Blackwell), DeepSpeed ✗.

## Matrice (sintesi)
| Acceleratore | Stato | Copertura | Nota |
|---|---|---|---|
| SageAttention INT8 | ✓ t2i FLUX.1 | solo t2i puro | GIUSTO off su inpaint/kontext (rumore INT8 sul blending, confermato) |
| SDPA-flash | ✓ tutti | 100% | kernel flash automatico su sm_120 |
| FBCache | OFF (NATIVE_FBCACHE) | — | regressione qualità confermata; eventuale riattivazione SOLO t2i con soglia 0.20 previo A/B |
| torch.compile | OFF (opt-in) | — | ~1.5×; skip su fp8-storage; → DA ATTIVARE con fallback |
| fp8 layerwise | ✓ auto | FLUX.1 fp8 | fatto in FASE C |
| VAE tiling | OFF | — | giusto così (≤1536, seam) |
| Upscale ESRGAN | eager PURO | — | NESSUNA ottimizzazione → autocast fp16 (+20-30%) |
| Segment DINO+SAM2 | eager PURO | — | idem |
| TeaCache | assente | — | per FLUX ROI basso/rischio alto → SCARTATO (lato immagini) |
| CUDA graphs / channels_last / DeepSpeed | assenti | — | scartati (shape dinamiche / marginale / solo training) |

## Interventi decisi
1. **torch.compile default ON** con `torch._dynamo.config.suppress_errors` (fallback eager automatico su
   qualsiasi errore di grafo → zero rischio job-killed). Skip già presente su fp8-storage/blockswap.
2. **autocast fp16** su upscale (tiled + once) e segment (GroundingDINO+SAM2 forward) — +20-30%, rischio minimo.
3. FBCache: resta OFF; protezione per-arch aggiunta per uso futuro con NATIVE_FBCACHE=1 + threshold 0.20.
4. TeaCache immagini / CUDA graphs / channels_last / flash_attn pacchetto / DeepSpeed: SCARTATI (motivazioni sopra).

Guadagno cumulativo atteso lato immagini: ~1.8× sui percorsi bf16 + upscale/segment più veloci.
