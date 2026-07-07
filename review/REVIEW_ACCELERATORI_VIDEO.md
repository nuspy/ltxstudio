# Audit ACCELERATORI — lato VIDEO/AUDIO/TRAINING (2026-07-07)

Venv verificati: ltx2_venv torch 2.8.0+cu128 + triton 3.4 + sageattention ✓ (flash_attn/xformers ✗);
inference_server_venv 2.11+cu128; ACE-Step venv15 2.11+cu128.

## Stato (sintesi): il video è GIÀ ben ottimizzato (~60% acceleratori attivi)
| Acceleratore | LTX-2 (t2v/i2v/a2v/iclora/vinpaint/extend) | MultiTalk/Wan | Musica | Training |
|---|---|---|---|---|
| SageAttention | ✓ ON (monkey-patch F.sdpa→sageattn, fallback SDPA; LTX_SAGE=0 per off) | ✓ (attention_mode=sageattn) | ✗ | — |
| fp8_scaled_mm | ✓ ON (+20-30% matmul) | ✗ | ✗ | — |
| GGUF Q4 dequant | ✓ auto (22B 27.8GB→14GB; blocca torch.compile: GGMLTensor non tracciabile) | ✗ | ✗ | — |
| block-swap | ✓ (per-layer) | ✓ auto | ✗ | — |
| VAE tiling | ✓ TilingConfig | ✓ hardcoded | ✗ | — |
| **TeaCache** | ✓ IMPLEMENTATO (workers/teacache.py + slider GUI) ma thresh **0.0 = OFF** | ✗ non esponibile (wrapper Wan, architettura diversa) | ✗ | — |
| torch.compile | ✗ (GGUF + subprocess one-shot: warm-up non ammortizzato) | ✗ | opt-in | opt-in |
| Gradient ckpt / adamw8bit / quantize | — | — | — | ✓ ON |
| FlashAttention | ✗ (wheel SECourses cp311+cu131 incompatibile con py312+cu128 → richiede migrazione #103) | | | |

## Interventi applicati OGGI
1. **TeaCache nei preset** (scrive NEI controlli GUI, regola utente): ⚡FAST → 0.05 (bozza, ~2×),
   🎯PRO → 0.03 (~1.6×, drift impercettibile a 24 step), 💎HQ → 0.0 (purezza assoluta).
   Lo slider manuale resta per il fine-tuning; calibrazione live consigliata al primo uso.
2. **Log GGUF selezionato** in _auto_gguf (prima fallback silenzioso).

## Scartati con motivazione
- TeaCache su MultiTalk: WanVideoWrapper non espone hook sul velocity-model → reverse-eng non banale.
- torch.compile su LTX-2: GGUF non tracciabile + worker one-shot (warm-up perso). Rivalutare se nasce
  il server video persistente.
- FlashAttention 2.8.4: richiede migrazione py3.11+cu131 (task #103, MAJOR, rinviato).
- DeepSpeed (solo training multi-GPU), xformers (coperto da Sage/SDPA), CUDA graphs (shape variabili),
  channels_last (marginale).

## Prossimi (quando si vuole)
- Calibrare i coefficienti TeaCache per LTX 2.3 (il paper usa 0.9) — 2h di A/B live.
- #103 migrazione CUDA 13.1 + py311 → sblocca flash_attn (+10-15%) [MAJOR].
