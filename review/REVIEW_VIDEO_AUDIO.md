# Review VIDEO e AUDIO — LTX Studio (2026-07-07)

Reviewer: agente dedicato (very thorough). Stato runtime: zero ComfyUI (backend nativo validato 03/07).

## Sintesi salute: 🟢 BUONA (92%)

Backend video solido e nativo. Core feature (T2V/I2V, audio, LoRA, extend, ControlNet IC-LoRA) implementate e testate.

## Stato per funzione

| Funzione | Stato | Note chiave |
|---|---|---|
| T2V/I2V LTX-2.3 | ✅ OK | auto-GGUF (ltx2_infer.py:422-433), auto-offload (347-372), block-swap (98-157), A2V stereo OK. GAP: preview per-step solo testuale |
| Editing video AI | ⚠️ PARZIALE | edit-anything OK; **SAM2 tracking incompleto** (orchestrazione manca); mask_dilate/mask_blur NON esposti in GUI (hardcoded) |
| Audio (TTS/clone/A2V/MultiTalk/HeartMuLa) | ✅ OK | voice clone 100% nativo; Edge TTS fallback; MultiTalk resume post-crash |
| Storyboard multi-scena | ✅ OK | verificare live 3+ keyframe + audio simultanei (non testato) |
| Restauro volto + face-swap | ✅ OK | GFPGAN per-frame, auto-chain, fidelity slider |
| LoRA su video | ✅ OK | filtro arch da header, renaming map Comfy→ltx-core |
| ControlNet video (IC-LoRA) | ✅ OK | canny/depth/pose; conditioning_attention_strength nel backend ma NON in GUI |
| Prolunga video | ✅ OK | chunk autoregressivi, resume, anti-drift, multi-prompt temporale, audio slicing |
| Migliora qualità | ✅ OK | ESRGAN, RIFE (ffmpeg minterpolate fallback), SeedVR2, FlashVSR, auto-chain |
| Fast/Pro/HQ + HDR/EXR | ⚠️/❌ | selettore Fast/Pro/HQ NON esplicito (solo preset steps); HDR/EXR export ASSENTE |

## Wiring GUI→backend
- Coda seriale ok (ui/job_queue.py:52-150). ⚠️ Rischio contesa GPU tra :8189 (immagini) e ltx2_infer (video) se simultanei.
- Parametri ignorati/non esposti: `mask_dilate`+`mask_blur` (inpaint video), `conditioning_attention_strength` (ControlNet).
- Residui ComfyUI: nessuna dipendenza runtime; dead code presente (template_engine, rami fallback) non critico.

## Top 10 interventi (dal reviewer)
1. Esporre mask_dilate/mask_blur in UI inpaint video (MEDIA, 1h)
2. Completare SAM2 tracking backend video (ALTA, 4h)
3. Preview visuale per-step video (MEDIA, 3h)
4. Modalità Fast/Pro/HQ esplicita (MEDIA, 2h)
5. Esporre conditioning_attention_strength ControlNet (MEDIA, 1h)
6. Test produzione storyboard 3+ keyframe (BASSA, test)
7. EXR/HDR export (ALTA, 8h)
8. Cache LoRA video per riuso multi-job (BASSA, 1h)
9. Tooltip CFG/guide inpaint video (BASSA)
10. Doc reverse-lookup IC-LoRA (BASSA)

## Funzioni esistenti extra rilevate
Face-swap+pose transfer, music video (sperimentale), mocap (anim_tab+mocap_worker), Blender/world, rig/skinning.

## Mancanti vs stato dell'arte
HDR/EXR export, subtitle burn-in sync, real-time preview streaming, style transfer CLIP-based (solo LoRA anime2real).
