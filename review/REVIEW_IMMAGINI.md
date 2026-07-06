# Review PIPELINE IMMAGINI — LTX Studio (2026-07-07)

Reviewer: agente dedicato (very thorough). Backend nativo :8189, zero ComfyUI per i task supportati.

## Stato per funzione

| Funzione | Stato | Note chiave |
|---|---|---|
| T2I (FLUX.1/2, Z-Image, Ernie) | ✅ OK | auto-settings sensati; GAP: negative_prompt saltato in silenzio dove non supportato (Z-Image); true_cfg solo per PuLID |
| Inpaint/Outpaint | ⚠️ FRAGILE | **GAP CRITICO: senza maschera → maschera bianca → rigenera TUTTA la foto col Fill** (serve modalità img2img/strength esplicita); mapping strength→denoise non documentato; guidance clamp invisibile all'utente; feather hardcoded |
| PuLID (volto) | ✅ OK | robusto; multi-foto via "\|"; errori silenziosi da loggare; symlink InsightFace da verificare su Windows |
| Auto-select modelli/componenti | ✅ OK | detection 3 livelli (metadata→fingerprint→nome); fingerprint fallisce su prefisso model.diffusion_model (fix nel backend, non nel detector); colori VRAM ok |
| Integra (Kontext/FLUX.2) | ✅ OK | dual-path mask-based (Fill g=18) vs Kontext (g=2.5/4) non evidente in GUI; verificare _reload al cambio engine |
| Componi multi-foto | ⚠️ GAP | multiref ok; **bottone "maschera foto" senza callback** (frame1_tab.py:768-770); stitch non implementato nel backend? |
| Upscale/definizione | ✅ OK | tiling 512/32 ok; stego+watermark post ok |
| Recolor | ✅ OK | prompt-based Kontext; color picker non interattivo (solo hex?) |
| Segmentazione | ✅ OK | multimask iou_scores appena fixato; SAM2 assente → fallback bbox silenzioso (loggare) |
| Stego + dither | ✅ OK | applicati su tutti i percorsi; best-effort (non fail-closed) — se stego fallisce nessun avviso |

## Wiring GUI→backend
- gen_engine (comfy/diffusers) di fatto ignorato (il router decide) — coerente ma da chiarire in GUI.
- gen_encoder/clip_type/vae spesso auto-risolti nel backend anche se scelti in GUI.
- Il resto dei parametri arriva correttamente (tabella completa nel report esteso).

## Top 10 interventi (dal reviewer)
1. **CRITICO**: modalità img2img SENZA maschera nel tab inpaint (strength-based) — l'utente l'ha chiesta esplicitamente
2. Loggare negative_prompt ignorato per arch
3. Sincronizzare guidance clamp GUI↔backend (warning visibile quando clampata)
4. Fallback callback preview con log visibile
5. Documentare strength→denoise nel tooltip
6. Esporre feather_size in GUI inpaint
7. Validare SAM2 multimask iou_scores (fatto in parte 07/07)
8. Indicatore visivo componenti auto-selezionati
9. Centralizzare _ARCH_DEFAULTS in model_catalog
10. Doc tecnica (fingerprint, dither ±1.7, PuLID intervals)

## Mancanti vs stato dell'arte
ControlNet immagini nativo, HDR tone mapping, faceswap in UI immagini, batch generation, inpaint multi-pass, img2img strength su t2i, custom scheduler curve.
