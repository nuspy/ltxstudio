# Report test GUI → backend REALE

Sessione: test guidati dalla GUI con backend reale, low-res, un job alla volta. Legenda:
✅ funziona · ⚠️ riparato in questa sessione · 🔒 pesi mancanti (non scaricabili ora) · ❌ rotto · ⏳ in corso.

## Infra (Fase 0)
- ✅ **Selettore motore video** GUI (LTX-2 nativo / Comfy) — `video_tab.video_engine_sel` + routing in `_generate_video_native` + i18n 15 lingue. Commit 056db8a.
- ✅ **Film Studio**: attivazione documentata (menu **Strumenti → 🎬 AI Film Studio**, non una tab). Commit e216d9c.
- ✅ **Probe pesi locali**: 20/32 feature con pesi locali; flag mancanti (sotto).

## Wiring GUI→backend (mocked Spy, 594/594 PASS)
La suite `tests/gui_regression/run_all.py` pilota la GUI e verifica che ogni controllo raggiunga il
`params` del backend. **594/594 verde** dopo allineamento casi integra (mask-based) + esclusione Redux.

## Riparazioni (Fase 1)
- ⚠️ **i2v `ref_strength` (Forza immagine)**: era hard-coded 1.0 in `ltx2_infer` → ora usato. Commit ffae095.
- ⚠️ **`i2v_method` inplace-soft**: ora cablato (inplace = riferimento morbido strength≤0.6). Commit ffae095.
- ⚠️ **Integra mask-based** (sessione precedente): usa inpaint+mask → fonde l'oggetto (non solo posiziona). Commit ff66c86.
- ⚠️ **Segment solo GroundingDINO** (use_sam): toggle GUI cablato. Commit d9c6aa8/f9c718e.

## Pesi mancanti (🔒 — non testabili senza download, connessione metered)
- voce: fish, vibevoice, cosyvoice2 · face-restore: codeformer/GFPGAN · RIFE ckpt · ControlNet IC-LoRA (canny/depth/pose) · Klein qwen3-8b.

## Pesi locali presenti (testabili)
- voce: edge, kokoro, handy, **f5tts, xtts2** · musica: heartmula, acestep, mmaudio · video: LTX-2 (gemma/vae/transformer), LoRA edit/inpaint/anime/real · SeedVR2 default · upscale: 23 modelli · faceswap insightface · FLUX.2 transformer.

## Matrice test reale (in aggiornamento)
| Feature | Esito | Note |
|---|---|---|
| (in corso) | ⏳ | |
