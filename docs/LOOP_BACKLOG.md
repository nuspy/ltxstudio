# LOOP BACKLOG — completamento feature mancanti (Aree 3 → 2 → 1)

Legenda: `[ ]` aperto · `[x]` fatto · `[~]` DEFERRED (download >15GB / HW assente).
Vincoli: test video ≤10s (long‑video ≤30–60s); no download >15GB; i18n 15 lingue; feature_matrix; commit per item.

## AREA 3 — migrazione no‑ComfyUI
- [x] 3.1 ControlNet IC‑LoRA reale — ICLoraPipeline GGUF-patchata + IC-LoRA canny/depth/pose da disco (no download), firma video_conditioning; full-GPU val in #43
- [~] 3.2 FLUX.2 t2i — TE è nvfp4 (no fp8): modulo dequant nvfp4→bf16 PRONTO (flux2_te.py, no download 48GB); resta validazione live swizzle su Blackwell → Task #44
- [ ] 3.3 SeedVR2 nativo — 3B 14.6GB IN DOWNLOAD; integrare pipeline DiT + handler video_upscale opzione seedvr2
- [x] 3.4 ReTake nativo — RetakePipeline nel worker + ramo GUI _retake_native (sezione [start,end]); full-GPU val #43
- [ ] 3.5 UI conversione modelli (3 script in Import Models)
- [ ] 3.6 text→audio (T2AOneStage) + HDR/EXR export
- [ ] 3.7 Durata‑illimitata multi‑prompt (run_long) [long‑video ≤60s]
- [ ] 3.8 FlashVSR v1.1 (JunhaoZhuang/FlashVSR-v1.1, <15GB) — scaricare + integrare
- [x] 3.9 SD3 — ABORTITA DEFINITIVAMENTE (verificato: nessuna feature usa il modello SD3; EmptySD3LatentImage/ModelSamplingSD3 sono nodi generici, "sd3" solo voce catalogo)
- [ ] 3.10 Test LIVE end‑to‑end path nativi (≤10s)

## AREA 2 — Videomusicale
- [ ] 2.1 Anteprima video inline (thumbnail)
- [ ] 2.2 Allineamento Whisper/forced‑align
- [ ] 2.3 Ghosting tagli (crossfade)
- [ ] 2.4 Parametri canzone (temp/cfg/seed) in fase Canzone
- [ ] 2.5 Riuso reale immagine fornita come scena 0
- [ ] 2.6 Transizioni tra spezzoni
- [ ] 2.7 Export NLE del videomusicale
- [ ] 2.8 Batch generazione asincrona robusta

## AREA 1 — AI Film Studio
- [ ] 1.1 Generazione immagini variazioni/abbigliamento
- [ ] 1.2 Rigenerazione inline singolo shot
- [ ] 1.3 Timeline visuale + transizioni + animatic
- [ ] 1.4 SFX puntuali per azione
- [ ] 1.5 Multilingua/doppiaggio
- [ ] 1.6 Storyboard/shot‑list export (documento)
- [ ] 1.7 Continuity: elementi vietati/forzati
- [ ] 1.8 Serie/casting UI
- [ ] 1.9 NOTA A: auto‑apertura AssetDialog su #variante
- [ ] 1.10 Seed locking per asset/shot
- [ ] 1.11 Completare 13 lingue minori
- [ ] 1.12 Multi‑GPU + pausa/ripresa coda
