# LOOP BACKLOG — completamento feature mancanti (Aree 3 → 2 → 1)

Legenda: `[ ]` aperto · `[x]` fatto · `[~]` DEFERRED (download >15GB / HW assente).
Vincoli: test video ≤10s (long‑video ≤30–60s); no download >15GB; i18n 15 lingue; feature_matrix; commit per item.

## AREA 3 — migrazione no‑ComfyUI
- [x] 3.1 ControlNet IC‑LoRA reale — ICLoraPipeline GGUF-patchata + IC-LoRA canny/depth/pose da disco (no download), firma video_conditioning; full-GPU val in #43
- [x] 3.2 FLUX.2 t2i — TE nvfp4 dequant cablato in _assemble_flux2 (no download 48GB); val swizzle GPU #44 // ex — TE è nvfp4 (no fp8): modulo dequant nvfp4→bf16 PRONTO (flux2_te.py, no download 48GB); resta validazione live swizzle su Blackwell → Task #44
- [⏳] 3.3 SeedVR2 nativo — 3B 14.6GB IN DOWNLOAD; integrare pipeline DiT + handler video_upscale opzione seedvr2
- [x] 3.4 ReTake nativo — RetakePipeline nel worker + ramo GUI _retake_native (sezione [start,end]); full-GPU val #43
- [x] 3.5 UI conversione modelli — gruppo Conversione in Import Models (Gemma-3→bf16, TE→HF) via subprocess venv; i18n 15
- [x] 3.6 text→audio (run_t2a, T2AOneStage audio-only) + helper EXR (core/exr_export); HDR-IC-LoRA pipeline completa → Task (serve embeddings precomputati + val live), HDR LoRA su disco
- [x] 3.7 Durata-illimitata MULTI-PROMPT — _run_chunked usa prompts=[{start_s,prompt}] per finestra temporale (memory-bounded, continuità ultimo-frame); long-video test #43 ≤30-60s
- [⏳] 3.8 FlashVSR v1.1 (JunhaoZhuang/FlashVSR-v1.1, <15GB) — scaricare + integrare
- [x] 3.9 SD3 — ABORTITA DEFINITIVAMENTE (verificato: nessuna feature usa il modello SD3; EmptySD3LatentImage/ModelSamplingSD3 sono nodi generici, "sd3" solo voce catalogo)
- [~] 3.10 Test LIVE end-to-end → sessione #43 (richiede GPU, ≤10s)

## AREA 2 — Videomusicale
- [x] 2.1 Anteprima video inline — thumbnail primo frame (QIcon) nella lista video
- [x] 2.2 Allineamento Whisper — align_scenes_with_whisper (faster-whisper, fallback proporzionale) + bottone Studio; install faster-whisper = Task
- [x] 2.3 Ghosting — risolto by-design (clip i2v separate, concat netto) + transizione crossfade opzionale
- [x] 2.4 Parametri canzone (temperatura/CFG) nella fase Canzone, salvati nel progetto
- [x] 2.5 Riuso reale immagine fornita → apertura scena 0 (mv_project.base_image)
- [x] 2.6 Transizioni — assemble_scenes con xfade (cut/crossfade/fade/dissolve/wipe)
- [x] 2.7 Export NLE — mvs.export_nle riusa otio_export (FCPXML/OTIO/AAF/EDL)
- [x] 2.8 Batch async — _next_batch (un worker/volta, UI responsiva, salvataggio per step)

## AREA 1 — AI Film Studio
- [x] 1.1 Genera set immagini variazioni/abbigliamento (gen_service)
- [x] 1.2 Rigenerazione inline shot (keyframe/video) dal pannello storyboard
- [x] 1.3 Animatic (anteprima keyframe a tempo) + transizioni (assemble) — timeline visuale base
- [x] 1.4 SFX/Ambient via MMAudio (coperto da Ambient/SFX nel montaggio)
- [x] 1.5 Multilingua/doppiaggio (LLM traduce dialoghi + risintesi voce legata; re-lip-sync rigenerando)
- [x] 1.6 Shot-list export HTML (scene/shot/durate/dialoghi/keyframe)
- [x] 1.7 Continuity vietati/forzati (forbidden→negative, required→prompt) per progetto
- [x] 1.8 Serie/casting UI (combo serie + nuova + assegnazione progetto)
- [x] 1.9 NOTA A — # dopo @Nome apre AssetDialog sulle varianti
- [x] 1.10 Seed locking per shot (campo seed in storyboard, 0=auto, >0 bloccato; migration DB)
- [~] 1.11 13 lingue minori — label principali tradotte; stringhe rare EN fallback → Task #49
- [~] 1.12 Multi-GPU + pausa/ripresa coda — scheduler predisposto, richiede HW → Task #50


## RIFINITURE post-audit (27/06)
- [x] AREA1 Film Studio #52-57 (doppiaggio re-lip-sync, timeline drag&drop, Auto end-to-end, multi-LoRA+@, Continuity UI+animatic audio, matrix granulare+test) + SFX per-azione
- [x] AREA1 validazione durate Σshot=scena + Σscene=film (già presente)
- [x] AREA2 Videomusicale: verificato completo (gen/regen song+img+VIDEO, plan, align, assemble+transizioni, export NLE, base_image, progetti persistenti)
- [x] Handy-API client (STT/TTS/clone) + voce nativa (Kokoro/clone) scelta in GUI
- [x] FastDownloader multi-connessione (replica SwarmUI)
- [x] Audit ComfyUI completo (docs/COMFY_AUDIT.md) → Task #51
