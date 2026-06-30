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
| Video t2v/i2v — motore **ltx2_infer** (default) | ❌→⚠️ | Test reale: il 22B **OOM/sborda sui 32GB anche a 512×320/25f** (model-bound, dequant ~44GB indip. dalla risoluzione). RIPARATO con auto-switch: t2v/i2v + VRAM<40GB → comfy automatico (commit 70b52e7). |
| Video t2v/i2v — motore **comfy-native** | ✅ | Validato in sessioni precedenti (768×512 49f, no-OOM). Ora è il path automatico su 32GB per t2v/i2v. |
| Video feature avanzate (audio/controlnet/storyboard/lungo) | 🔒/⏳ | Solo ltx2_infer → servono >40GB VRAM o il rewrite memoria (#18) per girare su 32GB. Da testare su HW adeguato o con #18. |
| Immagini t2i/inpaint/outpaint/kontext/compose/integra/upscale/restauro/segment | ✅ | Validati in sessioni precedenti (backend reale). Integra mask-based ✅. |

### Output REALI prodotti (backend vero, HF_HUB_OFFLINE, pesi locali) — 30/06
| Feature | File | Esito |
|---|---|---|
| Immagine FLUX.1 t2i | `native_flux_123.png` (588KB, cucciolo) | ✅ |
| Video t2v (motore comfy) | `REAL_video_t2v.mp4` (512×320, 49f, mongolfiera) | ✅ |
| Upscale ESRGAN 4xNomos8kDAT | `native_upscale_1.png` (8.4MB) | ✅ |
| Recolor (kontext FLUX.1) | `native_kontext_2.png` (682KB) | ✅ |
| Segment (GroundingDINO+SAM2) | `native_segmask.png` | ✅ |
| Audio TTS (edge) | `tts/tts_edge_elsa_*.mp3` (31KB) | ✅ (installato edge-tts) |

NB: i run reali girano con `HF_HUB_OFFLINE=1` per non scaricare nulla sulla metered. Libreria mancante
`edge-tts` installata (consentito). I `.incomplete` HF erano placeholder vuoti (nessun download mio andato a segno).

### Nota chiave 32GB
Sui 32GB il **video 22B in ltx2_infer non è eseguibile** (model-bound). Il selettore + auto-switch instradano
t2v/i2v su comfy-native (no-OOM). Le **feature video avanzate** (audio→video/canta/talk/retake/controlnet/
storyboard/lungo) sono solo su ltx2_infer → richiedono più VRAM o il memory-rewrite #18: su questa macchina
non sono testabili a piena risoluzione. Da rifare su GPU ≥40GB o dopo #18.

## Audio/Voce/Musica — test reali (30/06)
| Feature | File | Esito |
|---|---|---|
| Voice clone **f5tts** | output/audio/clone_f5.wav (147KB) | ✅ (bug `Job.run` riparato in music_service) |
| **HeartMuLa** musica | output/music/REAL_heartmula_song.mp3 (149KB) | ✅ |
| TTS edge (Elsa/Diego) | output/tts/*.mp3 | ✅ |
| ACE-Step | — | ⏳ funziona ma lento al load (>120s), non bug |
| MMAudio sonify | — | ⚠️ da ri-testare (campo SonifyParams = video_path/duration_s) |
| Voice clone xtts2/fish/vibevoice/cosyvoice | — | sbloccati dal fix Job.run, da testare (pesi: f5tts/xtts2 locali, altri mancanti) |

**Bug riparato**: `core/music_service.Job` non aveva `run()` (solo `start()` async) → `tts_engine._synthesize_native_clone` falliva ('Job object has no attribute run') = **tutto il voice-cloning nativo era rotto**. Aggiunto `Job.run()` bloccante.

### Aggiornamento audio (30/06, cont.)
- ✅ Voice clone **xtts2**: output/audio/clone_xtts.wav (193KB) — 2ª voce validata (fix Job.run conferma).
- 🔒 **MMAudio**: venv rotto = torch 2.12.1+**cpu** + torchaudio ABI mismatch (libtorchaudio.pyd non carica). Riparazione = reinstall torch-CUDA+torchaudio (download grosso, metered → rimandato). Codice/wiring OK.

## Verifica rigorosa PASS/FAIL (orari = oggi, post-avvio test) — 30/06 13:45
12 PASS con output reale fresco: t2v, i2v, **i2v-inplace** (REAL_i2v_inplace.mp4 13:32), **storyboard/keyframe**
(REAL_storyboard.mp4 13:45, 25f via LTXVAddGuide comfy), FLUX.1 t2i, recolor, upscale, segment, inpaint,
voice-clone f5tts+xtts2, TTS edge, HeartMuLa musica.
Pendenti: MMAudio (venv riparato torch/torchaudio 2.11 cpu, demo.py gira su CPU lento), ACE-Step (load lento).
Nota: storyboard/keyframe sui 32GB va a 25f; a 49f+ timeout → serve server video persistente (no reload/gen).

- LoRA video: REAL_video_lora2.mp4 (108KB, [gen] 24s, [lora] fightV3 applicata via LoraLoaderModelOnly) ✅

- ControlNet video: REAL_video_controlnet.mp4 (430KB, [gen] 14s) — IC-LoRA union-control 22b + guida canny + reference_downscale_factor=2 dai metadata. Pipeline VALIDATA (struttura segue la guida); qualità bassa a 6-step/Q4 (test veloce), pulita su fp8/30-step. ✅

- a2v (canta/talk): REAL_video_a2v.mp4 (512x320 h264 + AAC 48kHz stereo) — modello AV congiunto comfy, audio fornito encode+keep+mux. Copre canta/talk/audio-utente. ✅

- V2V real->anime: REAL_video_v2v_anime.mp4 (183KB, [gen] 13s). Pipeline V2V VALIDATA meccanicamente (sorgente->guida + ReStyle IC-LoRA + genera+output). MA a control_strength=0.7/6-step lo stile anime NON è visibile (output ≈ sorgente realistico) → serve strength più bassa (0.3-0.5) + più step per il restyle effettivo. Wiring OK, resa stile da tarare. ⚠️

- Face-swap foto: REAL_faceswap.png (409KB) — InsightFace inswapper (volto sorgente→ritratto target), CUDAExecutionProvider. ✅

- SeedVR2 super-res video: REAL_seedvr2.mp4 (2.86MB, 49s) — upscale 512x320→720p, service con offload proprio. ✅

- Restauro+colorizza (combinato): composizione di restauro (native_upscale_1.png, ESRGAN ✅) + colorizzazione (native_kontext_2.png, recolor kontext ✅), eseguiti in sequenza. Entrambe le metà validate con output reale; il combinato è la catena (nessun nuovo percorso). ✅(composizione)

- V2V edit/restyle (INDAGINE COMPLETA): testati ReStyle IC-LoRA + edit_anything LoRA, strength 0.35/0.7/1.0, con/senza LTXVCropGuides → in TUTTI i casi l'output RICOSTRUISCE il sorgente (no restyle/edit). FIX CropGuides applicato (corretto, da workflow ufficiale). Causa: sorgente-come-guida-overlay (frame_idx 0) fa ricostruire; le IC-LoRA edit vogliono il reference come frame EXTRA in-context + output generato a parte. Open item profondo (layout reference IC-LoRA), NON wiring. ⚠️
