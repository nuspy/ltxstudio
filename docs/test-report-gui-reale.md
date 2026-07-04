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

- V2V — DIAGNOSI DEFINITIVA: pipeline comfy STRUTTURALMENTE CORRETTA = identica al workflow IC-LoRA ufficiale (LTXVAddGuide APPENDE il reference in-context come frame extra in coda, frozen a strength 1.0; LTXVCropGuides tiene i primi F frame=output, rimuove la coda=guida). L'output sono i frame GENERATI, ma col distilled Q4+8step il modello RIPRODUCE il sorgente invece di editare. edit_anything era gia validato su ltx2_infer (#60, modello pieno >=40GB) -> l'effetto-edit serve modello pieno + piu step, NON e un bug di porting comfy. CHIUSA come limite hardware/step su 32GB, non difetto.

- Videomusicale (canzone->storyboard LLM->video coerente+scene+montaggio+audio): REAL_music_video.mp4 (495KB, 284s, 2 scene, video+audio). Pipeline end-to-end su 32GB (ltx2_infer GGUF+offload). Copre video-musicale e video-musicale-con-scene. ✅

- Prolunga/video lungo: REAL_video_extended.mp4 (270KB, 4.1s = base+2 segmenti autoregressivi via worker comfy). Catena memory-bounded (VRAM costante per segmento) → scalabile a 20s+ no-OOM. ✅

## 3D — TRELLIS immagine→mesh ✅ VALIDATO (30/06)
REAL_trellis_puppy/model.glb (4.2MB, 75171 vert · 140884 facce · texture). load 55s+diffusion 112s+to_glb 16s su 32GB.
Catena di fix necessari (TRELLIS.2-4B, applicati in E:\AI\Tools\TRELLIS2):
1. DINOv3 gated (facebook 403) → mirror non-gated `camenduru/dinov3-vitl16-pretrain-lvd1689m` + patch model_name in pipeline.json/texturing_pipeline.json
2. rembg `briaai/RMBG-2.0` gated → cambiato model_name in `ZhengPeng7/BiRefNet` (non-gated, scaricato)
3. pip install kornia + timm (richiesti dal codice remoto BiRefNet) nel venv TRELLIS2
4. BiRefNet.py: input cast al dtype del modello (`.to(next(self.model.parameters()).dtype)`) — fix RuntimeError float vs Half
5. image_feature_extractor.py: `self.model.layer` → `self.model.model.layer` (transformers DINOv3ViTModel.model=encoder)
Download 3D: DA-2 1.3GB, DINOv3 mirror 1.2GB, BiRefNet 445MB, WorldGen-LoRA 76MB.
WorldGen: 🔴 Windows friction (nunchaku FP4 path bug + FLUX-Fill bf16 24GB / FP4 6GB). Rig: ⚪ SkinTokens venv non installato.

## 3D — Rig (SkinTokens) e WorldGen: BLOCCATI (porting/infra)
- Rig: install_skintokens.py installa venv+torch cu128 OK, MA git clone del repo sorgente FALLISCE (github VAST-AI-Research/TokenRig = 404). rig_generate.py importa `skin_tokens.inference.auto_rig` (pacchetto assente); il codice SkinTokens funzionante e' vendorizzato nel custom node ComfyUI (ComfyUI-SkinToken/vendor/skintokens + sktn_nodes.py + blender_headless_bridge) con API diversa. => porting nativo INCOMPLETO, richiede adattare rig_generate al pacchetto vendorizzato + Blender bridge. Pesi in ComfyUI/models/skintoken (Qwen3-0.6B+experiments).
- WorldGen: friction Windows (nunchaku FP4 repo-id backslash bug; FLUX-Fill bf16 24GB o FP4 6GB; DA-2 senza config.json nel repo). Frozen.
- TRELLIS resta l'UNICO modulo 3D validato end-to-end (model.glb reale).

- Rig nativo (mesh->scheletro+skin): REAL_rig_puppy.glb (6MB, 1 armatura, 7 ossa Mixamo mixamorig:Hips/Spine/Neck/Head..., skin weights). Port del node ComfyUI-SkinToken come LIBRERIA (scripts/rig_native_skintoken.py, no ComfyUI server) nel venv ComfyUI + Blender headless. Inferenza 46s. rig_service ricablato (repo TokenRig 404 era morto). 2/3 moduli 3D validati (TRELLIS+Rig). ✅

- WorldGen — fix nunchaku Windows: venv/.../nunchaku/models/transformers/utils.py, le 2 chiamate hf_hub_download usano `repo_id=str(pretrained_model_name_or_path).replace(chr(92),chr(47))` (Path() su Windows trasformava il repo-id mit-han-lab/svdq-fp4-flux.1-fill-dev in backslash -> HFValidationError). FP4 ~7GB non-gated in download; al termine WorldGen testabile (ramo low_vram nunchaku FP4, evita i 24GB bf16). ⏳

## 3D secondari (30/06)
- Convertitore formati 3D: REAL_trellis_puppy.fbx (5.7MB, GLB->FBX via Blender headless, bl_common). ✅
- EXR/HDR export: REAL_hdr_*.exr (float32 HDR). FIX bug: OPENCV_IO_ENABLE_OPENEXR ora settato a livello modulo PRIMA di import cv2 (era dopo -> codec disabilitato -> scrittura falliva). ✅
- Retarget: serve un .bvh (input animazione) -> non testabile senza mocap/webcam. Mocap: richiede webcam (no headless). ⚪

## Animazione/Mocap (30/06) — pipeline completa
- Mocap DA VIDEO: headsmash.mp4 (1280x720) -> REAL_headsmash_mocap.bvh (225 frame, 19 ossa umanoidi @24fps, 92% detection, 7s). FIX: mediapipe>=0.10.21 ha rimosso mp.solutions -> install_mocap pinna mediapipe==0.10.18. ✅
- Retarget: BVH + mesh riggata (Mixamo) -> REAL_headsmash_retarget.glb (2 animazioni, 21 channels). FIX: nla.bake only_selected=True (POSE) filtrava le OSSA selezionate (0 selezionate -> bake vuoto) -> only_selected=False. ✅
- Convertitore 3D (GLB->FBX) + EXR/HDR export gia validati. => 4/4 moduli 3D secondari OK.

## Catena asset videogame su materiali utente (Headbang.png + headsmash.mp4) — 30/06
- TRELLIS cavaliere: REAL_trellis_knight/model.glb (140742 vert + texture 1024). ✅
- Rig cavaliere: REAL_knight_rigged.glb (46 ossa Mixamo + skin). ✅
- Mocap: REAL_headsmash_mocap.bvh (225 frame, asset pulito riutilizzabile). ✅
- Retarget BVH->cavaliere: REAL_knight_animated.glb (2 anim, 138 ch) — pipeline gira end-to-end MA deformazione CONTORTA (rest-pose mismatch tra BVH mediapipe e auto-rig SkinToken). Per resa pulita: retarget su Mixamo standard o aggiungere rest-pose alignment in retarget.py. ⚠️

- Retarget MIGLIORATO (correzione orientamento rest per-osso, R_tgt=C·R_src·C^-1 con C=T^-1·S): animazione cavaliere ora PULITA (posa di combattimento corretta, no contorsione). Risolve il rest-pose/roll mismatch tra BVH mediapipe e auto-rig SkinToken. REAL_knight_animated.glb. ✅

## Sessione 02/07 — bug da GUI live (immagini)
- **Inpaint/Kontext con transformer GGUF** (es. `fluxunchained-fill-hyp8-Q5_K_S.gguf`): prima falliva
  `Unable to load weights` (from_single_file trattava il .gguf come safetensors). FIX: `_assemble_flux`
  usa `GGUFQuantizationConfig` sui `.gguf` (diffusers key-map BFL→diffusers, pesi quantizzati in VRAM +
  dequant on-the-fly); `_load` instrada i .gguf all'assembler. **Validato LIVE**: inpaint reale col
  modello GGUF Fill → `native_inpaint_12345.png` (838KB, success 58s). Commit 0d4d49f. ✅
- **LoRA su base FLUX.2/Klein fp8**: prima crashava `Target module Fp8Linear() is not supported` (PEFT non
  inietta su moduli quantizzati custom). FIX: `apply_loras` rileva `Fp8Linear`/`Nvfp4Linear` e fallisce con
  messaggio chiaro/azionabile (usa base bf16 o fondi la LoRA); base bf16 invariata (path PEFT). Supporto
  LoRA nativo su fp8 = lavoro dedicato #17/#78. Commit 2315659. ⚠️ (guardia; feature completa parcheggiata)
- **FLUX.1 Krea [dev]** (#76): già integrato (detector flux1/dev/t2i; filtro dropdown FLUX include 'krea';
  pesi su disco). **Validato LIVE**: t2i → `native_flux_76076.png` (728KB, success 63s). ✅
- **Kontext** (FLUX.1, editing per istruzione, #14 ramo FLUX.1): LIVE → `native_kontext_555.png` (success 96s). ✅
- **Segment** (SAM2+GroundingDINO): LIVE → `native_segmask.png` (success 12s). ✅
- **Upscale** (4x-UltraSharpV2): LIVE → `native_upscale_0.png` (success 17s). ✅
- **Video ControlNet** (iclora depth, #24): LIVE → `test_cnet_depth.mp4` (25 frame @24fps, gen 91s, IC-LoRA depth 625M). ✅
- **Video ReTake** (#24, rigenera sezione): LIVE OK su sorgente ≤448x256 → `test_retake.mp4` (17 frame, gen 75s). ⚠️
  Fix anti-OOM: `_prep_retake_source` ridimensiona il sorgente al target + empty_cache. Ma **marginale su 32GB**:
  su sorgente grande auto-ridotto l'encoder-load resta al limite (OOM flaky). Fix profondo (eviction transformer
  durante load encoder) rimandato. Su hardware >32GB atteso stabile.

## Sessione 03/07 — richieste utente A/B
- **#88 Preview live per-step immagine**: callback latent2rgb (no VAE→no OOM) → preview_bus → GET /preview →
  native_driver.preview() → GenProgressDialog (QTimer) in t2i/inpaint/kontext. **Validato LIVE**: 8 preview
  su 12 step, gen ok. Commit 1c0f9c1. ✅
- **#89 Audio nel video t2v/i2v**: il motore comfy-native generava audio solo in a2v; ora gate
  (audio_path OR generate_audio=True) crea l'empty audio latent + concat + mux AAC. **Validato LIVE**:
  test_audio_t2v.mp4 → stream 0 h264 + stream 1 aac. Commit bc8b290. ✅

## Sessione 03/07 — loop #14/#24/#7/#8/#27 (backend reale, GPU pulita)
- **#14 FLUX.2 Klein inpaint** (flux2_inpaint NSFW): LIVE success 244s (fix check-encoder shardato 7b20f04). kontext funzionale ma memory-bound su 32GB. ✅
- **#24 video editing**: controlnet 91s · retake 75s · keyframe 87s · extend 67s (5s video). ✅
- **#7 v2v-edit** (edit-anything LoRA rank128): LIVE 435s. ✅
- **#8 qualità video**: RIFE fps (24→60) 3s · upscale ESRGAN 68s · restauro volti GFPGAN 4s (installato gfpgan+modello 333MB). ✅
- **#27**: SD3 abortito, conversione modelli già fatta, prompt-enhance funzionale (available()=True). ✅
- NOTA PERF: le lentezze osservate (video 10s→1h, keyframe 18min iniziale) erano SEMPRE contesa GPU da worker orfani (worker non ucciso alla chiusura GUI → #91), NON il codice: su GPU pulita tutto è ~1-2 min.

## Sessione 03/07 — loop #83/#51/#91/#90 (wiring + fix)
- **#83 wiring**: 7/7 casi Spy rappresentativi PASS (audio/music/video/vedit/restore/frame1/controlnet). ✅
- **#51**: rimossa ultima dipendenza ComfyUI ATTIVA (cosyvoice2 fallback → errore nativo). Runtime zero-comfy. Purge fisico dead-code deferito post-lancio. ✅(runtime)
- **#91**: kill worker/subprocesso alla chiusura GUI (taskkill /T albero) — validato: uccide padre+figlio venv-shim. ✅
- **#90 preview video**: LIVE — FASI ('Caricamento componenti…'→'Generazione…'→'Decodifica…') ✅ + barra credibile (step 1..16) ✅. Preview-IMMAGINE: OK per foto; per VIDEO bloccata dal latent format LTXAV (audio-video, rgb_factors=None) → enhancement separata (patch tentata rompeva la gen, revocata). ✅(fasi+barra) ⚠️(immagine video)

## Sessione 04/07 — #90 (video) + #87 (API endpoint)
- **#90 video preview**: fasi+barra LIVE ok; preview-IMMAGINE video BLOCCATA (latente LTX-2.3 tokenizzato, previewer LTXV crasha 45312x1 vs 128x3, rompe la gen) → revert sicuro bd151cf, decode custom = task #92.
- **#87 API :8200**: health/segment/rife/tts(edge)/film-script/rig ✅ (fix rig preflight 3f17d86). world: API ok, WorldGen non produce output → task #93. sonify/MMAudio non installato 🔒.

## Sessione 04/07 — #92 + #93 (loop "fino a verde")
- **#92 preview-immagine video**: FATTO+live. Decode del latente-token LTX-2.3 nel callback (fattori is_23=True KJNodes estratti, unpack video, F.linear+sigmoid). 7-8 anteprime/gen. Commit 61a0f03.
- **#93 WorldGen**: VERDE. 3 bug reali corretti (2 nunchaku loader legacy path-Windows + 1 pano_depth device str), resi persistenti nell'installer (patch_nunchaku/patch_worldgen_src). Cache HF su E: (HF_HOME). Pesi: FLUX.1-dev diffusers (23GB) + nunchaku fp4 + DA-2 + LoRA. Validato: worker diretto → mesh.glb 8MB; API :8200 /threed/world → success + download_url. Commit 76e2cfa/32240bc/b44c47d.
