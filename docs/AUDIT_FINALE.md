# AUDIT FINALE — copertura funzioni/sottofunzioni + dipendenze ComfyUI

Generato il 2026-06-28 (sessione completamento autonomo). Verifica reale della catena
**backend → GUI presente → settabile → nativo/ComfyUI** per ogni feature e sottofunzione.
Legenda: ✅ completo+nativo+GUI · ⚠️ funziona ma con dipendenza/limite · ⛔ DEFERRED (vedi fondo) · ❌ mancante.

## Metodo
- `backend_config.json`: image/video/enhance/segment/face = **native** → i path nativi sono il default.
- Ogni `driver.submit` (ComfyUI) in `ui/main_window.py` è preceduto da `if backend.is_native(...): return …native()` → ComfyUI è **fallback** non eseguito, TRANNE dove indicato ⚠️/⛔.

## IMMAGINI (Frame1Tab)
| Feature / sottofunzione | Backend | GUI settabile | Stato |
|---|---|---|---|
| Gen FLUX.1 t2i | native (_start_native_image) | combo+prompt+steps+cfg+seed | ✅ |
| FLUX.2 t2i | nativo dequant | combo | ⛔ FLUX.2 forward (rimandato) |
| Z-Image / Ernie | native | combo | ✅ |
| Inpaint / Outpaint (FLUX.1 Fill) | native | canvas mask + modello | ✅ |
| Kontext / integra | native (is_native image+kontext) | signal+UI | ✅ |
| Compose multi-reference | native | gruppo Componi (lista+maschere+metodo+bottone) | ✅ |
| Upscale / Restauro foto (ESRGAN) | native (spandrel) | tab Restauro | ✅ |
| Recolor / colorizzazione | FLUX.2 → ComfyUI fallback | tab Restauro | ⚠️ usa FLUX.2 (vedi note) |
| Segmentazione testo (GroundingDINO+SAM2) | native | signal | ✅ |
| **PuLID face-adapter** | native (image_flux:316) | gruppo PuLID (enable+foto+strength) | ✅ |
| **Face-swap su foto** (T5) | native InsightFace | tab Restauro op faceswap | ✅ |

## VIDEO (VideoTab)
| Feature / sottofunzione | Backend | GUI settabile | Stato |
|---|---|---|---|
| Gen t2v / i2v LTX-2.3 | native (ltx2_infer) | preset+fps+durata+res+img_strength | ✅ |
| Sampler/Seed/Steps/CFG/Shift | native | advanced_tab | ✅ |
| Qualità Fast/Pro/HQ | native | combo preset | ✅ |
| LoRA video | native | LoraVideoTab | ✅ |
| Prompt-enhance / encoder | native | advanced_tab | ✅ |
| Storyboard multi-scena (keyframe a tempo) | native | StoryboardPanel | ✅ |
| Keyframe-interpolation | native (worker keyframes) + RIFE | storyboard + RIFE | ✅ |
| Video lunghi (memory-bounded) | native (chunk) | durata | ✅ |
| HR upscale/refine inline (T3) | rimosso (ridondante) | nota → Definizione Video | ✅ |
| **audio→video / canto (a2v)** | native (A2VidPipelineTwoStage) | V2V Sing | ✅ |
| **HDR/EXR export** | exr_export pronto, pipeline non cablata | — | ⛔ HDR IC-LoRA (deferred) |

## V2V / EDITING VIDEO
| Feature | Backend | GUI | Stato |
|---|---|---|---|
| V2V Normal | native | V2VPanel | ✅ |
| V2V ReTake (rigenera sezione) | native (is_native video) | V2VPanel + slider start/end | ✅ |
| V2V Just_Talk / lip-sync | native a2v (main_window:2155) | V2VPanel Talk | ✅ |
| V2V Sing | native a2v | V2VPanel Sing | ✅ |
| MultiTalk lip-sync (Wan2.1) | native (wan_multitalk) | V2V talk/sing → _maybe_multitalk | ✅ |
| ControlNet (canny/depth/pose) | native | ControlNetTab | ✅ (verifica live #43) |
| Edit-anything (V2V prompt) | native LoRA | tab Modifica Video AI | ✅ |
| Inpaint oggetto (SAM2 tracking) | native (#59) | tab Modifica Video AI + CFG/guide/dilate | ✅ |
| Stile anime↔real | native LoRA | tab Modifica Video AI | ✅ |

## POST-PROCESS VIDEO (VideoDefTab)
| Feature | Backend | GUI | Stato |
|---|---|---|---|
| SeedVR2 super-res | native (validato 512→1152x720) | engine combo + res/blocks/batch | ✅ |
| **FlashVSR x4** (T4) | native (validato 512→2048, 22s) | engine combo | ✅ |
| ESRGAN per-frame | native (spandrel) | engine combo + scale | ✅ |
| RIFE (FPS up) | native | checkbox + moltiplicatore | ✅ |
| Restauro volti (CodeFormer/GFPGAN) | native (video_face) | model+detection+fidelity | ✅ |
| **Auto face-restore post-gen** (fix) | native (era ComfyUI, ora _face_restore_native) | auto-chain | ✅ |
| Face-swap video (InsightFace) | native | VideoTab fs_auto + picker | ✅ |

## VOCE / TTS / MUSICA
| Feature | Backend | GUI | Stato |
|---|---|---|---|
| TTS Edge (~400 voci) | native | AudioTab | ✅ |
| TTS Kokoro / Handy-API (STT+TTS+clone) | native (handy_api) | AudioTab/VoiceDialog | ✅ |
| Voice cloning CosyVoice2 | native (cosyvoice_service) | AudioTab | ✅ |
| **Voice cloning F5/XTTS/Fish/VibeVoice** | **ComfyUI custom_nodes** | AudioTab engine combo | ⛔ T9 (porting nativo deferred) |
| Generazione canzone (HeartMuLa) | native venv | MusicTab | ✅ |
| Lyrics AI (Gemma) | native (LM Studio) | MusicTab | ✅ |
| Sonorizzazione video (MMAudio) | native venv | MusicTab | ✅ |
| Videomusicale (canzone→storyboard→video) | native | MusicTab/MV | ✅ |

## 3D
| Feature | Backend | GUI | Stato |
|---|---|---|---|
| TRELLIS.2 (immagine→mesh) | **ComfyUI/comfy-env** (driver.submit:3248) | TrellisTab | ⛔ T(3D) porting nativo deferred |
| Rig 3D (SkinTokens) | **ComfyUI** (driver.submit:3350) | RigTab | ⛔ porting nativo deferred |
| Anim (MediaPipe→BVH, retarget Blender) | native venv mocap/blender | AnimTab | ✅ |
| World Gen (foto→scena 3D) | native venv | WorldTab | ✅ |
| Visore 3D | native (model-viewer) | embedded | ✅ |

## FILM STUDIO / LoRA / IMPORT
| Feature | Backend | GUI | Stato |
|---|---|---|---|
| Film Studio (Copione/Asset/Scenografia/Montaggio) | native (gen_service+llm) | FilmStudioWindow (7 tab) | ✅ |
| Doppiaggio + re-lip-sync, Continuity, Auto, Timeline | native | tab montaggio/storyboard | ✅ |
| LoRA training (FLUX) | native (ai-toolkit venv) | TrainTab | ✅ |
| LoRA Video/Image dataset+train | native | LoraVideo/ImageTab | ✅ |
| Import models / Weights download | native (FastDownloader) | ImportModelsTab/WeightsTab | ✅ |

---

## Dipendenze ComfyUI RESIDUE (da eliminare — stato)
1. **Voice cloning F5/XTTS/Fish/VibeVoice** (`core/tts_engine.py` `_synthesize_comfy`): runtime attivo → ⛔ T9 (porting nativo: download venv + ricerca per engine, di giorno).
2. **TRELLIS 3D** (`generate_trellis`→driver.submit): runtime attivo → ⛔ porting nativo (comfy-env, grosso).
3. **Rig 3D** (`generate_rig`→driver.submit): runtime attivo → ⛔ porting nativo.
4. **Fallback legacy** (12 `*_pipeline.py` + `template_engine.py` + `comfy_driver/launcher/compat`): NON eseguiti (tutto is_native) → da RIMUOVERE (FASE 5) dopo conferma che nessun path li usa con backend native.
5. Recolor/colorizzazione foto: usa FLUX.2 → quando FLUX.2 nativo sarà sbloccato passerà nativo; per ora ComfyUI fallback.

## Conclusione corrente
- Tutte le feature **immagine/video/post-process/voce(parziale)/musica/film/LoRA** sono **native e in GUI settabili**.
- Dipendenze ComfyUI residue: **3D (TRELLIS/Rig)** e **voice cloning F5/XTTS/Fish/VibeVoice** → DEFERRED (richiedono porting nativo con download/ricerca diurni).
- Codice fallback ComfyUI: da rimuovere (FASE 5) — non eseguito a runtime.
- DEFERRED tecnici: FLUX.2 forward, HDR IC-LoRA, Multi-GPU.
