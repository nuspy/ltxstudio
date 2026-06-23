# Migrazione a ZERO dipendenze ComfyUI — piano + INVENTARIO ESAUSTIVO

Obiettivo: eliminare ComfyUI come runtime. Runtime attuali:
- **ComfyUI** (8188, `core/comfy_driver.py`/`comfy_launcher.py`) → da eliminare.
- **Backend nativo** (8189, `inference_server/`, diffusers) → il sostituto; dispatch `core/backend.py` (`is_native`/`comfy_blocked`/`strict_native`) + `core/native_driver.py` + autostart `core/native_launcher.py`.
- **Standalone** (venv isolati, subprocess) → musica/training/world/mocap/TTS-edge: **già zero-comfy**, non toccare.

Aggiornato: 2026-06-23. Stato `Comfy_Nexus` rinominato `.bak` (ComfyUI assente) → indipendenza nativa VERIFICATA per la Fase A.

## INVENTARIO COMPLETO (43 feature) — stato migrazione
Legenda: ✅ nativo verificato · 🟢 nativo codice pronto (attende pesi) · 🟡 nativo parziale/fallback · 🔴 da migrare · ⬛ già standalone (no comfy).

### Immagini
| Feature | core | Stato | Note |
|---|---|---|---|
| Gen FLUX.1 / Z-Image | gen_pipeline → image_flux | ✅ | single-file locale |
| Gen FLUX.2 / Ernie | gen_pipeline → image_flux | 🟢 | repo diffusers (download) |
| Gen **SD3** | gen_pipeline | 🔴 | diffusers SD3Pipeline (raro) |
| Inpaint/Outpaint FLUX.1 Fill | inpaint_pipeline → image_flux | ✅ | FluxFillPipeline |
| Inpaint FLUX.2 Klein | inpaint_pipeline | 🔴 | FLUX.2 (download) |
| Kontext / Integra (1 ref) | kontext_pipeline → image_flux | ✅ | FluxKontextPipeline |
| Compose multi-foto (stitch) | kontext_pipeline → image_flux | ✅ | stitch PIL → Kontext |
| Compose multi-ref VERO | kontext_pipeline | 🔴 | FLUX.2 ReferenceLatent (download) |
| Recolor (prompt) | kontext_pipeline → image_flux | ✅ | Kontext |
| Upscale ESRGAN | upscale_pipeline → image_upscale | ✅ | spandrel |
| Restauro foto (ESRGAN) | restore_pipeline → image_upscale | ✅ | via upscale |
| Colorizzazione B&W | restore_pipeline | 🔴 | FLUX.2 Kontext (download) |
| Segmentazione testo | segment_pipeline → image_segment | 🟢 | GroundingDINO+SAM2 transformers (download pesi) |
| Segmentazione regione (click/box) | segment_pipeline → image_segment | 🟢 | SAM2 (download pesi) |
| **Prompt-enhance (QwenVL)** | gen_pipeline | 🔴 | LLM enhance prompt (valutare nativo) |

### Video (LTX-2.3) — runtime nostro template_engine, eseguito da ComfyUI
| Feature | core | Stato | Note |
|---|---|---|---|
| t2v | template_engine → video_ltx | 🟢 | LTX2Pipeline (download Gemma) |
| i2v | template_engine → video_ltx | 🟢 | LTX2ImageToVideoPipeline |
| **i2v keyframe multipli** | template_engine apply_i2v_multiframe | 🔴 | condiz. su più frame |
| Estensione/Prolunga | extend_pipeline (IAMCCS) | 🔴 | reimpl. context-latent+extend-sampler |
| **Durata illimitata / auto-extend** | template_engine + extend_pipeline (loop) | 🔴 | catena segmenti (NB: instabile anche su ComfyUI → rivedere) |
| ReTake (rigenera sezione) | template_engine apply_retake (v2v+mask) | 🔴 | LTX2 condition+mask |
| Just_Talk (lip-sync) | template_engine apply_just_talk | 🔴 | audio conditioning |
| ControlNet (depth/canny/pose) | template_engine collapse_controlnet | 🔴 | LTX2ConditionPipeline + preprocessor |
| **HR/upscale interno i2v** | template_engine apply_hr_pipeline | 🔴 | upscale in-generazione |
| **Scaling risoluzione** | template_engine apply_resolution_mode | 🔴 | parametrico |
| Definizione/enhance (SeedVR2) | video_enhance_pipeline | 🔴 | repo standalone |
| Upscale video (ESRGAN per-frame) | video_enhance_pipeline | 🔴 | image_upscale in loop + imageio/ffmpeg |
| Aumento FPS (RIFE) | video_enhance_pipeline | 🔴 | torch per-frame |
| **FlashVSR** | video_enhance_pipeline | 🔴 | opzionale |
| Restauro volti (CodeFormer/GFPGAN) | face_restore_pipeline | 🔴 | pip per-frame |
| Audio nativo LTX | template_engine | 🟢 | vocoder LTX2 (nel checkpoint locale) |
| **Voice-clone conditioning** | template_engine + tts_engine | 🔴 | LTXVReferenceAudio + CosyVoice venv |
| **Audio slicing extend** | main_window _slice_audio | 🔴 | supporto extend (Python, già no-comfy) |

### 3D / Audio / Meta
| Feature | core | Stato |
|---|---|---|
| TRELLIS.2 (img→3D) | trellis_pipeline | 🔴 → venv standalone (microsoft/TRELLIS) |
| Rigging (SkinTokens) | rig_pipeline | 🔴 → venv standalone + Blender |
| Mocap / Retarget / Import anim | mocap_service/blender_service | ⬛ standalone |
| Musica (HeartMuLa/MMAudio) | music_service | ⬛ standalone |
| Training LoRA | training_service | ⬛ standalone |
| Mondo 3D (WorldGen) | world_service | ⬛ standalone |
| TTS Edge | tts_engine | ⬛ standalone |
| TTS CosyVoice/F5 (clone) | tts_engine (custom_node) | 🔴 → venv standalone |
| **Watermark / Gating** | watermark.py / gating_apply.py | ⬛ Python post-process (no comfy) |
| **Conversione modelli** | convert_tab | 🔴 verificare (ComfyUI-Manager?) |
| Workflow analyzer / navigator/gallery | settings_analyzer / UI | ⬛ tool UI (no comfy) |

## Piano a fasi
**A — Immagini (DONE per FLUX.1/ESRGAN, verificato)**: gen, inpaint/outpaint, kontext/integra, recolor, upscale, restore, compose-stitch ✅. Segmentazione 🟢 (download pesi). Restano: SD3, prompt-enhance, FLUX.2 (inpaint/kontext/compose-multiref/colorize → download).
**B — Video core nativo**: t2v/i2v 🟢 (download Gemma) → keyframe-multipli, extend, durata-illimitata, ReTake, Just_Talk, ControlNet, HR, scaling. *La durata illimitata va RIPROGETTATA (instabile anche su ComfyUI).*
**C — Post-process video**: SeedVR2, ESRGAN-per-frame, RIFE, FlashVSR, face-restore (CodeFormer/GFPGAN), audio-slicing.
**D — 3D + voce a venv standalone**: TRELLIS, rigging, CosyVoice.
**E — Cleanup**: rimuovere driver/launcher/shim/path ComfyUI, convert_tab, semplificare backend.py.

**Principio**: ComfyUI resta solo fallback per-feature finché ogni feature non ha il percorso nativo/standalone **validato** (flag `is_native(module)`). `strict_native` blocca i fallback per scovare dipendenze.

## Abilitatori già fatti
`image=native`; autostart nativo (`native_launcher`); autostart ComfyUI OFF; `strict_native`/`comfy_blocked`; backend nativo INDIPENDENTE da `Comfy_Nexus` (pesi in `E:\AI\Models`). Handler nativi: `image_flux.py`, `image_upscale.py`, `image_segment.py`, `video_ltx.py`. HANDLERS = image/upscale/segment/video.

## Blocco
Download pesi su CDN-LFS HuggingFace ~3 Mbit/s (config volano, pesi LFS appesi). Vedi `docs/download_manuale.md`. Indipendente da banda/script/hf_transfer/mirror.
