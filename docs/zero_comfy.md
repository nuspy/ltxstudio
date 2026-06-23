# Migrazione a ZERO dipendenze ComfyUI — piano di finalizzazione

Obiettivo: eliminare ComfyUI come runtime. Oggi l'app usa **tre** runtime:
- **ComfyUI** (porta 8188, `core/comfy_driver.py` + `core/comfy_launcher.py`) — da eliminare.
- **Backend nativo** (porta 8189, `inference_server/`, diffusers) — il sostituto; dispatch via `core/backend.py` (`is_native(module)`) + `core/native_driver.py`.
- **Standalone** (venv isolati, subprocess) — musica/training/3D-scene/TTS: **già zero-ComfyUI**, non toccare.

Aggiornato: 2026-06-23.

## 1. Stato per componente

| Feature | Pipeline (costruisce grafo Comfy) | Handler GUI | Equivalente diffusers | Stato |
|---|---|---|---|---|
| Immagini (FLUX/Z-Image/Kontext/inpaint) | `core/gen_pipeline.py` | `generate_image` | `inference_server/backends/image_flux.py` | ✅ **NATIVO** (flag `image`) |
| Inpaint/Outpaint (FLUX Fill) | `core/inpaint_pipeline.py` | `generate_inpaint` | `FluxFillPipeline` (in image_flux, `inpaint`) | 🟡 nativo pronto, **cablare il flag** |
| Kontext / Integra | `core/kontext_pipeline.py` | `generate_integrate` | `FluxKontextPipeline` (in image_flux, `kontext`) | 🟡 nativo pronto per 1-img; **multi-ref/ReferenceLatent da reimplementare** |
| Compose multi-foto | `core/kontext_pipeline.py:compose` | `generate_compose` | come Kontext + multi-reference | 🟡 dipende da Kontext nativo |
| Upscale immagine | `core/upscale_pipeline.py` | `generate_upscale` | ESRGAN (spandrel) + rifinitura FLUX | 🔴 da migrare (semplice) |
| Restauro/colorazione foto | `core/restore_pipeline.py` | `generate_restore` | ESRGAN + Kontext colorize | 🔴 da migrare (dipende da Kontext) |
| Segmentazione testo (SAM+GroundingDINO) | `core/segment_pipeline.py` | `segment_by_text` | SAM2 + GroundingDINO via transformers | 🔴 da migrare (custom_node → libreria) |
| **Video t2v/i2v** | `core/template_engine.py` | `on_generate` | `LTX2Pipeline` (`video_ltx.py`, task #18) | 🟡 **scaffold nativo** (in corso) |
| **Video prolunga/long** | `core/extend_pipeline.py` | `generate_video_extend` | `LTXI2VLongMultiPromptPipeline` / logica IAMCCS da reimplementare | 🔴 nodi IAMCCS custom → reimplementare in diffusers |
| Video ReTake / Just_Talk | `core/template_engine.py` (v2v) | `on_generate` (v2v) | LTX2 condition + mask | 🔴 da reimplementare |
| Video ControlNet (copia movimenti) | `core/template_engine.py` (controlnet) | `on_generate` (cnet) | `LTX2ConditionPipeline` | 🔴 da migrare |
| Definizione/upscale video (SeedVR2/RIFE) | `core/video_enhance_pipeline.py` | `generate_video_enhance` | SeedVR2 (repo standalone) + RIFE (torch) | 🔴 da migrare (modelli estraibili) |
| Restauro volti video (CodeFormer/GFPGAN) | `core/face_restore_pipeline.py` | `generate_face_restore` | codeformer/gfpgan (pip) per-frame | 🔴 da migrare (librerie pip esistono) |
| 3D da immagine (TRELLIS.2) | `core/trellis_pipeline.py` | `generate_trellis` | `microsoft/TRELLIS` repo Python standalone | 🔴 migrare a venv standalone (come WorldGen), NON ComfyUI |
| Rigging (SkinTokens) | `core/rig_pipeline.py` | `generate_rig` | SkinTokens repo + Blender headless | 🔴 migrare a venv standalone |

## 2. Già ZERO-ComfyUI (non toccare)
- **Musica** (`core/music_service.py`): HeartMuLa + MMAudio, venv isolati `E:\AI\Tools\heartlib`, `...\MMAudio`.
- **Training LoRA** (`core/training_service.py`): ai-toolkit, venv `E:\AI\Tools\ai-toolkit`.
- **Mondo 3D** (`core/world_service.py`): WorldGen, venv `E:\AI\Tools\WorldGen`.
- **TTS** (`core/tts_engine.py`): default Edge TTS (no ComfyUI); il voice-clone CosyVoice usa un custom_node → va spostato su un venv standalone.

## 3. Dipendenze "fisiche" da rimuovere a fine migrazione
- Driver/launcher: `core/comfy_driver.py`, `core/comfy_launcher.py`, `core/comfy_compat.py` (shim).
- Shim custom node: `comfy_compat/ltxstudio_comfy_compat/`.
- Path hardcoded `E:\AI\Comfy_Nexus\*` (root, custom_nodes, input, `Comfy_Nexus\Output`).
- Custom node usati da sostituire con librerie/pipeline: `FluxKontextImageScale`+`ReferenceLatent`, `IAMCCS_LTX2_*`, `GroundingDinoSAMSegment`, `FaceRestoreCFWithModel`, `SeedVR2Upscaler`, RIFE VFI, `VHS_*` (I/O video → imageio/ffmpeg), `Trellis2*`, `SkinTokenRigTrimesh`.

## 4. Piano a fasi (priorità → sforzo)

**Fase A — chiudere le IMMAGINI (in corso)**
1. ✅ **FATTO** (commit 4be8f8a): inpaint/outpaint nativo FLUX.1 Fill (`_start_native_inpaint`, branch in `generate_inpaint`). FLUX.2 Klein/checkpoint → ComfyUI fallback.
2. ✅ **FATTO** (commit b05ed75) per Kontext 1-img integrate (`_start_native_kontext`, branch in `generate_integrate`). 🔸 multi-reference vero (compose N foto) POSTICIPATO: FLUX.1 single-ref; multi-ref serve FLUX.2 (download) o reimplementazione ReferenceLatent multi-latent.
3. ✅ **FATTO** (commit 666b3dc): **upscale** nativo ESRGAN via `spandrel` (`image_upscale.py`, validato 1024²→2048² 4s) + `HANDLERS['upscale']` + branch in `generate_upscale`. 🔸 rifinitura FLUX e **restauro/colorizzazione foto** (generate_restore) ancora da cablare.
4. 🟡 **Segmentazione**: handler `image_segment.py` (GroundingDINO+SAM2 transformers) **scritto e registrato** (commit a9e5841) → `HANDLERS['segment']`. Runtime in attesa dei pesi HF (download A.4 frenato dal CDN).
- ✅ **EXTRA migrati** (commit 3a2a62e, 60ef236): **recolor** (Kontext nativo), **restauro 'restore'** (ESRGAN nativo), **compose multi-foto** (FLUX.1 via stitch lato Python → Kontext nativo). compose multi-ref VERO + colorize → FLUX.2 (posticipato).

**STATO Fase A (immagini)**: tutto ciò che è FLUX.1/ESRGAN è **migrato in nativo e verificato indipendente da ComfyUI** (gen, inpaint/outpaint, kontext/integra, recolor, upscale, restore, compose-stitch). Restano solo: segmentazione (pesi in download), colorize/compose-multiref (FLUX.2 download), → Fase B+ video/3D.

**Abilitatori migrazione (FATTO 2026-06-23):**
- `backend.strict_native()` / `comfy_blocked()` — blocco NON distruttivo del fallback ComfyUI (commit 666b3dc).
- `core/native_launcher.py` + `_maybe_autostart_native` — l'app avvia il server nativo (8189, Sage+FBCache) da sola (commit b832826).
- `backend.set_backend('image','native')` attivato.
- Backend nativo reso INDIPENDENTE da `Comfy_Nexus` (i pesi sono in `E:\AI\Models`; rimosso il fallback in image_upscale).
- **Strategia utente: rinominare `E:\AI\Comfy_Nexus` → `.bak`** (non eliminare) per far emergere ogni dipendenza nascosta. Con `.bak`: nativo OK per immagini/inpaint(FLUX.1)/kontext(FLUX.1)/upscale; emergono da migrare → video (serve Gemma), FLUX.2, segmentazione, restauro/face video, 3D (TRELLIS/rig), CosyVoice. Path ComfyUI residui nel codice (comfy_launcher, runner, tts, 3D, main_window comfy_input/output) si romperanno: è lo scopo.

**Fase B — VIDEO core nativo (task #18, in corso)**
5. PoC **t2v** `LTX2Pipeline` (`video_ltx.py`) → poi **i2v** (`LTX2ImageToVideoPipeline`). Bloccato dal download Gemma (CDN HF lento).
6. **Long/extend** nativo: usare `LTXI2VLongMultiPromptPipeline` o reimplementare la catena IAMCCS (context-latent + extend-sampler) in diffusers. Sforzo: alto.
7. **ControlNet** video (`LTX2ConditionPipeline`) + **ReTake/Just_Talk** (condition+mask). Sforzo: alto.

**Fase C — POST-PROCESS video nativo**
8. **SeedVR2** (repo standalone) + **RIFE** (torch) + I/O video con **imageio/ffmpeg** (sostituisce VHS). Sforzo: medio.
9. **Restauro volti** per-frame con `codeformer`/`gfpgan` pip. Sforzo: medio.

**Fase D — 3D fuori da ComfyUI (a venv standalone, come WorldGen)**
10. **TRELLIS.2** → venv `microsoft/TRELLIS` (subprocess, come music/world). Sforzo: medio.
11. **SkinTokens** rigging → venv standalone + Blender headless (già usiamo Blender). Sforzo: medio.
12. **CosyVoice** voice-clone → venv standalone.

**Fase E — rimozione finale**
13. Eliminare driver/launcher/shim ComfyUI, path `Comfy_Nexus`, `core/backend.py` semplificato (sempre nativo). Aggiornare installer (niente ComfyUI). Aggiornare wizard pesi (organizzazione nativa).

## 5. Principio guida
ComfyUI resta SOLO come fallback finché ogni feature non ha il percorso nativo/standalone validato (flag `backend.is_native(module)` per modulo). Si smonta ComfyUI quando tutti i flag sono `native`/standalone. Nessuna regressione: ogni fase mantiene il fallback ComfyUI fino alla validazione.

## 6. Blocco corrente
Il PoC video nativo (Fase B) è frenato dal **download del TE Gemma-3** su CDN-LFS HuggingFace throttlato (~3 Mbit/s misurati, contro 150 Mbit della linea). Rimedi: mirror `hf-mirror.com`, HF Pro, VPN, o orari diversi.
