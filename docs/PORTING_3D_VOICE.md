# Piano esecutivo — Porting nativo 3D (TRELLIS/Rig) + Voice cloning (no ComfyUI)

Dal reverse-engineering dei custom nodes in `E:\AI\Comfy_Nexus.bak\ComfyUI\custom_nodes` (28/06).
Obiettivo: eliminare le ultime dipendenze ComfyUI runtime. Pattern di riferimento già validato:
**SeedVR2/FlashVSR** (subprocess in venv isolato, `core/seedvr2_service.py` + `scripts/install_*.py`).
⚠️ Richiede GPU + pesi (presenti) → esecuzione/validazione DIURNA (non di notte su mobile).

## 1. TRELLIS.2 (immagine→mesh 3D) — `generate_trellis`
- **Codice**: `custom_nodes/ComfyUI-TRELLIS2/nodes/` — il cuore è `stages.py` (`run_conditioning` DinoV3 → `image_to_shape` slat_flow DiT → export GLB). I nodi (`nodes_inference.py`) sono wrapper sottili.
- **Pesi PRESENTI** (25GB): `E:\AI\Comfy_Nexus.bak\ComfyUI\models\trellis2\ckpts\` (shape_dec_next_dc_f16c32_fp16, slat_flow_img2shape_dit_1_3B_{512,1024}_bf16). → spostare/symlink in `E:\AI\Models\TRELLIS2`.
- **Dipendenze da rimuovere**: `import comfy.model_management` (→ `torch.cuda` diretto + offload manuale), `comfy_api.latest.io` (solo per lo schema nodo, non serve nativo).
- **Passi**:
  1. `scripts/install_trellis.py`: clona/copia il pacchetto trellis + venv (torch cu128 sm_120) + `requirements.txt` del nodo (DinoV3, utils3d, ecc.).
  2. `core/trellis_service.py` (pattern seedvr2_service): subprocess venv → runner che importa `stages` (adattato senza comfy), carica i ckpt, `image→conditioning→shape→GLB`. preflight().
  3. `workers/trellis_run.py`: CLI parametrico (image, resolution mode, output GLB).
  4. main_window: `generate_trellis` → branch `_trellis_native()` (worker subprocess) come `_enhance_video_native`. Rimuovere `driver.submit` + `trellis_pipeline`.
  5. Test live: immagine→GLB, verifica mesh nel visore (TrellisTab già pronto).
- **GUI**: TrellisTab già completa (image picker + format) → solo ricablare il signal al worker nativo.

## 2. Rig 3D (SkinTokens) — `generate_rig`
- **Codice**: `custom_nodes/ComfyUI-SkinToken/` (SkinTokens; richiede Blender, auto-install winget). Pattern identico.
- **Passi**: `scripts/install_skintoken.py` (venv + Blender check) + `core/skintoken_service.py` (subprocess: mesh→scheletro+skinning) + `workers/skintoken_run.py` + branch nativo in `generate_rig`. RigTab GUI già pronta.

## 3. Voice cloning nativo
- **CosyVoice2**: GIÀ NATIVO (`core/cosyvoice_service.py` + venv). ✅ Solo validare.
- **F5-TTS**: repo `SWivid/F5-TTS` (pip `f5-tts`). venv isolato + runner (ref_audio+ref_text+gen_text → wav). Nessun custom node installato → integrare da upstream.
- **XTTS-2**: `coqui-ai/TTS` (`pip TTS`, modello `tts_models/multilingual/multi-dataset/xtts_v2`). venv + runner (speaker_wav + text → wav).
- **Fish-Speech**: repo `fishaudio/fish-speech`. venv + runner.
- **VibeVoice**: repo upstream. venv + runner.
- **Integrazione**: in `core/tts_engine.py` sostituire `_synthesize_comfy()` (e `_comfy_node_present`/`_scan_comfy_nodes`/`install_engine`) con un dispatch a runner nativi `core/<engine>_service.py` (pattern cosyvoice_service: subprocess venv, install on-demand). GUI AudioTab/VoiceDialog: combo motore (tutti nativi) + install button.
- ⚠️ Richiede download repo+modelli per engine (diurno).

## 4. Eliminazione finale ComfyUI (dopo 1-3)
Quando 3D+voice+recolor sono nativi: rimuovere `core/{gen,inpaint,integrate,kontext,upscale,segment,extend,
face_restore,video_enhance,restore,rig,trellis}_pipeline.py`, `core/template_engine.py`,
`core/comfy_driver.py`/`comfy_launcher.py`/`comfy_compat.py`, gli import/istanze in `ui/main_window.py`
(inpainter/segpipe/upscaler/integrator/driver/restorer/facerestorer/enhancer), e i path hardcoded
`E:\AI\Comfy_Nexus\ComfyUI`. Verifica: `grep -ri comfy` runtime = 0 + smoke 14/14 + app parte senza ComfyUI.

## Note risorse
- TRELLIS pesi 25GB già presenti (in Comfy_Nexus.bak/models/trellis2) → riusare, NON riscaricare.
- SkinToken: Blender (auto-install winget al primo uso).
- Voice: download repo+modelli per F5/XTTS/Fish/VibeVoice (diurno, no banda mobile notturna).
