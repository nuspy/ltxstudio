# Stato lavoro notturno — Video NATIVO via GGUF (26/06)

## TL;DR
**Il video LTX-2.3 22B ora gira NATIVO sui 32GB, veloce, ZERO ComfyUI.** Scoperto come ci riusciva
ComfyUI (modello GGUF Q4 + dequant on-the-fly) e replicato nel motore nativo. Martina canta, nativa,
con audio.

## Come ComfyUI ci riusciva (la tua domanda)
ComfyUI per il video usava il **GGUF distilled Q4_K_M (13.5GB)**, non il `dev-fp8` (27.8GB), + il nodo
**ComfyUI-GGUF** che tiene i pesi quantizzati e li **dequantizza per-layer al volo** nel forward. Così
il 22B sta nei 32GB. `ltx-core` (motore nativo) leggeva solo safetensors → ho portato la tecnica.

## Cosa ho fatto (committato su `feature/native-backend`)
- `workers/gguf_native.py`: loader GGUF + dequant (Q4_K/Q5_K/Q6_K/Q3_K…) + `patch_stage_to_gguf()`.
  Il transformer si carica dal GGUF (13GB su GPU, dequant al volo); Gemma/vae/vocoder dal fp8.
- `workers/ltx2_infer.py`: param `gguf_transformer`; `torch.set_grad_enabled(False)` globale.
- 3 bug chiave risolti: mutazione in-place dei nn.Linear (il preprocessor cattura patchify per
  riferimento); no_grad globale (il conditioning teneva un grafo da 37GB → sforava); disabilitato
  torch.compile/_quantization sullo stage GGUF.
- Risultato: VRAM stabile ~13GB, **niente spill**. t2v 256²: 2143s → **100s (21x)**.

## Video generati (in `output/`)
- `martina_canta_512.mp4` — 5s, 512², audio. **Qualità buona.**
- `martina_canta_512_10s.mp4` — 10s, 512².
- `martina_canta_512_20s.mp4` — 20s, 512².
- `martina_canta_FULL.mp4` — **la canzone completa (~98s)**, generata a SEGMENTI da ~19.6s
  concatenati con continuità (ultimo frame → conditioning del successivo). [in generazione stanotte]
- `TESTmartina_gguf256.mp4` — primo test 256² (2s).

## Chunking video lunghi (FATTO nel worker)
- `run_generation` ora: **auto-GGUF** di default (`_auto_gguf` trova il Q4) + **chunking in-process**
  (`chunk_frames>0` → `_run_chunked`): genera a chunk corti riusando la PIPE IN CACHE (carica una
  volta), continuità via ultimo-frame→conditioning, audio affettato per chunk, concat ffmpeg.
- Il full 98s **one-shot** NON entra (stage_2/upscale a lunghezza piena sborda; ComfyUI lo gestiva col
  **tiled-sampler** — qui risolto col chunking, ~60s/chunk da 5s). NB: la VRAM reale è ridotta da STT
  + processi di sistema → tenere chunk corti (5s/121 frame) e verificare `nvidia-smi memory.free`.

## CABLAGGIO GUI — FATTO (additivo, da validare con interfaccia aperta)
- **Tab Video → nativo**: ramo `if backend.is_native("video")` in `generate` → `_generate_video_native`
  → `_NativeLtx2Worker` (subprocess venv ltx2, auto-GGUF + chunking + a2v se audio), risultato su
  `_on_multitalk_done` (face-swap/RIFE/announce). **`backend_config` video=native** (flippato).
- **Lip-sync/canto → nativo**: `_start_native_multitalk` ora usa l'a2v nativo (non più Wan/ComfyUI).
- 14/14 smoke test GUI passano. Validato a import/costruzione + worker standalone; il flusso
  end-to-end in GUI va provato con l'app aperta.
- **Extend video → nativo** (`_extend_video_native`): i2v dall'ultimo frame + concat.
- **Enhance/definizione → nativo** (`_enhance_video_native`): upscale per-frame ESRGAN (handler video_upscale).
- **Ancora ComfyUI (warn/da cablare)**: storyboard #37 (avvisa, genera scena base; serve pipeline
  keyframe_interpolation nel worker).

## PESI già presenti in E:\AI\Models (verificato 26/06)
- SAM2 + GroundingDINO (#22 segmentazione) ✅ · FLUX.2 fp8 33GB + vae + mistral text-enc (#14) ✅ ·
  Ernie diffusers completo (#1) ✅ · **SD3 (#27) MANCA** (unico da scaricare).
- → #22/#14/#1 sbloccabili SENZA download: restano da validare/abilitare i handler nativi.

## ALTRO ANCORA DA FARE (storico)
Il motore video nativo è completo (`workers/ltx2_infer.py` + `core/ltx2_service.py` server :8190) ma
**NON è ancora collegato al tab Video**: la GUI genera video via `self.driver.submit(workflow ComfyUI)`
costruito da `TemplateEngine` (in `ui/main_window.py`: `generate_video*`, `generate_video_extend`, …).
Per migrare: in ogni metodo `generate_video*` aggiungere il ramo `if backend.is_native("video"):
params=...; core.ltx2_service.generate(params)` (additivo, non rompe il path comfy finché
`backend_config` video=comfy). Poi flip `backend_config` video=native e TEST in GUI. **Non fatto
stanotte perché non testabile senza interfaccia** (rischio di lasciare la GUI rotta).
Stesso discorso per storyboard #37 (→ pipeline `keyframe_interpolation` nativa).

## Modello usato
`E:\AI\Models\diffusion_models\LTXVideo\LTX-2\quantstack\ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf` (13.5GB)

Commit: `2c9e2a2` (integrazione GGUF), `759f012` (no_grad forward), `f830764` (set_grad_enabled globale).
