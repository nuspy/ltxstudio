# LTX Studio — REST API (nativa)

API REST **locale** (FastAPI, `http://127.0.0.1:8200`) che espone **tutte le funzioni native** —
immagini, video, film/videoclip, musica, modelli 3D, rig/skinning — come **job asincroni** che
ritornano **link di download**. Backend 100% nativo (FLUX/FLUX.2/Z-Image/Ernie · LTX-2.3 ·
SeedVR2/FlashVSR · HeartMuLa/ACE-Step/MMAudio · TRELLIS/WorldGen), zero ComfyUI dove possibile.

> Sostituisce la vecchia API `:8200` che pilotava ComfyUI (sincrona). Nuova semantica: **submit →
> polling → download**. Codice in `api/`. Contratti macchina + skill agenti in `agents/`.

## Avvio automatico e spegnimento

- **Autostart**: `python -m api.autostart` → stampa `READY http://127.0.0.1:8200` quando pronta
  (avvia il server se giù e attende `/health`). Lo skill degli agenti lo chiama a inizio sessione.
- **Auto-shutdown**: dopo **30 min** senza richieste **e** senza job in coda/esecuzione il server si
  spegne da solo (`LTX_API_IDLE_TIMEOUT`, watchdog). Al prossimo uso rilancia l'autostart.
- Avvio manuale: `python -m api.app`. Swagger: <http://127.0.0.1:8200/docs> · OpenAPI:
  <http://127.0.0.1:8200/openapi.json>.
- Env: `LTX_API_HOST` (127.0.0.1), `LTX_API_PORT` (8200), `LTX_API_IDLE_TIMEOUT` (1800),
  `LTX_API_PYTHON` (python che esegue l'API).

## Modello asincrono (sempre)

1. Ogni azione è un `POST` (body **JSON**) → `{ "job_id": "...", "status": "pending" }`.
2. **Polling** `GET /jobs/{job_id}` (backoff 2–5 s) finché `status` = `success` | `error`:
   - `running` → `progress` (0–1) + `message`; `pending` → `queue_position`.
   - `success` → `files: [{ path, name, download_url }]`.
   - `error` → `error` (spesso un prerequisito da installare).
3. **Download**: `GET /download?path=<path>` (servito solo dentro `output/` e `output/api_uploads/`).
4. **Coda seriale**: un solo job GPU alla volta. Non lanciare job in parallelo.

## File in ingresso + concatenamento

- Carica un file UNA volta: `POST /files/upload` (**multipart**, campo `file`) → `{ path, download_url }`.
  Riusa `path` nei campi `*_path` (`image_path`, `video_path`, `audio_path`, `mask_path`, `mesh_path`).
- L'output di un job (`files[0].path`) è un path locale sul server: passalo come input al passo
  successivo, senza download/upload intermedi.

## Infra

| Metodo | Endpoint | Scopo |
|---|---|---|
| GET | `/health` | `{api, native_backend_up}` |
| POST | `/files/upload` | carica un file (multipart) → `{path, download_url}` |
| GET | `/jobs/{job_id}` | stato job (+ `files[]` a success) |
| GET | `/download?path=` | scarica un file prodotto |

---

## IMMAGINI (`tags: immagini`) — backend nativo :8189

| Endpoint | Parametri principali (default) |
|---|---|
| `POST /image/generate` | `prompt*`, `model*`, `arch=flux` (flux/flux2/zimage/ernie), `negative`, `width=1024`, `height=1024`, `steps=20`, `guidance=3.5`, `seed=0`, `loras`, `pulid_face`, `pulid_strength=0.8` |
| `POST /image/edit` | `prompt*`, `image_path*`, `model*`, `arch=kontext` (kontext/flux2_kontext), `steps=20`, `guidance=2.5`, `seed=0`, `max_side=1536` |
| `POST /image/recolor` | `image_path*`, `target*`, `subject`, `model*`, `steps=20`, `guidance=2.5`, `seed=0` |
| `POST /image/inpaint` | `prompt*`, `image_path*`, `mask_path*`, `model*`, `arch=inpaint` (inpaint/flux2_inpaint), `steps=20`, `guidance=30.0`, `seed=0` |
| `POST /image/outpaint` | come inpaint (maschera della regione da espandere; `outpaint=true` interno) |
| `POST /image/upscale` | `image_path*`, `model*` (es. `4x-UltraSharpV2.safetensors`), `target_scale=4.0`, `seed=0` |
| `POST /image/segment` | `image_path*`, `text*` (oggetto), `threshold=0.3`, `use_sam=true` → maschera PNG |
| `POST /image/faceswap` | `target_path*`, `source_face_path*` (InsightFace) |

Nota: Z-Image usa SDPA (SageAttention non supporta `attn_mask`); per gli arch image-conditioned
(kontext/inpaint) le dimensioni sono derivate dall'immagine (multipli di 16, lato lungo ≤ `max_side`).

Esempio (t2i con Z-Image):
```bash
curl -s -X POST http://127.0.0.1:8200/image/generate -H "Content-Type: application/json" -d '{
  "prompt":"a humanized beet chasing a flying winged opossum, storybook",
  "arch":"zimage","model":"E:/AI/Models/diffusion_models/Z-Image/zimageTurboNSFWBy_2602NSFWBF16.safetensors",
  "width":1024,"height":768,"steps":10,"guidance":3.5,"seed":42 }'
# → {"job_id":"...","status":"pending"}  → poi:  curl http://127.0.0.1:8200/jobs/<job_id>
```

---

## VIDEO (`tags: video`) — worker `ltx2_infer` (venv ltx2) + servizi nativi

| Endpoint | Parametri principali (default) |
|---|---|
| `POST /video/generate` | `prompt*`, `negative`, `width=768`, `height=512`, `fps=25`, `length_s=5`, `steps=40`, `guidance=3.0`, `seed=42`, `loras`; **i2v**: `image_path` (+`ref_strength=1.0`, `i2v_method=addguide`); **a2v/canto**: `audio_path` (+`audio_scale=3.0`) |
| `POST /video/keyframes` | `prompt*`, `keyframes*` `[{path,frame,strength}]`, + base |
| `POST /video/extend` | `prompt*`, `video_path*`, `add_seconds=5`, `seg_seconds=0`, `steps=8` |
| `POST /video/retake` | `prompt*`, `video_path*`, `start_time`, `end_time`, `steps=40` |
| `POST /video/v2v_edit` | `prompt*`, `control_video*`, `edit_lora*`, `control_strength=1.0`, `length_s=5` |
| `POST /video/controlnet` | `prompt*`, `control_video*`, `control_type=canny` (canny/depth/pose), `control_strength=0.8`, `image_path?` |
| `POST /video/lipsync` | `image_path*`, `audio_path*`, `prompt`, `audio_scale=3.0`, `separate_vocals=false`, `talk_region` (mouth/face), `talk_mask_offset=0` |
| `POST /video/enhance` | `video_path*`, `engine=seedvr2` (seedvr2/flashvsr), `seedvr2_model`, `resolution=1080`, `batch_size=5`, `block_swap=0`, `scale=4.0` — `engine=h3_upsampler` (MiniMax H3, nativo; cap.video_h3_upsampler): `h3_profile*` (nvfp4/int4/int8/fp8, nessun ripiego), `h3_te`, `megapixels=2.0`, `denoise=0.1`, `guide=true`, `lora_strength=1.0`, `tile_tokens=0` (0 = automatico dalla VRAM libera, tetto 70000); pesi assenti → errore con id e GB, nessun download |
| `POST /video/rife` | `video_path*`, `factor=2` |
| `POST /video/faceswap` | `video_path*`, `source_face_path*` |

Note: `width`/`height` arrotondate a **multipli di 64** (richiesti da LTX-2); i video >150 frame sono
**segmentati automaticamente** (chunk memory-bounded, niente OOM su clip lunghe).

Esempio (i2v 10s dall'immagine generata):
```bash
curl -s -X POST http://127.0.0.1:8200/video/generate -H "Content-Type: application/json" -d '{
  "prompt":"the beet chases the flying opossum across a meadow, playful chase",
  "image_path":"E:/AI/ltxstudio/output/native_zimage_42.png",
  "width":768,"height":576,"fps":16,"length_s":10,"steps":10,"guidance":3.0,"seed":42 }'
```

---

## MUSICA / AUDIO (`tags: musica`)

| Endpoint | Parametri principali (default) |
|---|---|
| `POST /music/song` | `tags*` (stile), `lyrics`, `name=song`, `engine=heartmula` (heartmula/acestep), `duration_s=120`, `instrumental=false`, `version=3B`, `guidance=15.0` (acestep), `lazy_load=false` |
| `POST /music/sonify` | `video_path` (vuoto = testo→audio), `prompt`, `duration_s=8`, `cfg_strength=4.5`, `num_steps=25` (MMAudio) |
| `POST /music/music_video` | `lyrics*`, `image_path*`, `song_path`, `width=704`, `height=512`, `fps=25`, `scene_seconds=6`, `video_style`, `steps=8` |
| `POST /audio/tts` | `text*`, `engine=edge` (edge/kokoro/f5tts/xtts2/fish/vibevoice/cosyvoice2), `voice_id`, `language`, `sample_path?` (clone), `sample_transcript` |
| `POST /audio/separate_vocals` | `audio_path*` (demucs → vocals.wav) |
| `POST /audio/transcribe` | `audio_path*` (Handy/whisper → txt) |

Esempio (canzone ACE-Step):
```bash
curl -s -X POST http://127.0.0.1:8200/music/song -H "Content-Type: application/json" -d '{
  "name":"ballad","engine":"acestep","tags":"medieval ballad, lute, playful, duet",
  "lyrics":"[verse]\n...","duration_s":180,"guidance":15.0 }'
```

---

## 3D / RIG / MONDO (`tags: 3d`)

| Endpoint | Parametri principali (default) |
|---|---|
| `POST /threed/image_to_3d` | `image_path*`, `name=model`, `seed=1`, `ss_steps=12`, `slat_steps=12`, `simplify=0.95`, `texture_size=1024` (TRELLIS → GLB) |
| `POST /threed/rig` | `mesh_path*`, `name=rig`, `template=Mixamo`, `file_format=glb` (SkinToken → scheletro+skinning) |
| `POST /threed/world` | `name=world`, `image_path` (vuoto=testo→scena), `prompt`, `resolution=1600`, `low_vram=false`, `mesh=true` (WorldGen → GLB/PLY) |
| `POST /threed/convert` | `glb_path*`, `fmt=fbx` (fbx/obj, via Blender) |

> mocap (webcam live) e retarget/animazione Blender restano nella GUI (interattivi), non esposti qui.

---

## FILM (`tags: film`)

| Endpoint | Parametri principali (default) |
|---|---|
| `POST /film/script` | `subject*`, `style`, `max_tokens=2000` (LLM → trattamento+scene, .txt) |
| `POST /film/asset_image` | `prompt*`, `width=1024`, `height=1024`, `seed=7`, `arch=flux`, `face_ref` (PuLID), `ref_image` (Kontext), `loras`, `negative` |
| `POST /film/shot_video` | `prompt*`, `width=768`, `height=512`, `fps=25`, `duration_s=4`, `seed=7`, `keyframes`, `loras`, `audio_path`, `negative` |

> L'orchestrazione di PROGETTO completa (DB, scheduler, montaggio, doppiaggio, modalità Auto) resta
> nella GUI Film Studio (stateful). Il videoclip canzone→video è in `/music/music_video`.

---

## Prerequisiti (endpoint "gated")

Alcune funzioni richiedono venv/pesi isolati; se assenti il job torna `error` con l'hint di installazione:
SeedVR2, FlashVSR, TRELLIS, WorldGen, HeartMuLa, ACE-Step, MMAudio, TTS-clone (f5/xtts/fish/vibevoice),
face-swap (InsightFace), PuLID → `scripts/install_*.py`.

## Agenti (plugin + skill)

`agents/`: contratti macchina condivisi (`openapi.json`, `tools.openai.json`, `manifest.json`) + pack
per agente (`hermes` default, `openclaw`, `claude_code`, `antigravity`, `openai_codex`, `generic`),
ognuno con `SKILL.md` + `plugin.json`. Rigenera con `python -m agents.build_packs`; installa con
`python -m agents.install_skill [--agent <nome>]`.
