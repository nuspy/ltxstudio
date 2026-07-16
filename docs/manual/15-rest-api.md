# Local REST API

*Automate every creative function in AI Creative Center from your own scripts, the mobile companion, or an AI agent — all on your own machine, nothing in the cloud.*

> **Edition:** **Studio Pro only** · **Capability:** `api.rest`

The **local REST API** exposes every native function of AI Creative Center — images, video, film shots, music, voice, 3D, world generation, and LoRA training — as **asynchronous jobs** that return download links. It is the same engine that powers the [mobile companion app](README.md) and the multi‑agent skill packs. Everything runs on `http://127.0.0.1:8200`; no data leaves your computer.

See also [Advanced settings](12-advanced.md) and the [capability table](17-appendix.md).

---

## Starting and stopping the server

- **Autostart** — Run `python -m api.autostart`. It starts the server if it is down, waits for `/health`, and prints `READY http://127.0.0.1:8200` when ready. The agent skill packs call this at the start of a session.
- **Manual start** — `python -m api.app`.
- **Auto‑shutdown** — After **30 minutes** with no requests *and* no jobs queued or running, the server shuts itself down. The next call simply re‑launches it via autostart.
- **Interactive docs** — Swagger UI at `http://127.0.0.1:8200/docs`; the OpenAPI schema at `/openapi.json`.

Environment variables: `LTX_API_HOST` (default `127.0.0.1`), `LTX_API_PORT` (default `8200`), `LTX_API_IDLE_TIMEOUT` (default `1800` seconds), `LTX_API_PYTHON` (the Python interpreter that runs the API).

---

## The asynchronous job model

Every creative action is a `POST` with a JSON body that returns a **job id** immediately; you then poll for the result and download the files. GPU jobs run **one at a time** — do not fire jobs in parallel.

1. **Submit** — `POST` an action → `{ "job_id": "...", "status": "pending" }`.
2. **Poll** — `GET /jobs/{job_id}` (back off 2–5 s) until `status` is `success` or `error`:
   - `running` → reports `progress` (0–1) and a `message`.
   - `pending` → reports `queue_position`.
   - `success` → returns `files: [{ path, name, download_url }]`.
   - `error` → returns an `error` message (often a missing prerequisite to install).
3. **Download** — `GET /download?path=<path>` (served only from inside `output/` and `output/api_uploads/`).

**Chaining** — Upload a file once with `POST /files/upload` (multipart, field `file`) and reuse the returned `path` in any `*_path` field (`image_path`, `video_path`, `audio_path`, `mask_path`, `mesh_path`). A job's output path is a local path on the server, so you can feed it straight into the next step with no intermediate download.

---

## Gating: 403 with upsell

Each endpoint checks the same capability the desktop app checks. If your edition does not include a capability, the endpoint returns **HTTP 403** with upsell information (the minimum edition and how to unlock it) rather than running. The API itself is Studio Pro (`api.rest`); the tables below list the extra capability each endpoint requires.

---

## Endpoint reference

### Infra

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | `{ api, native_backend_up }` |
| POST | `/files/upload` | Upload a file (multipart) → `{ path, download_url }` |
| GET | `/jobs/{job_id}` | Job status (plus `files[]` on success) |
| GET | `/download?path=` | Download a produced file |

### Images (`tag: images`)

| Method · Path | Key parameters (defaults) | Capability |
|---|---|---|
| POST `/image/generate` | `prompt*`, `model*`, `arch=flux` (flux/flux2/zimage/ernie), `negative`, `width=1024`, `height=1024`, `steps=20`, `guidance=3.5`, `seed=0`, `loras`, `pulid_face`, `pulid_strength=0.8` | `cap.gen_images` |
| POST `/image/edit` | `prompt*`, `image_path*`, `model*`, `arch=kontext` (kontext/flux2_kontext), `steps=20`, `guidance=2.5`, `seed=0`, `max_side=1536` | `cap.gen_images` |
| POST `/image/recolor` | `image_path*`, `target*`, `subject`, `model*`, `steps=20`, `guidance=2.5`, `seed=0` | `cap.inpaint` |
| POST `/image/inpaint` | `prompt*`, `image_path*`, `mask_path*`, `model*`, `arch=inpaint` (inpaint/flux2_inpaint), `steps=20`, `guidance=30.0`, `seed=0` | `cap.inpaint` |
| POST `/image/outpaint` | As inpaint (mask marks the region to expand) | `cap.inpaint` |
| POST `/image/upscale` | `image_path*`, `model*` (e.g. `4x-UltraSharpV2.safetensors`), `target_scale=4.0`, `seed=0` | `cap.upscale_image` |
| POST `/image/segment` | `image_path*`, `text*`, `threshold=0.3`, `use_sam=true` → mask PNG | `cap.sam` |
| POST `/image/faceswap` | `target_path*`, `source_face_path*` (InsightFace) | `cap.gen_images` |

### Video (`tag: video`)

| Method · Path | Key parameters (defaults) | Capability |
|---|---|---|
| POST `/video/generate` | `prompt*`, `negative`, `width=768`, `height=512`, `fps=25`, `length_s=5`, `steps=40`, `guidance=3.0`, `seed=42`, `loras`; **i2v**: `image_path` (+`ref_strength=1.0`, `i2v_method=addguide`); **a2v**: `audio_path` (+`audio_scale=3.0`) | `cap.video` |
| POST `/video/keyframes` | `prompt*`, `keyframes*` `[{path,frame,strength}]` + base params | `cap.video` |
| POST `/video/extend` | `prompt*`, `video_path*`, `add_seconds=5`, `seg_seconds=0`, `steps=8` | `cap.extend_retake` |
| POST `/video/retake` | `prompt*`, `video_path*`, `start_time`, `end_time`, `steps=40` | `cap.extend_retake` |
| POST `/video/v2v_edit` | `prompt*`, `control_video*`, `edit_lora*`, `control_strength=1.0`, `length_s=5` | `cap.video` |
| POST `/video/controlnet` | `prompt*`, `control_video*`, `control_type=canny` (canny/depth/pose), `control_strength=0.8`, `image_path?` | `cap.controlnet` |
| POST `/video/lipsync` | `image_path*`, `audio_path*`, `prompt`, `audio_scale=3.0`, `separate_vocals=false`, `talk_region` (mouth/face), `talk_mask_offset=0` | `cap.video` |
| POST `/video/enhance` | `video_path*`, `engine=seedvr2` (seedvr2/flashvsr), `seedvr2_model`, `resolution=1080`, `batch_size=5`, `block_swap=0`, `scale=4.0` | `cap.restore` |
| POST `/video/rife` | `video_path*`, `factor=2` | `cap.restore` |
| POST `/video/faceswap` | `video_path*`, `source_face_path*` | `cap.video` |

> Width/height are rounded to multiples of **64** (required by LTX‑2). Clips over 150 frames are segmented automatically (memory‑bounded chunks, no out‑of‑memory on long clips).

### Music & Audio (`tag: music`)

| Method · Path | Key parameters (defaults) | Capability |
|---|---|---|
| POST `/music/song` | `tags*` (style), `lyrics`, `name=song`, `engine=heartmula` (heartmula/acestep), `duration_s=120`, `instrumental=false`, `version=3B`, `guidance=15.0` (acestep), `lazy_load=false` | `cap.music_song` |
| POST `/music/sonify` | `video_path` (empty = text→audio), `prompt`, `duration_s=8`, `cfg_strength=4.5`, `num_steps=25` (MMAudio) | `cap.music_sonify` |
| POST `/music/music_video` | `lyrics*`, `image_path*`, `song_path`, `width=704`, `height=512`, `fps=25`, `scene_seconds=6`, `video_style`, `steps=8` | `cap.music_video` |
| POST `/audio/tts` | `text*`, `engine=edge` (edge/kokoro/f5tts/xtts2/fish/vibevoice/cosyvoice2), `voice_id`, `language`, `sample_path?` (clone), `sample_transcript` | `cap.audio_tts_clone` |
| POST `/audio/separate_vocals` | `audio_path*` (demucs → vocals.wav) | `cap.audio_tts_clone` |
| POST `/audio/transcribe` | `audio_path*` (whisper → txt) | `cap.audio_tts_clone` |

### 3D / Rig / World (`tag: 3d`)

| Method · Path | Key parameters (defaults) | Capability |
|---|---|---|
| POST `/threed/image_to_3d` | `image_path*`, `name=model`, `seed=1`, `ss_steps=12`, `slat_steps=12`, `simplify=0.95`, `texture_size=1024` (TRELLIS → GLB) | `cap.threed` |
| POST `/threed/rig` | `mesh_path*`, `name=rig`, `template=Mixamo`, `file_format=glb` (SkinToken → skeleton + skinning) | `cap.threed` |
| POST `/threed/world` | `name=world`, `image_path` (empty = text→scene), `prompt`, `resolution=1600`, `low_vram=false`, `mesh=true` (WorldGen → GLB/PLY) | `cap.threed` |
| POST `/threed/convert` | `glb_path*`, `fmt=fbx` (fbx/obj, via Blender) | `cap.threed` |

> Live webcam mocap and Blender retarget/animation remain interactive in the desktop GUI and are not exposed over the API.

### Film (`tag: film`)

| Method · Path | Key parameters (defaults) | Capability |
|---|---|---|
| POST `/film/script` | `subject*`, `style`, `max_tokens=2000` (LLM → treatment + scenes, .txt) | `cap.filmstudio` |
| POST `/film/asset_image` | `prompt*`, `width=1024`, `height=1024`, `seed=7`, `arch=flux`, `face_ref` (PuLID), `ref_image` (Kontext), `loras`, `negative` | `cap.filmstudio` |
| POST `/film/shot_video` | `prompt*`, `width=768`, `height=512`, `fps=25`, `duration_s=4`, `seed=7`, `keyframes`, `loras`, `audio_path`, `negative` | `cap.filmstudio` |

> Full project orchestration (database, scheduler, editing, dubbing, Auto mode) stays in the stateful Film Studio GUI. The song→video music clip lives at `/music/music_video`.

### Training (`tag: training`)

| Method · Path | Purpose | Capability |
|---|---|---|
| POST `/train/lora_image` | Auto‑dataset photo LoRA training | `cap.lora_image_train` |
| POST `/train/lora_video` | Auto‑dataset video LoRA training | `cap.lora_video_train` |

### Catalog (`tag: catalog`)

| Method · Path | Purpose |
|---|---|
| GET `/features` | Activation state per capability (identical to desktop) + upsell info |
| GET `/models?task=` | Available models per task (`image` / `video` / `upscale` / `restore`) + LoRA + recommended model |
| GET `/components?model=` | Auto components for a chosen model (arch / VAE / text encoder / steps / CFG / compatible LoRA) |
| GET `/loras?model=` | LoRA list (all, or only those compatible with a given model) |

---

## Prerequisites (gated features)

Some features need isolated virtual environments or weights. If they are missing, the job returns `error` with an install hint: SeedVR2, FlashVSR, TRELLIS, WorldGen, HeartMuLa, ACE‑Step, MMAudio, TTS‑clone engines (f5/xtts/fish/vibevoice), face‑swap (InsightFace), and PuLID all resolve to a `scripts/install_*.py` helper.

## Multi‑agent skill packs

The `agents/` folder ships machine‑readable contracts (`openapi.json`, `tools.openai.json`, `manifest.json`) plus a skill pack per agent — **Hermes** is the default, alongside `openclaw`, `claude_code`, `antigravity`, `openai_codex`, and `generic`. Each pack contains a `SKILL.md` and a `plugin.json`. Rebuild them with `python -m agents.build_packs` and install one with `python -m agents.install_skill [--agent <name>]`. These packs let an AI agent drive the whole API — submit jobs, poll, and download results — using the same async model described above.

---

### Worked example — image then video

```bash
# 1. Generate an image (Z-Image)
curl -s -X POST http://127.0.0.1:8200/image/generate \
  -H "Content-Type: application/json" -d '{
    "prompt":"a humanized beet chasing a flying winged opossum, storybook",
    "arch":"zimage","model":"E:/AI/Models/diffusion_models/Z-Image/....safetensors",
    "width":1024,"height":768,"steps":10,"guidance":3.5,"seed":42 }'
# → {"job_id":"...","status":"pending"}   then poll:  GET /jobs/<job_id>

# 2. Animate the produced image (i2v, reuse its server-side path)
curl -s -X POST http://127.0.0.1:8200/video/generate \
  -H "Content-Type: application/json" -d '{
    "prompt":"the beet chases the flying opossum across a meadow, playful chase",
    "image_path":"E:/AI/ltxstudio/output/native_zimage_42.png",
    "width":768,"height":576,"fps":16,"length_s":10,"steps":10,"guidance":3.0,"seed":42 }'
```
