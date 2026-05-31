# LTX Studio — REST API

API REST (FastAPI) che espone **tutte le funzioni della GUI**: generazione/editing immagini e
video tramite ComfyUI (LTX‑2.3 + FLUX/FLUX.2 + SeedVR2).

## Avvio

1. Avvia **ComfyUI** (`run_nvidia.bat`) — deve essere su `http://127.0.0.1:8188`.
2. Avvia l'API: `server\run_api.bat`  (oppure `python -m uvicorn server.api:app --host 0.0.0.0 --port 8200`).
3. **Swagger UI** (prova interattiva): <http://localhost:8200/docs>
   **OpenAPI**: <http://localhost:8200/openapi.json>

## Convenzioni comuni

- Tutte le richieste sono **`multipart/form-data`** (campi `Form` + file).
- Ogni input‑file accetta **due modalità** (alternative):
  - **upload**: campo file (es. `image`, `mask`, `video`, `audio`).
  - **path locale del server**: campo stringa `*_path` (es. `image_path=E:\foto\a.png`). Utile se il
    file è già sul PC del server (nessun upload).
- **Output**: ogni endpoint risponde JSON con `files` (path locali sul server), `videos`, `images`,
  `prompt_id`, `node_count`, e `download` (URL pronti per `/download`).
- **Download**: `GET /download?path=<path locale>` scarica un file dal server.
- I file generati sono salvati in `E:\AI\ltxstudio\output`.

Esempio risposta:
```json
{ "prompt_id":"…", "node_count":13, "files":["E:\\AI\\ltxstudio\\output\\ltxstudio_gen_0001.png"],
  "images":["…_0001.png"], "videos":[], "download":["/download?path=E:\\AI\\ltxstudio\\output\\…png"] }
```

---

## Utilità

| Metodo | Endpoint | Descrizione |
|---|---|---|
| GET | `/health` | Stato (ComfyUI attivo?). |
| GET | `/models?folder=` | Elenco modelli in una cartella ComfyUI (`diffusion_models`, `loras`, `vae`, `upscale_models`…). |
| GET | `/models/seedvr2` | Modelli SeedVR2 disponibili (etichetta → file). |
| POST | `/files/upload` | Carica un file sul server → ritorna il `path` (riusabile come `*_path`). |
| GET | `/download?path=` | Scarica un file dal path locale del server. |
| POST | `/audio/upload` | Carica audio pre‑registrato (per voce/clone) → ritorna `path`. |

---

## Immagini

### POST `/image/generate` — generazione txt2img
Genera un'immagine da prompt. Campi: `prompt*`, `arch` (`flux`|`z-image`|`ernie`), `model*`,
`negative`, `width`, `height`, `steps`, `guidance`, `seed`, `loras` (JSON `[{"name","strength"}]`).

```bash
curl -F prompt="a cat on the beach" -F arch=flux \
  -F model="Flux\\FLUX_2_Dev_FP8_Mixed_Scaled.safetensors" -F width=1024 -F height=1024 \
  http://localhost:8200/image/generate
```

### POST `/image/edit` — image‑to‑image / integrazione (Kontext)
L'immagine è il **riferimento**: viene rigenerata/armonizzata secondo il `prompt` (non incolla).
Per **multi‑immagine → 1** componi prima le immagini (o usa `/image/inpaint`). Campi: `prompt`,
`image`|`image_path`, `engine` (`kontext`|`flux2`), `model` (se flux2), `negative`, `steps`, `seed`,
`aux_lora` (es. `MagicQuill\\puzzle_lora.safetensors`).

### POST `/image/inpaint` — inpainting (con maschera)
Rigenera l'area mascherata. Campi: `prompt*`, `image`|`image_path`, `mask`|`mask_path` (bianco =
area da rigenerare), `model` (checkpoint Fill), `steps`, `denoise`, `guidance`, `seed`, `negative`.

```bash
curl -F prompt="red dress" -F image=@photo.png -F mask=@mask.png http://localhost:8200/image/inpaint
```

### POST `/image/recolor` — cambia colore (Kontext)
Ricolora mantenendo texture/luce. Campi: `target*` (es. `#cc0000`), `subject` (es. "la maglietta"),
`image`|`image_path`, `engine`, `model`, `steps`, `seed`.

### POST `/image/upscale` — aumenta definizione immagine
Modello AI di super‑risoluzione + opzionale rifinitura FLUX. Campi: `image`|`image_path`, `model*`
(es. `4x-UltraSharpV2.safetensors`), `scale`, `refine`, `refine_model` (FLUX), `refine_steps`,
`refine_denoise`, `prompt`, `seed`.

### POST `/image/segment` — selezione oggetto (CLIPSeg + SAM)
Maschera dell'oggetto descritto. Campi: `image`|`image_path`, `text*`, `sam_model`, `threshold`.

---

## Video

### POST `/video/generate` — generazione video LTX‑2.3
T2V o **I2V** (invia `image`). Campi: `prompt*`, `negative`, `width`, `height`, `fps`, `length_s`,
`steps`, `cfg`, `sampler`, `seed`, `nsfw_mode` (`sfw`|`nsfw`), `audio_mode`
(`native`|`user`|`clone`|`mute`), `hr_refine` (rifinitura HR), `ref_strength` (forza immagine i2v),
`image`|`image_path`, `audio`|`audio_path` (per `user`/`clone`), `loras`.

```bash
# I2V con audio clonato da file
curl -F prompt="she smiles and waves" -F image=@frame1.png -F audio_mode=clone \
  -F audio=@voice.wav -F length_s=8 http://localhost:8200/video/generate
```

### POST `/video/enhance` — definizione video (restore + super‑risoluzione)
Campi: `video`|`video_path`, `engine` (`seedvr2`|`esrgan`|`flashvsr`), `seedvr2_model`,
`resolution`, `blocks_to_swap`, `batch_size`, `esrgan_model`, `target_scale`, `fps`, `rife`
(aumento FPS), `rife_multiplier`, `rife_ckpt`, `seed`. SeedVR2 scarica il modello al primo uso.

### POST `/video/extend` — prolunga video (LTX‑2.3 nativo)
Estensione multi‑segmento condizionata sulla coda del video. Campi: `prompt*`, `video`|`video_path`,
`width`, `height`, `fps`, `add_seconds`, `segments`, `overlap_frames`, `context_frames`, `steps`,
`seed`, `negative`.

---

## Note

- I tempi: immagini secondi/decine di secondi; video diversi minuti (modello 22GB).
- Gli endpoint sono **sincroni** (attendono il completamento del job; timeout 20–60 min secondo il tipo).
- Per riusare lo stesso file in più chiamate senza ricaricarlo: `/files/upload` (o `/audio/upload`)
  → usa il `path` ottenuto nei campi `*_path`.
