# LTX Studio — Catalogo completo delle funzioni

> Documento triplo-scopo: **(1) manuale utente**, **(2) base per REST API**, **(3) skill per agenti AI**.
> Tutto ricavato leggendo il codice reale (`inference_server/`, `core/`, `ui/`, `data/`) — nessuna funzione inventata.
> Dove un parametro non è verificabile con certezza dal codice è marcato `(verificare)`.

---

## Architettura

LTX Studio è un'app desktop **PySide6** (cartella `ui/`) che pilota due **server di inferenza nativi** (diffusers/torch, **zero ComfyUI** nella distribuzione):

| Server | Porta | Client | Cosa serve |
|---|---|---|---|
| **InferenceServer** (immagini, upscale, segmentazione, post-process video) | `:8189` | `core/native_driver.py` | moduli `image`, `upscale`, `segment`, `video_upscale`, `video_face`, `video_fps`, `video_edit`, `video_long` |
| **LTX-2.3 video server** | `:8190` | `core/ltx2_service.py` | generazione video LTX-2.3 22B (t2v/i2v/a2v/t2a/controlnet/inpaint/retake) |
| **Handy-API** (voce) | `:8720` | `core/handy_api.py` | TTS Kokoro, STT Whisper, voice-clone |

**Protocollo nativo `:8189`** (`core/native_driver.py`):
- `POST /submit` body `{"module": <str>, "params": <dict>}` → `{"prompt_id": ...}`
- `GET /history/{prompt_id}` → `{status: pending|running|success|error, output: [...]}`
- `GET /system_stats`
- Helper: `submit(module, params)`, `run_and_wait(module, params, poll=1.0, timeout=1800)`, `is_up()`.

Le pipeline immagine in `core/*_pipeline.py` **costruiscono un grafo** (dict `{node_id: {class_type, inputs}}`) che viene passato come `params["prompt"]` al modulo nativo corrispondente. In questo documento, per ogni funzione immagine, i **Parametri** elencati sono quelli del metodo `build(...)` della pipeline (l'interfaccia logica utile per REST/skill); l'output del server è sempre il path del file salvato.

**Routing** (`core/driver_router.py`): moduli nativi = `{image, video, upscale, segment, video_upscale, video_face, video_fps, video_edit, video_long}`.

**Convenzioni trasversali**
- Selettore **SFW/NSFW** in tutti i tab immagine/video: ri-filtra modelli e LoRA (badge 🔞).
- **LoRA**: catena `LoraLoaderModelOnly` con `{name, strength}` (strength default 1.0); catalogo `core/lora_catalog.py`, preferiti ⭐ via `core/user_prefs.py`.
- **Auto-detection architettura** modelli (`core/model_detect.py`, cache `data/model_detect.json`): da header (chiavi+forme+dtype+modelspec) ricava `family/variant/role/kind/precision`. I combo "Modello/Encoder/VAE/CLIP" si auto-popolano e si auto-selezionano.
- **Precisione/VRAM**: combo colorati verde/giallo/rosso (`core/model_precision.py`), filtro fp16/fp8/fp4/gguf, auto-select VRAM-aware.
- **KSampler standard FLUX** (immagini): `cfg=1.0, sampler="euler", scheduler="simple"` (inpaint usa `scheduler="normal"`).
- Output salvati in `E:\AI\ltxstudio\output\...`.

---

# IMMAGINI

Modulo nativo: **`image`** (server `:8189`). Pipeline: `core/gen_pipeline.py`, `inpaint_pipeline.py`, `integrate_pipeline.py`, `kontext_pipeline.py`. Esecutore backend: `inference_server/backends/image_flux.py`.

## 1. Genera immagine — FLUX (`arch=flux`)
- **Cosa fa**: testo→immagine con modelli FLUX.1 (e checkpoint FLUX derivati: getphat, fuxcapacity, krea…).
- **GUI**: tab **Immagine** → sotto-tab **Genera immagine** → pulsante *Genera* (`request_generate`).
- **Backend**: modulo `image`, `GenPipeline.build(arch="flux", ...)`. Output `SaveImage` prefix `ltxstudio_gen`.
- **Parametri**:

| nome | tipo | default | descrizione |
|---|---|---|---|
| `arch` | str | `flux` | architettura |
| `model` | str | `__auto__` | checkpoint/UNET (auto-select via `core.auto_select`) |
| `prompt` | str | — | descrizione |
| `negative` | str | neg string | prompt negativo |
| `width`/`height` | int | 1024/1024 | risoluzione |
| `steps` | int | 20 | passi (1–80) |
| `cfg`/`guidance` | float | 3.5 | guida (cfg override) |
| `sampler` | str | `euler` | euler/euler_ancestral_cfg_pp/dpmpp_2m/res_2s |
| `scheduler` | str | `simple` | karras/bong_tangent/beta/simple |
| `encoder` | str | t5xxl_fp16 | text encoder |
| `clip_type` | str | — | tipo CLIP |
| `vae` | str | `Flux\ae.safetensors` | VAE |
| `seed` | int | 0 | seme (0–2147483647) |
| `loras` | list[{name,strength}] | `[]` | LoRA |
| `enhance` | bool | false | prompt enhancer (Gemma) |
| `nsfw` | bool | false | modalità NSFW |

- **Output**: file PNG (path in `output`).
- **Esempio payload**:
```json
{"module":"image","params":{"prompt":{"__pipeline__":"gen","arch":"flux","model":"__auto__","prompt":"a red fox in snow","steps":20,"cfg":3.5,"seed":0,"loras":[]}}}
```
- **Note**: modelli FLUX gated (HF token solo per download). PuLID disponibile (funz. 5).

## 2. Genera immagine — FLUX.2 Dev (`arch=flux2`)
- **Cosa fa**: testo→immagine con **FLUX.2 Dev** nativo (diffusers, no ComfyUI). Encoder Mistral-3-small (`mistral_3_small_flux2`), VAE `flux2-vae`, `CLIPLoader type=flux2`. *(funzione recente di questa sessione: t2i FLUX.2 nativo funzionante con block-swap ~4GB VRAM)*.
- **GUI**: tab **Immagine** → **Genera immagine** (modello FLUX.2 selezionato; auto-detect imposta encoder flux2).
- **Backend**: modulo `image`, `GenPipeline.build(arch="flux2", ...)` (FLUX.2 rilevato anche da nome modello via `is_flux2`).
- **Parametri**: come funz. 1; differenze: `vae=Flux\flux2-vae.safetensors`, `clip_type=flux2`, encoder filtrato via `core.flux2_encoder.is_compatible_encoder`, defaults euler/simple/20/3.5.
- **Output**: PNG.
- **Esempio**:
```json
{"module":"image","params":{"prompt":{"__pipeline__":"gen","arch":"flux2","model":"flux2-dev","prompt":"cyberpunk alley, neon","steps":20,"seed":7}}}
```
- **Note**: block-swap dinamico per VRAM bassa.

## 3. Genera immagine — Z-Image (`arch=zimage`)
- **Cosa fa**: testo→immagine con **Z-Image Turbo** (costruito a mano come `type=lumina2`), veloce.
- **GUI**: **Immagine → Genera immagine** (modello Z-Image).
- **Backend**: `GenPipeline.build(arch="zimage")`. Encoder/VAE risolti da inventario (`set_inventory`).
- **Parametri**: come funz. 1; defaults: turbo `steps=8, cfg=1.0`, base `~30, cfg=4.0`, sampler `res_multistep`, scheduler `simple`.
- **Output**: PNG. **Note**: non gated.

## 4. Genera immagine — Ernie (`arch=ernie`)
- **Cosa fa**: testo→immagine con **Ernie** (Baidu) **nativo** (diffusers + shim transformers<5, `clip_type=flux2`).
- **GUI**: **Immagine → Genera immagine** (modello Ernie).
- **Backend**: `GenPipeline.build(arch="ernie")`. Defaults `steps=20, cfg=4.0, sampler=euler, scheduler=simple`.
- **Output**: PNG. **Note**: gira nativo (no ComfyUI); non gated.

## 5. PuLID — adapter identità volto (FLUX)
- **Cosa fa**: imprime l'identità di un volto (da foto) sulla generazione FLUX, agganciando i blocchi del FluxTransformer2DModel.
- **GUI**: **Immagine → Genera immagine** → campo *PuLID face* + slider *strength* (FLUX only).
- **Backend**: `core/pulid_adapter.py` (repo `E:\AI\Tools\PuLID`, pesi `pulid_flux_v0.9.1.safetensors`). `load() → extract_id(face) → attach(strength)`. Usa InsightFace antelopev2 + EVA-CLIP + IDFormer.
- **Parametri**: `pulid_face` (str path), `pulid_strength` (float, def 0.8).
- **Output**: generazione con somiglianza volto. **Note**: fix contatore `ca_idx` per-step + single-block (validato).

## 6. Inpaint (FLUX Fill)
- **Cosa fa**: rigenera l'area mascherata di un'immagine.
- **GUI**: **Immagine → Inpaint/Outpaint** → disegno maschera su canvas → *Inpaint* (`canvas.request_inpaint`).
- **Backend**: modulo `image`, `InpaintPipeline.build(mode="inpaint", ...)`. Branch FLUX Fill (`UNETLoader`/GGUF + `InpaintModelConditioning` + `DifferentialDiffusion`). Output prefix `ltxstudio_inpaint`.
- **Parametri**:

| nome | tipo | default | descrizione |
|---|---|---|---|
| `image_file` | str | — | immagine |
| `mask_file` | str | — | maschera (canale rosso) |
| `prompt` | str | — | cosa generare |
| `mode` | str | `inpaint` | inpaint/outpaint |
| `denoise` | float | 1.0 | forza |
| `steps` | int | 20 | passi |
| `guidance` | float | 30.0 | guida (Fill range 4–30) |
| `feather` | int | 24 | sfumatura maschera |
| `mask_grow` | int | 0 | dilatazione |
| `seed` | int | 0 | seme |
| `lora`/`lora_strength` | str/float | —/1.0 | LoRA |
| `unet`/`checkpoint` | str | `Flux\FLUX_DEV_Fill.safetensors` | modello |
| `retention` | float | — | 0..1 mappa su guidance |

- **Output**: PNG. **Note**: l'inpaint adotta le proporzioni reali della foto (fix proporzioni). Varianti modello: FLUX Fill GGUF NSFW (Hyper-8, 8 step).

## 7. Outpaint
- **Cosa fa**: estende la tela oltre i bordi (genera contesto nuovo).
- **GUI**: **Immagine → Inpaint/Outpaint** → *Outpaint* (`canvas.request_outpaint`) con margini.
- **Backend**: `InpaintPipeline.build(mode="outpaint", margins=(t,r,b,l))` → `ImagePadForOutpaint` (senza maschera).
- **Parametri**: come funz. 6 + `margins` (tuple di 4 int), `outpaint_prompt`.
- **Output**: PNG (immagine allargata).

## 8. FLUX.2 Klein inpaint (`arch=flux2_inpaint`)
- **Cosa fa**: inpaint/edit-instruct evoluto con **FLUX.2 Klein** (Miraclein), distinto dal FLUX.1 Fill. *(funzione recente)*. Encoder **Qwen3** (4B/8B Klein TE).
- **GUI**: **Immagine → Inpaint/Outpaint** → modello *FLUX.2 Klein NSFW Miraclein* (12 step default).
- **Backend**: modulo `image`, branch FLUX.2 edit-instruct: `VAEEncode` + `SetLatentNoiseMask` + `ReferenceLatent` (Mistral/flux2-vae). Arch `flux2_inpaint`.
- **Parametri**: come funz. 6; `steps` def 12, encoder Qwen3, guidance FLUX.2 range [2..6].
- **Output**: PNG. **Note**: modalità edit-instruct, ottimo per NSFW.

## 9. Integra (img2img mascherato)
- **Cosa fa**: armonizza un ritaglio incollato nella scena **preservandone il contenuto** (denoise basso + `SetLatentNoiseMask`). Diverso da Kontext (che rigenera).
- **GUI**: **Immagine → Inpaint/Outpaint** → *Integra* (`request_integrate`).
- **Backend**: modulo `image`, `IntegratePipeline.build(...)` (FLUX/FLUX.2 auto). Prefix `ltxstudio_integrate`.
- **Parametri**: `image_file`, `mask_file`, `model`, `prompt`, `denoise` (0.55), `steps` (20), `guidance` (3.5/2.5), `negative`, `seed`, `lora`/`lora_strength`. UI: `integrate_engine` kontext|flux2, `integrate_puzzle_lora` (FLUX.1 only).
- **Output**: PNG.

## 10. Kontext — editing per riferimento / recolor
- **Cosa fa**: editing guidato dell'immagine via **FLUX.1-Kontext** (o FLUX.2) con `ReferenceLatent` — rigenera/armonizza una regione secondo il prompt.
- **GUI**: **Immagine → Integra** (engine *kontext*); **→ Colore** per recolor.
- **Backend**: modulo `image`, `KontextPipeline.build(engine="kontext"|...)` (FLUX.2 = arch `flux2_kontext`, derivato dal nome modello). Prefix `ltxstudio_kontext`. Metodi: `.integrate()`, `.recolor()`, `.compose()`.
- **Parametri**: `image_file`, `prompt`, `engine` (kontext/flux2), `model`, `negative`, `steps` (20), `guidance` (2.5), `seed`, `aux_lora`/`aux_strength`.
- **Output**: PNG. **Note**: proporzioni gestite via `FluxKontextImageScale` (fix proporzioni).

## 11. Componi più foto (multi-reference / stitch)
- **Cosa fa**: combina 2+ foto in una scena coerente senza "incollaggio" (`multiref` = un `ReferenceLatent` per foto; `stitch` = griglia `ImageStitch`). Maschere per-foto opzionali.
- **GUI**: **Immagine → Componi più foto** → *Componi* (`request_compose`).
- **Backend**: modulo `image`, `KontextPipeline.compose(image_files, method, engine, ...)`. Prefix `ltxstudio_compose`. `prep_masked()` ritaglia su bbox.
- **Parametri**: `images` (list[str]), `masks` (dict opz), `method` (multiref/stitch), `prompt`, `engine` (kontext/flux2), `model`, `steps` (20), `guidance` (2.5), `seed`.
- **Output**: PNG.
- **Esempio**:
```json
{"module":"image","params":{"prompt":{"__pipeline__":"compose","method":"multiref","engine":"flux2","images":["a.png","b.png"],"prompt":"the two characters shaking hands"}}}
```

## 12. Cambia colore / Recolor
- **Cosa fa**: cambia il colore di un soggetto mantenendo texture e luce (Kontext prompt-based).
- **GUI**: **Immagine → Colore** → *Cambia colore* (`request_recolor`).
- **Backend**: `KontextPipeline.recolor(image_file, target, subject, ...)`.
- **Parametri**: `target` (hex), `subject` (str), `engine` (kontext/flux2), `steps` (20).
- **Output**: PNG.

## 13. Prompt enhancer
- **Cosa fa**: espande/migliora il prompt via Gemma-3 locale (sostituisce il nodo ComfyUI `QwenVL_PromptEnhancer`).
- **GUI**: checkbox *Migliora prompt* nei tab immagine/video.
- **Backend**: `core/prompt_enhance.py` → `enhance(prompt, kind="image"|"video", nsfw=False, max_new_tokens=220)`. Non lancia mai eccezioni (fallback = prompt originale).
- **Parametri**: `prompt`, `kind`, `nsfw`.
- **Output**: stringa prompt potenziato (~50 parole EN).

---

# SEGMENTAZIONE

Modulo nativo: **`segment`**. Pipeline `core/segment_pipeline.py`. Backend `inference_server/backends/image_segment.py`. Default: `GroundingDINO_SwinB`, `sam_hq_vit_l`.

## 14. Segmentazione da testo
- **Cosa fa**: produce una **maschera PNG** dell'oggetto descritto a parole (GroundingDINO + SAM-HQ).
- **GUI**: box **Selezione AI** sotto il canvas → testo + *Seleziona* (`request_segment_text`).
- **Backend**: modulo `segment`, `SegmentPipeline.build_text(...)`. Output `MaskToImage`→`SaveImage` prefix `ltxstudio_segmask`.
- **Parametri**:

| nome | tipo | default | descrizione |
|---|---|---|---|
| `image_file` | str | — | immagine |
| `text` | str | — | oggetto da selezionare |
| `detector` | str | `grounding_dino` | grounding_dino / grounding_dino_only / clipseg |
| `threshold` | float | 0.50 | soglia detection |
| `blur` | float | 7.0 | sfumatura maschera |
| `dilation` | int | 4 | dilatazione |
| `sam_model` | str | sam_hq_vit_l | modello SAM |

- **Output**: maschera PNG.
- **Esempio**:
```json
{"module":"segment","params":{"prompt":{"__pipeline__":"segment","image_file":"foto.png","text":"the dog","detector":"grounding_dino","threshold":0.5}}}
```

## 15. Segmentazione da regione (click/box)
- **Cosa fa**: selezione puntuale (click/rettangolo) → maschera, con SAM vanilla (rifiuta SAM-HQ).
- **GUI**: click sul canvas (selezione puntatore).
- **Backend**: `SegmentPipeline.build_region(image_file, region_mask_file, threshold=0.50)`.
- **Parametri**: `image_file`, `region_mask_file`, `sam_model`, `threshold` (0.5), `detection_hint` (center-1).
- **Output**: maschera PNG.

---

# RESTAURO / UPSCALE (immagini)

Modulo nativo: **`upscale`**. Pipeline `core/upscale_pipeline.py`, `restore_pipeline.py`. Backend `inference_server/backends/image_upscale.py`.

## 16. Upscale immagine (ESRGAN + refine opz.)
- **Cosa fa**: aumenta la risoluzione con modello ESRGAN, con eventuale rifinitura AI FLUX.2 img2img.
- **GUI**: **Immagine → Upscale** → *Upscale* (`request_upscale`).
- **Backend**: modulo `upscale`, `UpscalePipeline.build(...)`. Prefix `ltxstudio_upscale`.
- **Parametri**:

| nome | tipo | default | descrizione |
|---|---|---|---|
| `image_file` | str | — | immagine |
| `model_name` | str | (es. 4xNomos8kSCHAT-L) | modello ESRGAN da `E:\AI\Models\Upscale` |
| `target_scale` | float | 4.0 | fattore (2/3/4) |
| `refine` | bool | false | rifinitura AI |
| `refine_model` | str | — | refiner FLUX.2 |
| `refine_denoise` | float | 0.30 | denoise refine (0.05–0.70) |
| `refine_steps` | int | 12 | passi refine |
| `seed` | int | 0 | seme |

- **Output**: PNG ingrandito. **Note**: proporzioni preservate (fix).
- **Esempio**:
```json
{"module":"upscale","params":{"prompt":{"__pipeline__":"upscale","image_file":"foto.png","model_name":"4x-UltraSharpV2.safetensors","target_scale":4.0}}}
```

## 17. Restauro foto
- **Cosa fa**: restaura foto rovinata (super-res ESRGAN + denoise).
- **GUI**: tab **Restauro foto** → operazione *restore* (`request_restore`); anche pulsante *Restauro foto* nel tab Immagine.
- **Backend**: modulo `upscale`, `RestorePipeline.build_restore(...)`. Prefix `ltxstudio_restore`.
- **Parametri**: `image_name`, `upscale_model` (def `4xUltraSharp.pth`), `target_scale` (2.0), `denoise` (bool), `denoise_strength` (0.25), `seed`.
- **Output**: PNG restaurato.

## 18. Colorizza (B&N → colore)
- **Cosa fa**: colora una foto in bianco e nero via FLUX.2 img2img.
- **GUI**: **Restauro foto** → operazione *colorize*.
- **Backend**: `RestorePipeline.build_colorize(...)` (denoise 0.85, guidance 2.5). Prefix `ltxstudio_colorize`.
- **Parametri**: `image_name`, `flux2_model` (obbligatorio), `subject`, `style` (def "natural realistic colors"), `steps` (20), `seed`.
- **Output**: PNG a colori.

## 19. Restaura + colorizza
- **Cosa fa**: restauro ESRGAN + colorizzazione FLUX.2 in un passo.
- **GUI**: **Restauro foto** → *restore_and_colorize*.
- **Backend**: `RestorePipeline.build_restore_and_colorize(...)`. Prefix `ltxstudio_restored_colorized`.
- **Parametri**: unione di funz. 17+18.
- **Output**: PNG restaurato e colorato.

## 20. Face-swap su immagine
- **Cosa fa**: sostituisce il volto in una foto con quello di una foto sorgente (InsightFace inswapper).
- **GUI**: **Restauro foto** → operazione *faceswap* (`source_face`).
- **Backend**: `core/faceswap.py` → `swap_image(target, source_face, out)`. Modelli `E:\AI\Models\insightface` (antelopev2 + inswapper_128.onnx), onnxruntime-gpu.
- **Parametri**: `target_path`, `source_face_path`, `out_path`.
- **Output**: PNG con volto sostituito. **Note**: non usa VRAM diffusion (onnxruntime CUDA12).

---

# VIDEO

Server **LTX-2.3** `:8190` via `core/ltx2_service.generate(params)` — `params` è un **passthrough** a `workers/ltx2_infer.run_generation`. La modalità (`ptype`) è auto-selezionata in base ai parametri presenti. Modello path **ASSOLUTO** (es. `E:\AI\Models\diffusion_models\LTX\ltx-2.3-22b-dev-fp8.safetensors`).

**Parametri comuni video** (validi per tutte le modalità):

| nome | tipo | default | descrizione |
|---|---|---|---|
| `prompt` | str | `""` | prompt |
| `negative` | str | "worst quality, blurry" | negativo |
| `seed` | int | 42 | seme |
| `width`/`height` | int | 768/512 | risoluzione |
| `num_frames` | int | 121 | frame (deve essere 8n+1) |
| `fps` | float | 25.0 | frame rate |
| `steps` | int | 40 | passi |
| `guidance` | float | 3.0 | CFG |
| `stg` | float | 1.0 | spatio-temporal guidance |
| `model` | str | LTX-2.3 22B dev fp8 (path assoluto) | checkpoint |
| `quant` | str | auto (fp8_scaled_mm / fp8_cast) | quantizzazione |
| `offload` | str | `cpu` | offload |
| `loras` | list[{name,strength}] | `[]` | LoRA (rel. a `E:\AI\Models\LoRA`) |
| `enhance` | bool | false | prompt enhance nativo |
| `gguf_transformer` | str | auto (Q4_K_M) | transformer GGUF |
| `chunk_frames` | int | 0 | >0 = long-video in-process |
| `seconds` | float | 5.0 | usato se `num_frames` assente |

## 21. Testo → Video (t2v)
- **Cosa fa**: genera un video da solo testo (LTX-2.3 nativo, no ComfyUI). *(motore nativo recente)*.
- **GUI**: tab **Video** → prompt → *Genera* (`get_values`).
- **Backend**: `ltx2_service.generate(params)`, modalità `ti2v` (senza `image_path`).
- **Parametri**: comuni video. UI: preset risoluzione (9:16 704×1280 default, 16:9 1280×704, 1:1 768×768, …), `length_s` (def 8.0), sampler `euler_ancestral_cfg_pp`, scheduler `karras`/`LTXVScheduler`, steps suggeriti 32, cfg 1.0.
- **Output**: mp4.
- **Esempio**:
```json
{"params":{"prompt":"a drone shot over a forest at sunrise","width":1280,"height":704,"num_frames":121,"fps":25,"steps":32,"guidance":1.0,"seed":42}}
```
- **Note**: modelli LTX-2.3 22B distilled (STANDARD), dev (BASE), 10EROS (NSFW). GGUF fallback Q4_K_M.

## 22. Immagine → Video (i2v)
- **Cosa fa**: anima una foto (condiziona il primo frame).
- **GUI**: tab **Video** → carica immagine + prompt → *Genera*. `ref_strength` = forza riferimento (soft-reference).
- **Backend**: `ltx2_service.generate`, modalità `ti2v` con `image_path`. `i2v_method` = addguide|inplace.
- **Parametri**: comuni + `image_path` (str), `ref_strength` (0–1, def 0.50), `keyframes` (list[{path,frame,strength}]).
- **Output**: mp4.
- **Esempio**:
```json
{"params":{"prompt":"she turns and smiles","image_path":"E:/in/foto.png","ref_strength":0.5,"num_frames":121,"fps":25}}
```

## 23. Audio → Video (a2v / lip-sync nativo)
- **Cosa fa**: genera video sincronizzato su una traccia audio (parlato/canto). Base del lip-sync.
- **GUI**: tab **Video** (sorgente audio) / tab **Audio**.
- **Backend**: `ltx2_service.generate`, modalità `a2v` (trigger `audio_path`).
- **Parametri**: comuni (steps def 30, w/h 512×… ) + `audio_path` (str, reso stereo), `image_path` (opz), `ref_strength` (1.0), `audio_start_time` (0.0), `audio_max_duration` (float|None).
- **Output**: mp4 con audio.

## 24. Testo → Audio (t2a)
- **Cosa fa**: genera solo audio dal prompt (`T2AOneStagePipeline`).
- **GUI**: tab **Audio** → sorgente *native*.
- **Backend**: `ltx2_service.generate` con `t2a=True`. `guidance=7.0, steps=30, num_frames=seconds*fps`.
- **Parametri**: `prompt`, `negative`, `seed` (42), `seconds`.
- **Output**: file audio.

## 25. ControlNet — video da movimenti (canny/depth/pose)
- **Cosa fa**: genera un video guidato dai movimenti di un video di riferimento (IC-LoRA).
- **GUI**: tab **ControlNet** → abilita + video riferimento + tipo → *Genera*.
- **Backend**: `ltx2_service.generate`, modalità `iclora` (trigger `control_video`). IC-LoRA su disco per tipo.
- **Parametri**: comuni + `control_video` (str), `control_type` (canny/depth/pose), `control_strength` (0.8), `ref_strength` (1.0), `conditioning_attention` (1.0), `edit_lora` (opz). UI: `controlnet_type` 0=canny/1=depth/2=pose.
- **Output**: mp4.

## 26. Storyboard multi-scena
- **Cosa fa**: N immagini-keyframe a tempo → video continuo senza montaggio (catena `LTXVAddLatentGuide`).
- **GUI**: tab **Video** → pannello **Storyboard** (keyframe + durata per scena).
- **Backend**: `ltx2_service.generate` con `keyframes` (list[{path,frame,strength}]).
- **Parametri**: `keyframes`, + comuni.
- **Output**: mp4 continuo.

## 27. Prolunga video / Long video illimitato
- **Cosa fa**: estende un video a lunghezza (quasi) illimitata, catena autoregressiva memory-bounded (decode per-segmento + stitch overlap).
- **GUI**: tab **Prolunga video** → *Genera* (`request_extend`); auto-carica ultimo video.
- **Backend**: modulo nativo `video_long` / `core/extend_pipeline.py` (`ExtendPipeline.build`). Output `VHS_VideoCombine` prefix `ltxstudio_longvideo`.
- **Parametri**:

| nome | tipo | default | descrizione |
|---|---|---|---|
| `video_name` | str | — | video da estendere |
| `prompt` | str | — | continuazione |
| `add_seconds` | float | 20.0 (5.0 pipeline) | secondi da aggiungere |
| `seg_seconds` | float | 18.0 | lunghezza segmento |
| `overlap_frames` | int | 8 | sovrapposizione |
| `context_frames` | int | 16 | frame di contesto |
| `steps` | int | 8 | passi |
| `cfg` | float | 1.0 | guida |
| `nag` | bool | true | NAG |
| `seed` | int | 0 | seme |

- **Output**: mp4 lungo.

---

# EDITING VIDEO

Modulo nativo: **`video_edit`** / LTX-2 con edit-LoRA. Tab **Modifica Video AI** (`video_edit_ai_tab.py`, `request_edit`). Pannello **V2V** nel tab Video.

## 28. Edit-anything (V2V)
- **Cosa fa**: applica una modifica/stile a tutto il video (video→video). ✅ validato.
- **GUI**: **Modifica Video AI** → modo *edit*.
- **Backend**: `ltx2_service.generate` modalità `iclora`+`edit_lora` = `ltx23_edit_anything_global_rank128`.
- **Parametri**: `video`, `mode=edit`, `edit_lora`, `prompt`, `strength` (0.6/0.8/1.0, def 1.0), `guide` (0.1–1.0, def 0.4).
- **Output**: mp4.

## 29. Inpaint video (oggetto) + tracking SAM2
- **Cosa fa**: rimuove/sostituisce un oggetto nel video; la maschera lo segue (SAM2 tracking). Sperimentale.
- **GUI**: **Modifica Video AI** → modo *inpaint* + selezione parola oggetto.
- **Backend**: modalità `vinpaint` (richiede `inpaint_masks_dir` + `control_video` + `edit_lora` = `ltx23_inpaint_masked_t2v_rank128`).
- **Parametri**: `video`, `object`, `prompt`, `cfg_scale` (5.0/6.0), `num_steps` (30), `control_strength` (0.5), `mask_dilate` (0–64, def 24), `mask_blur` (0).
- **Output**: mp4. **Note**: bias contestuale LTX (es. auto→moto/cavallo a volte rifiutato).

## 30. anime2real / real2anime
- **Cosa fa**: conversione stile anime↔reale del video.
- **GUI**: **Modifica Video AI** → modi *anime2real* / *real2anime*.
- **Backend**: `edit_lora` = `ltx23_anime2real_rank64` / `ltx23_real2anime_rank64`.
- **Parametri**: come funz. 28.
- **Output**: mp4.

## 31. ReTake (rigenera sezione)
- **Cosa fa**: rigenera una porzione temporale del video (video e/o audio).
- **GUI**: tab **Video** → pannello **V2V** → modo *ReTake*.
- **Backend**: `ltx2_service.generate` modalità `retake` (richiede `retake`+`video_path`).
- **Parametri**: `video_path`, `start_time` (0.0), `end_time` (0.0), `steps` (40), `regenerate_video` (true), `regenerate_audio` (true).
- **Output**: mp4.

## 32. Just_Talk / MultiTalk — lip-sync
- **Cosa fa**: lip-sync su audio reale: foto/video + audio → bocca sincronizzata. (Just_Talk via `LTXVAudioVideoMask`; MultiTalk via worker Wan2.1+MultiTalk per video lunghi.)
- **GUI**: tab **Video** → V2V modo *Talk*; tab **Audio** (sorgente).
- **Backend**: LTX-2 `a2v` (funz. 23) per Just_Talk; worker MultiTalk (WanVideoWrapper-as-library) per lip-sync esteso.
- **Parametri**: `audio_path`, `image_path`/`video`, `ref_strength`.
- **Output**: mp4 con labiale sincronizzato.

---

# POST-PROCESS / QUALITÀ VIDEO

Moduli nativi: **`video_upscale`**, **`video_face`**, **`video_fps`**. Tab **Definizione video** (`video_def_tab.py`, `request_enhance`). Pipeline `core/video_enhance_pipeline.py`, `face_restore_pipeline.py`, `rife.py`. Servizi `seedvr2_service.py`, `flashvsr_service.py`.

## 33. SeedVR2 — super-risoluzione/restauro video
- **Cosa fa**: upscala/restaura video con SeedVR2 DiT (CLI standalone, no ComfyUI), block-swap dinamico + streaming chunk.
- **GUI**: **Definizione video** → engine *seedvr2* → *Genera* (`request_enhance`).
- **Backend**: modulo `video_upscale`; `core/video_enhance_pipeline.build(engine="seedvr2")` o `core/seedvr2_service.upscale_video(...)`. Pesi `E:\AI\Models\SeedVR2`. Output `VHS_VideoCombine` prefix `ltxstudio_enhanced` (audio preservato).
- **Parametri**:

| nome | tipo | default | descrizione |
|---|---|---|---|
| `video` | str | — | input |
| `engine` | str | `seedvr2` | seedvr2/esrgan/flashvsr |
| `seedvr2_model` | str | seedvr2_ema_7b_fp16 (svc: 3b_fp8) | modello DiT |
| `resolution` | int | 1080 | altezza target |
| `blocks_to_swap` | int | 0 | block-swap CPU |
| `batch_size` | int | 5 | deve essere 4n+1 |
| `seed` | int | 0 | seme |

- **Output**: mp4. **Note**: validato 512×320→1152×720 (~21min); VRAM auto (≥20GB no swap; <8GB 32 blocchi).

## 34. FlashVSR — super-risoluzione x4 veloce
- **Cosa fa**: VSR x4 in streaming (diffsynth), riaggancia l'audio originale.
- **GUI**: **Definizione video** → engine *flashvsr*.
- **Backend**: modulo `video_upscale`; `core/flashvsr_service.upscale_video(...)`. Pesi `E:\AI\Models\FlashVSR`.
- **Parametri**: `inp`, `out`, `scale` (4.0), `seed` (0), `sparse` (2.0), `local_range` (11).
- **Output**: mp4 con audio. **Note**: validato 512×320→2048×1280 (~22s).

## 35. ESRGAN per-frame (video)
- **Cosa fa**: upscale video frame-per-frame con modello ESRGAN.
- **GUI**: **Definizione video** → engine *esrgan*.
- **Backend**: modulo `video_upscale`; `VideoEnhancePipeline.build(engine="esrgan")`.
- **Parametri**: `esrgan_model` (def `4x-UltraSharpV2.safetensors`), `target_scale` (2.0).
- **Output**: mp4.

## 36. RIFE — interpolazione FPS / fluidifica
- **Cosa fa**: aumenta gli FPS interpolando frame intermedi (Practical-RIFE), audio rimuxato.
- **GUI**: **Definizione video** → checkbox RIFE + moltiplicatore; tab **Video** → *Fluidifica RIFE auto* (post-gen).
- **Backend**: modulo `video_fps`; `core/rife.interpolate_video(in, out, factor=2, target_fps=None)`. Pesi `Practical-RIFE`.
- **Parametri**: `factor` (2/3/4), `rife_ckpt` (rife47/48/46.pth), `target_fps` (opz).
- **Output**: mp4 a FPS maggiore (stessa durata).

## 37. Restauro volti su video
- **Cosa fa**: restaura i volti per-frame (CodeFormer/GFPGAN) preservando l'identità.
- **GUI**: **Definizione video** → *Restauro volti* (`request_face_restore`); tab **Video** → restauro volti auto.
- **Backend**: modulo `video_face`; `core/face_restore_pipeline.build(...)` (`facerestore_cf`). Prefix `ltxstudio_facerestore`.
- **Parametri**: `video`, `model` (codeformer-v0.1.0.pth / GFPGANv1.4 / v1.3), `detection` (retinaface_resnet50), `fidelity` (0–1, def 0.70).
- **Output**: mp4 (audio preservato).

## 38. Face-swap su video
- **Cosa fa**: sostituisce il volto in tutto il video (InsightFace).
- **GUI**: tab **Video** → face-swap auto (post-gen, `face` foto sorgente).
- **Backend**: `core/faceswap.swap_video(video, source_face, out)` (audio rimuxato).
- **Parametri**: `video_path`, `source_face_path`, `out_path`.
- **Output**: mp4.

## 39. MMAudio — sonorizza video (ambient/SFX/t2a)
- **Cosa fa**: genera audio sincronizzato per un video muto (o audio da testo).
- **GUI**: tab **Musica** → box *Sonorizza* → *Genera audio* (`_gen_sonify`).
- **Backend**: `core/music_service.make_sonify_job(SonifyParams)`. Tool `E:\AI\Tools\MMAudio`.
- **Parametri**: `video_path` (vuoto→t2a), `prompt`, `negative_prompt`, `duration_s` (8.0), `cfg_strength` (4.5), `num_steps` (25), `seed` (42).
- **Output**: file audio / video sonorizzato in `output/music/`.

---

# 3D

## 40. TRELLIS.2 — immagine → mesh 3D (PBR/GLB)
- **Cosa fa**: da una foto crea una mesh 3D con texture PBR.
- **GUI**: tab **Modello 3D** → *Genera* (`request_trellis`); visore 3D model-viewer.
- **Backend**: `core/trellis_pipeline.py` (grafo ComfyUI) o `core/trellis_service.make_job(...)` (subprocess venv `E:\AI\Tools\TRELLIS2`). Output `output/3d/<name>/model.glb`.
- **Parametri** (GUI): `resolution` (512/1024/1024_cascade[def]/1536_cascade), `file_format` (glb/obj/ply/stl), `remove_bg` (true), `detail`/max_tokens (16384–262144, def 49152), `guidance` (6.5), `steps` (12), `faces` (def 500000), `texture` (true), `tex_size` (2048), `seed`, `precision`/`attn`/`bg_color` (avanzate).
- **Output**: GLB (export FBX/OBJ via Blender). **Note**: pesi ~25GB, ~24GB VRAM consigliati; gated (HF token).

## 41. Rig 3D (SkinTokens)
- **Cosa fa**: da una mesh genera scheletro + skinning (auto-rigging).
- **GUI**: tab **Rig 3D** → *Genera* (`request_rig`).
- **Backend**: `core/rig_pipeline.py` (ComfyUI) o `core/rig_service.make_job(mesh)` (venv `E:\AI\Tools\SkinTokens`, Blender headless). Output `output/3d/<name>/rigged.glb`.
- **Parametri**: `skeleton` (Keep names/Mixamo/Unreal Engine 5), `file_format` (glb/fbx), `transfer` (true), `postprocess`, `gpv` (4), `smooth` (55°), + avanzate (num_beams 10, temperature 1.0, top_k 5, top_p 0.95, rep_pen 2.0).
- **Output**: GLB/FBX riggato. **Note**: ~4GB VRAM (Qwen3-0.6B backbone), richiede Blender 4.0+ (auto-install winget).

## 42. Mondo 3D (WorldGen)
- **Cosa fa**: da foto/testo crea una scena 3D (mesh GLB e/o Gaussian-splat .ply).
- **GUI**: tab **Mondo 3D** → *Genera scena* (`_gen`); player web three.js (orbit/WASD) + visore nativo; export FBX/OBJ.
- **Backend**: `core/world_service.make_job(WorldParams, mesh=True|False)` (venv `E:\AI\Tools\WorldGen`). Output `output/worlds/<name>/{mesh.glb,splat.ply}`.
- **Parametri**: `name` (world), `image_path` (vuoto→text-to-scene), `prompt`, `mode` (mesh/splat/both), `resolution` (1600), `low_vram`, `hf` token.
- **Output**: GLB/PLY. **Note**: usa FLUX.1-dev gated; Windows install non banale.

## 43. Mocap webcam → BVH
- **Cosa fa**: motion capture da webcam (MediaPipe) → animazione BVH.
- **GUI**: tab **Animazione** → sorgente *webcam* → start/stop (`request_mocap_start`/`stop`).
- **Backend**: `core/mocap_service.MocapService.start(out_bvh, preview, camera=0, fps=30, max_seconds=0)` (venv `mocap_env`).
- **Parametri**: `camera` (0–8), `fps` (30), `max_seconds`.
- **Output**: file BVH + frame preview.

## 44. Retarget animazione (Anim)
- **Cosa fa**: applica un'animazione (BVH/FBX) a un modello riggato.
- **GUI**: tab **Animazione** → *Applica* (`request_retarget`).
- **Backend**: Blender headless (`core/blender_service`).
- **Parametri**: `model` (.glb/.gltf/.fbx riggato), `anim` (bvh/fbx), `format` (fbx/glb/bvh), `template` (auto/Mixamo/Unreal Engine 5).
- **Output**: file animato.

## 45. Convertitore formati 3D
- **Cosa fa**: import/export universale di formati 3D via Blender headless.
- **GUI**: tab **Convertitore** → *Converti* (`request_convert`).
- **Backend**: `core/blender_service.BlenderWorker` + `scripts/blender/convert.py`.
- **Parametri**: `input`, `format` (glb/gltf/fbx/usd/usdz/obj/dae/stl/ply/abc/bvh/x3d/dxf), `options` (`scale` 1.0, `embed_textures` true, `animations` true).
- **Output**: file 3D convertito.

## 46. EXR/HDR export
- **Cosa fa**: salva frame HDR lineari in OpenEXR (per la pipeline HDR IC-LoRA).
- **GUI**: (interno alla pipeline HDR).
- **Backend**: `core/exr_export.save_exr_frames(frames, out_dir, prefix="hdr")`, `tonemap_reinhard(frame)`.
- **Parametri**: `frames` ([f,h,w,c] float), `out_dir`, `prefix`.
- **Output**: file .exr per frame.

---

# VOCE / TTS

Motore unificato `core/tts_engine.py` (libreria voci unica; output `output/tts/`). Engine clone in venv isolati. Tab **Audio** (`audio_tab.py`).

## 47. TTS Edge (default)
- **Cosa fa**: sintesi vocale da testo con Edge TTS (~400 voci, SSML), gratis.
- **GUI**: tab **Audio** → sezione TTS → engine *edge* → *Genera*.
- **Backend**: `tts_engine.synthesize(engine="edge", voice, text)` / `edge_synthesize(text, voice_id, rate, volume, pitch)`.
- **Parametri**: `voice_id`, `text`, `rate` (+0%), `volume` (+0%), `pitch` (+0Hz). Marker SSML `[pause:Xms]`, `<prosody>`, `<emotion>`.
- **Output**: MP3.

## 48. Voice clone — CosyVoice2
- **Cosa fa**: clonazione vocale zero-shot (campione + testo → voce clonata).
- **GUI**: tab **Audio** → engine *cosyvoice2* (voce con campione 🎤).
- **Backend**: `core/cosyvoice_service.make_job(text, prompt_wav, prompt_text, instruct)` (venv `E:\AI\Tools\CosyVoice`). Output `output/audio/<name>.wav`.
- **Parametri**: `text`, `prompt_wav`, `prompt_text` (transcript), `instruct` (opz).
- **Output**: WAV.

## 49. Voice clone — F5-TTS
- **Cosa fa**: clone vocale F5-TTS (SOTA EN, MIT).
- **GUI**: tab **Audio** → engine *f5tts*.
- **Backend**: `core/f5tts_service.make_job(text, prompt_wav, prompt_text)`.
- **Parametri**: `text`, `prompt_wav`, `prompt_text`.
- **Output**: WAV.

## 50. Voice clone — Fish-Speech / OpenAudio
- **Cosa fa**: clone vocale Fish (Apache-2.0).
- **GUI**: tab **Audio** → engine *fish*.
- **Backend**: `core/fish_service.make_job(text, prompt_wav, prompt_text)`.
- **Parametri**: `text`, `prompt_wav`, `prompt_text`.
- **Output**: WAV. **Note**: da validare al primo uso.

## 51. Voice clone — XTTS-v2 / Coqui
- **Cosa fa**: clone vocale multilingua (17 lingue, CC-BY-NC).
- **GUI**: tab **Audio** → engine *xtts2*.
- **Backend**: `core/xtts_service.make_job(text, prompt_wav, language="it")` (ignora prompt_text).
- **Parametri**: `text`, `prompt_wav`, `language`.
- **Output**: WAV.

## 52. VibeVoice (multi-speaker)
- **Cosa fa**: TTS multi-speaker per dialoghi lunghi (MIT, VibeVoice-1.5B).
- **GUI**: tab **Audio** → engine *vibevoice*.
- **Backend**: `core/vibevoice_service.make_job(text, prompt_wav)`.
- **Parametri**: `text`, `prompt_wav`.
- **Output**: WAV.

## 53. Handy-API — TTS / STT / clone (REST :8720)
- **Cosa fa**: servizio voce REST: TTS Kokoro, STT Whisper, voice-clone.
- **GUI**: usato internamente (STT videomusicale, voce Film Studio).
- **Backend**: `core/handy_api.py` → `tts(text, voice="if_sara", lang="it", speed=1.0)`, `transcribe(audio_path)`, `clone_voice(name, audio_path)`, `list_tts_models()`.
- **Parametri TTS**: `text`, `voice`, `lang`, `speed`. **STT**: `audio_path`.
- **Output**: WAV (TTS) / testo (STT).

---

# MUSICA

Tab **Musica** (`music_tab.py`). Pipeline `core/music_service.py`, `acestep_service.py`, `music_video_service.py`, `lyrics_llm.py`.

## 54. HeartMuLa — genera canzone/strumentale
- **Cosa fa**: genera una canzone (con voce) o brano strumentale da testo+tag.
- **GUI**: tab **Musica** → box Canzone → engine *HeartMuLa* → *Genera musica* (`_gen_song`).
- **Backend**: `core/music_service.make_song_job(SongParams)` (tool `E:\AI\Tools\heartlib`). Output `output/music/`.
- **Parametri** (`SongParams`): `name`, `lyrics`, `tags` (def "pop, upbeat, female vocals, synth, catchy"), `duration_s` (120), `instrumental` (false), `version` (3B/7B), `temperature` (1.0), `cfg_scale` (1.5), `topk` (50), `lazy_load` (false).
- **Output**: MP3.

## 55. ACE-Step — genera canzone (alternativa)
- **Cosa fa**: generazione musicale ACE-Step XL 1.5 (Apache-2.0).
- **GUI**: tab **Musica** → engine *ACE-Step*.
- **Backend**: `core/acestep_service.make_job(tags, lyrics, duration=60.0, steps=27, guidance=15.0)` (venv `E:\AI\Tools\ACE-Step`).
- **Parametri**: `tags`, `lyrics`, `duration`, `steps`, `guidance`.
- **Output**: WAV.

## 56. Genera testo canzone con AI
- **Cosa fa**: genera/adatta i testi della canzone via Gemma-3 (15 lingue).
- **GUI**: tab **Musica** → *✨ Genera testo con AI* (`_gen_lyrics_ai`).
- **Backend**: `core/lyrics_llm.py` (Gemma-3).
- **Parametri**: `lyr_prompt`, `lyr_lang`.
- **Output**: testo.

## 57. Videomusicale (song → storyboard → video)
- **Cosa fa**: da una canzone genera uno storyboard LLM coerente col testo → video nativo coerente, con cambi scena a ritmo.
- **GUI**: tab **Musica** → *🎬 Genera videomusicale* (`_gen_music_video`) → MusicVideoStudio.
- **Backend**: `core/music_video_service.py` → `generate_song_and_video(...)` / `make_music_video(...)`. Storyboard editabile (`plan_scenes`, `gen_scene_image`, `gen_scene_video`, `assemble_scenes`), allineamento Whisper, export NLE (fcpxml).
- **Parametri**: `lyrics`, `tags`, `image_path`, `width`/`height` (704×512), `scene_seconds` (6.0; pace slow~12/medium~7/fast~4/vfast~2.5), `video_style`, `fps` (25), `steps` (8).
- **Output**: mp4 (canzone + video). **Note**: LTX-2.3 GGUF per le clip; MAX_SCENES=40.

---

# FILM STUDIO

Modulo **🎬 AI Film Studio** (`ui/film_studio/`): DB SQLite + versioning, `llm_service` (Gemma/LM Studio), `gen_service` nativo, Continuity, Scheduler. Backend 100% nativo (cap. `filmstudio`).

## 58. Copione (Script)
- **Cosa fa**: da idea/tema/copione/interattivo genera un trattamento cinematografico via LLM.
- **GUI**: Film Studio → tab **Copione** → *✍️ Genera copione* (`_gen`), *↻ Rigenera*, *💾 Salva*.
- **Backend**: `core/film_studio` + `core/llm_service`.
- **Parametri**: `kind` (full/idea/theme/interactive), `styles` (~60), `genres` (~23), `inp` (input, @-reference), `text`.
- **Output**: copione (tabella `script`).

## 59. Asset Manager + Character/Asset Sheet
- **Cosa fa**: CRUD di asset (attore/prop/location/architettura), variazioni e guardaroba; genera "character sheet" (set di immagini di copertura via PuLID/Kontext).
- **GUI**: Film Studio → tab **Asset Manager** → *🎭 Genera Character Sheet* (`_gen_sheet`), *➕ Variazione*, *👗 Abbigliamento*.
- **Backend**: `core/lora_coverage` + `gen_service.gen_image` (PuLID/Kontext).
- **Parametri**: `type` (actor/prop/location/architecture), `name`, `desc` (@-ref), `tags`, `photo` (base).
- **Output**: asset + immagini sheet.

## 60. Scenografia / Storyboard breakdown
- **Cosa fa**: scompone scene→inquadrature (LLM JSON), genera keyframe e video per shot, con Continuity (ora/meteo/luce/mood) propagata.
- **GUI**: Film Studio → tab **Scenografia** → *🎬 Breakdown*, *🖼 Genera keyframe*, *🎥 Genera video*, *🎬 Genera film completo* (`_gen_all`), *📄 Shot-list*.
- **Backend**: `llm_service` + `gen_service` + `continuity` + `scheduler.enqueue("image"/"video", {...})`.
- **Parametri shot**: `desc` (@-ref), `shot_seed`, Continuity (`time` day/night/dawn/dusk/golden hour, `weather`, `light`, `mood`), `duration_s`, `face_ref`, `loras`, `keyframes`, `audio_path`.
- **Output**: keyframe + clip video per shot.

## 61. Montaggio / Master / Export NLE
- **Cosa fa**: timeline dai clip → master video; dialoghi, colonna sonora, ambient, SFX, doppiaggio; export verso NLE.
- **GUI**: Film Studio → tab **Montaggio** → *🔊 Genera dialoghi*, *🎵 Colonna sonora*, *🌧 Ambient*, *💥 SFX*, *🎞 Master*, *📦 Export NLE*, *🌐 Doppia*.
- **Backend**: `otio_export.render_master`/`export_nle`, `music_service`, `voice_studio`, `llm_service`, ffmpeg.
- **Parametri**: `aspect` (16:9/2.39:1/9:16/1:1), `fmt` (fcpxml/otio/aaf/edl), `dub_lang` (12 lingue), `dub_relip` (true).
- **Output**: master mp4 / progetto NLE.

## 62. Voce (Voice Studio)
- **Cosa fa**: crea voci da descrizione/parametri/clone e le lega agli attori.
- **GUI**: Film Studio → tab **Voce** → *🗣 da descrizione*, *🎚 da parametri*, *🎙 clone*, *🔗 Lega*.
- **Backend**: `core/film_studio/voice_studio` + `tts_engine`.
- **Parametri**: descrizione; oppure `gender`/`age`/`lang`/`rate`/`pitch`/`emotion`; oppure `sample` (clone); binding `actor`.
- **Output**: voce salvata + binding.

## 63. LoRA Studio (libreria film)
- **Cosa fa**: libreria LoRA condivisa con scoping (episodio/serie) e binding agli asset.
- **GUI**: Film Studio → tab **LoRA Studio** → *💾 Salva*.
- **Backend**: `core/film_studio.db` + file `output/loras/*.safetensors`.
- **Parametri**: `trigger`, `flavor` (image/video), `scope` (episode/series), `asset`.
- **Output**: LoRA registrata. **Note**: max 3 LoRA simultanee in composizione.

---

# LORA TRAINING

## 64. LoRA Video (LTX-2.3) — dataset auto + train
- **Cosa fa**: genera automaticamente un dataset di clip (copertura per tipo soggetto + situazioni custom + media reali; identità via PuLID keyframe→i2v; face-focus auto-crop) e allena un LoRA video con il trainer ufficiale LTX-2.
- **GUI**: tab **LoRA Video (LTX-2.3)** → *Genera dataset* / *Genera + Allena LoRA*; *Installa trainer*.
- **Backend**: `core/lora_video_service.py` → `make_lora(subjects, name, out_dir, rank, steps, clip_seconds, use_pulid, low_vram)`. Trainer `E:\AI\Tools\LTX-2\packages\ltx-trainer`.
- **Parametri**: soggetti (`type` person/place/object/concept, `name`=trigger, `photos`+angle, coverage, custom, real_media, `face_focus`), `rank` (16), `steps` (2000), `clip_seconds` (3.0), `use_pulid` (true), `low_vram` (true).
- **Output**: `.safetensors` LoRA video.

## 65. LoRA Foto (FLUX.2) — dataset auto + train
- **Cosa fa**: come funz. 64 ma genera immagini (volti via FLUX.1+PuLID, oggetti/luoghi via FLUX.1 Kontext) e allena LoRA FLUX.2 via ai-toolkit.
- **GUI**: tab **LoRA Foto (FLUX.2)** → *Genera + Allena LoRA FLUX.2*; *Installa ai-toolkit*.
- **Backend**: `core/lora_image_service.make_lora(subjects, name, out_dir, base="flux2-dev", rank, steps, width=1024, height=1024)`.
- **Parametri**: soggetti; `base` (flux2-dev/klein-4b/klein-9b/flux1-dev), `rank` (16), `steps` (2000), `img_res` (1024).
- **Output**: `.safetensors` LoRA immagine.

## 66. Allena LoRA (ai-toolkit, FLUX.1/2)
- **Cosa fa**: training LoRA completo da un dataset manuale (FLUX.1/FLUX.2) via ostris ai-toolkit.
- **GUI**: tab **Allena LoRA** → *Avvia training* (`ts.TrainingRun`); *Installa modulo training*.
- **Backend**: `core/training_service.py` (venv `E:\AI\Tools\ai-toolkit`). Output → `E:\AI\Models\LoRA\Flux`.
- **Parametri** (`TrainParams`): `name`, `dataset_path`, `base_model` (flux1-dev/flux2-klein-4b/flux2-klein-9b/flux2-dev), `model_path`, `rank` (16), `alpha` (16), `lr` (1e-4), `steps` (2000), `batch_size` (1), `save_every`/`sample_every` (250), `resolutions` ([512,768,1024]), `quantize` (true), `low_vram`, `train_text_encoder`, `trigger_word`, `seed` (42).
- **Output**: LoRA `.safetensors`. **Note**: optimizer adamw8bit, flowmatch, bf16.

## 67. Caption dataset (Florence-2)
- **Cosa fa**: genera automaticamente le didascalie del dataset (Florence-2).
- **GUI**: tab **Allena LoRA** → *Genera caption (Florence-2)*.
- **Backend**: `scripts/caption_dataset.py`.
- **Parametri**: `dataset_path`, `trigger`, `overwrite` (bool).
- **Output**: file `.txt` di caption.

---

# IMPORT / MODELLI / DETECTION

## 68. Scarica pesi modelli (Weights)
- **Cosa fa**: wizard di download pesi da HuggingFace (local-first), per modulo.
- **GUI**: tab **Pesi modelli** → *Cerca esistenti*, *Seleziona mancanti*, *Scarica selezionati*.
- **Backend**: `core/weights_catalog.py` + `scripts/download_weights.py`; catalogo `data/weights_catalog.json` (15 voci: flux1-dev, flux1-fill, flux1-kontext, flux2-dev, mistral-flux2-te, zimage-turbo, ernie-image, ltx23, da2, mmaudio, heartmula-gen/3b, heartcodec, qwen3-8b/4b-klein-te).
- **Parametri**: `base` dir, `search` dir, `hf` token, `ids` (selezione).
- **Output**: pesi su disco. **Note**: gated (flux*, ltx23, mistral-te) richiedono HF token solo per download.

## 69. Import Models / catalog tooling
- **Cosa fa**: classifica e sposta file modello, arricchisce un catalogo HTML (Civitai/HF), gestisce path configurabili e patch compat ComfyUI.
- **GUI**: tab **Import Models** → *Analizza*, *Sposta*, *Scansiona*, *Arricchisci*, *Genera HTML*, *Ingest completo*; conversioni Gemma/TE.
- **Backend**: `core/catalog_paths.py`, `core/catalog_tooling.CatalogTooling`, `core/comfy_compat.py`.
- **Parametri**: path (models/workflow/dictionary/inbox/tools/hashes), source dir.
- **Output**: catalogo aggiornato, file riposizionati.

## 70. Auto-detection architettura modelli
- **Cosa fa**: rileva automaticamente l'architettura di ogni file modello dall'header (chiavi+forme+dtype+modelspec) e la mette nei dropdown (Modello/Encoder/VAE/CLIP) auto-popolati e auto-selezionati. *(funzione recente: detection dropdown + componenti auto)*.
- **GUI**: trasversale (tutti i combo modello). Persistenza precisione (fix: FP8 non nasconde più i bf16).
- **Backend**: `core/model_detect.py`, cache `data/model_detect.json` (campi `family/variant/role/kind/precision/n_double/n_single/_sig`). Famiglie: ltxv, wan21/wan22, hunyuan_video, dreamo, ernie, flux/flux2, lumina2…
- **Parametri**: path file modello.
- **Output**: dict descrittore arch (cache su file).

## 71. Conversione Gemma / Text-Encoder → HF (de-Comfy)
- **Cosa fa**: converte pesi ComfyUI (Gemma-3 → bf16, text encoder → formato HF) per il backend nativo.
- **GUI**: tab **Import Models** → *Gemma-3 → bf16*, *Text encoder → HF*.
- **Backend**: `scripts/convert_gemma_comfy.py`, `scripts/convert_te_to_hf.py` (in `inference_server_venv`).
- **Parametri**: file sorgente.
- **Output**: pesi convertiti per diffusers nativo.

## 72. ComfyUI compat shim (patch)
- **Cosa fa**: patch opzionale che porta LTX-2.3 / sblocco prompt-enhancer su ComfyUI (per chi usa ancora ComfyUI). Nella distribuzione il backend è nativo.
- **GUI**: tab **Import Models** → *Installa patch* / *Disinstalla patch*.
- **Backend**: `core/comfy_compat.py`.
- **Parametri**: `prompt_unlock` (bool).
- **Output**: patch installata/rimossa.

---

# Tabella riassuntiva (funzione → modulo/servizio → arch/task → GUI)

| # | Funzione | Modulo/Servizio | Arch / Task | GUI Tab |
|---|---|---|---|---|
| 1 | Genera FLUX | `image` / gen_pipeline | flux | Immagine→Genera |
| 2 | Genera FLUX.2 Dev | `image` / gen_pipeline | flux2 | Immagine→Genera |
| 3 | Genera Z-Image | `image` / gen_pipeline | zimage (lumina2) | Immagine→Genera |
| 4 | Genera Ernie | `image` / gen_pipeline | ernie | Immagine→Genera |
| 5 | PuLID volto | pulid_adapter | flux+pulid | Immagine→Genera |
| 6 | Inpaint | `image` / inpaint_pipeline | inpaint (FLUX Fill) | Immagine→Inpaint |
| 7 | Outpaint | `image` / inpaint_pipeline | inpaint(outpaint) | Immagine→Inpaint |
| 8 | FLUX.2 Klein inpaint | `image` / inpaint_pipeline | flux2_inpaint (Qwen3) | Immagine→Inpaint |
| 9 | Integra | `image` / integrate_pipeline | flux/flux2 img2img | Immagine→Integra |
| 10 | Kontext edit/recolor | `image` / kontext_pipeline | kontext / flux2_kontext | Immagine→Integra/Colore |
| 11 | Componi multi-ref | `image` / kontext_pipeline | multiref/stitch | Immagine→Componi |
| 12 | Recolor | `image` / kontext_pipeline | kontext | Immagine→Colore |
| 13 | Prompt enhancer | prompt_enhance | gemma | (trasversale) |
| 14 | Segment testo | `segment` / segment_pipeline | grounding_dino+sam | Immagine→Selezione AI |
| 15 | Segment regione | `segment` / segment_pipeline | sam region | Immagine (click) |
| 16 | Upscale immagine | `upscale` / upscale_pipeline | esrgan(+refine) | Immagine→Upscale |
| 17 | Restauro foto | `upscale` / restore_pipeline | restore | Restauro foto |
| 18 | Colorizza | `upscale` / restore_pipeline | colorize (flux2) | Restauro foto |
| 19 | Restaura+colorizza | `upscale` / restore_pipeline | restore+colorize | Restauro foto |
| 20 | Face-swap immagine | faceswap | insightface | Restauro foto |
| 21 | t2v | ltx2_service (:8190) | ti2v | Video |
| 22 | i2v | ltx2_service | ti2v+image | Video |
| 23 | a2v (audio→video) | ltx2_service | a2v | Video/Audio |
| 24 | t2a (testo→audio) | ltx2_service | t2a | Audio |
| 25 | ControlNet video | ltx2_service | iclora (canny/depth/pose) | ControlNet |
| 26 | Storyboard multi-scena | ltx2_service | keyframes | Video→Storyboard |
| 27 | Prolunga video | `video_long` / extend_pipeline | autoregressive | Prolunga video |
| 28 | Edit-anything | `video_edit` / ltx2 | iclora+edit_lora | Modifica Video AI |
| 29 | Inpaint video | ltx2 | vinpaint+SAM2 | Modifica Video AI |
| 30 | anime↔real | ltx2 | edit_lora | Modifica Video AI |
| 31 | ReTake | ltx2 | retake | Video→V2V |
| 32 | Lip-sync (Just_Talk/MultiTalk) | ltx2 a2v / MultiTalk worker | a2v | Video→V2V/Audio |
| 33 | SeedVR2 upscale | `video_upscale` / seedvr2_service | seedvr2 | Definizione video |
| 34 | FlashVSR upscale | `video_upscale` / flashvsr_service | flashvsr x4 | Definizione video |
| 35 | ESRGAN video | `video_upscale` / video_enhance | esrgan | Definizione video |
| 36 | RIFE FPS | `video_fps` / rife | rife | Definizione video/Video |
| 37 | Restauro volti video | `video_face` / face_restore_pipeline | codeformer/gfpgan | Definizione video |
| 38 | Face-swap video | faceswap | insightface | Video |
| 39 | MMAudio sonorizza | music_service | mmaudio | Musica |
| 40 | TRELLIS mesh | trellis_service/pipeline | trellis2 | Modello 3D |
| 41 | Rig 3D | rig_service/pipeline | skintokens | Rig 3D |
| 42 | Mondo 3D | world_service | worldgen | Mondo 3D |
| 43 | Mocap | mocap_service | mediapipe→bvh | Animazione |
| 44 | Retarget | blender_service | retarget | Animazione |
| 45 | Convert 3D | blender_service | convert | Convertitore |
| 46 | EXR/HDR export | exr_export | openexr | (interno) |
| 47 | TTS Edge | tts_engine | edge | Audio |
| 48 | Clone CosyVoice2 | cosyvoice_service | cosyvoice2 | Audio |
| 49 | Clone F5-TTS | f5tts_service | f5tts | Audio |
| 50 | Clone Fish | fish_service | fish | Audio |
| 51 | Clone XTTS-v2 | xtts_service | xtts2 | Audio |
| 52 | VibeVoice | vibevoice_service | vibevoice | Audio |
| 53 | Handy-API voce | handy_api (:8720) | kokoro/whisper/clone | (interno) |
| 54 | HeartMuLa song | music_service | heartmula | Musica |
| 55 | ACE-Step song | acestep_service | acestep | Musica |
| 56 | Testo canzone AI | lyrics_llm | gemma | Musica |
| 57 | Videomusicale | music_video_service | song→storyboard→ltx2 | Musica |
| 58 | Copione | llm_service/film_studio | script | Film Studio→Copione |
| 59 | Asset/Character Sheet | gen_service/lora_coverage | pulid/kontext | Film Studio→Asset |
| 60 | Storyboard breakdown | llm_service/scheduler | image/video enqueue | Film Studio→Scenografia |
| 61 | Montaggio/Master/NLE | otio_export/music/voice | render/export | Film Studio→Montaggio |
| 62 | Voice Studio | voice_studio/tts_engine | voce | Film Studio→Voce |
| 63 | LoRA Studio | film_studio.db | lora scope | Film Studio→LoRA |
| 64 | LoRA Video train | lora_video_service | ltx-trainer | LoRA Video |
| 65 | LoRA Foto train | lora_image_service | ai-toolkit flux2 | LoRA Foto |
| 66 | LoRA train (ai-toolkit) | training_service | flux1/flux2 | Allena LoRA |
| 67 | Caption Florence-2 | scripts/caption_dataset | florence2 | Allena LoRA |
| 68 | Scarica pesi | weights_catalog | download | Pesi modelli |
| 69 | Import Models | catalog_tooling | place/scan/enrich | Import Models |
| 70 | Auto-detection arch | model_detect | header detect | (trasversale) |
| 71 | Conversione Gemma/TE | scripts convert | de-comfy | Import Models |
| 72 | ComfyUI compat shim | comfy_compat | patch | Import Models |

---

## Note per agenti AI (skill)

- **Punto di partenza → funzione**:
  - solo testo + voglio immagine → funz. 1–4 (scegli arch: flux generale, flux2 qualità/edit, zimage veloce, ernie alt).
  - una foto + voglio modificarla → inpaint (6), edit per riferimento (10), integra (9), recolor (12), componi più foto (11).
  - foto del volto da imporre → PuLID (5) o face-swap (20).
  - foto/testo + voglio video → i2v (22) / t2v (21).
  - audio + foto → lip-sync (32) / a2v (23).
  - video di movimenti → ControlNet (25).
  - video da migliorare → SeedVR2 (33)/FlashVSR (34)/RIFE (36)/volti (37).
  - video troppo corto → prolunga (27).
- **NSFW**: usare modelli/encoder abliterated + `nsfw=true` (re-filtra modelli e LoRA); FLUX.2 Klein (8) e LTX 10EROS per contenuti espliciti.
- **VRAM**: tarata dinamicamente (block-swap auto); FLUX.2 t2i ~4GB; SeedVR2 7B/3B; TRELLIS ~24GB; Rig ~4GB.
- **Path**: gli engine nativi riusano i path locali del server (no upload). Modello video = path ASSOLUTO.

---

**Riepilogo**: documentate **72 funzioni** in **12 sezioni** (IMMAGINI, SEGMENTAZIONE, RESTAURO/UPSCALE, VIDEO, EDITING VIDEO, POST-PROCESS/QUALITÀ VIDEO, 3D, VOCE/TTS, MUSICA, FILM STUDIO, LORA TRAINING, IMPORT/MODELLI/DETECTION) + tabella riassuntiva finale e note skill per agenti AI.
