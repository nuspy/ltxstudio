# Images

*Generate images from text, edit them with inpaint/outpaint, recolor and upscale, cut objects out with AI selection, and impose real faces — all around a single shared canvas.*

The **Images** tab (`tab.images`) is the heart of still-image creation in AI Creative Center. It is organised as four sub-tabs down the left — **Generate image**, **Inpaint / Outpaint**, **Color**, and **Definition** (upscale) — plus an always-visible **mask canvas** on the right that every operation draws from. Below the canvas sits the **AI selection** panel (SAM segmentation), and the Generate sub-tab adds the **Face adapter (PuLID)** and **Compose multiple photos** panels.

## How image generation works

An image model turns your text **prompt** into pixels through a number of **diffusion steps**. You choose an **architecture** (FLUX, Z-Image or Ernie), a specific **model** file, and optionally a **text encoder** and **VAE** — or leave those on **auto** and let the app pick the matching pair. Two extra dials shape the result: **CFG** (how strictly the image obeys the prompt) and the **seed** (the random starting point; fix it to reproduce a result exactly).

The editing sub-tabs work differently from generation:

- **Inpaint / Outpaint** use a **mask** (drawn on the canvas) to mark the area to regenerate or the borders to extend, using a Fill-type model.
- **Integrate** and **Compose** are *edit-by-reference* operations (Kontext / FLUX.2): the extra images are handed to the model as *references*, not pasted, and the model harmonizes the scene. **No mask is needed** for these — only for Inpaint/Outpaint.
- **Recolor** changes the colour of a described subject from the prompt alone.

All model weights are downloaded once from Hugging Face; nothing runs in the cloud. See [Import models](11-import-models.md) for managing them.

---

### Generate an image
> **Edition:** **Demo+** · **Capability:** `cap.gen_images`
> *(Demo is capped at 768 px and 20 generations; Supporter at 1024 px and 200; Full and above are unlimited. Demo/Supporter outputs carry the "AI Tools Forge" watermark; removed from Full up.)*

**What it does** — Creates a brand-new image from a text prompt using the model and architecture you select.
**Where to find it** — **Images → Generate image** (`f1.subtab.gen`).
**Step by step**
1. Set **Content** to *Standard* or *18+* — this filters the model and LoRA lists. (The adult option requires **Unlocked (18+) mode**; see edition note below.)
2. Choose an **Architecture** (`f1.gen.arch_lbl`): FLUX, Z-Image or Ernie.
3. Pick an **Engine** (`f1.gen.engine_lbl`): *Comfy (fast, dynamic VRAM)* (recommended) or *Diffusers*.
4. Choose a **Model** (`f1.model_lbl`). Use the precision filter beside it to narrow by fp16/fp8/fp4/gguf; colours flag VRAM feasibility. Leave *auto* to let the app choose.
5. Leave **Text encoder**, **CLIP type** and **VAE** on *auto* unless you have a reason to override.
6. Type your **Prompt img** (`f1.gen.prompt_lbl`) and optionally a **Negative** prompt.
7. Set **Steps**, **CFG** and the **Seed** (untick **🎲 random** to fix it).
8. Click **🖼️ Generate image** (`f1.gen.btn`).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Content** (`f1.gen.content_lbl`) | Dropdown | Standard | Standard / 18+ | Filters models & LoRAs. The 18+ option requires the Unlocked (18+) mode. |
| **Architecture** (`f1.gen.arch_lbl`) | Dropdown | FLUX | FLUX / Z-Image / Ernie | Determines compatible models, encoders and LoRAs. |
| **Engine** (`f1.gen.engine_lbl`) | Dropdown | Comfy | Comfy / Diffusers | Comfy scales VRAM dynamically (recommended); Diffusers is the classic pipeline. |
| **Model** (`f1.model_lbl`) | Dropdown | First available (or ⚡ Auto) | Installed models | The generation checkpoint. Colours = VRAM feasibility. |
| **precision filter** | Dropdown | All | All / FP16 / FP8 / FP4 / GGUF | Narrows the model list by weight precision. |
| **Text encoder** (`f1.gen.encoder_lbl`) | Dropdown | auto | auto + compatible encoders | Interprets the prompt; auto matches the model. |
| **CLIP type** (`f1.gen.clip_lbl`) | Dropdown | auto | auto + types | CLIP variant; auto is safe. |
| **VAE** (`f1.gen.vae_lbl`) | Dropdown | auto | auto + VAEs | Decodes the latent; auto uses the model's own. |
| **Sampler** (`f1.sampler_lbl`) | Dropdown | auto | auto / euler / euler_ancestral_cfg_pp / dpmpp_2m / res_2s | Sampling algorithm; euler/dpmpp are reliable. |
| **Scheduler** (`f1.scheduler_lbl`) | Dropdown | auto | auto / karras / bong_tangent / beta / simple | Strongly affects image look; bong_tangent/kl_optimal often best on FLUX. |
| **🪄 Enhance prompt (LLM Qwen3)** (`f1.gen.enhance`) | Checkbox | Off | On / Off | A local LLM expands short prompts into detailed descriptions before generating. |
| **Steps** (`f1.steps_lbl`) | Spinbox | 20 | 1–80 | More steps = more detail, slower. |
| **CFG** (`f1.cfg_lbl`) | Spinbox | 3.5 | 0.0–20.0, step 0.1 | Prompt adherence; too high causes artifacts. |
| **Seed** (`f1.seed_lbl`) | Spinbox + checkbox | 0, random on | 0–2,147,483,647 | Fix the seed for reproducible results. |
| **Prompt img** (`f1.gen.prompt_lbl`) | Text | (empty) | free text | Image-specific prompt; empty falls back to the global prompt. |
| **Negative** (`f1.neg_lbl`) | Text | *pc game, cartoon, ugly, deformed, low quality, blurry, watermark, text…* | free text | Elements to avoid. |

**Specializations** — **Classic vs. Pro:** the Pro interface (`ImportModelsTabPro`, and the Pro image layout) reorganizes these controls for denser workflows but exposes the same parameters; any UI change is mirrored across both variants. A **🧱 Prompt Builder** button next to the prompt opens a structured field-to-prompt helper.
**Tips for best results**
> - Leave encoder/VAE on **auto** — the app pairs them to the architecture automatically.
> - Keep **CFG** near the model's recommended value (FLUX ≈ 3.5); raise steps before raising CFG for more detail.
> - Turn on **Enhance prompt** for short prompts; it fills in visual detail (first use downloads Qwen3-4B, ~3 GB).

---

### Face adapter (PuLID)
> **Edition:** **Demo+** (with image generation) · **Capability:** `cap.gen_images`

**What it does** — Injects the identity of a face from a photo into the generated image, so the result looks like that person (FLUX architecture only).
**Where to find it** — **Images → Generate image**, the **Face adapter (PuLID)** group (`f1.pulid.group`).
**Step by step**
1. Tick **Enable PuLID (face from photo)** (`f1.pulid.enable`).
2. In **Face photo** (`f1.pulid.face_lbl`), click **Browse…** and choose a clear face image (you can pass several paths separated by `|`).
3. Set **Strength** (0.6–0.9 recommended), and optionally **Start step** and **True CFG**.
4. Generate as usual.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Enable PuLID** (`f1.pulid.enable`) | Checkbox | Off | On / Off | Turns on face-identity injection (FLUX only). |
| **Face photo** (`f1.pulid.face_lbl`) | File picker | (empty) | image file(s) | The identity source. |
| **Strength** (`f1.pulid.strength_lbl`) | Spinbox | 0.8 | 0.0–3.0, step 0.05 | How strongly the face identity is applied (0.6–0.9 recommended). |
| **Start step** (`f1.pulid.start_lbl`) | Spinbox | 0 | 0–10 | When to start injecting identity. 1–4 lets pose/composition form first → more faithful pose. |
| **True CFG** (`f1.pulid.truecfg_lbl`) | Spinbox | 1.0 | 1.0–10.0, step 0.5 | 1.0 = off/fast; >1 improves prompt adherence and reduces identity dominance (slower). |

**Specializations** — Only available with the **FLUX** architecture; the panel disables for Z-Image/Ernie (`f1.pulid.only_flux`). REST: `POST /image/generate` with `pulid_face` and `pulid_strength` (default 0.8). See [REST API](15-rest-api.md).
**Tips for best results**
> - Use a sharp, front-facing, well-lit face photo; several photos of the same person improve robustness.
> - If the pose looks locked, raise **Start step** to 2–4 so the prompt sets the composition first.

---

### Inpaint & Outpaint
> **Edition:** **Full+** · **Capability:** `cap.inpaint`

**What it does** — Regenerates a masked area of an image (**inpaint**) or extends the image beyond its borders (**outpaint**) using a Fill-type model, keeping the rest intact.
**Where to find it** — **Images → Inpaint / Outpaint** (`f1.subtab.inpaint`). Draw the mask on the canvas first.
**Step by step**
1. Load your image onto the canvas and use the **Brush** tool to paint the area to change (for outpaint, set a larger canvas aspect so there is empty border).
2. Open the **Inpaint / Outpaint** sub-tab.
3. Choose an **Inpaint model** (`f1.inp.model_lbl`) — e.g. *FLUX_DEV_Fill*.
4. Type the **Inpaint prompt** (what to draw in the mask) or the **Outpaint prompt** (what to generate at the edges).
5. Set **Strength (retention)**, **Guidance (prompt adherence)** and **Edge feather** as needed.
6. Click **🎨 Generate inpaint** (`f1.inpaint.btn`) or **⬜ Generate outpaint** (`f1.outpaint.btn`).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Inpaint prompt** (`f1.inpaint.prompt_lbl`) | Text | (empty) | free text | What to draw in the masked area. |
| **Outpaint prompt** (`f1.outpaint.prompt_lbl`) | Text | (empty) | free text | What to generate beyond the edges. |
| **Negative** (`f1.neg_lbl`) | Text | default negative | free text | What to avoid in the regenerated area. |
| **Strength (retention)** (`f1.inpaint.strength_lbl`) | Slider | 0.75 | 0.00–1.00 | How much of the ORIGINAL is preserved. Internally `denoise = 1 − 0.4×strength` (min 0.5). |
| **Guidance (prompt adherence)** (`f1.inpaint.guidance_lbl`) | Spinbox | 30.0 | 1.0–30.0, step 0.5 | How strongly the edit follows the prompt. Auto-adjusts per model (Fill dev ≈30, distilled ≈2.5, FLUX.2 Klein ≈4). |
| **Edge feather (px)** (`f1.inpaint.feather_lbl`) | Spinbox | 0 = Auto | 0–16 | Blend between edit and original. Auto = proportional to the photo. |
| **Inpaint model** (`f1.inp.model_lbl`) | Dropdown | FLUX_DEV_Fill | Installed Fill/compatible models | The editing checkpoint. |
| **LoRA** (`f1.inpaint.lora_lbl`) | Dropdown | (none) | Installed LoRAs | Optional style/subject LoRA during the edit. |
| **Steps** (`f1.steps_lbl`) | Spinbox | 20 | 1–80 | Diffusion steps (auto-set from model). |
| **Seed** (`f1.seed_lbl`) | Spinbox + checkbox | 0, random on | 0–2,147,483,647 | Fix for reproducibility. |
| **✨ Enhance prompt (AI)** (`f1.edit.enhance`) | Checkbox | Off | On / Off | A local LLM enriches the edit prompt. Invisible, optional. |

**Specializations** — REST: `POST /image/inpaint` / `POST /image/outpaint` with `arch` (`inpaint` | `flux2_inpaint`), `steps` (20), `guidance` (30.0). **Retention** vs **Guidance** are distinct: retention controls how much original survives; guidance controls prompt obedience.
**Tips for best results**
> - Paint a mask slightly larger than the object and use a small **Edge feather** for seamless blends.
> - For subtle edits, raise **Strength (retention)** toward 1.0; to fully replace content, lower it.

---

### Integrate images (edit-by-reference)
> **Edition:** **Full+** · **Capability:** `cap.inpaint`

**What it does** — Blends extra images pasted on the canvas into the scene coherently (light, shadow, perspective) using Kontext (FLUX.1) or FLUX.2 — no mask required.
**Where to find it** — **Images → Inpaint / Outpaint**, the **🧩 Integrate images** group (`f1.integrate.group`).
**Step by step**
1. Paste the extra image(s) onto the canvas as layers.
2. In **Integrate engine**, choose *Kontext (FLUX.1)* (recommended) or *FLUX.2 (experimental)*.
3. Pick the matching **model**.
4. Optionally describe how to blend them in **Integrate prompt**.
5. Set **Guidance** and click **🧩 Integrate images** (`f1.integrate.btn`).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Integrate engine** (`f1.integrate.engine_lbl`) | Dropdown | Kontext (FLUX.1) | Kontext / FLUX.2 | Editing engine. Kontext is recommended. |
| **Integrate model** | Dropdown | First available | Kontext or FLUX.2 models | The editing checkpoint for the chosen engine. |
| **Integrate prompt** (`f1.integrate.prompt_lbl`) | Text | (empty) | free text | How to blend the images; empty = automatic. |
| **Guidance** (`f1.integrate.guidance_lbl`) | Slider | 2.5 | 1.0–10.0 | Edit strength (FluxGuidance). Kontext ≈2.5, FLUX.2 ≈4.0. |
| **use puzzle_lora** (`f1.integrate.puzzle`) | Checkbox | Off | On / Off | Experimental FLUX.1-only LoRA; leave off (tends to add a frame). |
| **Steps** (`f1.steps_lbl`) | Spinbox | 20 | 1–80 | Diffusion steps. |
| **Seed** | Spinbox + checkbox | 0, random on | 0–2,147,483,647 | Fix for reproducibility. |

**Tips for best results**
> - Kontext harmonizes without any mask — describe the target scene, not the cut-out.
> - Keep **puzzle_lora** off; it is experimental and usually degrades the blend.

---

### Compose multiple photos
> **Edition:** **Full+** · **Capability:** `cap.inpaint`

**What it does** — Combines 2+ photos into one image *without* pasting them, so hidden parts aren't lost — using multiple references or a side-by-side stitch.
**Where to find it** — **Images → Generate image** (and Inpaint tab), the **🧩 Compose multiple photos** group (`f1.compose.group`).
**Step by step**
1. Click **➕ Add photo** (`f1.compose.add`) to add each source (order matters — the 1st is the base).
2. Optionally select a photo and click **🎭 Mask** to load a PNG mask marking which part to use.
3. Choose a **Method**: *Multiple references (recommended)* or *Side-by-side (stitch)*.
4. Describe the final scene in the prompt.
5. Click **🧩 Compose** (`f1.compose.btn`).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Photos list** | List | (empty) | image files | Sources to combine; the 1st is the base. |
| **➕ Add photo / ➖ Remove / 🎭 Mask** | Buttons | — | — | Manage the source list and optional per-photo masks. |
| **Method** (`f1.compose.method_lbl`) | Dropdown | Multiple references | Multiple references / Side-by-side (stitch) | References = the model blends each photo naturally; stitch = photos placed next to each other as one reference. |
| **Compose prompt** | Text | (empty) | free text | Describe the final composition. |

**Specializations** — Works with FLUX.1 and FLUX.2. Masks are optional and only indicate *which part* of a photo to use.
**Tips for best results**
> - Use **Multiple references** for natural blends (e.g. *"the person from photo 1 in the landscape of photo 2, same lighting"*).
> - Use **Side-by-side** only when you want the photos literally adjacent.

---

### Recolor / restyle
> **Edition:** **Full+** · **Capability:** `cap.inpaint`

**What it does** — Changes the colour or style of a described subject from the prompt alone, leaving the rest of the image unchanged.
**Where to find it** — **Images → Color** (`f1.subtab.color`).
**Step by step**
1. Load the image on the canvas.
2. Enter the target **Color** (name or hex, or use **🎨 choose**).
3. Name the **Subject** to recolour (e.g. *"the shirt"*), or use AI selection / a mask.
4. Choose an **Engine** (Kontext or FLUX.2) and set **Steps**.
5. Click **🎨 Recolor / Restyle** (`f1.color.btn`).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Color** (`f1.color.color_lbl`) | Text + picker | (empty) | name / hex | Desired colour, e.g. *navy blue*, `#cc0000`. |
| **Subject** (`f1.color.subject_lbl`) | Text | (empty) | free text | What to recolour. |
| **Engine** (`f1.engine_lbl`) | Dropdown | Kontext | Kontext / FLUX.2 | Recolor engine. |
| **Steps** (`f1.steps_lbl`) | Spinbox | 20 | 1–80 | Diffusion steps. |

**Specializations** — REST: `POST /image/recolor` with `target`, `subject`, `arch` (default `kontext`), `guidance` (2.5). See [REST API](15-rest-api.md).

---

### AI upscale (increase definition)
> **Edition:** **Supporter+** · **Capability:** `cap.upscale_image`

**What it does** — Raises an image's resolution with an ESRGAN model, optionally adding a FLUX.2 refinement pass for extra detail.
**Where to find it** — **Images → Definition** (`f1.subtab.def`), the **🔍 Increase definition** group (`f1.upscale.group`).
**Step by step**
1. Load the image on the canvas.
2. Choose an **Upscale model** and a **Factor** (2×/3×/4×).
3. Pick a **Mode**: *Model only (fast)* or *Model + AI refinement (more detail)*.
4. For refinement, choose a **Refinement model** and set **Refinement strength** and refine **Steps**.
5. Click **🔍 Increase definition** (`f1.upscale.btn`).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Upscale model** (`f1.upscale.model_lbl`) | Dropdown | First preferred (Nomos/UltraSharp…) | Installed ESRGAN models | The super-resolution network. |
| **Factor** (`f1.upscale.factor_lbl`) | Dropdown | 4× | 2× / 3× / 4× | Magnification. |
| **Mode** (`f1.upscale.mode_lbl`) | Dropdown | Model only (fast) | fast / refine | Refine adds a FLUX.2 pass for more detail. |
| **Refinement model** (`f1.upscale.refine_model_lbl`) | Dropdown | First FLUX.2 | FLUX.2 models | Model for the AI refinement pass (refine mode only). |
| **Refinement strength** (`f1.upscale.refine_strength_lbl`) | Slider | 0.30 | 0.05–0.70 | High redraws more, low preserves the original. |
| **Steps** (`f1.steps_lbl`) | Spinbox | 12 | 4–40 | Refinement steps (refine mode only). |

**Specializations** — REST: `POST /image/upscale` with `model` and `target_scale` (default 4.0). A **🔧 Restore / Colorize photo…** button links to the [Enhance quality](04-quality-restore.md) Photo tab with the current image pre-loaded.
**Tips for best results**
> - Use *fast* mode for a clean resolution bump; switch to *refine* only when you want extra invented detail.
> - Keep refinement strength ≤ 0.30 to avoid altering faces and text.

---

### Object segmentation (AI selection with SAM)
> **Edition:** **Full+** · **Capability:** `cap.sam`

**What it does** — Selects an object as an editable mask, either from a text description (GroundingDINO + SAM) or by clicking/boxing on the canvas (SAM).
**Where to find it** — **Images**, the **AI selection** panel (`f1.seg.group`) below the canvas.
**Step by step**
1. To select by text: type the object in **Object to select** (e.g. *person, car, sky*) and click **🎯 Select from text** (`f1.seg.btn`).
2. To select by pointer: choose the **SAM model (pointer)** and use the **SAM Point** canvas tool to click on the object.
3. Adjust **Threshold** if the selection is too wide or too narrow.
4. The resulting mask is editable by hand with the brush/eraser, then usable by inpaint, recolor, etc.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Object to select** (`f1.seg.text_lbl`) | Text | (empty) | free text | Object description for text selection. |
| **Detector** (`f1.seg.detector_lbl`) | Dropdown | GroundingDINO + SAM-HQ | +SAM-HQ (refined edge) / DINO only (faster) | How the text selection is found and refined. |
| **Threshold** (`f1.seg.threshold_lbl`) | Spinbox | 0.50 | 0.10–0.95, step 0.05 | High = tighter/precise; low (0.20–0.35) = wider. Lower it if a click grabs only a patch. |
| **SAM model (pointer)** (`f1.seg.sam_lbl`) | Dropdown | First vanilla SAM | Installed SAM models | Used only by pointer/box selection on the canvas. |

**Specializations** — REST: `POST /image/segment` with `text`, `threshold` (0.3), `use_sam` (true). Pointer selection needs a *vanilla* SAM model, not SAM-HQ.
**Tips for best results**
> - For text selection, use a single concrete noun (*"the dog"*); combine with +SAM-HQ for a clean edge.
> - If the pointer grabs too little, lower the **Threshold**.

---

### Face-swap (InsightFace)
> **Edition:** **Full+** · **Capability:** `cap.restore`

**What it does** — Replaces the face in an image with a face from another photo using InsightFace.
**Where to find it** — Available from the [Enhance quality](04-quality-restore.md) Photo sub-tab (**Face-swap** operation), reachable via the **🔧 Restore / Colorize photo…** button on the Definition sub-tab.
**Step by step**
1. Click **🔧 Restore / Colorize photo…** on the Definition sub-tab (or open **Enhance quality → Photo**).
2. Choose **Face-swap** and pick the **Source face** photo.
3. Process the image.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Source face** | File picker | (empty) | image file | The face to impose onto the target. |

**Specializations** — REST: `POST /image/faceswap` with `target_path` and `source_face_path`.

---

### The mask canvas
> **Edition:** **Full+** (masks feed inpaint/segmentation) · **Capability:** `cap.inpaint` / `cap.sam`

**What it does** — The shared drawing surface where you load images, paint and erase masks, arrange layers, and set the working aspect ratio.
**Where to find it** — Right side of the **Images** tab, always visible.
**Step by step**
1. Load an image with **📁**, or paste extra layers via the layers panel.
2. Pick a tool from the vertical toolbar and (for the brush) set the **Brush** size.
3. Paint the mask (red) with **Brush**; remove with **Erase**; use **Fill**, **Lasso** or **SAM Point** for faster selections.
4. **Undo/Redo** with ↶/↷; **Clear Mask** to start over; **Save Mask** to persist it.
5. Use the aspect-ratio menu to set the output size (or a custom W×H).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Brush** | Slider | 40 px | 3–200 px | Brush/eraser diameter. |
| **Tools** | Buttons | Move | Move / Brush / Erase / Fill / Lasso / SAM Point | Move layers, paint/erase the mask, flood-fill, freehand lasso, or click-to-select with SAM. |
| **Aspect ratio** | Menu | — | 1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3, 5:4, 4:5, 2:1, 21:9, 2.39:1, custom | Sets the working canvas dimensions (multiples of 32). |
| **Layers panel** | List + buttons | Base photo | Add / Up / Down / Delete | Manage stacked images used as references. |

**Tips for best results**
> - The mask you draw here is what Inpaint/Outpaint and Recolor operate on — a clean, slightly-oversized mask gives the best edits.
> - Set the aspect ratio *before* generating so the output matches your target format.

---

## Troubleshooting

- **"Template missing for <arch>."** The selected architecture has no image template; switch to FLUX (always available) or install the required workflow. See [Import models](11-import-models.md).
- **PuLID panel is greyed out.** PuLID works only with the FLUX architecture — switch **Architecture** to FLUX.
- **Pointer selection does nothing / warns "need a vanilla SAM."** Load a vanilla SAM model in the **SAM model (pointer)** dropdown; SAM-HQ is only used for text-based edge refinement.
- **Inpaint result looks noisy or over-cooked.** Lower **Guidance** toward the model's recommendation (distilled ≈2.5, FLUX.2 Klein ≈4, FLUX Fill dev ≈30) and raise **Strength (retention)** to keep more of the original.
- **Integrate adds an unwanted frame around the subject.** Turn **use puzzle_lora** off — it is experimental and FLUX.1-only.
- **Colours bleed during Recolor.** Name the subject precisely, or draw/segment a mask so only that region changes.
- **Outputs carry an "AI Tools Forge" watermark.** That is the Demo/Supporter brand watermark; it is removed from the **Full** edition upward (`cap.watermark`).
