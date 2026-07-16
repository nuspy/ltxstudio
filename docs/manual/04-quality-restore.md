# Enhance quality

*Rescue and upgrade your media after it's made: restore, upscale, colorize and face-swap photos, and apply AI super-resolution, face restoration and smooth frame interpolation to video.*

The **Enhance quality** tab (`tab.quality`, shown as *"✨ Migliora qualità"*) is a post-processing workbench. It never regenerates your content from scratch — it takes an existing image or video and improves its resolution, colours, faces or fluidity while keeping the original intact. The tab is split into two nested sub-tabs:

- **🖼 Photo** (`tab.quality.photo`) — restore, upscale, colorize black-and-white, or face-swap a single image.
- **🎬 Video** (`tab.quality.video`) — AI super-resolution (SeedVR2 / FlashVSR / ESRGAN), per-frame face restoration, RIFE frame interpolation, and an EXR/HDR export path for VFX.

Every operation here runs locally and preserves your source file; results are written as new files.

---

## Photo sub-tab

Load an image, choose one operation, tune its settings, and press **🔧 Process image**. The four operations are mutually exclusive — the settings groups for the others grey out automatically.

### Restore a photo (ESRGAN super-resolution)
> **Edition:** **Supporter+** · **Capability:** `cap.upscale_image` / `cap.restore`

**What it does** — Increases a photo's resolution and sharpness with an ESRGAN super-resolution model, optionally applying a light denoise to clean up compression artifacts.
**Where to find it** — **Enhance quality → 🖼 Photo**, the **Operation** group with **🔧 Quality restore (ESRGAN super-resolution)** selected.
**Step by step**
1. Load a source image: drag it onto the field, click **📂 Browse…** (`res.input.browse.btn`), or click **🖼 Use canvas** (`res.input.canvas.btn`) to pull the current [Images](02-images.md) canvas image.
2. In **Operation**, choose **🔧 Quality restore (ESRGAN super-resolution)** (`res.op.restore`, the default).
3. Under **Restore settings (ESRGAN)**, pick an **Upscale model** and a **Target scale**.
4. Optionally tick **Denoising** and set its strength.
5. Click **🔧 Process image** (`res.run.btn`).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Upscale model** (`res.restore.model_lbl`) | Dropdown | First "ultra/real/nomos" model | Installed ESRGAN models | The super-resolution network. Colours indicate VRAM feasibility. |
| **Target scale** (`res.restore.scale_lbl`) | Spinbox | 2.0× | 1.0×–4.0×, step 0.5 | How much to enlarge. 4× for very small sources; 2× for a gentle boost. |
| **Denoising** (`res.restore.denoise`) | Checkbox | Off | On / Off | Removes compression artifacts; may soften detail slightly. |
| **Denoise strength** (`res.restore.denoise_str_lbl`) | Spinbox | 0.20 | 0.05–0.50, step 0.05 | Strength of the denoise pass; active only when the checkbox is on. |

**Tips for best results**
> - ESRGAN needs no extra download beyond the upscale models you already have.
> - Start at 2× with denoise off; enable denoise only if you see JPEG blocking.

---

### Colorize a black-and-white photo (FLUX.2)
> **Edition:** **Full+** · **Capability:** `cap.restore`

**What it does** — Adds realistic, period-accurate colour to a black-and-white photo using a FLUX.2 diffusion model guided by your description.
**Where to find it** — **Enhance quality → 🖼 Photo**, **Operation → 🎨 B&W colorization (FLUX.2 — requires a FLUX.2 model)** (`res.op.colorize`).
**Step by step**
1. Load the black-and-white image.
2. Select **🎨 B&W colorization** in **Operation**.
3. Under **Colorization settings (FLUX.2)**, choose a **FLUX.2 model**.
4. Describe the **Subject** and adjust **Color style**, **Steps**, **Strength (denoise)** and **Adherence (guidance)** as needed.
5. Click **🔧 Process image**.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **FLUX.2 model** (`res.color.model_lbl`) | Dropdown | First FLUX.2 model | Installed FLUX.2 models | The colorization model. Requires a FLUX.2 checkpoint. |
| **Subject** (`res.color.subject_lbl`) | Text | (empty) | free text | Main subject, e.g. *"man in uniform"* — helps the model choose plausible colours. |
| **Color style** (`res.color.style_lbl`) | Text | *natural realistic colors, period-accurate* | free text | Desired palette and look. |
| **Steps (FLUX.2)** (`res.color.steps_lbl`) | Spinbox | 20 | 5–50 | Diffusion steps: more = more quality, slower. |
| **Strength (denoise)** (`res.color.denoise_lbl`) | Spinbox | 0.85 | 0.30–1.00, step 0.05 | High = bolder colours but less faithful to structure; low = structure better preserved. |
| **Adherence (guidance)** (`res.color.guidance_lbl`) | Spinbox | 2.5 | 1.0–6.0, step 0.5 | How strictly the result follows your subject/style description. |

**Specializations** — Choose **✨ Restore + Colorization (sequential)** (`res.op.both`) to run ESRGAN restore and FLUX.2 colorization back-to-back in one click.
**Tips for best results**
> - Name the era and subject in **Subject** (e.g. *"1940s family portrait"*) for believable, period-accurate tones.
> - If colours bleed or look invented, lower **Strength** toward 0.5 to keep the original structure.

---

### Face-swap a photo (InsightFace)
> **Edition:** **Full+** · **Capability:** `cap.restore`

**What it does** — Replaces the face in the target photo with a face from another photo using InsightFace.
**Where to find it** — **Enhance quality → 🖼 Photo**, **Operation → Face-swap (impose a face)** (`res.op.faceswap`).
**Step by step**
1. Load the target image (the one whose face you want to replace).
2. Select **Face-swap** in **Operation**.
3. Under **Face-swap**, click **Choose face…** (`res.faceswap.source.btn`) and pick the source face photo.
4. Click **🔧 Process image**.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Source face** (`res.faceswap.source_lbl`) | File picker | (empty) | image file | The face to impose onto the target. |

**Specializations** — The REST equivalent is `POST /video/faceswap` with `video_path` (image or video) and `source_face_path`. See [REST API](15-rest-api.md).
**Tips for best results**
> - Use a clear, front-facing source face with even lighting for the cleanest swap.

---

## Video sub-tab

Load a video, pick a super-resolution **Engine**, tune it, and press **✨ Enhance video**. The Video sub-tab also hosts face restoration, RIFE interpolation and EXR export. The original audio is preserved throughout.

### AI video super-resolution (SeedVR2 / FlashVSR / ESRGAN)
> **Edition:** **Full+** · **Capability:** `cap.restore`

**What it does** — Upscales a video to a higher resolution with your choice of three engines: SeedVR2 (AI video model, best quality), FlashVSR (node-based AI), or ESRGAN (fast per-frame).
**Where to find it** — **Enhance quality → 🎬 Video**, the **Restore / super-resolution** group (`def.restore.group`).
**Step by step**
1. Click **📁 Load video…** (`def.pick.btn`) and choose your clip.
2. Pick an **Engine** (`def.engine_lbl`). The relevant controls enable for that engine.
3. For **SeedVR2**: choose the model, **Target resolution**, and tune **BlockSwap**, **Batch** and **Temporal overlap** for your VRAM.
4. For **ESRGAN / FlashVSR**: choose the ESRGAN model and **Scale**.
5. Optionally tick **🔗 Then restore faces (in sequence)** to chain face restoration (and RIFE) automatically.
6. Click **✨ Enhance video** (`def.run.btn`).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Engine** (`def.engine_lbl`) | Dropdown | SeedVR2 | **SeedVR2 (AI, recommended)** · **ESRGAN per-frame (fast)** · **FlashVSR (requires node)** | The super-resolution method. SeedVR2 = best; ESRGAN = fastest. |
| **SeedVR2 model** (`def.seedvr2_model_lbl`) | Dropdown | First available | SeedVR2 variants | Model variant (SeedVR2 only). Downloads on first use. |
| **Target resolution** (`def.resolution_lbl`) | Spinbox | 1080 px | 256–8192, step 64 | Target vertical resolution (SeedVR2). 1080 = FullHD. |
| **BlockSwap (VRAM)** (`def.blocks_lbl`) | Spinbox | 0 | 0–36 | >0 lowers VRAM use (slower); 0 = everything on GPU (fine on 32 GB). |
| **Batch (4n+1)** (`def.batch_lbl`) | Spinbox | 5 | 1–121, step 4 | Frames per batch (1, 5, 9…). Higher = more temporal coherence and VRAM. |
| **Temporal overlap** (`def.overlap_lbl`) | Spinbox | 3 | 0–16 | Overlap frames between SeedVR2 chunks; reduces judder at cuts on long clips. |
| **ESRGAN model** (`def.esrgan_model_lbl`) | Dropdown | First available | Installed ESRGAN models | Per-frame upscaler (ESRGAN engine). |
| **Scale (ESRGAN)** (`def.scale_lbl`) | Spinbox | 2.0× | 1.0×–8.0×, step 0.5 | Upscale factor for ESRGAN / FlashVSR. |
| **🔗 Then restore faces (in sequence)** (`def.chain_face`) | Checkbox | Off | On / Off | After super-resolution, auto-runs face restoration then RIFE — no manual re-run. |

**Specializations** — REST: `POST /video/enhance` with `engine` (`seedvr2`|`flashvsr`), `seedvr2_model`, `resolution` (default 1080), `batch_size` (default 5), `block_swap` (default 0), `scale`. See [REST API](15-rest-api.md).
**Tips for best results**
> - **SeedVR2** gives the most natural motion-aware upscale; keep **Batch** at 5 unless you have spare VRAM.
> - On long videos with visible seams at cuts, raise **Temporal overlap** to 4–6.
> - SeedVR2 downloads its model on first use and always keeps the original audio.

---

### Restore faces in a video (CodeFormer / GFPGAN)
> **Edition:** **Full+** · **Capability:** `cap.restore`

**What it does** — Reconstructs faces frame-by-frame to fix the "plastic" look and identity drift of generated video, while preserving identity (it does not re-run diffusion).
**Where to find it** — **Enhance quality → 🎬 Video**, the **🙂 Face restoration (CodeFormer / GFPGAN)** group (`def.face.group`).
**Step by step**
1. Load the video.
2. If the status shows the node isn't installed, click **📥 Install face-restore** (`def.face.install.btn`) and restart when prompted.
3. Choose a **Model** and **Face detector**.
4. Set **Fidelity** (≈0.70 recommended).
5. Click **🙂 Restore video faces** (`def.face.run.btn`).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Model** (`def.face.model_lbl`) | Dropdown | First available | CodeFormer (quality) / GFPGAN (speed) | The face-restoration network. |
| **Face detector** (`def.face.detection_lbl`) | Dropdown | First available | Available detectors | Locates faces before restoring them. |
| **Fidelity** (`def.face.fidelity_lbl`) | Slider | 0.70 | 0.00–1.00 | 1.0 preserves the original more; 0.0 reconstructs more (sharper, may drift). ~0.7 is a good balance. |

**Tips for best results**
> - Apply face restoration as the *last* step, after super-resolution.
> - This can also be triggered automatically from the [Video](03-video.md) tab's automatic face-restore option.

---

### Increase FPS with frame interpolation (RIFE)
> **Edition:** **Full+** · **Capability:** `cap.restore`

**What it does** — Generates in-between frames with RIFE to raise a video's frame rate, reducing judder and adding fluidity.
**Where to find it** — **Enhance quality → 🎬 Video**, the **FPS increase (RIFE, optional)** group (`def.rife.group`).
**Step by step**
1. Load the video.
2. Tick **Interpolate frames (more fluidity)** (`def.rife.check`).
3. Choose an **FPS multiplier**.
4. Run it via **✨ Enhance video** (or automatically as part of the chain).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Interpolate frames** (`def.rife.check`) | Checkbox | Off | On / Off | Enables RIFE interpolation. |
| **FPS multiplier** (`def.rife_mult_lbl`) | Dropdown | 2× | 2× / 3× / 4× | 2× doubles the frame rate, 4× quadruples it. |
| **RIFE model** (`def.rife_ckpt_lbl`) | Dropdown | (fixed) | — | Non-selectable: the native RIFE uses a single Practical-RIFE model. |

**Specializations** — REST: `POST /video/rife` with `factor` (2, 3 or 4). See [REST API](15-rest-api.md).
**Tips for best results**
> - Use 2× to turn a 24 fps clip into a smooth 48 fps; reserve 4× for slow-motion looks.

---

### Export HDR / EXR frames (VFX)
> **Edition:** **Full+** · **Capability:** `cap.restore`

**What it does** — Exports every frame of a video as OpenEXR float32 for a professional VFX pipeline (Nuke, Resolve, Blender).
**Where to find it** — **Enhance quality → 🎬 Video**, the **HDR / EXR export (VFX)** group (`def.exr.group`).
**Step by step**
1. Load the video.
2. Choose whether to **Linearize** (de-gamma to linear light).
3. Click **Export EXR frames** (`def.exr.run`) — frames are written to a `<video>_exr` folder.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Linearize (de-gamma BT.709 → linear light)** (`def.exr.linear`) | Checkbox | On | On / Off | On = standard compositing (linear light); Off = display-referred (gamma) values. |

**Tips for best results**
> - For a *true* HDR result, generate the video with the HDR IC-LoRA in the [Video](03-video.md) tab first, then export here.

---

## Troubleshooting

- **Colorization says a FLUX.2 model is required.** Install a FLUX.2 checkpoint via [Import models](11-import-models.md); ESRGAN restore does not need one, but colorization does.
- **Face restoration is greyed out or errors.** The `facerestore_cf` node isn't installed — click **📥 Install face-restore** and restart when prompted. Its models (~350 MB) download on first use.
- **SeedVR2 runs out of memory.** Raise **BlockSwap** above 0 and/or lower **Batch** toward 1; both reduce VRAM at the cost of speed.
- **The RIFE model dropdown won't let me choose.** That's expected — the native RIFE uses one fixed Practical-RIFE model.
- **Upscaled video looks over-sharp.** Prefer SeedVR2 over ESRGAN for moving footage; ESRGAN is per-frame and can accentuate flicker.
