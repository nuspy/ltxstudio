# Video

*Turn a prompt, a photo, or an audio track into moving pictures with LTX-2 — then guide motion with ControlNet, extend clips to any length, edit them with AI, re-shoot sections, and lip-sync speech.*

The **Video** tab (`tab.video`) generates video from text (t2v), from a starting image (i2v), or from an audio track (a2v). It shares its prompt, model selector, LoRA panel and sampler/seed group with several companion tabs that specialize in editing and post-production: **ControlNet**, **Extend video**, **AI Video Edit**, plus the in-tab **V2V** panel (Normal / Talk / Sing / ReTake) and **Storyboard** for multi-scene shots.

## How video generation works

You write a **prompt** (and optionally a **negative prompt**), pick a **preset** that sets resolution, FPS, duration and step count, choose a **model**, and press generate. LTX-2 samples the video over a number of **steps** using the **LTXVScheduler** with **CFG 1.0** — the guidance is provided by NAG, so you don't raise CFG the way you would for images.

- **Text-to-video (t2v):** prompt only.
- **Image-to-video (i2v):** add a guide image; the **i2v method** and **image strength** control how the video starts from it.
- **Audio-to-video (a2v):** attach an audio track from the [Audio & voice](05-audio-voice.md) tab; the model animates in time with it.

Type **`\`** in any prompt field to open the tag palette (pauses, emotions, speed). Advanced sampler, scheduler and SHIFT controls live in the embedded Sampler/Seed group and the [Advanced settings](12-advanced.md) tab.

---

### Generate a video (text / image / audio)
> **Edition:** **Full+** · **Capability:** `cap.video`

**What it does** — Produces a video clip from your prompt, optionally conditioned on a starting image and/or an audio track.
**Where to find it** — **Video** tab (`tab.video`).
**Step by step**
1. Type your **Prompt** (`vid.prompt.lbl`) and, if needed, a **Negative prompt** (`vid.prompt.neg_lbl`).
2. Pick a **Preset** (`vid.preset.lbl`) — it fills resolution, FPS, duration and steps for you.
3. Set **Content** (Standard / 18+) and choose a **Model** (`vid.model_lbl`); use the precision filter and VRAM colours to fit your GPU.
4. Adjust **Resolution**, **W/H**, **FPS** and **Duration** if the preset isn't exactly what you want.
5. For image-to-video, load a guide image, choose an **i2v method** and set **Image strength**.
6. For audio-to-video, set the audio source in the [Audio & voice](05-audio-voice.md) tab.
7. Optionally add **LoRA (LTX-2)**, a **Storyboard**, or automatic face restoration.
8. Press **🎬 Generate**.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Prompt** (`vid.prompt.lbl`) | Text | (empty) | free text | Describes the video. Type `\` for the tag palette. |
| **Negative prompt** (`vid.prompt.neg_lbl`) | Text | *pc game, cartoon, ugly, deformed, low quality, blurry, watermark, text* | free text | What to avoid. |
| **Preset** (`vid.preset.lbl`) | Dropdown | — | See preset table below | Applies resolution/FPS/duration/steps. |
| **Content** (`vid.content_lbl`) | Dropdown | Standard | Standard / 18+ | Filters models. The 18+ option requires the Unlocked (18+) mode. |
| **Model** (`vid.model_lbl`) | Dropdown | Auto | Installed LTX models | Generation model; colours flag VRAM feasibility. |
| **Resolution** (`vid.res_lbl`) | Dropdown | Vertical 9:16 (704×1280) | 9:16, 16:9, 1:1, 4:3, 3:4, 21:9, 2.39:1, 2.35:1, 1.85:1, small variants, Custom | Aspect/resolution; Custom enables manual W/H. |
| **W** (`vid.w_lbl`) | Spinbox | 704 | 256–2048, step 32 | Width (multiple of 32). |
| **H** (`vid.h_lbl`) | Spinbox | 1280 | 256–2048, step 32 | Height (multiple of 32). |
| **FPS** (`vid.fps_lbl`) | Spinbox | 24 | 1–60 | 24 = cinematic, 30 = TV, 60 = fluid. |
| **Duration** (`vid.dur_lbl`) | Spinbox | 8.0 s | 1.0–60.0 s | Clip length. |
| **Image strength (soft reference)** (`vid.img_strength_lbl`) | Slider | 0.50 | 0.00–1.00 | How faithful the video stays to the guide frame. 0.45–0.60 = natural motion; 0.80+ = tight but rigid. |
| **Image→video method** (`vid.i2v_method_lbl`) | Dropdown | AddGuide multi-keyframe (default) | AddGuide multi-keyframe / Inplace soft | AddGuide = attention guide, supports multiple keyframes; Inplace = writes the image into frame 1 with a noise mask (physical start, natural motion). |
| **Video engine** (`vid.engine_lbl`) | Dropdown | LTX-2 native (all features) | LTX-2 native / Comfy (dynamic VRAM, t2v/i2v) | Native = every feature (audio/keyframe/controlnet); Comfy = dynamic VRAM but t2v/i2v only. |
| **Fast cache (TeaCache)** (`vid.teacache_lbl`) | Spinbox | 0.0 (off) | 0.0–0.15, step 0.01 | Experimental; keep 0 — any value >0 degrades LTX 2.3 video. |

**Specializations** — REST equivalents are `POST /video/generate` (t2v/i2v/a2v) with `i2v_method` (`addguide`|`inplace`), `ref_strength`, `audio_path`, `audio_scale`, etc. See [REST API](15-rest-api.md). **Downscale/Upscale/Refine HR** checkboxes are currently inert (generation is native full-resolution); use the [Enhance quality](04-quality-restore.md) Video tab for super-resolution.

**Available presets** (from the built-in preset list):

| Preset | Steps | Resolution | FPS | Duration | Notes |
|---|---|---|---|---|---|
| **⚡ FAST (draft ~1 min)** | 8 | 704×384 | 16 | 4.0 s | Low-res draft to validate the idea. |
| **🎯 PRO (balanced)** | 24 | 1280×704 | 24 | 6.0 s | Best speed/quality compromise. |
| **💎 HQ (max quality)** | 44 | 1280×704 | 24 | 8.0 s | No cache, highest quality, slowest. |
| **🎯 Best quality** | 40 | 1280×704 | 24 | 8.0 s | Max quality, 40 steps. |
| **⚡ Quick preview** | 8 | 480×832 | 16 | 4.0 s | Fast draft (~1 min). |
| **📺 Broadcast 16:9** | 30 | 1280×704 | 25 | 6.0 s | TV/YouTube, 25 fps PAL. |
| **📱 Social 9:16** | 30 | 704×1280 | 30 | 6.0 s | Reels/TikTok/Shorts vertical. |
| **🎬 Cinemascope** | 35 | 1280×544 | 24 | 10.0 s | 2.35:1 cinematic. |
| **🎥 Anamorphic 2.39:1** | 35 | 1280×544 | 24 | 8.0 s | Anamorphic cinema format. |
| **🚀 GGUF fast (low RAM)** | 8 | 704×1280 | 24 | 5.0 s | Optimized for quantized GGUF. |
| **🌟 Balanced** | 24 | 704×1280 | 24 | 8.0 s | Everyday quality/speed balance. |

**Tips for best results**
> - Start from **🎯 PRO** for a balance, **⚡ FAST** to test an idea, **💎 HQ** for the final render.
> - For i2v, keep **Image strength** around 0.45–0.60 for natural motion; raise it only if the subject drifts.
> - Leave **TeaCache** at 0 — on LTX 2.3 any value visibly degrades the video.

---

### First frame from a face (PuLID → i2v)
> **Edition:** **Full+** · **Capability:** `cap.video`

**What it does** — Generates the first frame carrying a face identity (PuLID-FLUX), then uses it as the i2v starting frame so the person's face is preserved through the video.
**Where to find it** — **Video** tab, the **Face → first frame (PuLID)** control (`vid.pulid_kf_lbl`).
**Step by step**
1. In **Face → first frame (PuLID)**, browse to one or more face photos (`vid.pulid_kf.browse`).
2. Click **🧑 First frame from face** (`vid.pulid_kf.gen`) — the identity-carrying frame is generated and loaded.
3. Press **Generate video**: i2v animates the frame while keeping the face.

**Tips for best results**
> - Provide several clean photos of the same person for a more robust identity.

---

### Automatic face restoration, face-swap and smoothing
> **Edition:** **Full+** · **Capability:** `cap.restore`

**What it does** — Post-processes the generated video automatically: restores faces (CodeFormer/GFPGAN), swaps in a face from a photo (InsightFace), and/or smooths motion with RIFE.
**Where to find it** — **Video** tab: the **🙂 Automatic face restoration** group (`vid.fr_auto.group`), the **Face-swap** option (`vid.faceswap.auto`), and the **Smooth (RIFE)** option (`vid.rife.auto`).
**Step by step**
1. To fix "plastic" faces, tick **Restore faces at the end of generation (CodeFormer / GFPGAN)** (`vid.fr_auto`), choose a **Model**, and set **Fidelity** (~0.70).
2. To impose a specific face, tick **Face-swap: impose a face from a photo** (`vid.faceswap.auto`) and **Choose face…** (`vid.faceswap.face.btn`).
3. To smooth motion, tick **Smooth (RIFE): less judder, more fps** (`vid.rife.auto`) and pick an **Interpolation** factor.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Restore faces at the end** (`vid.fr_auto`) | Checkbox | Off | On / Off | Per-frame face restoration preserving identity (no re-diffusion). |
| **Model** (`vid.fr_auto_model_lbl`) | Dropdown | First available | CodeFormer / GFPGAN | Face-restoration network. |
| **Fidelity** (`vid.fr_auto_fidelity_lbl`) | Slider | 0.70 | 0.00–1.00 | 1.0 preserves the original more; 0.0 reconstructs more. |
| **Face-swap: impose a face** (`vid.faceswap.auto`) | Checkbox | Off | On / Off | Replaces every frame's face with the chosen photo (InsightFace). |
| **Smooth (RIFE)** (`vid.rife.auto`) | Checkbox | Off | On / Off | Interpolates in-between frames (anti-judder). |
| **Interpolation** (`vid.rife.factor_lbl`) | Dropdown | x2 | x2 / x4 | x2 doubles the fps, x4 quadruples them. |

**Tips for best results**
> - Face restoration adds a second pass — the final video is the restored one. See also [Enhance quality](04-quality-restore.md).

---

### The Sampler / Seed group
> **Edition:** **Full+** · **Capability:** `cap.video`

**What it does** — Exposes the sampling engine (quality preset, steps, CFG, sampler, scheduler, SHIFT), the text encoder, prompt enhancement, and a fixed-seed option for reproducibility.
**Where to find it** — **Video** tab, the embedded Sampler/Seed group (see also [Advanced settings](12-advanced.md)).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Quality** | Dropdown | Custom | Custom / Fast (8, CFG 1.0) / Pro (30, CFG 3.0) / HQ (50, CFG 4.0) | One-click quality tier that writes Steps/CFG. |
| **Steps** | Spinbox | 32 | 1–60 | Sampling steps; 32 is the reference quality value. |
| **CFG** | Spinbox | 1.0 | 0.0–15.0, step 0.1 | Keep 1.0 — NAG provides the guidance on LTX. |
| **Sampler** | Dropdown | euler_ancestral_cfg_pp | euler_ancestral_cfg_pp / euler_cfg_pp / euler / dpmpp_2m | Sampling algorithm verified for LTX-2.3. |
| **Scheduler** | Dropdown | LTXVScheduler (shift) | LTXVScheduler / linear_quadratic / bong_tangent / karras / beta | The video scheduler; LTXVScheduler is the correct default. |
| **Base Shift** | Spinbox | 0.95 | 0.10–6.00, step 0.05 | LTXVScheduler shift lever — a real, zero-cost quality dial. |
| **Max Shift** | Spinbox | 2.05 | 0.50–12.00, step 0.05 | Upper shift bound. |
| **Text encoder** | Dropdown | auto | auto / Gemma full / Gemma abliterated | auto picks the full encoder for standard content and the abliterated encoder in Unlocked (18+) mode. |
| **Enrich prompt (local AI)** | Checkbox | Off | On / Off | LLM enriches the prompt before generation. |
| **Fixed seed** + **Seed** | Checkbox + spinbox | Off, 777 | 0–2,147,483,647 | Fix the seed for reproducible runs. |

**Tips for best results**
> - The single biggest quality lever at zero cost is **Base Shift** — nudge it if detail degenerates on native resolution.

---

### Storyboard / keyframes (multi-scene)
> **Edition:** **Full+** · **Capability:** `cap.video`

**What it does** — Builds one continuous video from several keyframe images anchored at different times; the model interpolates between scenes, with no editing.
**Where to find it** — **Video** tab, the **🎬 Multi-scene storyboard** group (`vid.sb.group`).
**Step by step**
1. Tick **Enable storyboard (multiple scenes)** (`vid.sb.enable`).
2. For each scene, click **📷 Image** to pick a keyframe, set the time (**at … s**) and the **strength**.
3. Click **+ Add scene** (`vid.sb.add`) for more scenes (at least 2 with images are required).
4. Generate — you get a single continuous clip.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Enable storyboard** (`vid.sb.enable`) | Checkbox | Off | On / Off | Turns on multi-scene interpolation. |
| **at (time)** (`vid.sb.at`) | Spinbox | — | 0.0–3600.0 s, step 0.5 | When to anchor this keyframe. |
| **strength** (`vid.sb.force`) | Spinbox | 0.8 | 0.0–1.0, step 0.05 | 1 = rigid (little motion), 0 = free. |

**Specializations** — REST: `POST /video/keyframes` with a `keyframes` list of `{path, frame, strength}`. See [REST API](15-rest-api.md).
**Tips for best results**
> - Space keyframes evenly and keep strengths around 0.7–0.8 so motion flows between scenes.

---

## ControlNet tab — copy motion from a reference video
> **Edition:** **Full+** · **Capability:** `cap.controlnet`

**What it does** — Copies the *motion/shape* of a reference video into your generated video; the content comes from the Video-tab prompt, the motion from the reference.
**Where to find it** — the **ControlNet** tab (`tab.controlnet`).
**Step by step**
1. Tick **✅ Enable ControlNet (copy the motion from the video below)** (`cn.enable`).
2. Choose a **Type** and load the **reference video** (`cn.video.pick.btn`).
3. In the **Video** tab, write the prompt describing the *content* you want.
4. Set **Control strength** and **Guide attention**, then press **🎬 Generate**.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Enable ControlNet** (`cn.enable`) | Checkbox | Off | On / Off | The generated video follows the reference motion. |
| **Type** (`cn.type_lbl`) | Dropdown | Canny | Canny (edges) / Depth (3D geometry) / Pose (human skeleton, DWPose) | Canny for rigid objects, Depth for full scenes, Pose for body motion. |
| **Control strength** (`cn.strength_lbl`) | Spinbox | 0.8 | 0.0–1.0, step 0.05 | 1.0 strictly follows the reference; lower = more creative freedom. |
| **Guide attention** (`cn.attn_lbl`) | Spinbox | 1.0 | 0.1–1.0, step 0.05 | `conditioning_attention_strength`: lower (0.5–0.8) allows freedom while preserving motion. |

**Specializations** — REST: `POST /video/controlnet` with `control_type` (`canny`|`depth`|`pose`) and `control_strength` (0.8). See [REST API](15-rest-api.md).
**Tips for best results**
> - Use **Pose** for dance/sport, **Depth** for environments, **Canny** for products and rigid shapes.
> - The reference resolution/duration don't matter — the workflow resamples to your Video-tab settings.

---

## Extend video tab — unlimited-length continuation
> **Edition:** **Full+** · **Capability:** `cap.extend_retake`

**What it does** — Adds time to an existing video with an autoregressive, memory-bounded chain — no length limit; up to 20 s is one pass, beyond that it stitches overlapping segments.
**Where to find it** — the **Extend video** tab (`tab.extend_video`).
**Step by step**
1. Load a video (`ext.pick.btn`) or use the last generation.
2. Write the **Continuation prompt** (what happens next).
3. Set **Total duration** to add, and — only if over 20 s — the **Segment** length, **Overlap** and **Context**.
4. Set **Steps** and click **➕ Extend video** (`ext.run.btn`).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Total duration** (`ext.duration_lbl`) | Spinbox | 20.0 s | 1.0–3600.0 s | Time to add. No limit — generated in chained chunks. |
| **Segment (over 20 s)** (`ext.seg_lbl`) | Spinbox | 18.0 s | 2.0–20.0 s | Chunk length when exceeding 20 s. ≤20 s → single pass (no seam). |
| **Overlap (frames)** (`ext.overlap_lbl`) | Spinbox | 8 | 0–48 | Overlap frames between chunks for motion continuity. |
| **Context (frames)** (`ext.context_lbl`) | Spinbox | 16 | 8–64 | Tail frames used as context for the next chunk. Higher = more anchored to the past. |
| **Steps** (`ext.steps_lbl`) | Spinbox | 8 | 1–60 | Sampling steps per chunk. |

**Specializations** — REST: `POST /video/extend` with `add_seconds` (5.0), `seg_seconds` (0 = default), `steps` (8). SSML tags are supported in the prompt.
**Tips for best results**
> - Keep additions ≤20 s for a single seamless pass; longer requests are stitched automatically.

---

## AI Video Edit tab — instruction, inpaint, style, replace character
> **Edition:** **Full+** · **Capability:** `cap.controlnet` / `cap.extend_retake`

**What it does** — Edits an existing video in five modes: global edit-by-instruction, object inpaint (SAM2 tracking), anime↔real style transfer, and character replacement.
**Where to find it** — the **✨ AI Video Edit** tab (`tab.video_edit_ai`).
**Step by step**
1. Choose a video (`vedit.pick`).
2. Pick a **Mode** (`vedit.mode_lbl`).
3. Fill the mode-specific fields (object, reference image, etc.).
4. Write the **Prompt** and set the edit parameters.
5. Click **✨ Generate edit** (`vedit.go`).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Mode** (`vedit.mode_lbl`) | Dropdown | Edit by instruction | Edit by instruction (edit-anything) / Replace object (inpaint, experimental) / Style: anime→real / Style: real→anime / Replace character | The kind of edit to apply. |
| **Object (by words)** (`vedit.obj_lbl`) | Text | (empty) | free text | Object to select (Inpaint) or character to limit to (Replace character). |
| **Prompt** (`vedit.prompt_lbl`) | Text | (empty) | free text | What to generate/change. |
| **Edit strength** (`vedit.strength_lbl`) | Dropdown | 1.0 | 0.6 / 0.8 / 1.0 | How strongly to apply the edit. |
| **Replacement image (character)** (`vedit.ref_lbl`) | File picker | (empty) | image | First video frame with the other character (Replace character). |
| **Identity strength (reference)** (`vedit.ref_strength_lbl`) | Spinbox | 0.8 | 0.1–1.0, step 0.05 | How much the reference identity counts vs. the video's motion. |
| **CFG (prompt adherence) — inpaint** (`vedit.cfg_lbl`) | Spinbox | 6.0 | 1.0–10.0, step 0.5 | Prompt adherence for inpaint. |
| **Guide strength (anchor to original)** (`vedit.guide_lbl`) | Spinbox | 0.4 | 0.1–1.0, step 0.1 | How tightly the edit anchors to the original. |
| **Mask dilation (px)** (`vedit.dilate_lbl`) | Spinbox | 24 | 0–64 | Grows the tracked mask; more room for the new object. |
| **Mask feather (px)** (`vedit.blur_lbl`) | Spinbox | 6.0 | 0.0–30.0, step 1.0 | 0 = hard edge; 6–12 = smooth transition. |
| **✨ Enhance prompt (AI)** (`vedit.enhance`) | Checkbox | Off | On / Off | LLM enriches the edit prompt. Invisible, optional. |

**Specializations** — REST: `POST /video/v2v_edit` with `edit_lora`, `control_strength`. **Replace character** keeps the original audio and preserves scene/motion via canny; long videos are processed in segments.
**Tips for best results**
> - **Replace object (inpaint)** is experimental — SAM2 tracks the object across the clip and regenerates its area.
> - For **Replace character**, load an image that is the first video frame with only the character changed.

---

## V2V panel — Normal, Talk (lip-sync), Sing, ReTake
> **Edition:** **Full+** · **Capability:** `cap.extend_retake`

**What it does** — In-tab video-to-video operations: re-shoot a section (**ReTake**), lip-sync speech (**Talk**), make a face sing a song (**Sing**), or pass through (**Normal**).
**Where to find it** — **Video** tab, the **🎬 Video editing (V2V)** panel (`vid.v2v.group`), **Mode** selector (`v2v.mode_label`).

**Specializations**

- **ReTake** (`v2v.mode_retake`) — regenerate a section. Set the **Section to regenerate** with **Start**/**End** sliders (constraint: at least 3 s of video before and after). Choose section **Audio**: **Keep original** (`v2v.audio_keep`) or **Regenerate** (`v2v.audio_regen`). REST: `POST /video/retake` with `start_time`/`end_time`.
- **Talk** (`v2v.mode_talk`) — lip-sync. With a video chosen, it lip-syncs an existing clip; with no video, it generates speech from the Frame-1 image (i2v). Pick the **Face mask**: **Mouth/lips (faithful lip-sync)** (`v2v.region_mouth`) or **Whole face (more natural)** (`v2v.region_face`), with a **Mask offset (px)**. The voice follows the Audio-tab decision tree (uploaded file → selected TTS voice + text → dialogue in the main prompt). REST: `POST /video/lipsync` with `audio_scale` (3.0), `separate_vocals`, `talk_region`, `talk_mask_offset`.
- **Sing** (`v2v.mode_sing`) — from the Frame-1 photo, generate a video of the face singing a real song with lip-sync and beat-driven motion. **Choose song…** (`v2v.sing_pick_audio`), tick **Isolate vocals (recommended)** (`v2v.sing_isolate_vocals`), and set **Sync strength** (1.5–3.0 recommended; default 2.0).
- **Normal** (`v2v.mode_normal`) — standard pass-through mode.

**Parameters (ReTake / Talk / Sing highlights)**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Mode** (`v2v.mode_label`) | Radio | Normal | Normal / Talk / Sing / ReTake | Selects the V2V operation. |
| **Start / End** (`v2v.start_label` / `v2v.end_label`) | Sliders | — | within the clip (3 s guard each side) | The section to re-shoot (ReTake). |
| **Section audio** | Radio | Keep original | Keep original / Regenerate | Whether to keep or regenerate the section's audio. |
| **Face mask** (`v2v.face_mask_label`) | Dropdown | Mouth/lips | Mouth/lips / Whole face | Region for lip-sync (Talk). |
| **Mask offset (px)** (`v2v.mask_offset_label`) | Spinbox | 0 | 0–128 | Shifts the face mask (Talk). |
| **Isolate vocals** (`v2v.sing_isolate_vocals`) | Checkbox | On | On / Off | Separates voice from music (demucs) for accurate lip-sync (Sing). |
| **Sync strength** (`v2v.sing_audio_scale`) | Slider | 2.0 | 0.5–3.0 | Lip-sync strength on the audio (Sing). |

**Tips for best results**
> - For lip-sync, choose **Mouth/lips** for accuracy or **Whole face** for a more natural look.
> - In **Sing**, keep **Isolate vocals** on and **Sync strength** at 1.5–3.0.

---

### LoRA (LTX-2) panel
> **Edition:** **Full+** · **Capability:** `cap.video`

**What it does** — Applies one or more LTX-2 video LoRAs (style, subject, effects) filtered to the current video model's architecture.
**Where to find it** — **Video** tab, the **LoRA (LTX-2)** group (`vid.lora.group`).
**Step by step**
1. Search/filter the LoRA list (adult-LoRA visibility follows the video Content selector).
2. Tick the LoRAs you want and set each one's strength (default 0.6, range 0.0–2.0).
3. Generate — triggers are added automatically if the auto-trigger option is on.

**Tips for best results**
> - Keep total LoRA strength moderate; stacking several strong LoRAs can wash out the model.

---

## Troubleshooting

- **The video degenerates into "dots" or dematerializes on native resolution.** This is a known LTXVScheduler shift issue on native token counts — adjust **Base Shift** in the Sampler/Seed group, and don't force high CFG (keep 1.0).
- **CFG has no effect / high CFG looks over-cooked.** On LTX, guidance comes from NAG — leave **CFG at 1.0** and change the preset/steps instead.
- **Downscale/Upscale/Refine HR checkboxes do nothing.** They are temporarily inert; generation is native full-resolution. Use the [Enhance quality](04-quality-restore.md) Video tab for super-resolution.
- **ControlNet output ignores the motion.** Raise **Control strength** toward 1.0 and make sure the reference has clearly readable motion for the chosen type.
- **Extend produces a black seam.** Keep the addition ≤20 s (single pass); for longer clips the overlap/context settings smooth the joins.
- **Faces look "plastic."** Enable automatic face restoration (Fidelity ~0.70) or run the [Enhance quality](04-quality-restore.md) face restorer as a final pass.
- **A2V/Talk has no dialogue.** Set the audio source correctly in the [Audio & voice](05-audio-voice.md) tab, or write the spoken line into the main prompt for native generation.
