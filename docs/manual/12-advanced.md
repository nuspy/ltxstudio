# Advanced settings

*The power-user control room: quality presets, the exact sampler/scheduler/SHIFT levers that drive video quality, text-encoder selection, the prompt enhancer, fixed seed, unlimited-length auto-extend, VRAM calibration, and the gallery/workflow/analyzer tools.*

The **Advanced** tab is where AI Creative Center exposes the low-level dials that the rest of the interface keeps sensible for you. Nothing here is required for everyday work — the defaults are tuned to the verified high-quality LTX-2.3 pipeline — but when you want deterministic runs, faster drafts, sharper detail, or truly long videos, this is the tab you reach for.

> **Edition:** The **Advanced** tab as a whole requires **Full+**. It groups the sampler/seed controls (which also live inline in the [Video](03-video.md) tab), the long-video engine, the VRAM calibration tool, and the gallery/workflow utilities. The **Import Models** sub-tab shown next to **Tools** is a separate module — see [Import models](11-import-models.md) — and requires **Studio Pro**.

The tab is split into two sub-tabs: **🛠 Tools** (all the settings and utility buttons documented below) and **📦 Import Models** (documented in its own chapter).

---

## How presets work in this app (read this first)

There is one rule that governs every "preset" and "optimize" action in AI Creative Center, and it will save you confusion:

> **Presets write into the GUI controls — they do not silently reconfigure the pipeline.**

When you pick a quality preset, the app fills in the **Steps** and **CFG** fields (and, for the Video-tab presets, resolution / FPS / duration). Those fields then remain fully editable by hand. Nothing is hidden: what you see in the controls is exactly what the model receives. This means you can start from a preset and then fine-tune a single value without fighting an invisible override.

---

## The Sampler / Seed group

> **Edition:** Full+ · **Capability:** `cap.video` (video sampling)

**What it does** — Controls how many denoising steps run, how strongly the prompt is enforced (CFG), which sampler and scheduler are used, the LTX SHIFT quality lever, which text encoder reads your prompt, whether the prompt is auto-enhanced, and the random seed.

**Where to find it** — This group (internally the **Sampler / Seed** group) is embedded directly in the [Video](03-video.md) tab, and its settings are documented here because they are the "advanced" heart of video generation. The heading reads **Sampler / Seed**.

**Step by step**
1. (Optional) Choose a **Quality (preset):** entry to fill Steps and CFG with a known-good combination.
2. Adjust **Steps** and **CFG** if you want finer control.
3. Leave **Sampler / Sched.** on the defaults unless you have a specific reason to change them.
4. Tune the **Shift (LTX)** **base** / **max** values — this is the highest-impact, zero-cost video quality lever.
5. Choose a **Text encoder** (leave on **Auto** to let content mode decide).
6. Optionally enable **Migliora prompt (LTX enhancer)** (prompt enhancement) for short prompts.
7. Enable **Seed fisso** (Fixed seed) and set a **Seed** value if you need reproducible output.

### Parameters

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Steps** | Number | 32 | 1–60 | Denoising steps. More = more detail and coherence, slower. Selecting a distilled model lowers this automatically (e.g. 8). |
| **CFG** | Number | 1.0 | 0.0–15.0 (step 0.1) | Classifier-free guidance strength. Kept at **1.0** by default because negative guidance is supplied by NAG; raise it only if the prompt is being ignored. High CFG (2.5–4.5) tends to "over-cook" LTX video. |
| **Quality (preset):** | Dropdown | **Custom** | Custom · **Fast (8 steps)** · **Pro (balanced, 30 steps)** · **HQ (max, 50 steps)** | Shortcut that writes Steps + CFG per the official LTX-2.3 modes. **Custom** changes nothing. Fast = 8 steps / CFG 1.0; Pro = 30 / 3.0; HQ = 50 / 4.0. Values remain hand-editable afterward. |
| **Sampler** | Dropdown | **euler_ancestral_cfg_pp** | euler_ancestral_cfg_pp · euler_cfg_pp · euler · dpmpp_2m | The sampling algorithm. The default matches the AuthorStorm reference and gives the best quality on LTX. See the note below on RES samplers. |
| **Scheduler** | Dropdown | **LTXVScheduler (shift)** | LTXVScheduler (shift) · linear_quadratic · bong_tangent · karras · beta | Noise schedule. The default is LTX's own flow-/shape-aware scheduler and enables the SHIFT lever below. Picking a classic scheduler routes the sampler through a BasicScheduler and **ignores** the LTX SHIFT values. |
| **base** (Shift) | Number | 0.95 | 0.10–6.00 (step 0.05) | LTXVScheduler **base shift**. Higher = more steps spent on structure and **motion** (temporal coherence, less flicker); lower = finer detail but more risk of frame-to-frame inconsistency. This is quality lever #1 for LTX video. |
| **max** (Shift) | Number | 2.05 | 0.50–12.00 (step 0.05) | LTXVScheduler **max shift** — the resolution/length-dependent upper bound of the shift. Raise it for longer or larger clips when motion looks uncertain. |
| **Text encoder:** | Dropdown | **Auto** | Auto (by content mode) · Gemma 3 12B — full (max quality) · Gemma 3 12B — abliterated | Which encoder reads your prompt; affects prompt **adherence** (composition, subject, action) far more than texture. Both are 12B, so there is no speed difference. See specializations. |
| **Migliora prompt (LTX enhancer)** | Checkbox | Off | On / Off | Prompt enhancement. When on, a local LLM (Gemma) expands your prompt into a detailed cinematic description before generation. Useful for short prompts; leave off when your prompt is already detailed. |
| **Seed fisso** (Fixed seed) | Checkbox | Off | On / Off | When on, the **Seed** value below is used verbatim, giving reproducible results. When off, a fresh random seed is used each run. |
| **Seed** | Number | 777 | 0 to 2³¹−1 | The random seed. Only applied when **Fixed seed** is on. |

> **Note on samplers:** The RES family (`res_2s`, `res_multistep`, …) is deliberately **not** offered. Although excellent in theory for flow-matching, in practice these break LTX-2.3 and produce green/cyan noise (verified in A/B testing). Stick to the four samplers listed.

**Specializations — text encoders**
- **Auto** — the content mode picks the encoder for you: standard content uses the **full** Gemma encoder; Unlocked (18+) mode uses the **abliterated** (uncensored) encoder.
- **Gemma 3 12B — full** — maximum semantic fidelity; softens explicit content.
- **Gemma 3 12B — abliterated** — an uncensored variant, much better on explicit prompts. An encoder marked absent (⛔) means the weight file has not been downloaded yet.

**Specializations — model-aware suggestions**
When you switch models in the Video tab, the app may suggest steps, CFG, sampler, and scheduler values tuned for that model. Suggestions only override controls you have not manually changed, so your explicit choices are respected.

**Tips for best results**
> - Leave **CFG at 1.0** and **Scheduler on LTXVScheduler (shift)** unless you have a concrete problem to solve — this is the verified reference pipeline.
> - To sharpen a soft result at **zero extra cost**, adjust the SHIFT **base**: lower it slightly for more fine detail, raise it for steadier motion and less flicker.
> - Use **Fixed seed** while dialing in a shot so that each change you make is the only variable.
> - Turn the **prompt enhancer** off once your prompts are detailed; it can steer short prompts helpfully but adds its own interpretation.

---

## Quality presets

> **Edition:** Full+

The **Quality (preset):** dropdown in the Sampler / Seed group is the fast way to set a coherent Steps + CFG pair:

| Preset | Steps | CFG | Use it for |
|---|---|---|---|
| **Custom** | — (unchanged) | — (unchanged) | Full manual control; the preset makes no changes. |
| **Fast (8 steps)** | 8 | 1.0 | Quick drafts to validate an idea. |
| **Pro (balanced, 30 steps)** | 30 | 3.0 | Everyday balanced quality. |
| **HQ (max, 50 steps)** | 50 | 4.0 | Final, highest-quality renders (slowest). |

The Video tab additionally offers richer named presets (for example **⚡ FAST**, **🎯 PRO**, **💎 HQ**, plus aspect-ratio presets such as **📺 Broadcast 16:9**, **📱 Social 9:16**, **🎬 Cinemascope**) that also set resolution, FPS, and duration. Those live in the [Video](03-video.md) tab; the same "presets write into the controls" rule applies.

---

## Long-video auto-extend (unlimited length)

> **Edition:** Full+ · **Capability:** `cap.video`

**What it does** — Produces videos of **unlimited length** at constant VRAM by generating a base pass and then chaining automatic image-to-video extension segments, stitched together with overlap.

**Where to find it** — **Advanced ▸ Tools**, the **🎞 Long videos (unlimited length)** group.

**Step by step**
1. Leave auto-extend enabled (it is on by default).
2. Set the **single-pass threshold** — videos longer than this trigger automatic segmenting.
3. Tune segment length, overlap, and context to trade continuity against motion.
4. Request your target duration in the Video tab and generate; the app handles the rest.

### Parameters

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| Auto-generate videos longer than the threshold (in segments) | Checkbox | On | On / Off | Master switch. When on, any Video-tab duration above the threshold is produced as a base pass plus automatic extensions. |
| Single-pass threshold | Number (s) | 20.0 | 4–30 | Maximum duration of a single LTX pass. Beyond this, auto-extension kicks in. LTX-2.3 reaches ~20 s natively, so 20 is the recommended value. |
| Extension segment length | Number (s) | 10.0 | 4–15 | Length of each chained image-to-video segment. Longer = fewer joins = fewer transitions. 10 s recommended. |
| **Overlap/drop (frame):** | Number | 24 | 0–48 | Frames dropped at the start of each segment to hide the image-to-video transition from the static frame. 24 = 1 s at 24 fps. Higher = cleaner transitions but more discarded content. |
| **Contesto (frame):** (context frames) | Number | 16 | 8–64 | Tail frames from the previous segment used as context for the next one. |
| **Forza contesto:** (context strength) | Number | 0.5 | 0.0–1.0 (step 0.05) | How strongly the context constrains the next segment: 1.0 = heavily anchored (less drift, less motion), 0.0 = free (more motion, more drift). |
| **Scene anti-drift (photo):** | Number | 0.25 | 0.0–1.0 (step 0.05) | Re-injects the original photo's appearance into every segment to curb cumulative degradation (blur, over-exposure, artifacts) in very long videos. 0 = off; 0.25 recommended; high values freeze the look and reduce motion. |
| **Re-anchor face every (segments):** | Number | 0 | 0–20 | Every N segments, restarts from the original photo instead of the last frame (resets face drift). 0 = never (maximum continuity). Use 2–4 only if the face drifts too much. |
| **Resume after crash** | Checkbox | On | On / Off | Saves every generated segment to disk; if generation is interrupted, on restart it regenerates only the missing segments instead of starting over. |

**Tips for best results**
> - Keep the **single-pass threshold at 20 s** — that is LTX-2.3's native ceiling; pushing it higher degrades the base pass.
> - If faces slowly drift over a long clip, first raise **Scene anti-drift** to ~0.25; only if that is not enough, set **Re-anchor face every** to 2–4.
> - Raise **context strength** toward 1.0 for locked-down, low-motion scenes; lower it for dynamic action where you want more movement.
> - Leave **Resume after crash** on for long overnight renders so a single interruption doesn't cost you the whole video.

---

## VRAM calibration

> **Edition:** Full+ · **Capability:** `cap.video` / general memory management

**What it does** — Analyzes all your local image and video models, measures and caches each model's **real VRAM footprint**, and reuses those measurements to decide what to keep on the GPU and what to offload — so you get the largest models your card can run without out-of-memory errors.

**Where to find it** — **Advanced ▸ Tools**, the **VRAM memory management** group, button **Calibrate model VRAM…**.

**Step by step**
1. Open **Advanced ▸ Tools**.
2. Click **Calibrate model VRAM…**.
3. Let the calibration run — it is incremental, so subsequent runs only re-measure new or changed models.
4. The cached footprints are then used automatically for placement decisions on every job.

**Tips for best results**
> - Run this once after installing new large models (or after a driver change) so placement decisions stay accurate.
> - Calibration is incremental with invalidation, so re-running it is cheap.

---

## Tools: Navigator, Workflow presets, and Settings analyzer

> **Edition:** Full+

These three buttons live in **Advanced ▸ Tools** and open utility dialogs.

### Navigator Gallery

**What it does** — Browses your previous generations with previews, full metadata, and one-click parameter reuse.
**Where to find it** — **Advanced ▸ Tools ▸ 🗂 Navigator Gallery…**, shortcut **Ctrl+G**.
**Tips**
> - Use it to recover the exact seed and settings of a shot you liked, then reuse them for a variation.

### Workflow presets

**What it does** — Opens a library of 35 optimized workflows (for example Talking Head, Cinematic Landscape, Product Showcase, Action Scene, Cyberpunk, Fashion, and more) as ready-made starting points.
**Where to find it** — **Advanced ▸ Tools ▸ 🎬 Workflow presets…**.

### Settings analyzer

**What it does** — Checks your current steps, CFG, resolution, LoRA strength, and any content-mode / architecture conflicts, and suggests automatic fixes where possible.
**Where to find it** — **Advanced ▸ Tools ▸ 🔍 Analyze settings…**.
**Tips**
> - Run the analyzer before an expensive render if you have been experimenting — it catches mismatched settings that would waste a long generation.

---

*Related chapters: [Video](03-video.md) for where these sampler/seed controls appear inline, [Getting started](01-getting-started.md) for the app-wide Settings dialog and VRAM limit, and the [Appendix](17-appendix.md) for the full capability→edition table.*
