<!-- cover -->
<div class="cover">

# AI Creative Center

## Complete User Manual

**Local, offline AI studio for images · video · music · voices · 3D · film**

*Master edition — documents every module and function across all editions*

Manual revision: 1.0 · Applies to the current desktop release

</div>

---

## About this manual

**AI Creative Center** is a fully local, offline creative suite. Everything — image and video
generation, music, voices, 3D models, complete films — runs on your own GPU; nothing is uploaded to
the cloud. The only optional network use is a one-time download of model weights (and, in the *Studio
Pro* edition, importing models from Civitai).

This is the **master manual**: it documents every feature across every edition. Each function is
tagged with the **minimum edition** required to use it, so you always know what your edition unlocks.

### How to read a function entry

Every function in this manual follows the same layout:

> ### Function name
> **Edition:** *badge* · **Capability:** `cap.id`
>
> **What it does** — a one-line summary.
> **Where to find it** — the exact tab / sub-tab / menu path.
> **Step by step** — a numbered, do-this-then-that procedure.
> **Parameters** — a table listing every control, its default, its range, and when to change it.
> **Specializations** — sub-modes, variants, and Classic vs. Pro interface differences.
> **Tips for best results** — practical advice to get professional output.

### Edition badges at a glance

| Badge | Edition | Highlights |
|-------|---------|-----------|
| **Demo** | Free | Image generation (capped), invisible provenance signature. Outputs carry a visible brand watermark. |
| **Supporter** | $3 | Adds image upscaling; higher image caps. Brand watermark still applied. |
| **Full** | €15 | Unlocks video, music, audio/voice, 3D, Film Studio, quality tools, LoRA auto-training. No brand watermark. |
| **Unlocked 18+** | €25 | Everything in *Full* plus the Unlocked (18+) mode (uncensored models/encoders). |
| **Studio Pro** | €45 | Everything, plus model import (own files + Civitai), per-node overrides, heavy FLUX LoRA training, and the local REST API. |

A trailing **+** (e.g. *Full+*) means "this edition and all higher ones".

---

## Table of contents

1. **[Getting started](01-getting-started.md)** — install, first run, editions & upsell, Classic vs. Pro interface, 15 languages, generation queue, settings, VRAM.
2. **[Images](02-images.md)** — Generate · Inpaint/Outpaint · Recolor · Upscale · Segmentation (SAM) · Face adapter (PuLID) · Multi-reference compose · Face-swap.
3. **[Video](03-video.md)** — Text-to-video · Image-to-video · Audio-to-video · presets · LoRA · sampler/seed · Storyboard/keyframes · ControlNet · Extend · ReTake · AI Video Edit · lip-sync · long video.
4. **[Enhance quality](04-quality-restore.md)** — Photo restore/upscale/colorize/face-swap · Video super-resolution (SeedVR2, FlashVSR) · frame interpolation (RIFE).
5. **[Audio & voice](05-audio-voice.md)** — TTS engines · voice cloning · SSML markers · the shared voice library.
6. **[Music](06-music.md)** — Songs & instrumentals (HeartMuLa / ACE-Step) · video sonification (MMAudio) · post-processing · pitch correction · music LoRA.
7. **[Music video](07-music-video.md)** — song → LLM storyboard → generated scenes → mux · NLE export.
8. **[3D](08-3d.md)** — Image→3D model (TRELLIS.2) · Rig 3D · format Converter · Animation/Mocap · 3D World (WorldGen).
9. **[Cinema — AI Film Studio](09-cinema-studio.md)** — projects · Script · Storyboard/Scenography · Montage · Asset Manager · Voice · LoRA Studio · Continuity · Scheduler.
10. **[Training your own LoRA](10-training.md)** — image LoRA (ai-toolkit) · auto-dataset photo LoRA (FLUX.2) · auto-dataset video LoRA (LTX-2.3).
11. **[Import models](11-import-models.md)** — Discover · Import · Library · Advanced · Civitai · content gating · configurable paths.
12. **[Advanced settings](12-advanced.md)** — quality presets · sampler/scheduler/SHIFT · text encoders · prompt enhancer · fixed seed · long-video auto-extend · VRAM calibration · Navigator/Workflow/Analyzer.
13. **[Provenance signature (steganography)](13-steganography.md)** — invisible watermark · settings · standalone detector · robustness.
14. **[Wizards](14-wizards.md)** — first-run setup wizard · selective installer · guided creation wizard.
15. **[REST API](15-rest-api.md)** — local automation server, async job model, full endpoint reference, agent packs.
16. **[Tips & tricks](16-tips-and-tricks.md)** — best-results recipes for every module.
17. **[Appendix](17-appendix.md)** — glossary · keyboard shortcuts · file locations · capability→edition table · supported languages · model catalog · troubleshooting.
