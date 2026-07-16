# AI Creative Center — User Manual (source)

This folder contains the **complete user manual** for **AI Creative Center**, written in English and
organized one chapter per Markdown file. It is the single, authoritative reference for every module,
function, parameter and workflow in the application.

## How the manual is organized

| File | Chapter |
|------|---------|
| [`00-cover-toc.md`](00-cover-toc.md) | Cover, version, full table of contents, how to read this manual |
| [`01-getting-started.md`](01-getting-started.md) | Installation, editions, interface, languages, queue, settings |
| [`02-images.md`](02-images.md) | Image generation, inpaint/outpaint, recolor, upscale, face tools |
| [`03-video.md`](03-video.md) | Video generation, i2v/t2v/a2v, presets, ControlNet, extend, retake, AI edit |
| [`04-quality-restore.md`](04-quality-restore.md) | Enhance quality: photo restore/upscale + video super-resolution & interpolation |
| [`05-audio-voice.md`](05-audio-voice.md) | Text-to-speech, voice cloning, SSML, the voice library |
| [`06-music.md`](06-music.md) | Songs & instrumentals, video sonification, post-processing |
| [`07-music-video.md`](07-music-video.md) | Music Video Studio (song → storyboard → video) |
| [`08-3d.md`](08-3d.md) | Image→3D, rigging, format conversion, mocap animation, 3D worlds |
| [`09-cinema-studio.md`](09-cinema-studio.md) | AI Film Studio (text-to-film): script, storyboard, montage, continuity |
| [`10-training.md`](10-training.md) | Train your own LoRA (image, photo, video datasets) |
| [`11-import-models.md`](11-import-models.md) | Import checkpoints/LoRA, Civitai browser, catalog, paths |
| [`12-advanced.md`](12-advanced.md) | Advanced tab: sampler/scheduler, encoders, prompt enhancer, VRAM |
| [`13-steganography.md`](13-steganography.md) | Invisible provenance signature & the detector |
| [`14-wizards.md`](14-wizards.md) | First-run setup wizard, selective installer, guided creation wizard |
| [`15-rest-api.md`](15-rest-api.md) | Local REST API for automation (Studio Pro) |
| [`16-tips-and-tricks.md`](16-tips-and-tricks.md) | Best-results guide, per module |
| [`17-appendix.md`](17-appendix.md) | Glossary, shortcuts, file locations, editions table, languages, troubleshooting |

## How to read the edition badges

Every function is tagged with the **minimum edition** that unlocks it. Public edition names:

| Badge | Edition id | Price | 18+ | Automation |
|-------|-----------|-------|-----|------------|
| **Demo** | `free` | Free | — | — |
| **Supporter** | `supporter` | $3 | — | — |
| **Full** | `plus` | €15 | — | — |
| **Unlocked 18+** | `premium` | €25 | Yes | — |
| **Studio Pro** | `studio` | €45 | Yes | REST API |

A badge such as **Full+** means the feature is available in *Full* and every higher edition. The
authoritative capability→edition table lives in [`17-appendix.md`](17-appendix.md).

## Producing the two-column commercial PDF

The Markdown files are the source of truth. To render the polished, **two-column A4 booklet** for
commercial distribution, see [`build/BUILD.md`](build/BUILD.md). It concatenates the chapters, applies
[`build/style.css`](build/style.css) (which uses `column-count: 2` with A4 page geometry) and prints
to PDF via Pandoc or headless Chrome.
