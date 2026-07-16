# Getting started

*Your on-ramp to AI Creative Center: what it is, what your machine needs, how to install and launch it, how editions unlock features, and how to find your way around the interface.*

Welcome to **AI Creative Center** — a complete, self-contained creative studio that generates images, video, music, voices, 3D models, and entire films on your own computer. This chapter gets you from a fresh install to confidently navigating the app. When you're ready to create, jump to the module chapters: [Images](02-images.md), [Video](03-video.md), [Audio & voice](05-audio-voice.md), [Music](06-music.md), [3D](08-3d.md), and [Cinema — AI Film Studio](09-cinema-studio.md).

---

## What AI Creative Center is

AI Creative Center is a **100% local, offline** desktop application. Every generation — every image, frame of video, note of music, and 3D mesh — is computed on your own GPU. Nothing you create is uploaded to a cloud service, and no account or internet connection is needed to *use* the app.

There are only two optional moments where the app touches the network, and both are under your control:

1. **One-time model-weight downloads** — the first time you use a module, its AI model weights may need to be fetched. Once downloaded, they live on your disk and are reused offline forever after. The setup wizard handles this; see [Wizards](14-wizards.md).
2. **Civitai import (Studio Pro only)** — the *Studio Pro* edition can search and download community models from Civitai. This is an opt-in feature of the [Import models](11-import-models.md) tab, not something the app does on its own.

Everything else — including the language you read this in, the models you run, and the files you produce — stays on your machine.

---

## System requirements & GPU / VRAM notes

AI Creative Center runs on Windows with an **NVIDIA GPU**. The single most important factor for what you can do — and how fast — is your GPU's **VRAM**.

- **Image generation** is the lightest workload and runs on modest cards.
- **Video, 3D, and film** are the heaviest and reward more VRAM (24 GB is comfortable for the largest models).
- The app measures your GPU's memory without ever creating a CUDA context in the interface process, so the main window stays stable even when heavy sub-processes are running.

### Running on limited VRAM (quantization)

You do **not** need a top-tier card to run large models. AI Creative Center supports several quantized weight formats that trade a small amount of quality for dramatically lower memory use, and it can **auto-select the right format for your VRAM**:

| Format | Typical VRAM | Quality vs. full precision | Good for |
|---|---|---|---|
| NVFP4 (mixed) | ~14 GB | Almost identical | 32 GB cards (fastest) |
| FP8 scaled (mixed) | ~16–18 GB | Almost identical | Universal compromise |
| GGUF Q8_0 | 14–16 GB | 98–99% | 16 GB cards |
| GGUF Q6_K | 11–13 GB | ~95% | 12 GB cards |
| GGUF Q5_K_M | 9–11 GB | ~90% | 8 GB cards (sweet spot) |
| GGUF Q4_K_M | 7–9 GB | 75–85% | 8 / 6 GB cards |

VRAM-saving techniques (text-encoder offload, VAE offload, block-swap, model unloading between jobs) stack on top of quantization. For a full walkthrough of running the largest image model on constrained hardware, see **`docs/FLUX2_LOW_VRAM.md`**. You can also cap the memory the app is allowed to use and let it place models automatically — see **VRAM calibration** in [Advanced settings](12-advanced.md).

---

## Installation & first launch

1. Run the installer for your platform and follow the prompts. The installer is **selective**: it can link the models you already have on disk and download only the weights your edition actually needs, keeping the footprint small.
2. Launch AI Creative Center. On first run, the **setup wizard** opens automatically to connect your local models and fetch any missing weights. Follow it end to end — it is the fastest path to a working install. Full details are in [Wizards](14-wizards.md).
3. When the wizard finishes, the main window opens and you are ready to create.

You can re-open the wizard at any time from the **Tools ▸ Wizard** menu.

---

## Editions & upsell

AI Creative Center ships as a single application whose features unlock according to your **edition**. Higher editions include everything below them.

| Edition | Price | What it unlocks |
|---|---|---|
| **Demo** (`free`) | €0 | Image generation only (capped at 768 px and 20 generations), invisible provenance signature. Outputs carry a visible **"AI Tools Forge"** brand watermark. |
| **Supporter** | $3 | Adds image upscaling; image caps raised to 1024 px / 200 generations. Brand watermark still applied. |
| **Full** (`plus`) | €15 | Unlocks video, audio/voice, music, 3D, Film Studio, quality tools, LoRA auto-training, and the Advanced tab. Brand watermark removed. |
| **Unlocked 18+** (`premium`) | €25 | Everything in *Full* plus the **Unlocked (18+) mode** (`cap.adult_mode`): uncensored models and text encoders. |
| **Studio Pro** (`studio`) | €45 | Everything, plus model import (`cap.import_models`), Civitai import (`cap.civitai_import`), per-node model overrides (`cap.node_overrides`), the heavy FLUX LoRA training tab, and the local REST API. |

Throughout this manual, a trailing **+** on an edition badge (e.g. *Full+*) means "this edition and every higher one."

### How locked features appear

Locked features are never hidden away — they are shown in the interface with an **upsell** prompt so you always know what the next edition unlocks:

- A tab or panel you haven't unlocked shows a teaser: *"This feature is ready: unlock it and use it right away, in the same window."*
- A locked control shows a strip reading **🔒 {feature} — Available in the {tier} plan** with an **Unlock →** button.
- Choosing to unlock opens a dialog explaining that the feature is **already installed and ready** — switching to the indicated plan enables it immediately, with an **🔓 Unlock on Patreon** button.

Because features are already installed, upgrading is instant: no re-download, no reinstall.

### Checking your edition

Your active edition drives which tabs and functions are visible. Feature gating is applied at startup, so if a module is missing it is because your edition doesn't include it — the upsell strip will tell you the minimum tier. (In developer builds, everything is visible.)

---

## The interface

When the main window opens you'll see three areas: the **tab navigation** across the top, the **work area** (with a live canvas/preview on the left), and the **generation queue** dock on the right.

### The tab bar

Navigation is a set of large, wrapping buttons (each with an emoji) that switch between modules. In order, the modules are:

| Emoji | Module | What it does |
|---|---|---|
| 🖼️ | **Images** | Generate, inpaint/outpaint, recolor, upscale, segmentation, face adapter, compose, face-swap → [Images](02-images.md) |
| 🎙️ | **Audio / Voice** | TTS, voice cloning, the shared voice library → [Audio & voice](05-audio-voice.md) |
| 🎬 | **Video** | Text/image/audio-to-video, presets, LoRA, sampler/seed, storyboard → [Video](03-video.md) |
| 🎛️ | **ControlNet** | Guided video generation (canny/depth/pose) → [Video](03-video.md) |
| ⏩ | **Extend video** | Extend an existing clip → [Video](03-video.md) |
| ✨ | **AI Video Edit** | In-place video editing, tracking, inpaint → [Video](03-video.md) |
| 💎 | **Enhance quality** | Photo restore/upscale/colorize and video super-resolution → [Enhance quality](04-quality-restore.md) |
| 🧊 | **3D Model** | Image → textured 3D model (TRELLIS.2) → [3D](08-3d.md) |
| 🦴 | **3D Rig** | Add a skeleton to a mesh → [3D](08-3d.md) |
| 🔄 | **Converter** | Convert 3D formats → [3D](08-3d.md) |
| 🕺 | **Animation** | Animate a rig from your webcam (mocap) → [3D](08-3d.md) |
| 🎓 | **LoRA Train Img.** | Train an image LoRA → [Training](10-training.md) |
| 🎞️ | **LoRA Wizard Vid.** | Auto-dataset video LoRA → [Training](10-training.md) |
| 🖌️ | **LoRA Wizard Img.** | Auto-dataset photo LoRA → [Training](10-training.md) |
| 🎵 | **Music** | Songs, instrumentals, and video sonification → [Music](06-music.md) |
| 🌍 | **3D World** | Explorable 3D scene from a photo → [3D](08-3d.md) |
| 📦 | **Model weights** | Download and manage model weights → [Wizards](14-wizards.md) |
| ⚙️ | **Advanced** | Sampler/scheduler settings, long-video engine, tools, and (Studio Pro) Import Models → [Advanced](12-advanced.md) |

The **Enhance quality** tab further splits into **🖼 Photo** and **🎬 Video** sub-tabs, and the **Advanced** tab splits into **🛠 Tools** and **📦 Import Models**.

The tabs you see depend on your edition — the Demo edition, for example, shows only the Images module.

### The live canvas / preview

The Images module's left-hand **canvas** doubles as your live preview and your editing surface: it shows the current image, and it's where you paint masks for inpainting, isolate subjects, and feed results into other modules (for example "Use generated image" in the [3D](08-3d.md) tab).

### The generation queue dock

Generations run through a **serial queue** — one job at a time, so jobs never fight over your GPU. The **Generation queue** dock on the right lists every job with a status icon:

| Icon | Status |
|---|---|
| ⏳ | queued |
| ▶ | running |
| ✅ | done |
| ❌ | error |
| ⛔ | cancelled |

You can **Cancel selected** to stop a queued or running job and **Clear finished** to tidy the list. When the queue is empty it reads *"No generations queued."* Toggle the dock's visibility with the 📋 button in the navigation bar. Job labels are descriptive (for example *Image*, *Video*, *Super-resolution (SeedVR2)*, *3D model (TRELLIS)*, *Lip-sync*).

### The Tools menu

The **Tools** menu (labeled **Strumenti**) collects the app-wide utilities:

- **Wizard** — re-run the setup wizard ([Wizards](14-wizards.md)).
- **Settings** — the model/engine settings dialog (below).
- **🗂 Navigator Gallery…** (Ctrl+G) — browse past generations with metadata and parameter reuse.
- **🔍 Analyze settings…** — check your current parameters for conflicts.
- **🎬 Workflow presets…** — 35 ready-made workflows.
- **🎬 AI Film Studio** — open the film studio ([Cinema](09-cinema-studio.md)).
- **🎨 Interface** — switch the interface variants (below).

---

## Classic vs. Pro interface variants

From the **Tools ▸ 🎨 Interface** menu you can switch two modules between a **Classic** and a **Pro ✨** variant, independently:

- **Models Tab** — Classic / Pro ✨
- **AI Film Studio** — Classic / Pro ✨

The **Pro** variants add refinements on top of the Classic ones: the Pro Import Models tab introduces a segmented layout, lock badges, and lazy-loading thumbnails ([Import models](11-import-models.md)), while the Pro Film Studio adds workflow improvements ([Cinema](09-cinema-studio.md)). Both variants **respect the feature matrix** identically — switching interface style never changes what your edition unlocks. The switch happens live, without restarting the app.

---

## Languages

AI Creative Center's interface is available in **15 languages**:

Arabic (العربية) · German (Deutsch) · Greek (Ελληνικά) · English · Spanish (Español) · French (Français) · Hebrew (עברית) · Hindi (हिन्दी) · Hungarian (Magyar) · Italian (Italiano) · Korean (한국어) · Portuguese (Português) · Russian (Русский) · Ukrainian (Українська) · Simplified Chinese (中文).

**To switch language**, click the flag button in the status bar (bottom of the window) and pick your language from the popup. A confirmation appears — *"The language will be applied the next time the application starts."* — so restart the app to see the new language everywhere. Your choice is remembered between sessions.

---

## Automatic state persistence

AI Creative Center remembers your work between sessions so you never start from a blank slate:

- **Tab values** — the settings of every control (dropdowns, text fields, numbers, checkboxes, sliders) across all tabs are saved and restored automatically.
- **Window geometry and scroll position** are restored.
- **Favorites and custom presets** — starred LoRAs and models, your saved per-LoRA strengths, and your custom video presets come back exactly as you left them.

This is fully automatic; there is no "save layout" button to remember. (For reproducible *generations*, use **Fixed seed** in [Advanced settings](12-advanced.md).)

---

## The Settings dialog

Open it from **Tools ▸ Settings**. It has three sections:

### Generation engine (native)

Choose how models are loaded. **ComfyUI-native** uses dynamic VRAM management that avoids out-of-memory errors; the alternatives use native pipelines.

| Control | Type | Default | Options | What it does |
|---|---|---|---|---|
| **Image engine:** | Dropdown | ComfyUI-native | ComfyUI-native (dynamic VRAM, no-OOM) · Diffusers (native pipeline) | How image models are loaded. |
| **Video engine:** | Dropdown | ComfyUI-native | ComfyUI-native (dynamic VRAM, no-OOM) · LTX-2 official (ltx2_infer) | How video models are loaded. |

### Provenance

| Control | Type | Default | What it does |
|---|---|---|---|
| **Invisible provenance signature (recommended)** | Checkbox | On | Embeds an invisible, imperceptible signature into every generated photo and video, verifiable with the ATF Detector, without visibly altering the output. See [Provenance signature](13-steganography.md). |
| **Signature text (max 5)** | Text (≤5 chars) | empty | An optional custom code/handle embedded in the signature. Empty = a valid signature without text. The 5-character limit keeps the mark robust to screenshots and compression. |

### Model overrides (ComfyUI template) — Studio Pro

> **Capability:** `cap.node_overrides`

For each loader node in the current template you can pick a specific model file from the files available on your system. This is a *Studio Pro* power-user feature; if the template has no loader nodes you'll see *"No loader node in the current template."*

**Where your settings live:** the app stores your preferences in `data/user_prefs.json` (favorites, per-LoRA strengths, custom presets, engine and signature choices), tab/window state in `data/gui_state.json`, and your language in `data/gui_lang.json`.

---

*Next: dive into [Images](02-images.md) to make your first generation, or read [Advanced settings](12-advanced.md) to understand the sampler, VRAM, and long-video controls. The [Appendix](17-appendix.md) has the full capability→edition table, file locations, and troubleshooting.*
