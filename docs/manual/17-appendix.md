# Appendix

*Reference material: glossary, shortcuts, file locations, languages, the capability matrix, editions, and troubleshooting.*

---

## Glossary

| Term | Meaning |
|---|---|
| **LoRA** | A small add‑on file that adapts a base model toward a style, subject, or effect, at an adjustable strength. |
| **Checkpoint** | A full base model — the main set of weights that generates images or video. |
| **VAE** | Variational auto‑encoder; converts between the compact "latent" space the model works in and real pixels. |
| **Text encoder** | The component that turns your prompt text into the conditioning the model understands (e.g. Gemma, Qwen3, Mistral). |
| **CFG** | Classifier‑free guidance; how strongly the model follows the prompt. LTX video uses CFG 1.0 (guidance comes from NAG). |
| **Steps** | How many denoising iterations run; more steps = more refinement and more time. |
| **Sampler** | The algorithm that removes noise each step (e.g. `euler_ancestral_cfg_pp`). |
| **Scheduler** | The schedule of noise levels across steps (e.g. `LTXVScheduler (shift)`); strongly affects quality. |
| **t2v / i2v / a2v** | Text‑to‑video / image‑to‑video / audio‑to‑video generation modes. |
| **Inpaint / Outpaint** | Regenerate a masked region inside an image (inpaint) or extend the image beyond its edges (outpaint). |
| **ControlNet** | Guides generation with a structural reference (canny edges, depth, or pose). |
| **Upscale** | Increase resolution while adding plausible detail. |
| **Quantization (fp8 / fp4 / gguf)** | Compressed weight formats that cut VRAM use with small quality loss. |
| **PuLID** | A method that transfers a face's identity from a reference photo onto a generated image. |
| **SAM2** | Segment Anything 2; produces precise object masks, and tracks them across video frames. |
| **Keyframe** | A fixed image pinned to a specific frame/time to anchor a video. |
| **Storyboard** | A sequence of timed keyframes that produce one continuous video without manual editing. |
| **Mocap** | Motion capture; driving a rig from live webcam motion. |
| **Rigging** | Adding a skeleton to a 3D mesh so it can be posed and animated. |
| **Skinning** | Binding a mesh's surface to its skeleton so it deforms correctly when posed. |
| **Splat** | A Gaussian‑splat representation of a 3D scene (points with color and shape), used by world generation. |
| **OTIO** | OpenTimelineIO; an interchange format for editing timelines. |

---

## Keyboard shortcuts

| Shortcut | Action |
|---|---|
| **Ctrl+G** | Open the Navigator |

Beyond this, AI Creative Center is driven through its buttons, tabs, and context menus rather than keyboard shortcuts.

---

## File locations

| What | Where |
|---|---|
| Generated outputs | `output/` (and API uploads in `output/api_uploads/`) |
| Models root | `E:\AI\Models` (organized by type: `diffusion_models`, `TextEncoders`, `LoRA`, `VAE`, `Checkpoints`, …) |
| GUI state (tab settings, scroll positions) | `data/gui_state.json` |
| User preferences (favorites, provenance signature settings, custom presets) | `data/user_prefs.json` |
| Weights configuration / first‑run flag | `data/weights_config.json` |
| AI Film Studio project database | `core/film_studio/studio.db` (SQLite) |
| Hugging Face cache (one‑time gated downloads) | `E:\hf_cache` (via `HF_HOME`) |

> The app is 100% local at runtime. A Hugging Face token is used only for one‑time downloads of gated weights, never for cloud inference.

---

## Supported languages

The interface is fully translated into **15 languages**:

Arabic · German · Greek · English · Spanish · French · Hebrew · Hindi · Hungarian · Italian · Korean · Portuguese · Russian · Ukrainian · Simplified Chinese.

Switch language from the interface settings; the whole UI, including tooltips, is localized.

---

## Capability → minimum edition

| Feature | Capability | Min edition |
|---|---|---|
| Image generation (Demo capped 768 px/20; Supporter 1024 px/200; Full+ unlimited) | `cap.gen_images` | Demo |
| Invisible provenance signature | `cap.stego` | Demo (all) |
| Forced brand watermark on outputs | `cap.watermark` | Demo & Supporter only (removed Full+) |
| Image upscale | `cap.upscale_image` | Supporter |
| Audio / TTS / voice clone | `cap.audio_tts_clone` | Full |
| Video (t2v / i2v / a2v) | `cap.video` | Full |
| Extend video + ReTake | `cap.extend_retake` | Full |
| ControlNet | `cap.controlnet` | Full |
| Inpaint / Outpaint / Recolor | `cap.inpaint` | Full |
| Object segmentation (SAM) | `cap.sam` | Full |
| Enhance quality / restore / super‑res / face | `cap.restore` | Full |
| Song / instrumental | `cap.music_song` | Full |
| Video sonification | `cap.music_sonify` | Full |
| Music video | `cap.music_video` | Full |
| Image→3D model | `cap.model3d` (`cap.threed`) | Full |
| Rig 3D | `cap.rig3d` | Full |
| Mocap animation | `cap.mocap` | Full |
| 3D World (WorldGen) | `cap.world_gen` | Full |
| AI Film Studio (+ asset/script/scenografia/montaggio/voice/lora sub‑caps) | `cap.filmstudio` | Full |
| Auto‑dataset photo LoRA | `cap.lora_image_train` | Full |
| Auto‑dataset video LoRA | `cap.lora_video_train` | Full |
| HDR/EXR frame export | `cap.exr_export` | Full |
| Unlocked (18+) mode | `cap.adult_mode` | Unlocked 18+ (premium) |
| Import own checkpoints/LoRA | `cap.import_models` | Studio Pro |
| Civitai import (search + download) | `cap.civitai_import` | Studio Pro |
| Per‑node model overrides | `cap.node_overrides` | Studio Pro |
| Heavy generic FLUX LoRA training tab | `cap.train_lora` | Studio Pro |
| Local REST API | `api.rest` | Studio Pro |

---

## Editions

| Edition | Price | Unlocked (18+) | Automation |
|---|---|---|---|
| **Demo** | €0 | — | — |
| **Supporter** | $3 | — | — |
| **Full** | €15 | — | — |
| **Unlocked 18+** | €25 | Yes | — |
| **Studio Pro** | €45 | Yes | Local REST API |

Each edition includes everything in the tiers below it. See the capability table above for exactly which features each tier unlocks.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| **Out of memory (OOM)** | Resolution/steps too high, or a large unquantized model. | Lower resolution or steps; switch to a quantized model (fp8/fp4/gguf); use **/free** to release VRAM before the next job. |
| **Very slow first run** | On‑demand weights are downloading. | Wait for the one‑time download to finish; later runs are fast. |
| **Green / cyan noise in video** | Wrong sampler for LTX. | Use **`euler_ancestral_cfg_pp`** with `LTXVScheduler (shift)`; avoid RES samplers on video. |
| **Plastic‑looking faces** | Distilled/quantized model, or reference too loose. | Use a **base bf16** model and raise the image/reference strength (PuLID ~0.8). |
| **A model does not appear in the dropdown** | It is not linked or catalogued yet. | Rescan in the **Import Models** / **Weights** tab so the detector picks it up. |

> For deeper tuning of samplers, schedulers, and per‑module settings, see [Tips & tricks](16-tips-and-tricks.md) and [Advanced settings](12-advanced.md).
