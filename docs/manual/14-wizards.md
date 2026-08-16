# Setup & Guided Wizards

*The three assistants that get you from a fresh install to your first render: the first‑run model setup, the selective installer, and the guided creation wizard.*

AI Creative Center ships with three wizards. Two are one‑time helpers that prepare your models and disk; the third is an everyday assistant that turns a goal ("make a video from a photo") into a ready‑to‑run tab. This chapter documents all three. See also [Getting started](01-getting-started.md).

---

## First‑run Setup Wizard

> **Edition:** **All editions**

**What it does** — On the very first launch, this wizard links the AI models you already have on disk (without copying or re‑downloading them) and then downloads only the weights your edition's modules actually need.

**Where to find it** — Appears automatically at first launch (`ui/setup/first_run.py`). It reuses the app's **Weights** tab inside a guided frame.

**When it appears** — Only when setup has never been completed *and* no base models are already present. It is skipped automatically once either condition changes.

**Step by step**

1. **Base folder** — Point the wizard at the folder where you already keep your AI models.
2. **Scan for existing** — The wizard recursively scans that folder for known model files.
3. **Link found / Import unknown** — For recognized files it offers **"🔗 Link found"**, which creates a filesystem link (a junction or hardlink — **no copy, no re‑download**, and no administrator rights required). For files it does not recognize, it offers **"＋ Import unknown"** so you can classify them.
4. **Select missing + Download** — Finally, tick the weights that are still missing and download only those required by your edition's modules.

Two buttons finish the wizard: **"Done ✓"** (which marks setup complete) and **"Skip for now"** (which also marks it done so it does not reappear on every launch).

> **Tips for best results**
> - Point it at your top‑level models folder (for example `E:\AI\Models`) and let it link everything at once — linking is instant and uses no extra disk space.
> - To bypass the wizard entirely (for example on an automated or headless install), set the environment variable `LTX_NO_SETUP=1`.
> - Both **Done** and **Skip** set the internal `setup_done` flag, so the wizard will not nag you again. You can always reopen model management from the **Weights** tab.

---

## Selective installer

> **Edition:** **All editions** (edition‑aware download)

**What it does** — The selective installer links your local models and downloads **only** the weights for the modules included in your edition, then produces a native, signed installer. It avoids pulling gigabytes you will never use.

**Where to find it** — `installer/install.py`; the models come from the unified registry `data/weights_catalog.json` (`install_role` × edition capabilities). The backend is fully native: no ComfyUI prerequisite and no custom nodes (the optional `comfy_compat` shim for using LTX‑2.3 inside your own ComfyUI lives in the **Import models** tab).

**How it works**

1. **Detect the edition** — The installer reads your edition manifest (or an explicit `--edition` override) to learn which modules and adult‑mode weights are allowed.
2. **Check prerequisites** — Python 3.10+, NVIDIA driver + CUDA, a compatible PyTorch, and ffmpeg are verified up front.
3. **Build the plan (only‑wanted)** — From the weights catalog it selects only the models your edition wants: `editions` filters per tier, adult‑sensitivity weights download only when your edition allows Unlocked (18+) mode, and `on_demand` weights are deferred to first use rather than downloaded now.
4. **Show a sized summary** — Before anything is downloaded you see the full plan with estimated sizes and are asked to confirm. `--dry-run` prints the plan without downloading.
5. **Install** — Base weights for your edition are fetched; large on‑demand models (FLUX.2 encoder, FLUX Fill, TRELLIS.2, SkinTokens, SeedVR2, and so on) are pulled the first time you use the feature that needs them.

**Native installers & signing** — Distribution builds wrap the app in a native installer per platform — **Inno Setup** on Windows, a **DMG** on macOS, and an **AppImage** on Linux — each code‑signed for its platform so it installs without security warnings.

> **Tips for best results**
> - Run with `--dry-run` first to see exactly what will be downloaded and how large it is.
> - Because on‑demand weights are deferred, expect a one‑time download the first time you open a heavy module (3D, video upscale, world generation). This is normal.
> - Your edition determines which weights are eligible: higher tiers unlock more modules and therefore more downloadable weights. See the [capability table](17-appendix.md).

---

## Guided creation Wizard

> **Edition:** **All editions** (individual goals still respect your edition's capabilities)

**What it does** — Takes you from "what do you want to create?" to the correct tab with the right fields pre‑filled. It is the fastest way to start when you are not sure which tab to open.

**Where to find it** — `ui/wizard.py`. Open it from the main window; it applies your choices by switching to the correct tab (or opening AI Film Studio) and filling in the fields.

**Page 1 — Choose your goal**

Pick one of nine goals and set the content mode (a standard/Unlocked toggle). A contextual description explains each goal as you select it.

| Goal (label) | What it produces | Needs a starting image? | Audio mode | Routes to |
|---|---|---|---|---|
| **Text → Video** | A video from a text description. | No | Native | Video tab |
| **Image → Video** | Animates a starting image into a video. | Yes | Native | Video tab |
| **Image → Video + voice (lipsync)** | A photo plus a cloned voice, with synced lips. | Yes | Clone | Video tab |
| **Audio → Video** | A video driven by an audio track. | No | User‑supplied audio | Video tab |
| **Text → Image** | An image from a text description. | No | — | Images tab |
| **Image → 3D model** | Turns a photo into a 3D mesh/GLB. | Yes | — | 3D Model tab |
| **Generate music / song** | Composes a song or an instrumental track. | No | Native | Music tab |
| **Clone voice / speech synthesis** | Synthesizes or clones a voice from a sample. | No | Native | Audio/Voice tab |
| **AI Film Studio (text → film)** | A full film: storyboard, scenes, editing. | No | Native | AI Film Studio |

**Page 2 — Prompt & duration**

The second page adapts to the goal you chose:

1. A **prompt** box (shown only for goals that need a description — hidden, for example, for Image → 3D).
2. A **duration** field in seconds (shown only for goals where length matters — video and music). Its range is **1–60 s**, defaulting to **8 s**.
3. A blue **contextual hint** telling you where the choice will land (for example, "The prompt and length will be applied to the Video tab").

When you finish, the wizard returns your goal, category, content mode, whether an image is needed, the audio mode, your prompt, and the length — and the main window applies them.

> **Tips for best results**
> - The content‑mode toggle (standard vs Unlocked 18+) is only effective if your edition includes `cap.adult_mode`; otherwise it stays in standard mode.
> - The wizard fills fields but does not lock them — after it hands you to a tab you can fine‑tune everything before generating.
> - For **Image → Video + voice**, prepare your voice sample first (see [Audio & voice](05-audio-voice.md)); the wizard sets the clone audio mode but you still supply the sample in the Video tab.
