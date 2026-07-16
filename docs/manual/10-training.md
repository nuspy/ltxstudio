# Training Your Own LoRA

*Teach the app new faces, characters, objects, places and styles by training your own LoRA adapters — from a simple images folder, or from a dataset the app builds for you automatically.*

A **LoRA** is a small "add-on" file that steers a base model toward a specific subject or look. Once trained, a LoRA plugs into image and video generation everywhere in **AI Creative Center**: you type its **trigger word** in a prompt and the subject appears, consistently, take after take.

This chapter covers the three training tools the app offers, from the most hands-on to the most automated:

| Tool | Where | What you supply | Output | Edition |
|------|-------|-----------------|--------|---------|
| **Image LoRA training** (`tab.train_tab`) | *Train LoRA* tab | Your own images folder + captions | A FLUX image LoRA | **Studio Pro** |
| **Photo LoRA (auto-dataset)** (`lora_image_tab`) | *Photo LoRA (FLUX.2)* tab | A subject description + a few reference photos | A FLUX.2 image LoRA | **Full+** |
| **Video LoRA (auto-dataset)** (`lora_video_tab`) | *Video LoRA (LTX-2.3)* tab | A subject + reference photos | An LTX-2.3 video LoRA | **Full+** |

> All three train **locally**, inside isolated environments. The heavy generic trainer and both auto-dataset trainers use the **ai-toolkit** engine under the hood. The first time you use any of them, you must run its one-time **Install** step (several GB of download).

The two auto-dataset tools are the same ones the [Cinema — AI Film Studio](09-cinema-studio.md) LoRA Studio drives for character consistency; anything you train here appears in the shared LoRA library and in the Film Studio's LoRA list.

---

## 1. Image LoRA training (bring your own images)

### Image LoRA training
> **Edition:** **Studio Pro** · **Capability:** `cap.train_lora`

**What it does** — Trains a professional-grade image LoRA from a folder of images you already have, with full control over the base architecture, captioning, rank, steps, learning rate and multi-resolution training. This is the "power user" trainer: it does not generate a dataset for you — you provide the pictures.

**Where to find it** — The **Train LoRA** tab (`tab.train_tab`). This tab is only present in the **Studio Pro** edition.

**Step by step**
1. Open the **Train LoRA** tab.
2. Press **Install training module** the first time. Confirm the prompt — *ai-toolkit + PyTorch (cu128)* download into an isolated environment (several GB). Wait for **Training module installed ✓**.
3. Under **Dataset**, click **Browse…** and pick the **Folder** that holds your images. The hint reminds you it should be a *folder with images + matching .txt caption files*. The tab reports **{n} images · {c} captions**.
4. Optionally set a **Trigger word** (for example `ohwx`) — a rare activation token you will later type to summon the subject.
5. If your images have no captions, generate them: choose a **caption engine**, choose a **caption mode**, tick **Overwrite existing** only if you want to replace captions already present, then press **Generate captions**. Use **📝 Review caption** to read and hand-edit each caption in a non-blocking window (it auto-saves).
6. Under **Base model**, pick the **Architecture** and, if needed, fill the **HF repo / folder** and **HF token** (required for gated models such as FLUX.1/2-dev).
7. Set the training **Parameters** (see table). The defaults are production-safe.
8. Enter one **Preview prompt** per line so the trainer renders validation samples as it learns.
9. Press **Start training**. Watch the **Progress** bar, the step counter (**Step {s}/{t}**), the live log and the sample-preview image (refreshed every few seconds). Press **Stop** to abort.
10. When you see **Training complete ✓**, your `.safetensors` LoRA is saved and ready to select in [Images](02-images.md) and [Video](03-video.md).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---------|------|---------|-----------------|-------------------------------|
| **Folder** (Dataset) | Folder picker | — | Any folder | Your source images; each image may have a same-named `.txt` caption. |
| **Trigger word** | Text | *(empty)* | Free text (e.g. `ohwx`) | The rare token you type to invoke the subject. Leave blank for a pure style LoRA. |
| **Caption engine** | Dropdown | First available | **JoyCaption**, **Florence-2** | JoyCaption = best for training (uncensored, tags/descriptive); Florence-2 = lightweight and fast. |
| **Caption mode** | Dropdown | **Descriptive** | **Descriptive**, **Booru tags**, **SD prompt** | JoyCaption caption style. Ignored by Florence-2. Tags suit character/style LoRAs; descriptive suits scenes. |
| **Overwrite existing** | Checkbox | Off | On / Off | Replace captions that are already present. Leave off to only fill in missing captions. |
| **Architecture** (Base model) | Dropdown | First entry | Installed base architectures (FLUX.1, FLUX.2, …) | The model your LoRA attaches to. Must match the model you will generate with. |
| **HF repo / folder** | Text | — | HF id or local path | Where to fetch the base weights. |
| **HF token** | Password | — | Your token | Required only for gated models (FLUX.1/2-dev). |
| **Rank (dim)** | Number | **16** | 2 – 256 | Capacity of the adapter. Higher = more detail but bigger file and more VRAM; 16–32 is a sweet spot. |
| **Alpha** | Number | **16** | 1 – 256 | Scaling factor; commonly set equal to rank. |
| **Steps** | Number | **2000** | 100 – 100000 (step 100) | Training length. More steps = stronger learning but risk of overfitting. |
| **Learning rate** | Editable dropdown | **1e-4** | 1e-4, 2e-4, 5e-5, 1e-3, 8e-5 (or type your own) | How fast it learns. Lower is safer for faces; higher trains faster but can destabilize. |
| **Batch size** | Number | **1** | 1 – 16 | Images per step. Raise only if you have spare VRAM. |
| **Seed** | Number | **42** | 0 – 2,000,000,000 | Fixes randomness for reproducible runs. |
| **Save every** | Number | **250** | 50 – 10000 (step 50) | How often a checkpoint is written, so you can pick an earlier, less-overfit version. |
| **Preview every** | Number | **250** | 50 – 10000 (step 50) | How often validation samples are rendered. |
| **Resolutions** | Checkboxes | **512, 768, 1024 all on** | 512 / 768 / 1024 | Multi-scale training. Keep all on for versatility; drop the largest to save VRAM. |
| **Quantize (less VRAM)** | Checkbox | **On** | On / Off | Trains in a memory-light mode. Turn off only if you have abundant VRAM and want maximum fidelity. |
| **Low VRAM (CPU offload)** | Checkbox | Off (auto-set on low-VRAM machines) | On / Off | Offloads to system RAM. Slower but fits big models on small GPUs. |
| **Also train the text encoder** | Checkbox | Off | On / Off | Learns the trigger more strongly at the cost of flexibility. Leave off for most subjects. |
| **Preview prompts** | Multi-line text | — | One prompt per line | Prompts used to render validation samples during training. |

**Specializations**
- **Caption workflow** — You can train with your own hand-written captions, auto-generate them, or auto-generate then refine in the **📝 Review caption** window. For characters, JoyCaption in **Booru tags** mode with a trigger word usually gives the cleanest identity.
- **Checkpoint picking** — With **Save every** at 250, a 2000-step run leaves you several intermediate LoRAs; if the final one looks "baked", pick an earlier checkpoint.

**Tips for best results**
> - Aim for a clean, varied dataset: many angles, expressions and lighting conditions of the *same* subject; remove blurry or duplicated shots.
> - Always set a **rare trigger word** so the concept attaches to the token, not to a common word.
> - Start from the defaults (rank 16, 2000 steps, LR 1e-4). Only push steps higher if the subject is complex.
> - Keep **Quantize** on unless you see quality issues — it dramatically lowers VRAM use.
> - Use the **Preview prompts** to test *poses your dataset does not contain* — that reveals whether the LoRA generalizes or just memorizes.

---

## 2. Photo LoRA — auto-dataset (FLUX.2)

### Photo LoRA (auto-dataset)
> **Edition:** **Full+** · **Capability:** `cap.lora_image_train`

**What it does** — Trains a **FLUX.2** image LoRA without you having to assemble a dataset. You describe one or more subjects and add a few reference photos; the app **generates** a rich, best-practice image dataset (many views, expressions and lighting), captions it, and trains the LoRA for you.

**Where to find it** — The **Photo LoRA (FLUX.2)** tab (`ilora.train` — "LoRA training (FLUX.2)").

**Step by step**
1. Open the **Photo LoRA (FLUX.2)** tab.
2. Press **Install ai-toolkit** the first time and let the environment download.
3. In **LoRA subjects**, press **➕ Add subject** and fill the **Subject details**: **Type** (Character / Place / Object / Concept), **Name/Trigger** (this name *is* the trigger token), and — for people — tick **Face focus** to learn the face rather than clothes or background.
4. Add **Reference photos**: for each photo choose its **Angle** and press **Add photo…**. More angles produce a more coherent subject.
5. Review the **Automatic coverage** checklist — the cases the app will generate (views, expressions, lighting). Tick or untick to taste. (Motion-only cases are hidden here because this trainer produces still images.)
6. Optionally add **Custom situations** (one per line, unlimited) and any **real material** (extra photos of the subject).
7. Set the **LoRA training** options: **LoRA name**, **Base model**, **Rank**, **Steps**, **Resolution** and **Captioner**.
8. Press **🎬 Generate dataset** to build (and inspect) the dataset only, or **🎓 Generate + Train FLUX.2 LoRA** to run the whole pipeline end to end.
9. Watch the log and the sample-preview image. When finished, the LoRA is saved under the app's `output/loras` folder and registered in the library.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---------|------|---------|-----------------|-------------------------------|
| **Type** | Dropdown | Character | **Character**, **Place**, **Object**, **Concept** | Chooses which best-practice coverage set applies. |
| **Name/Trigger** | Text | — | Free text (e.g. `Martina`) | The subject's name *and* its trigger token. |
| **Face focus** | Checkbox | Off | On / Off | Learns the face only (varies clothes/background across the dataset for robustness). Use for people. |
| **Angle** (per photo) | Dropdown | front | front, three_quarter, profile, back, any | Tags the reference photo so the right one anchors each generated view. |
| **Automatic coverage** | Checkboxes | Balanced default set | Per-type case list | Which views/expressions/lighting the dataset covers. |
| **Custom situations** | Multi-line text | — | One per line, unlimited | Extra scenes to generate (e.g. "runs on the beach"). |
| **LoRA name** | Text | `my_flux_lora` | Free text | Output file name. |
| **Base model** | Dropdown | **FLUX.2 [dev]** | FLUX.2 [dev], FLUX.2 Klein 4B, FLUX.2 Klein 9B, FLUX.1 [dev] | The base the LoRA attaches to. Klein is lighter/faster; [dev] is highest quality. |
| **Rank** | Number | **16** | 4 – 128 | Adapter capacity. |
| **Steps** | Number | **2000** | 200 – 20000 (step 200) | Training length. |
| **Resolution** | Dropdown | **1024** | 768, 1024, 1280 | Dataset/training resolution. Higher = more detail, more VRAM/time. |
| **Captioner** | Dropdown | **Procedural (trigger+tags)** | Procedural, JoyCaption, Florence-2 | How dataset images are described. JoyCaption = realistic VLM captions (best, uncensored); Procedural = trigger + tags, no model. |
| **Caption mode** | Dropdown | Descriptive | Descriptive, Booru tags, SD prompt | JoyCaption style (when JoyCaption is chosen). |

**Specializations**
- **Klein vs [dev]** — FLUX.2 Klein 4B/9B train and run faster and lighter; FLUX.2 [dev] gives the highest fidelity but is the heaviest to train.
- **Dataset-only mode** — **🎬 Generate dataset** lets you build and review the generated images before committing to a full training run.

**Tips for best results**
> - Give **3–6 varied reference photos** at different angles; even for **Face focus**, variety prevents a "frozen" look.
> - Keep **Procedural** captioning for a clean, trigger-anchored identity; switch to **JoyCaption** when you want the LoRA to also understand context and clothing.
> - Prefer **FLUX.2 Klein** for quick iterations, then re-train on **[dev]** for the final asset.

---

## 3. Video LoRA — auto-dataset (LTX-2.3)

### Video LoRA (auto-dataset)
> **Edition:** **Full+** · **Capability:** `cap.lora_video_train`

**What it does** — Trains an **LTX-2.3** *video* LoRA so a subject stays consistent in motion — talking, turning, moving through a shot. The app auto-generates short training **clips** covering many views, expressions and camera moves from your reference photos, then trains the video LoRA.

**Where to find it** — The **Video LoRA (LTX-2.3)** tab (`tlora.train` — "LoRA training").

**Step by step**
1. Open the **Video LoRA (LTX-2.3)** tab.
2. Press **Install trainer** the first time.
3. In **LoRA subjects**, press **➕ Add subject** and set **Type**, **Name/Trigger** and, for people, **Face focus**.
4. Add **Reference photos** (more angles = more coherent video), choosing an **Angle** for each.
5. Tune the **Automatic coverage** checklist — for video this includes motion cases (talking, turning the head, camera pans/dollies, turntables). Add **Custom situations** and optional **real material** (subject videos or photos).
6. Set **LoRA training** options: **LoRA name**, **Rank**, **Steps**, **Clip length**, **PuLID**, **Low VRAM** and **Captioner**.
7. Press **🎬 Generate dataset** to build the clips only, or **🎓 Generate + Train LoRA** for the full run.
8. Watch the log and sample previews. The finished video LoRA is saved and registered in the library.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---------|------|---------|-----------------|-------------------------------|
| **Type** | Dropdown | Character | **Character**, **Place**, **Object**, **Concept** | Selects the coverage set. |
| **Name/Trigger** | Text | — | Free text (e.g. `Martina`) | Subject name and trigger token. |
| **Face focus** | Checkbox | Off | On / Off | Learns only the face; varies outfit and backdrop across clips. |
| **Angle** (per photo) | Dropdown | front | front, three_quarter, profile, back, any | Anchors the correct reference to each generated case. |
| **Automatic coverage** | Checkboxes | Balanced default set (incl. motion) | Per-type case list | Cases to generate; video adds pan/dolly/orbit/turntable/talking/nod/turn-head. |
| **Custom situations** | Multi-line text | — | One per line, unlimited | Extra actions to film (e.g. "dances happily"). |
| **Optional real material** | File list | — | Videos/photos | Real subject footage to strengthen the dataset. |
| **LoRA name** | Text | `my_lora` | Free text | Output file name. |
| **Rank** | Number | **16** | 4 – 128 | Adapter capacity. |
| **Steps** | Number | **2000** | 200 – 20000 (step 200) | Training length. |
| **Clip length** | Number | **3.0 s** | 2.0 – 6.0 s (step 0.5) | Duration of each generated training clip. Longer clips teach motion better but cost more. |
| **PuLID (strong face likeness)** | Checkbox | **On** | On / Off | Locks facial identity during dataset generation. Keep on for people. |
| **Low VRAM** | Checkbox | **On** | On / Off | Memory-light mode. Turn off for maximum speed if you have the VRAM. |
| **Captioner** | Dropdown | Procedural | Procedural, JoyCaption, Florence-2 | How clips/frames are captioned. |

**Specializations**
- **Motion coverage** — The video coverage set adds movement cases (camera **pan**, **dolly**, **orbit**, object **turntable**, and for people **talking**, **nod**, **turn-head**, plus a multi-angle **rotation** clip). These are what make the LoRA hold up *in motion*, not just in stills.
- **Shared subject editor** — The subject editor (types, multi-angle photos, coverage, custom situations, real material, face-focus) is identical to the Photo LoRA tab, so a subject you define one way transfers to the other.

**Tips for best results**
> - Keep **PuLID** on and provide clear frontal + three-quarter reference photos — faces drift most in motion.
> - Start with the default **3.0 s** clip length; raise it only if the subject's action needs more time to read.
> - Leave **Low VRAM** on unless training is too slow and your GPU has headroom.
> - Train a **Photo LoRA first** for a rock-solid still identity, then a **Video LoRA** for motion — this "image → video" order is exactly what the [Film Studio LoRA Studio](09-cinema-studio.md) automates.

---

## Where trained LoRAs go

All three trainers save `.safetensors` files and register them in the app's shared LoRA library. From there you can:

- Select them in the **LoRA** picker on the [Images](02-images.md) and [Video](03-video.md) tabs (filtered by architecture).
- Manage, re-scope and re-train them from the **LoRA Studio** inside the [AI Film Studio](09-cinema-studio.md).
- Import externally trained LoRAs, or download community LoRAs, via [Import models](11-import-models.md).

> **Reminder** — Match the LoRA to the base model it was trained on. A FLUX.2 photo LoRA belongs on FLUX.2 image generation; an LTX-2.3 video LoRA belongs on LTX-2 video generation. The pickers already filter by architecture to keep you on the right track.
