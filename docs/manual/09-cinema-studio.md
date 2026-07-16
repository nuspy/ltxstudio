# Cinema — AI Film Studio

*Turn a book, a short story, a one-line idea, or a random theme into a finished film — screenplay, storyboard, generated shots, voices, music and a final master or NLE project — all inside one guided, non-destructive workspace.*

The **AI Film Studio** (**🎬 AI Film Studio**) is the flagship module of **AI Creative Center**. It orchestrates every generative engine in the app — image, video, voice, music, sound effects, lip-sync and LoRA training — behind a single, linear, film-production workflow: **Script → Set design → Montage → Master**. Every stage is re-enterable, every artifact is regenerable, and nothing is ever silently overwritten: each regeneration creates a new **version** you can restore.

> **Edition:** **Full+** · **Capability:** `cap.filmstudio`
>
> The whole module, and every sub-module inside it, requires the **Full** edition or higher. Individual sub-capabilities are noted per section: Script `cap.filmstudio_script`, Asset/Characters `cap.filmstudio_asset`, Set design/Storyboard `cap.filmstudio_scenografia`, Montage `cap.filmstudio_montaggio`, Voice `cap.filmstudio_voice`, LoRA Studio `cap.filmstudio_lora`.

---

## Opening the Film Studio

The Film Studio opens in its own window titled **🎬 AI Film Studio** (in the Pro interface, **🎬 AI Film Studio — Pro ✨**). Across the top runs the **project bar**; below it are the workflow tabs, in order:

1. **Script** (*Copione*)
2. **Set design** (*Scenografia* — storyboard)
3. **Montage** (*Montaggio*)
4. **Asset Manager**
5. **Voice**
6. **LoRA Studio**

A beginner **Flow** banner reminds you of the path: *① Script (idea → Story → screenplay) → ② Storyboard (keyframe → video) → ③ Editing. Use @Name for asset consistency.* (The banner is hidden in the Pro interface, which shows a compact flow strip instead.)

Two global run-modes govern how much the app asks you before proceeding:

- **Supervised** — every stage waits for your explicit approval (the professional default).
- **Auto** — the app advances stage to stage with sensible defaults, stopping only on strong errors, then presents the result for final review. Even in Auto, every artifact stays regenerable.

---

## Projects

### Create, open and manage projects
> **Edition:** **Full+** · **Capability:** `cap.filmstudio`

**What it does** — A project is one film (or one episode of a series). It stores the input, screenplay, scenes, shots, continuity, timeline and every generated artifact.

**Where to find it** — The **project bar** at the top of the Film Studio window, and the **📂 Projects** dialog (*Projects — AI Film Studio*).

**Step by step**
1. Type a name in the **New project…** field.
2. Set the **Duration**, the **Coherence** path, and the run-mode.
3. Press **➕ Create**. The project is added to the selector as *Name (#ID)*.
4. To reopen or manage projects later, press **📂 Projects**, select one, and use **📂 Open**, **📋 Duplicate** or **🗑 Delete**.
5. Use the small **✏️ Rename**, **⧉ Duplicate**, **🗑️ Delete** buttons in the project bar for quick actions on the current project.
6. Optionally attach the project to a **Series** (choose from the selector or press **＋** to create one) so it can share cast and LoRAs with other episodes.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---------|------|---------|-----------------|-------------------------------|
| **New project…** | Text | — | Free text | The project name. |
| **Duration** | Number (seconds) | **60 s** | 5 – 7200 s | Target film length. Scenes and shots are constrained to sum to this. |
| **Coherence** (path_mode) | Dropdown | **Fast** | **Fast**, **LoRA**, **Mixed** | How character/asset consistency is achieved (see below). |
| **Run mode** | Dropdown | **Supervised** | **Supervised**, **Auto** | Whether the app pauses for approval at each stage. |
| **Series** | Dropdown | — none — | Existing series or **＋** | Groups episodes that share cast/locations/LoRAs. |
| **forbidden (negative)** | Text | — | Free text | Project-wide negative elements injected into every prompt. |
| **must-have (always)** | Text | — | Free text | Project-wide required elements injected into every prompt. |

**Specializations — the three coherence paths**
- **Fast** — no training. Consistency comes from the asset's **reference images** plus **PuLID** for faces and multi-reference conditioning. Instant, zero training cost; ideal for drafts, animatics, minor assets and short projects.
- **LoRA** — a trained **LoRA** (image and video) per asset gives strong, repeatable identity. `@Name` resolves to the LoRA's **trigger** token. Best for a main cast, series and final quality.
- **Mixed** *(recommended)* — the main cast on **LoRA**, extras on **Fast**, in the same job. Each asset can declare its own path.

**Tips for best results**
> - Start every serious production in **Supervised** mode so you can approve the screenplay before spending GPU time on shots.
> - Use **Series** for anything episodic — the cast LoRAs you train once carry over automatically.
> - Fill **must-have** and **forbidden** early (e.g. an aspect look you always want, or artifacts you never want) — they are injected everywhere.

---

## Script (Copione) tab

### Write or generate the screenplay
> **Edition:** **Full+** · **Capability:** `cap.filmstudio_script`

**What it does** — Turns any of four input kinds into a cinematic **screenplay**, opening with a short **plot summary**, respecting your **duration**, **genre(s)** and **style(s)**, and supporting `@Name` asset references throughout.

**Where to find it** — The **Script** (*Copione*) tab.

**Step by step**
1. Pick an **Input type** and enter your material in the input box (*Paste the story, idea or theme… (@Name for assets)*).
2. Choose one or more **Style** and **Genre** values (both fields accept comma-separated multiples and are editable).
3. (Optional) Open the **🎲 Writers' table (branches/beats)** to co-write interactively: the LLM proposes story beats with alternative branches; pick a branch per beat and it weaves them into the Story.
4. (Optional) Press **📖 Generate Story** to produce an editable cinematic **prose** treatment first.
5. Press **✍️ Generate script** to produce the screenplay. It appears with a **Plot summary** on top and the editable **Script** below.
6. Refine freely by hand, right-click a passage to **regenerate a segment** (similar, or "what to change"), or press **↻ Regenerate** for a fresh full pass with a new seed.
7. Press **💾 Save**. The screenplay is versioned (**Script saved (v{n})**).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---------|------|---------|-----------------|-------------------------------|
| **Input type** | Dropdown | Full story | **Full story**, **Mini-idea**, **Random theme**, **Interactive** | How the LLM treats your input (reduce / expand / invent / build step by step). |
| **Style** | Editable dropdown | — | ~60 visual styles (comma-separated) | Visual look (e.g. noir, minimalist, hyperreal). |
| **Genre** | Editable dropdown | — | ~23 genres (comma-separated) | Genre(s); can be combined (e.g. comedy + horror). |
| **Story input** | Multi-line text | — | Free text, supports `@Name` | Your source material. |

**Specializations — the four input kinds**
- **Full story** — a book/short story: the LLM reduces it to cinematic format.
- **Mini-idea** — a short premise: the LLM expands it.
- **Random theme** — just a theme: the LLM invents and expands.
- **Interactive** — build the story conversationally; the **🎲 Writers' table** proposes branching beats you weave into the Story.

The tab exposes three output areas: **Cinematic story** (editable prose), **Plot summary**, and the editable **Script**. The **@Name** system works in every text area — type `@` to open the asset picker; type `#` after a name to jump to its variants.

**Tips for best results**
> - Reference recurring characters and places with `@Name` from the first draft — the breakdown and continuity engine rely on them.
> - Use **📖 Generate Story** before **✍️ Generate script** when you want to lock the narrative tone before the technical treatment.
> - Regenerate *segments* rather than the whole screenplay when you only want to fix one beat — you keep everything else intact.

---

## Set design / Storyboard (Scenografia) tab

### Break down, storyboard and shoot the film
> **Edition:** **Full+** · **Capability:** `cap.filmstudio_scenografia`

**What it does** — Turns the approved screenplay into **Scenes**, each split into **Shots**, then generates the **keyframe** images and the **shot videos**, all governed by the continuity engine. This is where the film is actually rendered.

**Where to find it** — The **Set design** (*Scenografia*) tab.

**Step by step**
1. Press **🎬 Scene/shot breakdown (LLM)**. The scene/shot tree fills with **S1, S2…** scenes and their shots, each with a duration.
2. Select the **Image model** and **Video model** (both default to **(auto)**).
3. Press **🖼 Generate keyframes** to render the opening (and intermediate/final) reference images for every shot.
4. Press **🎥 Generate videos** to render the shot clips (LTX-2.3), conditioned on the keyframes and, in the LoRA path, on the subjects' video LoRAs.
5. Or press **🎬 Generate full film** to queue the entire keyframe + video pipeline at once (a cost estimate is shown; then go to **Montage → Master**).
6. Use the per-shot/scene panel to edit descriptions (with `@Name`), set the **Seed**, and regenerate keyframe or video individually.

**Key controls**

| Control | Type | Default | Range / Options | What it does & when to change |
|---------|------|---------|-----------------|-------------------------------|
| **Image model** | Dropdown | **(auto)** | Installed image models | Which checkpoint renders keyframes. |
| **Video model** | Dropdown | **(auto)** | Installed video (GGUF) models | Which transformer renders shot videos. |
| **🔗 Continuous match** | Checkbox | Off | On / Off | Anchors each shot to the previous shot's last frame (same scene) for cuts without a hard jump. |
| **Seed (0=auto/lock)** | Number | 0 | 0 – 2³¹−1 | Per-shot seed; lock a value for reproducible regenerations. |
| **Shot type** | Dropdown | (auto) | wide, close-up, medium, insert, … | Shot grammar for the selected shot. |
| **Camera move** | Dropdown | (auto) | pan, tilt, dolly, zoom, … | Camera motion for the selected shot. |
| **Transition** | Dropdown | cut → | cut, fade, in black, dissolve, wipe | Transition to the next shot. |
| **In / Out (trim)** | Number (s) | 0 | 0 – 3600 s (step 0.1) | Trim the shot's usable range for the timeline. |

**Specializations — creative assistants**
- **🎨 Style Bible** — define the film's **emotional logline**, **color palette**, **look (grain/optics/light)** and **reference directors/DoP**. Once saved, it is injected into every prompt.
- **🧐 Continuity supervisor** — re-reads the breakdown and flags continuity jumps (180°/eyeline, props, time of day, conditions).
- **🎥 DoP: suggest shot** — the director of photography proposes camera + lighting per shot for the selected scene, drawn from the Style Bible; apply the ones you like (writes `shot.camera`).
- **🎬 Expand dialogue coverage** — turns a multi-line dialogue shot into proper shot-reverse-shot coverage (one shot per line, alternating over-the-shoulder / clean single angles).
- **🎞 Take history** — every keyframe/video regeneration is kept as a numbered take you can restore; nothing is lost.
- **📄 Shot-list** — export the breakdown as a production shot-list.

**Chain-linking for smooth cuts** — With **🔗 Continuous match** on, the app extracts the last frame of shot *N* and feeds it as the frame-0 conditioning of shot *N+1* (at a low strength, so motion still evolves), giving raccord cuts within a scene.

**Tips for best results**
> - Set the **Style Bible** before generating anything — it is the single biggest lever on a consistent look.
> - Run the **🧐 Continuity supervisor** after the breakdown and again after editing shots, before committing GPU time to videos.
> - Keyframe first, review, *then* video. Regenerating a keyframe is far cheaper than a bad video.
> - Turn on **🔗 Continuous match** only within continuous action; across a hard cut you usually want a fresh frame.

---

## Continuity engine

The continuity engine makes consistency an **explicit state**, not a lucky side effect of prompting.

**Per-scene / per-shot state fields**

| Field | Default | Options / Notes |
|-------|---------|-----------------|
| **Time of day** | day | day, night, dawn, dusk, golden hour |
| **Weather** | clear | clear, cloudy, rain, snow, fog, storm |
| **Lighting** | natural soft | natural soft, hard sun, low key, high key, candlelight, neon |
| **Color grade** | neutral | narrative color grade |
| **Mood** | neutral | neutral, tense, joyful, melancholic, epic, romantic, scary |
| **Lens** | *(empty)* | 24mm wide, 35mm, 50mm, 85mm portrait, anamorphic — *persists across scenes* |
| **Film look** | *(empty)* | warm cinematic, cool teal-orange, bleach bypass, vintage film grain, high contrast noir, pastel — *persists* |
| **Per-asset state** | — | for each `@character`: **variation**, **wardrobe**, **condition** (e.g. wounded, wet), **emotion** |

**How it works**
- **Forward propagation** — state propagates forward from scene to scene and shot to shot. Photographic choices (lens, film look) persist; character changes (e.g. an actor gets muddy in scene 2) carry until you change them.
- **Prompt injection** — the relevant state is composed into a phrase (time, weather, lighting, color grade, mood, lens, film look, plus each asset's condition/emotion/wardrobe) and injected into every image/video prompt, so you never re-type it.
- **Warnings** — the engine flags incoherences (e.g. *time of day changes: day → night*, or *a character's condition lost without reason*) so you can catch jumps before rendering.

To edit continuity, select a **scene** in the storyboard, set the fields in the **Scene continuity** panel, and press **Apply + propagate**. Use **👤 Character state…** to set wardrobe / condition / emotion per character; it is saved and propagated to following scenes.

---

## Montage tab

### Assemble, score and export the film
> **Edition:** **Full+** · **Capability:** `cap.filmstudio_montaggio`

**What it does** — Builds an OpenTimelineIO (OTIO) timeline from the generated shots, lets you reorder/trim clips and add audio, then either renders a finished **Master** or exports a **project** for an external editor.

**Where to find it** — The **Montage** (*Montaggio*) tab.

**Step by step**
1. Press **↻ Refresh timeline** to (re)build the timeline from the generated clips.
2. Drag clips to reorder; trim in/out on the timeline; ripple-delete is reversible (**↩ Restore excluded clips**).
3. Generate audio: **🔊 Generate audio (dialogue+music)**, **🎵 Soundtrack**, **🌧 Ambient/SFX**, **💥 Action SFX**.
4. Set the **Output** format: **Aspect**, default **Transition** and its duration, and the **NLE format**.
5. Press **🎞 Master (A)** to render the finished video (all audio mixed, transitions applied), or **📦 Export NLE (B)** to export the timeline for an external editor.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---------|------|---------|-----------------|-------------------------------|
| **Aspect** | Dropdown | **16:9** | 16:9, 2.39:1 Cinemascope, 9:16 Vertical, 1:1 | Output aspect ratio (letterbox/pillarbox, no crop). |
| **Transition** (default) | Dropdown | **Cut** | Cut, Dissolve, Dissolve to black, Dissolve, Wipe ←, Slide →, Circle | Default transition between clips. |
| **Transition duration** | Number (s) | **0.5 s** | 0.1 – 3.0 s (step 0.1) | Length of transitions. |
| **NLE format** | Dropdown | **FCPXML** | FCPXML (Filmora/Resolve), OpenTimelineIO, AAF (Premiere/Avid), EDL | Interchange format for the NLE export. |

**Specializations**
- **Audio tracks** — the timeline carries separate **dialogue**, **music** and **ambient** tracks, mixed with ducking. Dialogue is synthesized from the shot's `@Name`→voice bindings and lip-synced; music and ambient are generated from the scene mood; **Action SFX** are anchored per shot.
- **Master (A)** — renders a single mixed file at the chosen aspect/transitions. The delivered master is the one point where the invisible provenance signature and the visible brand watermark are applied (see [Provenance signature](13-steganography.md)).
- **Export NLE (B)** — emits **FCPXML / AAF / EDL / OTIO** via OpenTimelineIO, with a self-contained media + manifest package. If OTIO is unavailable, a JSON manifest + minimal EDL fallback is written. Import into **Premiere** (AAF/XML), **DaVinci Resolve** (FCPXML), Filmora and others to finish.
- **Review assistants** — **🎞 Animatic** (fast keyframe-sequence preview with dialogue), **🔍 Second look (critic)** (an AI critic's final creative notes), **📉 Emotional rhythm** (tunes shot durations to a per-scene tension curve while keeping scene totals), and **🌐 Dub to language…** (regenerate dialogue in another language, optionally re-lip-syncing the videos).

**Tips for best results**
> - Render an **🎞 Animatic** first to judge pacing before spending GPU on full videos.
> - Produce **both** outputs when in doubt: a quick **Master** to share, and an **NLE export** to finish properly.
> - Use **📉 Emotional rhythm** to shorten cuts in high-tension scenes automatically — it preserves each scene's total duration.

---

## Asset Manager

### Create reusable actors, props, locations and architectures
> **Edition:** **Full+** · **Capability:** `cap.filmstudio_asset`

**What it does** — A CRUD library of first-class assets that stay consistent across scenes and projects, each with a generated **Character/Asset Sheet**, **variations**, and (for actors) **wardrobe**.

**Where to find it** — The **Asset Manager** tab.

**Step by step**
1. Press **➕ New**, pick the **Type**, and fill **Name**, **Description** (supports `@Name`), **Tags** and an optional **Base photo**.
2. Press **💾 Save**, then **🎭 Generate Character Sheet** to produce a coherent multi-angle turnaround (front, three-quarter, side, close-up, full body, detail).
3. Curate the gallery: **✓ Accept/Remove** individual images, or **↻ Regenerate this angle** (the old version is kept in take history).
4. Add **Variations / Sub-locations** (e.g. *wounded*, *dirty*, an interior clearing) and press **🎭 Generate set** for each.
5. For actors, add **Wardrobe** items and press **👗 Wardrobe** to auto-generate a six-angle outfit turnaround.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---------|------|---------|-----------------|-------------------------------|
| **Type** | Dropdown | Character | **Character**, **Object** (prop), **Place** (location), **Architecture** | The asset category. |
| **Name** | Text | — | Free text | Used as the `@Name` reference and (in LoRA path) the trigger alias. |
| **Description** | Multi-line text | — | Free text, supports `@Name` | The prompt the sheet is generated from. |
| **Tags** | Text | — | comma-separated | Search tags (LIKE with AND/OR/NOT). |
| **Base photo** | File | — | Image | Optional real photo to anchor the appearance. |

**Specializations — reference resolution by path**
- **Fast** — `@Name` resolves to the asset's **reference images** plus **PuLID** (faces) or Kontext (props/places).
- **LoRA** — `@Name` resolves to the asset's trained **LoRA + trigger**.
- **Mixed** — per-asset choice within the same job.

Variations are addressed as **`@Name#Variant`**; deleting `#Variant` from a reference reopens the variant picker, deleting `@Name` reopens the asset picker. Replacing an asset in a *scene* description propagates to every shot in that scene.

**Tips for best results**
> - Anchor important characters with one or two clean **Base photos** for a stronger, more stable identity.
> - Build the variations you know the story needs (wounded, wet, night) up front — the continuity engine can then switch to them by name.
> - Accept/curate the Character Sheet carefully: it doubles as the **training dataset** for the asset's LoRA.

---

## Voice tab

### Give characters voices and sync their lips
> **Edition:** **Full+** · **Capability:** `cap.filmstudio_voice`

**What it does** — Creates voice profiles — from a **description**, from **parameters**, or **cloned** from a sample — and binds them to actors so their dialogue is synthesized automatically and lip-synced onto their shots.

**Where to find it** — The **Voice** tab.

**Step by step**
1. **From description:** name the voice, describe it (e.g. *old man, baritone, authoritative tired tone, northern accent*) and press **🗣 Create voice from description**.
2. **From parameters:** name it, set gender/age/language/emotion/rate/pitch and press **🎚 Create voice from parameters**.
3. **Cloned:** name it, load a **Sample…** (`.wav`/`.mp3`/`.flac`) and press **🎙 Register cloned voice**.
4. **Bind:** pick an **Actor** and press **🔗 Bind**. Dialogue attributed to that `@Name` will use this voice, with lip-sync applied to the shot.

**Parameters (voice from parameters)**

| Control | Type | Default | Range / Options |
|---------|------|---------|-----------------|
| **Gender** | Dropdown | Female | Female, Male |
| **Age** | Dropdown | Adult | Child, Young, Adult, Old |
| **Language** | Editable dropdown | it | it, en, es, fr, de, pt, ru, zh, ar |
| **Emotion** | Dropdown | Neutral | Neutral, Happy, Sad, Angry, Whisper, Calm |
| **Rate** | Dropdown | Normal | Very slow, Slow, Normal, Fast, Very fast |
| **Pitch** | Dropdown | Medium | Very low, Low, Medium, High, Very high |

**Specializations**
- **Dialogue timing** — the app checks that a line fits the shot's duration and warns if a line is longer than its shot (extend the shot or trim the text).
- **Dubbing** — from Montage you can regenerate all dialogue in another language while keeping each character's bound voice, then re-lip-sync.

**Tips for best results**
> - Bind a voice to every speaking actor before generating dialogue — unbound `@Name`s fall back to a generic voice.
> - For cloning, use a clean 10–30 second sample with no background noise or music.

---

## LoRA Studio

### Train and manage character LoRAs (image → video)
> **Edition:** **Full+** · **Capability:** `cap.filmstudio_lora`

**What it does** — Hosts the shared LoRA library for the project and drives the **character pipeline** that turns an asset's photos into a trained image LoRA and then a video LoRA — the backbone of the **LoRA** coherence path.

**Where to find it** — The **LoRA Studio** tab. Actual training happens in the app's **Video LoRA** and **Photo LoRA** tabs (see [Training your own LoRA](10-training.md)); trained LoRAs appear here automatically.

**Step by step**
1. Select an asset in **Character pipeline (image → video)**; optionally point **photos folder** at a folder (empty = use the asset's reference images).
2. Choose **Supervised (gate)** or **Automatic**, then press **▶ Start pipeline**.
3. In Supervised mode, a **gate** appears — **Pick the good photos** that represent the character well, then continue.
4. When finished, the LoRA is **registered** in the library. Set its **Trigger**, **Type**, **Scope** and **Asset**, then **💾 Save**.

**The pipeline state machine**

| State | Meaning |
|-------|---------|
| **new** | Pipeline created, not started. |
| **img_training** | Generating the dataset and training the **image** LoRA. |
| **review** | Paused at the photo-selection gate (Supervised) — waiting for your picks. |
| **vid_training** | Training the **video** LoRA from the selected material. |
| **registered** | Both LoRAs trained and registered in the library. |

(An **error** state captures failures; the pipeline is idempotent and resumable.)

**Parameters (LoRA details)**

| Control | Type | Default | Range / Options | What it does & when to change |
|---------|------|---------|-----------------|-------------------------------|
| **Trigger** | Text | — | Free text | The rare token that summons the subject in prompts. |
| **Type** (flavor) | Dropdown | — | image, video | Whether the LoRA is for image or video generation. |
| **Scope** | Dropdown | **Episode** | Episode, Series | Episode-only vs. shared across the whole series. |
| **Asset** | Dropdown | — | Project assets | The asset this LoRA represents. |
| **Weight** | — | balanced default | per-LoRA | How strongly the LoRA is applied. |

**Specializations**
- **Base arch & flavor** — each asset gets an **image** LoRA (FLUX-family) and a **video** LoRA (LTX-2.3), trained from the same image base and sharing the same trigger, so `@Name` works identically in stills and motion.
- **Multi-LoRA warning** — when a shot uses several LoRAs at once, the app warns above a small threshold (*⚠ … simultaneous LoRA — interference risk, consider splitting*) and suggests splitting the shot or using the Fast path for minor subjects.

**Tips for best results**
> - Train the **image LoRA first**, review its samples, then let the pipeline train the **video LoRA** — the still identity anchors the motion one.
> - Keep no more than ~3 LoRAs active in a single shot; beyond that, faces start to bleed.
> - Scope recurring series cast as **Series** so every episode reuses them with pinned versions.

---

## Job Scheduler

Every heavy generation — images, videos, LoRA training, audio — passes through a **priority queue**. The queue does not block the UI: you keep working on the script or assets while it runs, and it recovers orphaned jobs after a crash.

**Job types and time estimates** (used for the cost estimate shown before **Generate full film**):

| Job type | Estimate |
|----------|----------|
| `image` | 25 s |
| `video` | 90 s |
| `lora_train` | 1800 s (30 min) |
| `audio_voice` | 20 s |
| `audio_music` | 120 s |
| `audio_sfx` | 30 s |
| `lipsync` | 60 s |

Jobs carry a **priority** (default 5; lower runs first) and a status of *pending → running → done / error / retry*. Before launching a batch, the app sums these estimates so you can decide whether to proceed. The Pro interface shows a live **🗓 Job** indicator (*{r} running · {w} waiting*).

---

## Generation service

Under the hood, two native functions do the actual rendering. You never call them directly, but their parameters explain what the UI controls map to.

**`gen_image`** — renders one image:
- `prompt`, `out_png`, `width` (**1024**), `height` (**1024**), `seed` (**7**)
- `face_ref` → **PuLID** face conditioning (strength 0.85) · `ref_image` / `ref_images` → **Kontext** multi-reference (used when there is no face_ref)
- `loras` (optional) · `arch` (**flux**) · `model` (checkpoint) · `negative` (extends the default negative with continuity's forbidden elements)
- Internal defaults: 22 steps, guidance 3.5.

**`gen_shot_video`** — renders one shot clip (LTX-2.3 GGUF):
- `prompt`, `out_mp4`, `width` (**768**), `height` (**512**), `fps` (**25**), `duration_s` (**4.0**), `seed` (**7**)
- `keyframes` = `[{path, frame, strength}]` (opening/intermediate/final conditioning) · `loras` (optional) · `audio_path` (for lip-sync) · `model_gguf` (project's chosen transformer) · `negative`
- Internal defaults: 8 steps, guidance 3.0, CPU offload; long shots are chunked automatically. If a persistent video worker is running, the model stays loaded in VRAM between shots.

---

## Style Bible

The **🎨 Style Bible** (in the Set design tab) is the project's visual constitution: an **emotional logline**, a **color palette**, a **look** (grain / optics / light) and **reference directors/DoP**. Once saved, it is injected into *every* image and video prompt, giving the whole film a coherent identity without you repeating yourself. Combine it with the continuity engine's persistent **lens** and **film look** for a locked cinematographic style.

---

## A full worked example: idea → finished film

1. **Create the project.** In the project bar, name it *"The Last Lighthouse"*, set **Duration** to 120 s, **Coherence** to **Mixed**, **Run mode** to **Supervised**, and press **➕ Create**.
2. **Cast.** In **Asset Manager**, create an actor `@Mara` (description + one base photo), generate her Character Sheet, accept the best angles, and add a *storm-soaked* variation. Create a location `@Lighthouse`.
3. **Voices.** In **Voice**, create a weary female voice from parameters and **🔗 Bind** it to `@Mara`.
4. **Script.** In **Script**, pick **Mini-idea**, type *"On the last night before the lighthouse is decommissioned, @Mara keeps the lamp burning through a storm."*, set Style *"moody, teal-orange, film grain"* and Genre *"drama"*. Press **📖 Generate Story**, then **✍️ Generate script**, review the summary, and **💾 Save**.
5. **Style Bible.** In **Set design**, open **🎨 Style Bible**: logline *"solitude versus duty"*, palette *"cold blues, amber lamp"*, look *"35mm grain, anamorphic flares"*, reference *"Deakins"*. Save.
6. **Breakdown.** Press **🎬 Scene/shot breakdown**, then **🧐 Continuity supervisor** to catch jumps. Set scene continuity to **night / storm / low key**.
7. **Storyboard.** Press **🖼 Generate keyframes**, curate, then **🎥 Generate videos** with **🔗 Continuous match** on for the action beats.
8. **Coverage & LoRA (optional).** For the close dialogue, use **🎬 Expand dialogue coverage**; for maximum consistency, train `@Mara`'s LoRAs from **LoRA Studio**.
9. **Montage.** In **Montage**, press **↻ Refresh timeline**, generate **🔊 dialogue**, **🎵 Soundtrack** and **🌧 Ambient**, set **Aspect** to **2.39:1**, then render **🎞 Master (A)** — or **📦 Export NLE (B)** as FCPXML to finish in Resolve.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| **"LLM unavailable (pick Gemma or connect LM Studio)"** | No text engine connected. | In the LLM engine panel choose **Local Gemma**, or select **LM Studio**, set the **Port** and press **Connect**. |
| **"Create/select a project."** | No active project. | Create or open a project in the project bar / **📂 Projects**. |
| **"Generate the script first."** | Trying to break down with no screenplay. | Generate and **💾 Save** the script in the **Script** tab first. |
| **Scene/film durations don't add up** (⚠ warnings) | Shot durations ≠ scene total, or scenes ≠ film total. | Adjust shot/scene durations; the warning shows the exact mismatch. |
| **Dialogue longer than the shot** | A line exceeds the shot duration. | Extend the shot or trim the line (the app flags which shots overflow). |
| **"⚠ … simultaneous LoRA — interference risk"** | Too many LoRAs in one shot. | Split the shot, or move minor subjects to the **Fast** path. |
| **Continuity supervisor flags jumps** | Time-of-day / eyeline / condition mismatch. | Open the flagged scene, fix continuity, **Apply + propagate**. |
| **Master render is slow / reloads the model each shot** | No persistent video worker. | Start the persistent LTX-2 worker so the model stays loaded between shots; otherwise expect per-shot reloads. |
| **NLE export won't open in my editor** | Wrong interchange format. | Choose the matching **NLE format** (FCPXML for Resolve/Filmora, AAF for Premiere/Avid, or OTIO/EDL as fallback). |
| **Faces drift in video** | Fast path over a long sequence. | Switch the character to the **LoRA** path (train image + video LoRA) for strong, repeatable identity. |

> **See also:** [Images](02-images.md) for the keyframe engine, [Video](03-video.md) for shot generation and LTX controls, [Training your own LoRA](10-training.md) for the training tabs the LoRA Studio drives, and [Import models](11-import-models.md) to bring in your own checkpoints and community LoRAs.
