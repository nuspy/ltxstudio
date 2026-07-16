# Music Video

*Turn a song into a finished music video: AI Creative Center writes a shot list from your lyrics, generates an opening image and a video clip for each shot, then concatenates everything and adds the audio track.*

The **Music Video Studio** is a project-based workflow. Each project is saved to disk, so you can build a video in stages, close the app, and come back to exactly where you left off. The Studio walks through four phases — Song, Shots/Images, Videos, Assembly — each of which you can review and edit before moving on.

> **Edition:** Full+ · **Capability:** `cap.music_video`. Studio Pro adds nothing extra here except model import.

The whole pipeline runs locally with native engines (no cloud): a local language model plans the storyboard from your lyrics, the image engine renders each opening frame, and the LTX-2 video engine animates it into a clip.

---

## How the workflow fits together

1. **Song** — Generate (or regenerate) the track with HeartMuLa. All songs made for the project are listed; you pick the active one.
2. **Shots / Images** — A local LLM reads your lyrics and builds a shot list, distributing shots across the song's duration. You edit the list — reorder, add, delete, adjust timing and prompts — then generate an opening image per shot.
3. **Videos** — Each opening image is animated into an image-to-video clip. You can regenerate, reorder or delete individual clips.
4. **Assembly** — The clips are concatenated in order and the active song's audio is muxed in, producing the final `music_video.mp4`.

Because the shots are separate image-to-video clips, scene changes are clean hard cuts by design; optional transitions can soften them at assembly time.

---

## Starting a project

You can launch the Studio from the [Music](06-music.md) tab:

- **🎬 Generate music video** — Creates a new project from the current song fields (name, lyrics, style tags, starting image, scene pace, video style, duration) and opens the Studio at the Song phase.
- **📂 Music video projects** — Opens the projects manager.

**The projects manager** (**Music video projects** dialog) lists every saved project with its current phase and song count. From here you can **📂 Open**, **📋 Duplicate**, or **🗑 Delete** a project. Opening one launches the Studio at its saved phase.

When the Studio opens, it inherits the image model from the [Images](02-images.md) tab and the video model and resolution from the [Video](03-video.md) tab, so generated frames and clips match the models you were already using.

---

## Phase 1 — Song

> **Where to find it** — Music Video Studio → **1 · Song** tab.

**What it does** — Generates the song for the project and lets you keep several takes, choosing which one is active.

**Step by step**
1. Set **Temperature** and **CFG** for the song generation.
2. Click **🎵 Generate/Regenerate song**. Each run is added to the list.
3. Select a track and use **▶ Play** to audition it, **✓ Use this** to make it the active song, or **🗑** to delete it.
4. Click **→ Go to shots** to continue.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Temperature** | Number | 1.0 | 0.1–2.0 | Creative randomness of the song generation. |
| **CFG** | Number | 1.5 | 1.0–10.0 | Adherence to the project's tags and lyrics. |

The lyrics, tags and target duration come from the project (set when you created it). For full control over the song itself, generate it in the [Music](06-music.md) tab first.

---

## Phase 2 — Shots / Images (storyboard)

> **Where to find it** — Music Video Studio → **2 · Shots / Images** tab.

**What it does** — Builds and edits the shot list, then generates the opening image for each shot.

**Step by step**
1. Click **1) Build shot list (LLM)**. The language model reads the project's lyrics and produces a list of shots spread across the song, each with an image prompt and a video prompt.
2. *(Optional)* Click **🎯 Align to vocals** to snap each shot's start time to the real sung onsets (requires speech-to-text; if unavailable you are prompted to install it).
3. Edit the list on the left: **↑ / ↓** to reorder, **➕ end** / **➕ after** to add a shot, **🗑** to delete.
4. Select a shot to edit it in the **Shot detail** panel: adjust **Start (s)**, rewrite the **Opening image prompt** and **Video prompt**, then **💾 Apply changes**.
5. Regenerate that shot's image with **🖼 Regenerate image**, or **📁 Replace…** it with your own file.
6. When the list is ready, click **2) Generate images** to render every missing opening image.
7. Click **→ Go to videos** to continue.

**Shot-detail controls**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Start (s)** | Number | (auto) | 0 and up | Start time of the shot; lengths recompute automatically so each shot runs to the next one. |
| **Length:** | Read-only | (auto) | — | Duration of the shot (distance to the next shot's start). |
| **Opening image prompt** | Multi-line text | (from LLM) | free text | Prompt for the still frame that opens the shot. |
| **Video prompt** | Multi-line text | (from LLM) | free text | Prompt for the motion in the clip. |

**Batch memory** — A **Batch memory** control (**Auto** / **Sticky (fast batch)**) sits above the shot list. *Sticky* keeps the image model resident in VRAM for the whole image batch instead of reloading it per shot — much faster with large models. The default is Sticky and your choice is remembered.

> If the starting image you supplied when creating the project exists, it is reused as the opening image of the first shot.

---

## Phase 3 — Videos

> **Where to find it** — Music Video Studio → **3 · Videos** tab.

**What it does** — Animates each opening image into a clip, then assembles the final video with audio.

**Step by step**
1. Click **3) Generate videos** to animate every shot that has an image but no clip yet.
2. Manage clips in the list: **↑ / ↓** to reorder, **🎬 Regenerate video** to redo one, **🗑** to delete. Thumbnails show the first frame of each clip.
3. Choose a **Transition** for the joins.
4. Click **4) Assemble + audio** to concatenate the clips in order and mux the active song. The result is saved as `music_video.mp4`.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Transition** | Dropdown | Hard cut | Hard cut · Crossfade · Fade · Dissolve · Wipe | How consecutive clips are joined at assembly. Hard cut is the clean default; the others soften the join with an FFmpeg crossfade. |

---

## Exporting to an editor (NLE)

> **Where to find it** — Music Video Studio → **3 · Videos** tab → format dropdown + **📦 Export NLE**.

**What it does** — Exports the shot list as an editing timeline you can open in a professional non-linear editor, keeping each clip on its own segment for further editing.

**Step by step**
1. Generate the video clips first.
2. Pick a format from the dropdown: **FCPXML**, **OTIO**, **AAF** or **EDL**.
3. Click **📦 Export NLE**. The timeline file is written to the project's `export` folder.

Use FCPXML for Final Cut Pro, AAF for Avid/Premiere, EDL for broad compatibility, and OTIO as an open interchange format.

---

## Parameters used across the pipeline

When driven through the REST API (`/music/music_video`) or when a project is created, these values control the whole pipeline:

| Parameter | Default | What it does |
|---|---|---|
| `lyrics` | (required) | The song lyrics; drive both the song and the LLM storyboard. |
| `song_path` | empty | An already-generated song; empty means generate it first. |
| `image_path` | (required via API) | Starting/background image reused as the first shot's opening frame. |
| `width` × `height` | 704 × 512 | Frame size of every clip (inherited from the Video tab in the Studio). |
| `fps` | 25 | Frame rate. |
| `scene_seconds` | 6.0 (API) / from pace preset | Average shot length = how often scenes change. Smaller = faster cuts. |
| `video_style` | empty | A look applied to every shot (e.g. "cartoon", "black & white, sad"). |
| `image_hint` | empty | Extra guidance for the LLM storyboard. |
| `steps` | 8 | Denoising steps per video chunk. |

In the Studio, the **Scene-change pace** you chose when creating the project maps to `scene_seconds`: Slow (~12s), Medium (~7s), Fast (~4s, default) or Very fast (~2.5s).

---

## Tips for best results

> - Write lyrics with clear `[Verse]` / `[Chorus]` sections — the storyboard follows the order of the lyrics, so structure improves shot-to-lyric alignment.
> - Review and edit the shot list *before* generating images; regenerating a single shot later is cheap, but re-planning the whole list resets your edits.
> - Set a **Video style** to keep every shot visually consistent (e.g. "anime", "film noir").
> - Use **🎯 Align to vocals** for lyric-driven videos so cuts land on the sung phrases.
> - Faster **scene-change pace** suits energetic tracks; slower pace suits ballads.

---

## Related

- Generate or fine-tune the song itself in [Music](06-music.md).
- The clips use the same engine as the [Video](03-video.md) tab, and the opening frames use the [Images](02-images.md) engine.
- For scripted, dialogue-driven films, see [Cinema — AI Film Studio](09-cinema-studio.md).
