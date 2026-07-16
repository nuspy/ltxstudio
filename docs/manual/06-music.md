# Music

*Compose full songs and instrumentals, add a soundtrack or foley to any video, and clean up the result — all locally, with no cloud services.*

The **Music** tab groups every audio-generation function of AI Creative Center. It has two main generators — **Song / Instrumental** (sung or instrumental music from lyrics and style tags) and **Sonify video (MMAudio)** (a matching soundtrack or sound effects for a video) — plus three finishing panels: audio post-production, vocal pitch correction, and a Music LoRA trainer.

> **Edition:** All music functions require **Full+**. Studio Pro adds nothing extra to this tab except model import. Adult (18+) content follows the same rules as elsewhere in the app.

The audio engines run as isolated background processes, so generation never freezes the interface. The first time you use an engine, install it with the dedicated button (it downloads the engine, its PyTorch runtime, and several gigabytes of model weights).

---

## Song / Instrumental

> **Edition:** Full+ · **Capability:** `cap.music_song`

**What it does** — Turns a set of style tags and (optionally) lyrics into a complete audio track: a sung song with vocals, or a purely instrumental piece.
**Where to find it** — Music tab → **Song / Instrumental** panel (top of the tab).

**Step by step**
1. Give the track a **Name** — this becomes the output file name.
2. Type **Style tags** (comma-separated), or click the 🎨 palette button to pick them, or the 🎲 button to load a complete random example (tags, lyrics and metadata).
3. Optionally enable **✨ Enhance style (AI)** so a local language model expands your tags into a rich style description before generating.
4. Write your **Lyrics**, using the section chips to insert structure markers, or click **✨ Generate lyrics with AI** to write them from a theme.
5. Choose the **Engine**, **Model**, **Duration** and the other parameters below.
6. If this is your first run, click **Install HeartMuLa** (or **Install ACE-Step**) and wait for the download to finish.
7. Click **Generate music**. Progress appears in the [Progress](#progress-playback-and-results) panel at the bottom.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Name** | Text | `song` | free text | Output file name (`.mp3` for HeartMuLa; `.wav`/other for ACE-Step). |
| **Style tags** | Text | `pop, upbeat, female vocals, synth, catchy` | comma-separated | Genre, mood, instruments, vocal type. Put the genre first — it carries the most weight. |
| 🎨 (palette) | Button | — | — | Opens the **Musical styles palette** to pick tags visually. |
| 🎲 (random example) | Button | — | — | Fills tags, lyrics and metadata with a random, always-different style example. |
| **✨ Enhance style (AI)** | Checkbox | Off | On / Off | Rewrites your tags into a full style caption via a local LLM before generating. Invisible and optional. |
| **Lyrics** | Multi-line text | empty | free text with `[Section]` markers | The words to sing. Disabled when **Instrumental** is on. |
| Section chips | Buttons | — | Intro · Verse · Pre-Chorus · Chorus · Bridge · Instrumental · Guitar Solo · Outro | Insert the corresponding `[Section]` marker at the cursor. |
| **Song theme** + **✨ Generate lyrics with AI** | Text + button | empty | free text; language dropdown (15 languages) | Writes lyrics from a theme in the chosen language using a local Gemma-3 model, then inserts them into the Lyrics box. |
| **Instrumental (no vocals)** | Checkbox | Off | On / Off | Produces a purely instrumental track and disables the Lyrics box. |
| **Engine** | Dropdown | HeartMuLa | HeartMuLa · ACE-Step · ACE-Step 1.5 | Chooses the generation model (see [Engines](#engines)). Available controls change with the engine. |
| **Duration** | Number (s) | 120 | 5–360 | Target length of the track in seconds. |
| **Model** | Dropdown | 3B | 3B · 7B (whichever are downloaded) | HeartMuLa checkpoint size. Larger = higher quality, more VRAM. HeartMuLa only. |
| **Temperature** | Number | 1.0 | 0.1–2.0 | Creative randomness. Higher = more variety, less predictable. HeartMuLa only. |
| **CFG** | Number | 2.5 (HeartMuLa) / 15.0 (ACE-Step) / 1.0 (1.5) | HeartMuLa 1.0–8.0 · ACE-Step 1.0–30.0 · 1.5 1.0–10.0 | Adherence to your tags/lyrics. The scale differs per engine and is reset automatically when you switch engine. On HeartMuLa keep it around 2.5 (beyond ~4–5 risks artifacts). |
| **Low VRAM (lazy load)** | Checkbox | Off | On / Off | Offloads weights to system RAM/CPU during generation — much less VRAM, but slower. |
| **Style adherence** | Number | 3.0 | 0.0–10.0 | *ACE-Step only.* Weights the style tags via Double Condition Guidance. Only active above 1.0; 3–5 = tags weigh more; 0/≤1 = off. |
| **Lyrics adherence** | Number | 3.0 | 0.0–10.0 | *ACE-Step only.* Weights the lyrics. Together with Style adherence enables double guidance — both must be above 1.0 to take effect. |
| **Planner 5Hz (think)** | Checkbox | On | On / Off | *ACE-Step 1.5 only.* Runs a language-model planner that reasons over metadata and structure before generating, improving coherence. |

**The estimate badge** — Below the parameters, a colored dot estimates the VRAM load (green = OK, yellow = near the limit, red = offload/OOM risk) based on duration, batch, steps and quantization. The time estimate is indicative and depends on your GPU.

**Presets** — Save the current style configuration under a name with 💾, reload it from the **Preset** dropdown, delete it with 🗑, or pull settings back from a previously generated track with **From track…**. The **↺ Reset default** button restores the optimal values for the selected engine.

### Engines

- **HeartMuLa** — The default engine. Produces complete songs (vocals + lyrics) or instrumentals. Offers the **Model** (3B/7B) and **Temperature** controls; its only adherence lever is **CFG**.
- **ACE-Step** — An alternative engine with finer control. Adds **Style adherence** and **Lyrics adherence** (Double Condition Guidance), an **Advanced (ACE-Step)** panel (scheduler, CFG type, omega, guidance interval/decay, ERG, output format, seed, batch, quantized model, overlapped decoding, torch.compile, LoRA), and an **Operation (ACE-Step)** panel for editing tasks (Retake, Repaint, Edit, Extend, Audio→Audio). Only ACE-Step supports [Music LoRA training](#music-lora-trainer).
- **ACE-Step 1.5** — A turbo DiT model with a 5 Hz planner. Metadata are passed as structured parameters (see below) rather than appended to the caption. CFG stays near 1.0; adherence comes from the planner and the metadata.

### Musical metadata

*Collapsible panel: **🎼 Musical metadata (BPM, key, time signature, language)**.*

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **BPM** | Number | (auto) | 0–300 (0 = auto) | Tempo. 60–80 slow, 90–120 medium, 120–150 pop/dance/rock, 150–180 very energetic. |
| **Key** | Dropdown | (auto) | (auto) + common major/minor keys | Key/scale. Minor keys sound darker; major keys brighter. |
| **Time signature** | Dropdown | (auto) | (auto) · 4/4 · 3/4 · 6/8 · 2/4 | Meter. 4/4 for most pop/rock, 3/4 waltz, 6/8 ballads. |
| **Vocal language** | Dropdown | (auto) | (auto) + 12 languages | Singing language. Auto is detected from the lyrics; set it if detection is wrong or the lyrics mix languages. |

For HeartMuLa and ACE-Step these values are appended to the style tags; for ACE-Step 1.5 they are passed as structured fields.

**Specializations** — For the ACE-Step editing operations (Retake / Repaint / Edit / Extend / Audio→Audio), see the **Operation (ACE-Step)** panel. Selecting a region on the [waveform player](#progress-playback-and-results) after generation automatically loads that time range into the Repaint operation.

**Outputs** — A generated song is saved as an audio file (`.mp3` with HeartMuLa) in the output folder, accompanied by a `.lyrics.txt` and a `.tags.txt` file holding the exact inputs, plus a metadata sidecar so the parameters can be reused later. Use **📝 Generate LRC** in the Progress panel to produce synchronized karaoke lyrics from the audio.

**Tips for best results**
> - Put the genre first in your tags — it dominates the style.
> - Leave metadata on **(auto)** unless you have a specific tempo/key in mind; the model usually picks well.
> - For the strongest tag/lyric adherence on ACE-Step, raise both **Style adherence** and **Lyrics adherence** above 1.0.
> - If you hit VRAM limits (red badge), enable **Low VRAM (lazy load)** or, on ACE-Step, the **Quantized model** option in the Advanced panel.
> - Use `{option a|option b|option c}` wildcards in tags or lyrics to randomize a choice at generation time.

---

## Sonify video (MMAudio)

> **Edition:** Full+ · **Capability:** `cap.music_sonify`

**What it does** — Generates an audio track that is synchronized to a video (soundtrack, sound effects, or both). With no video, it works as a text-to-audio generator.
**Where to find it** — Music tab → **Sonify video (MMAudio)** panel.

**Step by step**
1. Set the **Video** field: click **Browse…**, or **Last video** to reuse the last video generated in the app. Leave it empty for text-to-audio.
2. Describe the sound you want in **Prompt** (e.g. "rain and thunder, footsteps on gravel").
3. Optionally add a **Negative prompt** to exclude unwanted sounds.
4. Adjust **Duration**, **Steps**, **CFG** and **Seed** as needed.
5. If this is your first run, click **Install MMAudio**.
6. Click **Generate audio**.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Video** | File path | empty | video file, or empty | The video to sonify. Empty = text-to-audio. |
| **Last video** | Button | (disabled until available) | — | Loads the most recent video generated in the app. |
| **Prompt** | Text | empty | free text | Describes the sound to generate. |
| **Negative prompt** | Text | empty | free text | Sounds to avoid. |
| **Duration** | Number (s) | 8 | 1–120 | Length of the generated audio. |
| **Steps** | Number | 25 | 10–100 | Denoising steps. More = cleaner audio, slower. |
| **CFG** | Number | 4.5 | 1.0–10.0 | Adherence to the prompt. |
| **Seed** | Number | 42 | 0–2,000,000,000 | Fix it for reproducible results. |

**Tips for best results**
> - Describe concrete sound events ("thunderclap", "waves crashing") rather than abstract moods for tighter sync.
> - Keep the duration close to the video length; longer requests take more VRAM and time.
> - For a full musical score, describe the genre and instruments; for foley, describe the physical events on screen.

---

## Post-processing

*Collapsible panel: **🎚 Audio post-production (normalize · trim · fade · convert)**. No download required — it runs on any existing audio file via FFmpeg.*

**What it does** — Cleans up and standardizes a finished track: loudness normalization, silence trimming, fades, and format conversion.
**Where to find it** — Music tab, below the Sonify panel.

**Step by step**
1. Set the **Source**: drag an audio file in, **Browse**, or click **Last track** to use the last generated track.
2. Enable the operations you want.
3. Click **🎚 Process**.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Normalize loudness (EBU R128, -14 LUFS)** | Checkbox | On | On / Off | Brings perceived volume to a uniform streaming level without clipping. |
| **Trim silence (start/end)** | Checkbox | Off | On / Off | Removes leading/trailing silence and long internal gaps. |
| **Fade in** | Number (s) | 0.0 | 0.0–15.0 | Fade-in duration (0 = none). |
| **Fade out** | Number (s) | 0.0 | 0.0–15.0 | Fade-out duration (0 = none). |
| **Format** | Dropdown | (unchanged) | (unchanged) · wav · mp3 · flac · ogg | Output format; (unchanged) keeps the source format. |

---

## Pitch correction (DiffPitcher)

*Collapsible panel: **🎤 Vocal pitch correction (DiffPitcher)**.*

**What it does** — Corrects an out-of-tune vocal by transferring the pitch curve from an in-tune guide vocal singing the same phrase.
**Where to find it** — Music tab, below Post-production.

**Step by step**
1. Set **Off-key vocal** (the take to fix): drag in, Browse, or **Last track**.
2. Set **Guide vocal** — an in-tune recording of the same phrase.
3. Adjust **Steps** and **Transpose** if needed.
4. Click **🎤 Correct pitch**.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Steps** | Number | 50 | 10–100 | Diffusion steps. More = finer correction, slower. |
| **Transpose** | Number (semitones) | 0 | -12 to +12 | Shifts the reference pitch in semitones (0 = none). |

> This module is a preview feature; results may vary depending on your material.

---

## Music LoRA trainer

*Collapsible panel: **🎓 Train Music LoRA (ACE-Step)**. Only available with the ACE-Step engine — with HeartMuLa the panel shows a note and stays disabled.*

**What it does** — Trains a custom LoRA on your own audio dataset so the ACE-Step engine can reproduce a specific style, voice or instrument.
**Where to find it** — Music tab, standalone panel between the Sonify panel and the Progress panel.

**Step by step**
1. Select **ACE-Step** as the engine so the panel becomes active.
2. Drag audio files or whole folders into the **Audio dataset** list. If a file has a matching `<name>_prompt.txt` / `<name>_lyrics.txt` next to it, those are used; otherwise the global fields below apply.
3. Fill in **Global tags** and, optionally, **Global lyrics**.
4. Set the training parameters.
5. Click **🎓 Train LoRA** and confirm.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **LoRA name** | Text | `music_lora` | free text | Output folder name for the trained LoRA. |
| **Global tags** | Text | empty | free text | Style tags applied to files without their own prompt file. |
| **Global lyrics** | Multi-line text | empty | free text | Lyrics applied to files without their own lyrics file (optional). |
| **Max steps** | Number | 2000 | 100–100000 | Training steps. More = more learning and more time. |
| **Learning rate** | Number | 1e-4 | 1e-6–1e-2 | A good starting point is 1e-4. |
| **Repeat dataset** | Number | 2000 | 1–100000 | Dataset repetitions: high for few files, lower for many. |
| **Precision** | Dropdown | bf16-mixed | bf16-mixed · 16-mixed · 32 | bf16-mixed is fast and light on recent GPUs. |
| **Batch (accum)** | Number | 1 | 1–16 | Gradient accumulation (effective batch); raise it if you have lots of VRAM. |

Once trained, apply the LoRA from the **Advanced (ACE-Step)** panel's LoRA field when generating.

---

## Progress, playback and results

The **Progress** panel at the bottom of the tab shows a progress bar, a status line, and a live log. After generation you can:

- **Play** the result in your system player, or use the in-app **waveform player** (drag-select a region to send it to the ACE-Step Repaint operation).
- **Open folder** to reveal the output files.
- **📊 Spectrum / A-B** to inspect the spectrogram and compare two tracks by spectral centroid.
- **📚 Library** to browse your generation history — replay, reuse the parameters, or delete tracks.
- **📝 Generate LRC** to transcribe the audio into synchronized karaoke lyrics.
- **Stop** to cancel a running generation.

---

## Related

- To turn a song into a full video clip, see [Music Video](07-music-video.md).
- For cinematic, scripted projects with dialogue and scoring, see [Cinema — AI Film Studio](09-cinema-studio.md).
- For voice synthesis and cloning used by other modules, see the [Audio & voice](05-audio-voice.md) chapter.
