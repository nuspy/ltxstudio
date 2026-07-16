# Audio & Voice

*Turn text into natural speech, clone a voice from a short sample, and attach a spoken track to any video — all locally, with your choice of seven speech engines.*

The **Audio / Voice** tab (`tab.audio_voice`) is where you decide what your videos *say* and *how* they say it. It combines three things in one place: a **source selector** that chooses where a video's audio comes from, a full **multi-engine text-to-speech (TTS) studio**, and a shared **voice library** that stores custom and cloned voices for reuse across the whole application (Video, Music Video, and the AI Film Studio all draw voices from this same library).

## How audio works in AI Creative Center

When you generate a video, the model can produce sound in several ways. The **Audio source** radio group at the top of the tab picks exactly one of them:

- **Native LTX audio** — the video model (LTX-2) invents the audio itself, driven by your video prompt. Good for ambient sound and simple spoken lines described in the prompt.
- **Audio I provide** — you supply a ready-made track: a file from disk, a live microphone recording, or a clip you synthesized with the built-in TTS and sent over with **Use as video audio**.
- **Clone voice** — you pick a saved voice (with its reference sample) from the library. With text in the TTS box, that text is synthesized in the chosen voice; with the text left empty, only the voice *timbre* is handed to LTX, which then generates the spoken lines from your video prompt.
- **Mute** — no audio at all.

The TTS studio below the source selector works independently: you can synthesize, preview, and reuse speech at any time, whether or not you are making a video.

---

### Choose the audio source for a video
> **Edition:** **Full+** · **Capability:** `cap.audio_tts_clone`

**What it does** — Selects where the audio of your generated video comes from: the model itself, a file you provide, a cloned voice, or silence.
**Where to find it** — **Audio / Voice** tab, the **Audio source** group (`aud.source.lbl`) at the top. The label reads *"Audio source (only one active; the other options disable themselves)"*.
**Step by step**
1. Open the **Audio / Voice** tab.
2. Under **Audio source**, click one radio button:
   - **Native LTX audio (the model generates the audio)** (`aud.rb.native`)
   - **Audio I provide (ready file)** (`aud.rb.user`)
   - **Clone voice (use a saved voice with its sample)** (`aud.rb.clone`)
   - **Mute (no audio)** (`aud.rb.mute`)
3. The unrelated controls grey out automatically so you only see what applies to your choice.
4. Optionally tick **Speak but don't sing** (`aud.no_sing`) to force spoken delivery of dialogue with no singing.
5. Go to the **[Video](03-video.md)** tab and generate — the video inherits this audio source.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Audio source** | Radio group | Native LTX audio | Native / I provide / Clone voice / Mute | Selects the audio pipeline for the next video. |
| **Speak but don't sing** | Checkbox | Off | On / Off | Forces spoken pronunciation of dialogue and suppresses singing. Turn on when the model tends to sing lines you want spoken. |

**Specializations** — In **Clone voice** mode, the decision tree is: text present in the TTS box → the text is synthesized in that voice; text empty → only the timbre is passed to LTX, which speaks your video prompt.
**Tips for best results**
> - Use **Audio I provide** when you already have a clean recording — it is the most predictable path to perfect lip-sync in the [Video](03-video.md) tab's Talk mode.
> - **Native LTX audio** is best for ambience and background; use TTS or cloning for clear dialogue.

---

### Provide a ready-made audio track (file or microphone)
> **Edition:** **Full+** · **Capability:** `cap.audio_tts_clone`

**What it does** — Loads an existing audio file or records a fresh clip from your microphone to use as the video's soundtrack.
**Where to find it** — **Audio / Voice** tab, the **Ready audio track** group (`aud.file.group`), active when **Audio I provide** is selected.
**Step by step**
1. Select **Audio I provide (ready file)** in the source group.
2. Click **Load audio file…** (`aud.file.pick.btn`) and pick a file, **or**
3. Click **● Record** (`aud.file.rec.btn`) to capture from the microphone; the button changes to **■ Stop** (`aud.rec.stop`) while recording, showing *"recording…"* (`aud.rec.recording`).
4. When done, the status shows the loaded/recorded clip; an empty slot reads **(no audio)** (`aud.file.none.lbl`).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Load audio file…** | Button | — | Audio files | Opens a file picker for a ready track. |
| **● Record** | Button | — | Toggle record/stop | Records from the default microphone. Disabled and labelled *"Record (microphone unavailable)"* (`aud.file.rec_na.btn`) when no mic is detected. |

**Tips for best results**
> - Record 5–15 seconds of clean, single-speaker audio for best downstream results.
> - You can also fill this slot by synthesizing speech with TTS and clicking **Use as video audio** (below).

---

### Generate speech from text (multi-engine TTS)
> **Edition:** **Full+** · **Capability:** `cap.audio_tts_clone`

**What it does** — Converts typed text into spoken audio using any of seven speech engines, in the voice you select from the library.
**Where to find it** — **Audio / Voice** tab, the **TTS — Generate audio from text** group (`aud.tts.group`).
**Step by step**
1. Choose an **Engine** (`aud.tts.engine_lbl`) from the dropdown.
2. If the engine's runtime isn't installed yet, click **📥 Install** (`aud.tts.install.btn`) — it clones the required node and installs dependencies.
3. Pick a **Voice** (`aud.tts.voice_lbl`) from the library dropdown (or create one — see *Voice library*).
4. Type your line in the **Text** field (`aud.tts.text_lbl`). Type **`\`** to open the SSML tag palette (see *SSML markers*).
5. Click **🎙️ Generate TTS** (`aud.tts.gen.btn`).
6. Click **▶ Play** (`aud.tts.play.btn`) to audition, and **⤴ Use as video audio** (`aud.tts.use.btn`) to send the clip to the video's audio slot.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Engine** | Dropdown | Edge | **Edge**, **Kokoro**, **F5-TTS**, **XTTS2**, **Fish**, **VibeVoice**, **CosyVoice2** | The synthesis backend. Edge uses Microsoft preset voices; the others (F5-TTS, XTTS2, Fish, VibeVoice, CosyVoice2) are cloning-capable and use a voice sample. Change to a cloning engine when you need a custom timbre. |
| **📥 Install** | Button | — | — | Installs the selected engine's node and dependencies on first use. |
| **Voice** | Dropdown | — | Library voices | The voice used for synthesis. Manage entries with the ➕ / ✎ / ⧉ / 🗑 buttons beside it. |
| **Insert** | Dropdown/menu | — | SSML markers | Inserts an SSML tag (pause / emotion / prosody) at the cursor. |
| **Text** | Multi-line text | (empty) | free text | The words to speak. Type `\` for the SSML tag palette. |
| **🎙️ Generate TTS** | Button | — | — | Synthesizes the text into audio. |
| **▶ Play** | Button | — | — | Plays the last synthesized clip. |
| **⤴ Use as video audio** | Button | — | — | Sends the clip to the **Audio I provide** slot for the next video. |

**Specializations** — The engine selector's REST equivalent (`POST /audio/tts`) accepts `engine` (`edge | kokoro | f5tts | xtts2 | fish | vibevoice | cosyvoice2`), `voice_id` (default `it-IT-ElsaNeural`), `language` (default `it-IT`), plus `sample_path` and `sample_transcript` for cloning. See [REST API](15-rest-api.md).
**Tips for best results**
> - **Edge** is the fastest, needs no sample, and has a large catalogue of preset languages/voices — ideal for narration.
> - **F5-TTS**, **XTTS2**, **CosyVoice2**, **Fish** and **VibeVoice** reproduce a specific timbre from a sample — use them for cloned characters.
> - A short, readable preview updates under the text field (`aud.tts.preview.lbl`) as you edit, so you can see how SSML tags will be interpreted before generating.

---

### Build and manage the voice library
> **Edition:** **Full+** · **Capability:** `cap.audio_tts_clone`

**What it does** — Creates, edits, duplicates, and deletes reusable voices — including cloned voices built from a reference sample — shared across the whole app.
**Where to find it** — **Audio / Voice** tab, the buttons beside the **Voice** dropdown open the voice editor dialog (`ui/voice_dialog.py`).
**Step by step**
1. Click the **➕ new** button (tooltip *"Save a new custom voice, e.g. cloned"*, `aud.tts.new.tt`) to open the **New voice** dialog (`voice.new_title`).
2. Enter a **Name** (`voice.name_label`), e.g. *"Calm Alice"*.
3. Choose an **Engine** (`voice.engine_label`). Cloning-capable engines are tagged *"clonable"* (`voice.cap_clonable`); Edge is *"preset only"* (`voice.cap_preset_only`).
4. For Edge, pick a **Base voice** (`voice.base_voice_label`). For a cloning engine, attach a **Voice sample** (`voice.sample_header`) — 5–15 seconds of clean, single-speaker speech — with **📁 Load file…** (`voice.load_file_btn`) or **● Record** (`voice.record_btn`).
5. Optionally add a **Sample transcript** (`voice.transcript_label`) — writing exactly what is heard improves the clone.
6. Add free-form **Notes** (`voice.note_label`) if you like, then click **💾 Save** (`voice.save_btn`).

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Name** | Text | (empty) | free text | The voice's display name. Required. |
| **Engine** | Dropdown | — | Edge / CosyVoice2 / F5-TTS / XTTS / … | Synthesis engine for this voice. |
| **Base voice** | Dropdown | — | Edge preset voices | Only for Edge (preset) voices. |
| **Voice sample** | File / recording | (none) | wav/mp3/flac/m4a | 5–15 s reference clip for cloning engines. Required for clonable engines. |
| **▶ Preview** | Button | — | — | Auditions the attached sample. |
| **Sample transcript** | Text | (empty) | free text | Exact transcript of the sample; optional but improves clone accuracy. |
| **Notes** | Text | (empty) | free text | Free notes for your own reference. |

**Specializations** — The row of small buttons next to the **Voice** dropdown are: ➕ new (`aud.tts.new.tt`), ✎ edit (`aud.tts.edit.tt`), ⧉ duplicate (`aud.tts.dup.tt`), and 🗑 delete (`aud.tts.del.tt`). The dialog validates that a name is present and that cloning engines have a sample attached (`voice.warn_sample_text`).
**Tips for best results**
> - Voices you create here are available everywhere — the [Video](03-video.md) Talk/lip-sync flow, [Music video](07-music-video.md), and the [AI Film Studio](09-cinema-studio.md) all read from this one library.
> - Keep samples dry (no background music), mono, and consistent in tone for the most faithful clone.

---

### Voice cloning
> **Edition:** **Full+** · **Capability:** `cap.audio_tts_clone`

**What it does** — Reproduces a real person's timbre from a short reference clip, then speaks any text (or lends the timbre to LTX) in that voice.
**Where to find it** — Two entry points: the **Clone voice** radio in the audio source group, and any cloning-capable engine in the TTS voice editor.
**Step by step**
1. Create a cloning voice as described above (a cloning engine + a 5–15 s sample + optional transcript).
2. To synthesize dialogue in that voice: select it in the TTS **Voice** dropdown, type text, and click **🎙️ Generate TTS**.
3. To lend only the timbre to a video: select **Clone voice** in the audio source group, choose the voice, and leave the TTS text empty — LTX will speak your video prompt in that timbre.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **Reference sample** | File / recording | (none) | 5–15 s clean speech | The timbre source. Longer isn't better — clean and consistent is. |
| **Transcript** | Text | (empty) | free text | The exact words in the sample; raises clone fidelity for engines that use it. |

**Specializations** — Via REST, cloning is `POST /audio/tts` with `sample_path` (and optionally `sample_transcript`) set on a cloning engine. See [REST API](15-rest-api.md).
**Tips for best results**
> - CosyVoice2, F5-TTS and XTTS2 give the most natural clones; try each on your sample and keep the best.
> - Always fill the transcript for cloning engines that support it — it noticeably improves pronunciation and prosody.

---

### SSML markers (pauses, emotion, prosody)
> **Edition:** **Full+** · **Capability:** `cap.audio_tts_clone`

**What it does** — Inserts inline SSML-style tags into your text to control pauses, emotion, and speaking rate/pitch (prosody).
**Where to find it** — In the **Text** field, type **`\`** to open the filtering tag palette, or use the **Insert** dropdown (`aud.tts.insert_lbl`).
**Step by step**
1. Place the cursor where you want a marker.
2. Type **`\`** — a palette opens and filters progressively as you keep typing (`aud.tts.text.ph`).
3. Pick a marker category (pause, emotion, prosody) and value; it is inserted at the cursor.
4. Watch the readable preview below the field update to confirm the effect.

**Parameters**

| Control | Type | Default | Range / Options | What it does & when to change |
|---|---|---|---|---|
| **`\` tag palette** | Inline picker | — | pause / emotion / prosody markers | Inserts SSML tags; filters as you type. |
| **Insert** | Dropdown | — | Marker categories | Menu alternative to the `\` palette. |

**Tips for best results**
> - Use short pauses between sentences for a natural cadence; over-tagging can make delivery robotic.
> - Not every engine honours every marker — Edge and the cloning engines interpret prosody differently; audition after tagging.

---

## Troubleshooting

- **The Record button says "microphone unavailable."** No input device was detected. Connect a microphone and restart the app; the button re-enables (`aud.file.rec_na.btn`).
- **An engine won't synthesize.** Click **📥 Install** next to the **Engine** dropdown to fetch its node and dependencies; the first run of a cloning engine also downloads its model.
- **The clone doesn't sound like the person.** Use a cleaner 5–15 s single-speaker sample, fill in the **Sample transcript**, and try a different cloning engine (CosyVoice2 / F5-TTS / XTTS2).
- **My video has no dialogue even though I typed text in TTS.** Make sure the **Audio source** is set to **Clone voice** (with a selected voice) or that you clicked **⤴ Use as video audio** after generating — otherwise the video uses the native or muted source.
