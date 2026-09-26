---
name: ltx-avatar-film
description: Trasforma un testo libero (storia, sceneggiatura, monologo, dialogo, spot) in un filmato completo con una persona che parla, usando la REST API locale di LTX Studio (POST /avatar/generate). Costruisce la richiesta intera — immagine iniziale (da testo, da foto, da personaggio con modifiche o da più riferimenti), voce TTS/clone/dialogo con i marcatori espressivi (pause, emozioni, velocità e accelerando, tono, volume, enfasi, sussurro), musica, effetti, prompt video, motore (LongCat 1.5 o MiniMax H3), formato e post-produzione — la invia, segue il job e restituisce il PATH del video creato. Usalo quando l'utente vuole «far parlare» qualcuno, un avatar parlante, un video-messaggio, un monologo/dialogo filmato o un breve film parlato a partire da un testo.
---

# LTX Studio — filmato con avatar parlante da un testo

Tu (l'agente) trasformi la richiesta dell'utente in **una** richiesta JSON per `POST /avatar/generate` della
REST API locale di LTX Studio (`http://127.0.0.1:8200`), la invii con lo script dello skill e restituisci
all'utente il **path del video** (più il link di download).

```
python <skill>/scripts/avatar_film.py --options                 # cosa è installato (motori, voci, profili)
python <skill>/scripts/avatar_film.py --markup "testo con [pause:500ms] marcatori"   # controlla i marcatori
python <skill>/scripts/avatar_film.py --request req.json --dry-run   # valida senza spendere GPU
python <skill>/scripts/avatar_film.py --request req.json        # invia, attende → stampa  VIDEO: <path>
```
Lo script avvia l'API se è spenta (si spegne da sola dopo 30 min di inattività), carica con `/files/upload` i
file locali dell'utente, controlla i marcatori, fa polling di `/jobs/{id}` e stampa `VIDEO:` / `DOWNLOAD:`
(e `IMAGE:` / `AUDIO:` intermedi). **Il path da restituire all'utente è la riga `VIDEO:`.**
Codici d'uscita: 0 ok · 1 errore del job (leggi il messaggio: spesso pesi da scaricare) · 2 richiesta non valida
· 3 API irraggiungibile. La coda GPU è seriale: un filmato alla volta.

## 1. Procedura

1. **Leggi le opzioni installate** (`--options`) una volta per sessione: motori (`engines.longcat.installed`,
   `engines.minimax_h3.profiles_installed`), voci della libreria (`voices[].id`), motori TTS (`tts_engines`),
   lingue. Non promettere ciò che non è installato.
2. **Analizza il testo dell'utente** e ricava: chi parla (1 o 2 persone, aspetto), cosa dice (battute), come lo
   dice (emozioni, ritmo, pause), dove (scena, luce, inquadratura), atmosfera sonora (musica, effetti), durata
   e formato. Se il testo è una storia in prosa, **scrivi tu il copione** (battute parlate) e la descrizione
   visiva; non leggere ad alta voce le didascalie.
3. **Scegli il motore** (tabella §2), **l'immagine iniziale** (§3), **l'audio** (§4 + marcatori §5), il **video** (§6).
4. Scrivi la richiesta in un file JSON, fai `--dry-run`, correggi gli errori, poi invia.
5. Rispondi con il path della riga `VIDEO:` (ed eventualmente il link `DOWNLOAD:`). Se il job fallisce per pesi
   mancanti, dillo chiaramente con gli id indicati nel messaggio: si scaricano dal tab «Pesi modelli» dell'app.

## 2. Scelta del motore (`engine`)

| Situazione | Motore |
|---|---|
| Parlato lungo (oltre 15 s: nessun limite, la durata segue l'audio), voce precisa della libreria o clonata, due persone che dialogano, controllo fine della voce coi marcatori | **`longcat`** (LongCat-Video-Avatar 1.5) |
| Clip breve 5–15 s, voce naturale generata dal modello insieme al video (labiale perfetto) più suoni d'ambiente/musica nello stesso passaggio; timbro opzionale da una registrazione | **`minimax_h3`** |

Con **LongCat** la voce la produce la catena TTS dell'app (tutti i marcatori §5 vengono eseguiti davvero); con
**MiniMax H3** la voce la genera il modello: i marcatori diventano indicazioni di recitazione (§5.3).
Licenza MiniMax H3: territori esclusi UE, Regno Unito, USA, Corea del Sud — se l'utente è lì, usa `longcat`.

## 3. Immagine iniziale (`image`)

| `image.mode` | Quando | Campi |
|---|---|---|
| `text` (A) | la persona non esiste ancora | `prompt` (descrizione dettagliata: età, capelli, abiti, espressione, sfondo, luce, inquadratura a mezzo busto frontale), `aspect` |
| `image` (B) | l'utente dà una foto | `image_path` (path locale: lo script lo carica) |
| `character` (C) | stesso personaggio, cambia pettinatura/abiti/sfondo… | `character: {ref_images: [foto] \| asset: "@Nome", changes: {hair, outfit, background, expression, pose, lighting, extra}, keep_background}` |
| `character_multi` (D) | come C ma con più foto dello stesso personaggio | `character.ref_images` fino a 10 (Qwen-Image 2.1) |
| `none` | solo LongCat senza foto: persona e scena nascono dal prompt video | — |

`engine` dell'immagine: `qwen_image21` (default, fino a 10 riferimenti, licenza non commerciale) oppure
`krea2_edit` (≤2 riferimenti). `profile`: `nvfp4` (RTX 50xx) · `fp8` · `int8`; `turbo`: `""|"4"|"6"`.
Per un avatar parlante: soggetto frontale, bocca visibile, niente mani davanti al viso.

## 4. Audio (`audio`)

| `audio.mode` | Uso |
|---|---|
| `tts` | una persona: `text` (con marcatori §5), `speakers: [ {…} ]` |
| `dialogue` | due persone: `text` = copione con righe `Nome: battuta` (o `A:` / `B:`), `speakers: [A, B]` con `name` uguale ai nomi del copione; `gap_ms` = pausa fra le battute |
| `file` | audio già pronto: `audio_path` (e `audio2_path` per la persona 2 con `turn_mode` `add`/`para`) |

**Speaker** (`speakers[]`): `name`, `engine` (`kokoro` · `edge` · `handy_clone` · `cosyvoice2` · `f5tts` ·
`xtts2` · `fish` · `vibevoice`), UNA sola fonte di voce fra `voice_id` (id della libreria da `--options`, o id del
motore: `it-IT-ElsaNeural`, `it-IT-DiegoNeural`, `if_sara`, `im_nicola`…), `voice_description` (es. «uomo anziano,
voce calda e roca»: l'LLM sceglie la voce) o `clone: {sample_path, transcript}` (clonazione, motori con clone);
poi `language` (`it-IT`, `en-US`…), `speed` 0.5–2, `pitch` semitoni −12…+12, `volume_db` −30…+12, `emotion`
(emozione di base di tutta la voce; i marcatori la sovrascrivono localmente).

Altri campi: `pad_start_ms`, `pad_end_ms` (silenzio prima/dopo), `target_duration_s` + `fit` (`pad` · `speed` ·
`error`), `normalize_lufs` (es. −16), `music: {prompt, engine: "acestep"|"heartmula", volume_db: -18, ducking: true}`
(musica di sottofondo generata, si abbassa quando si parla), `sfx: {mode: "auto"|"prompt"|"none", prompt,
volume_db}` (effetti sonori; `auto` = dedotti dal video finale), `vocal_separation` (pulisce la voce da musica
nei file dati), `voice_reference_path` (solo H3: registrazione di cui copiare il timbro).

## 5. Marcatori espressivi della voce (uguali per tutti i motori)

Si scrivono **dentro il testo** delle battute (`audio.text` o `video.h3.dialogue_lines[].text`). Si possono
annidare; i valori si combinano con quelli dello speaker (velocità ×, tono +, volume +). Sempre chiudere i tag.

| Effetto | Marcatore | Note |
|---|---|---|
| pausa | `[pause:800ms]` `[pause:1.5s]` `[pause]` | `[pause]` = 500 ms; massimo 10 s |
| respiro | `[breath]` | ≈ 350 ms, prima di una frase importante |
| emozione | `[emotion:happy]…[/emotion]` | `neutral` `happy` `sad` `angry` `whisper` `calm` `excited` `fearful` `surprised` `tender` `serious` |
| sussurro | `[whisper]…[/whisper]` | uguale a `[emotion:whisper]` |
| velocità | `[speed:1.2]…[/speed]` | 0.5–2.0 (1 = normale) |
| accelerando / rallentando | `[accel:0.9-1.4]…[/accel]` | la velocità passa gradualmente dal primo al secondo valore, frase per frase (usa virgole per dare i punti di passaggio) |
| tono | `[pitch:+2]…[/pitch]` | semitoni −12…+12 |
| volume | `[volume:-6]…[/volume]` | dB −30…+12; scorciatoie `[loud]…[/loud]` (+5 dB) e `[soft]…[/soft]` (−6 dB) |
| enfasi | `[emph]parola[/emph]` | +3 dB e +1 semitono sulla parola |
| SSML compatibile | `<prosody rate="+20%" pitch="+2st" volume="-3dB">…</prosody>` `<emotion type="sad">…</emotion>` | accettati e convertiti |

### 5.1 Come usarli bene
- La punteggiatura conta quanto i marcatori: virgole e punti danno il respiro naturale; usa `[pause]` per le
  pause *drammatiche* (suspense, battuta comica, cambio di argomento), non dopo ogni frase.
- Le emozioni vanno sulle frasi, non su tutto il testo: un monologo credibile alterna `calm` → `excited` →
  `serious`. Per l'emozione di fondo usa `speakers[].emotion`.
- `[accel]` è il modo per l'ansia che cresce, una lista che accelera, un climax; `[speed:0.85]` per solennità.
- Enfasi con parsimonia: una o due parole chiave per frase.
- Non mettere marcatori dentro le parole e non scrivere didascalie tra parentesi tonde: verrebbero lette.
- Controlla sempre con `--markup` prima di inviare.

### 5.2 Esempio (LongCat, una persona)
```
[emotion:calm]Signore e signori,[/emotion] [pause:600ms] stasera vi confesso una cosa.
[breath] [whisper]Il mio gatto sa usare il telecomando.[/whisper] [pause:800ms]
[accel:0.95-1.35]Prima i documentari sui pesci, poi quelli sugli uccelli, poi le televendite di tonno,[/accel]
[emotion:surprised][loud]e ieri ha ordinato un acquario![/loud][/emotion] [pause:500ms] [emph]Online.[/emph]
```

### 5.3 Con MiniMax H3
H3 non ha un TTS: genera lui la voce. Gli stessi marcatori vengono tradotti in **indicazioni di recitazione**
(«angrily», «in a whisper», «speeding up», «with deliberate pauses»…) fuori dal tag `<d>` del prompt, e il testo
detto resta pulito. Quindi: marca l'**emozione prevalente** della battuta e il ritmo; le micro-regolazioni
(pitch di 1 semitono, pause precise al millisecondo) con H3 non hanno effetto garantito. Una battuta per riga in
`video.h3.dialogue_lines: [{speaker, language, text}]`; lingue: Italian English Chinese Spanish French German
Portuguese Japanese Korean Russian Arabic. Tieni il parlato entro la durata (≈ 2,5 parole al secondo).

## 6. Video (`video`)

Campi comuni: `prompt` (in inglese rende meglio: persona, azione, scena, luce, movimento di camera, gesti
naturali, «talks to the camera»), `seed` (−1 = casuale), `postprocess: {face_restore: "codeformer"|"gfpgan",
interpolate_fps: 50}`.

**LongCat**: `resolution` `480p` (veloce) | `720p`; `aspect` (senza foto) `16:9 9:16 1:1 4:3 3:4`;
`resize_mode` `crop`|`default`; `max_duration_s` (0 = tutto l'audio); `precision_profile` `int8`|`fp8`|`bf16`;
`ref_img_index` 0–30 (basso = più fedele alla foto, 30 = meno gesti ripetuti); `mask_frame_range` 1–8;
`offload_kv_cache` per video lunghi a 720p; `negative`; `multi_person.bboxes` per due persone
(`{person1: [y0,x0,y1,x1], person2: […]}` in frazioni 0–1, altrimenti metà sinistra/destra automatiche).
Tempi indicativi su RTX 5090: ~2,5 min di calcolo per ogni 3,2 s di video a 480p.

**MiniMax H3**: `precision_profile` `nvfp4`|`fp8`|`int8`; `h3: {duration_s: 5–15, dialogue_lines, soundscape
(suoni d'ambiente), music_desc (musica di sottofondo), turbo_steps: ""|"4"|"8" (8 = consigliato: veloce e
dettagliato), text_encoder}`; per il timbro di una voce reale: `audio.voice_reference_path` (+ una foto).

## 7. Richieste complete di esempio

**A · da testo, monologo con voce della libreria (LongCat):**
```json
{"engine": "longcat",
 "image": {"mode": "text", "engine": "qwen_image21", "profile": "nvfp4", "aspect": "16:9",
           "prompt": "photo of a cheerful 70-year-old Italian grandmother with silver curls and round glasses, flowery apron, cosy kitchen with copper pans, soft window light, medium shot, facing the camera"},
 "audio": {"mode": "tts", "text": "[emotion:tender]Tesoro,[/emotion] [pause:500ms] il ragù non si fa in fretta. [breath] [emph]Sei ore.[/emph] [pause] [emotion:angry]E niente panna![/emotion]",
           "speakers": [{"name": "Nonna", "engine": "edge", "voice_id": "it-IT-ElsaNeural", "language": "it-IT", "speed": 0.95}],
           "music": {"prompt": "gentle mandolin folk, warm", "volume_db": -20, "ducking": true}},
 "video": {"prompt": "The grandmother talks to the camera with warm expressive gestures, stirring a pot, natural head movements", "resolution": "480p", "seed": -1}}
```
**B · da foto dell'utente, dialogo a due (LongCat):**
```json
{"engine": "longcat",
 "image": {"mode": "image", "image_path": "C:/Users/me/Pictures/coppia.jpg"},
 "audio": {"mode": "dialogue", "gap_ms": 300,
           "text": "Marco: [emotion:excited]Ho prenotato il viaggio![/emotion]\nGiulia: [emotion:surprised]Dove?[/emotion] [pause:400ms]\nMarco: [whisper]Al Polo Nord.[/whisper]\nGiulia: [emotion:angry][loud]Marco![/loud][/emotion]",
           "speakers": [{"name": "Marco", "engine": "edge", "voice_id": "it-IT-DiegoNeural"},
                        {"name": "Giulia", "engine": "edge", "voice_id": "it-IT-ElsaNeural"}]},
 "video": {"prompt": "A couple on a sofa talking to each other, lively expressions", "resolution": "720p"}}
```
**C · personaggio con cambi (H3, voce generata dal modello):**
```json
{"engine": "minimax_h3",
 "image": {"mode": "character", "character": {"asset": "@Dorian", "changes": {"outfit": "astronaut suit without helmet", "background": "inside a space station, Earth in the window"}, "keep_background": false}},
 "video": {"prompt": "Live-action, cinematic medium close-up, the astronaut floats gently and talks to the camera", "precision_profile": "nvfp4",
           "h3": {"duration_s": 8, "turbo_steps": "8", "soundscape": "soft hum of life support, faint beeps",
                  "dialogue_lines": [{"speaker": "Dorian", "language": "Italian", "text": "[emotion:excited]Da quassù l'Italia sembra uno stivale che balla![/emotion]"}]}}}
```
**D · più riferimenti + timbro di una voce registrata (H3 Ref2VA):**
```json
{"engine": "minimax_h3",
 "image": {"mode": "character_multi", "character": {"ref_images": ["C:/foto/a1.jpg", "C:/foto/a2.jpg", "C:/foto/a3.jpg"], "changes": {"hair": "short bob", "outfit": "black turtleneck"}}},
 "audio": {"voice_reference_path": "C:/audio/mia_voce.wav"},
 "video": {"prompt": "The woman presents a new product on a minimal stage, confident gestures", "h3": {"duration_s": 10, "turbo_steps": "4", "dialogue_lines": [{"language": "Italian", "text": "Vi presento [emph]Luna[/emph]. [pause] La lampada che si accende quando sorridete."}]}}}
```

## 8. Regole
- Mai inventare path: usa solo file che l'utente ha indicato o path restituiti da job precedenti.
- Motore e modelli sempre espliciti: se qualcosa manca, il job fallisce con gli id da scaricare — riferiscilo,
  non cambiare motore di nascosto.
- Contenuti per adulti solo se l'utente lo chiede e l'edizione lo consente (NSFW), mai con persone reali
  riconoscibili senza consenso.
- Risposta finale all'utente: breve descrizione di cosa è stato creato + il path `VIDEO:` (+ link).
