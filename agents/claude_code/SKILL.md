---
name: ltxstudio
description: Pilota la REST API LOCALE di LTX Studio per CREARE e MODIFICARE immagini, video, film/videoclip, musica, modelli 3D, rig/skinning e animazioni. Job asincroni che ritornano link di download. Copre testo/immagine/audio->immagine e video (t2v/i2v/a2v, inpaint, kontext, upscale, lip-sync, controlnet, extend), canzoni (HeartMuLa/ACE-Step), TTS/voce, immagine->3D (TRELLIS), rig, mondo 3D, e shot di film. Usa quando l'utente vuole generare/modificare foto, video, musica, film o modelli 3D da QUALSIASI input. L'API si avvia da sola e si spegne dopo 30 min di inattivita.
target: Claude Code
---

# LTX Studio — skill per Claude Code

Guidi la generazione/modifica di **immagini, video, film, videoclip, musica, modelli 3D, rig/skinning e
animazioni** tramite la **REST API locale di LTX Studio** su `http://127.0.0.1:8200` (Swagger `http://127.0.0.1:8200/docs`).

## 0. Avvio automatico e spegnimento
- **Prima di ogni sessione**, assicura che l'API sia attiva: esegui `python -m api.autostart`
  (ritorna `READY http://127.0.0.1:8200` quando pronta) oppure `GET http://127.0.0.1:8200/health`. Se `health` non risponde,
  l'API è spenta: **avviala** con l'autostart, non fallire.
- L'API **si spegne da sola dopo 30 minuti** di inattività (nessuna richiesta e nessun job). È normale:
  al prossimo uso rilancia l'autostart. Non tenerla viva artificialmente.

## 1. Modello di esecuzione (ASINCRONO)
1. Ogni azione è un `POST` che ritorna `{ "job_id": "...", "status": "pending" }`.
2. **Fai polling** di `GET http://127.0.0.1:8200/jobs/{job_id}` finché `status` = `success` o `error`.
   - `status=running` include `progress` (0–1) e `message`.
   - a `success` la risposta ha `files: [{ path, name, download_url }]`.
   - a `error` la risposta ha `error` (spesso indica un prerequisito da installare).
3. **Consegna** all'utente i `download_url` (`GET http://127.0.0.1:8200/download?path=...`).
4. **Un solo job GPU alla volta**: la coda è seriale. Non lanciare job in parallelo; attendi il precedente.

## 2. File in ingresso
- Carica i file dell'utente UNA volta con `POST http://127.0.0.1:8200/files/upload` (multipart, campo `file`) →
  ritorna `{ path }`. Riusa quel `path` nei campi `*_path` (`image_path`, `video_path`,
  `audio_path`, `mask_path`, `mesh_path`, ...).
- **Concatenamento**: l'output di un job (`files[0].path`) è un path locale sul server: passalo
  direttamente come input del passo successivo, senza download/upload intermedi.

## 3. Scelta dell'endpoint (intento → chiamata)
- descrizione → immagine: `POST /image/generate` (`arch` flux/flux2/zimage/ernie, `model`)
- modifica foto (istruzione): `POST /image/edit`; colore: `/image/recolor`; area precisa:
  `/image/inpaint` (serve `mask_path`; creala con `/image/segment`); estendi bordi: `/image/outpaint`
- upscale/restauro foto: `/image/upscale`; volto da foto su un'altra: `/image/faceswap`;
  identità del volto in generazione: `image/generate` con `pulid_face`
- descrizione → video: `POST /video/generate` (t2v). foto → video: stesso con `image_path` (i2v).
  audio → video / canto: stesso con `audio_path` (a2v)
- keyframe a tempo: `/video/keyframes`; prolunga: `/video/extend`; rigenera sezione: `/video/retake`;
  modifica da istruzione/stile: `/video/v2v_edit`; copia movimenti: `/video/controlnet`
- lip-sync (foto+voce): `/video/lipsync` (`separate_vocals` per isolare la voce, `talk_region` mouth/face)
- avatar parlante / filmato parlato da un testo: `POST /avatar/generate` (LongCat 1.5 o MiniMax H3; immagine da
  testo/foto/personaggio, voce TTS/dialogo con marcatori `[pause:500ms]` `[emotion:happy]…[/emotion]`
  `[speed:1.2]` `[accel:0.9-1.3]` `[pitch:+2]` `[volume:-6]` `[whisper]` `[emph]` `[lang:en]`; `language`
  = lingua del TESTO, distinta dalla lingua della voce: indicala sempre); opzioni installate
  `GET /avatar/options`. Guida completa + client: skill **ltx-avatar-film** (`skills/ltx-avatar-film`)
- qualità video: `/video/enhance` (SeedVR2/FlashVSR), +fps `/video/rife`, volto `/video/faceswap`
- musica/canzone: `/music/song` (engine heartmula|acestep); sonorizza video: `/music/sonify`;
  videoclip canzone→video: `/music/music_video`; voce TTS: `/audio/tts`; trascrivi: `/audio/transcribe`
- 3D da foto: `/threed/image_to_3d`; scheletro+skinning: `/threed/rig`; scena 3D: `/threed/world`;
  converti mesh: `/threed/convert`
- film a inquadrature: `/film/script` (trattamento), `/film/asset_image` (personaggi/keyframe),
  `/film/shot_video` (shot dello storyboard)

## 4. Regole
- **SFW/NSFW**: default SFW. Per contenuti adulti chiedi conferma esplicita UNA volta; non dedurre il
  consenso dal prompt.
- **Selezione modello = compito tuo**: proponi default sensati; chiedi all'utente solo il necessario.
- **Prerequisiti**: alcuni endpoint (SeedVR2, FlashVSR, TRELLIS, WorldGen, HeartMuLa, ACE-Step, MMAudio,
  TTS clone, face-swap, PuLID) richiedono venv/pesi installati. Se un job torna `error` con "non
  installato/preflight", spiega all'utente come installarlo (script in `scripts/install_*.py`).
- **Job lunghi** (video/film/musica = minuti): annuncia l'attesa, fai polling con backoff (2–5 s),
  non ripetere il POST.

## 5. Contratto macchina
- OpenAPI completo: `http://127.0.0.1:8200/openapi.json` (o `agents/openapi.json`).
- Tool function-calling: `agents/tools.openai.json`. Catalogo neutro: `agents/manifest.json`.
