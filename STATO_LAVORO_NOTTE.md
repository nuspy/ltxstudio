# Stato lavoro notturno — V2V (ReTake / Just_Talk) + fix video lunghi

> Tutto è **validato a secco** (sintassi, import, struttura del prompt contro `object_info` in cache).
> **NON ho potuto lanciare generazioni** (ComfyUI era spento) → i test visivi finali sono da fare tu.
> **Riavvia l'app** per caricare tutte le modifiche.

---

## ✅ FATTO e VALIDATO a secco

### 1. ReTake (rigenera una sezione di un video) — COMPLETO
- **UI**: tab Video → riquadro **«🎬 Editing video (V2V)»** → radio **Normal / Talk / ReTake**.
  Scegliendo **ReTake** si apre l'accordion: picker video (o **«Usa ultima generazione»**), anteprima,
  **slider Inizio/Fine** (bloccati a [3s, durata−3s]), scelta **audio** (Tieni originale / Rigenera).
- **Backend** (`template_engine.apply_retake` → `_v2v_source_mask`): carica il video, lo VAE-encoda,
  applica `LTXVAudioVideoMask` sul range [start,end] → il sampler **rigenera SOLO la sezione**,
  preserva il resto. Riusa il superset (modello/sampler/scheduler/decode già a posto).
- **Orchestrazione** (`main_window.on_generate`): valida (dialog «manca X»), carica il video in
  ComfyUI (`_upload_video`), imposta la risoluzione = quella del sorgente, e (se «Tieni audio
  originale») muxa l'audio sorgente sul risultato a fine generazione.
- **Validazione a secco**: prompt costruito con **0 errori strutturali**.
- **DA TESTARE**: genera un ReTake e verifica che la sezione cambi e il resto resti intatto, e che la
  giunzione sia pulita. Se la sezione non si raccorda, possiamo aggiungere le ancore latenti ai bordi
  (LTXVAddLatentGuide inizio/fine, già disponibile dal task #60).

### 2. Just_Talk (lip-sync) — PARZIALE (manca solo la maschera VOLTO)
- **UI**: radio **Talk** → accordion: picker+anteprima, **maschera** (Bocca/Volto), **offset**,
  toggle **Re-Imagine**. Voce = **albero del tab Audio** (file → voce TTS+testo → dialogo nel prompt).
- **Backend** (`apply_just_talk`): maschera **temporale** + **audio** (lip-sync guidato dall'audio),
  validato a secco. ✅
- **⚠ DA COMPLETARE LIVE — maschera SPAZIALE del volto**: il workflow di riferimento usa il nodo
  `FaceSegment` che **non è nell'object_info in cache** (forse custom node non caricato), e
  `MediaPipeFaceMask` ha un parametro `regions` dinamico non determinabile a secco. Finché non lo
  verifichi con ComfyUI acceso, **Talk si comporta come Re-Imagine** (rigenera la finestra temporale
  col lip-sync dall'audio, senza limitare alla sola bocca). L'hook è documentato in `apply_just_talk`.
- **DA FARE con ComfyUI**: aprire il workflow `V2V_Just_Talk` di riferimento, vedere il vero nodo di
  maschera volto (FaceSegment o MediaPipe) e i suoi parametri, e completare il wiring spaziale.

### 3. Fix VIDEO LUNGHI (richiesti prima di dormire)
- **Segmento nero** → risolto da #60 (`LTXVAddLatentGuide` invece di `LTXVAddGuidesFromBatch`). ✅
- **Giunzione incoerente** → `latent_idx` 0 (la coda va all'inizio del nuovo segmento). ✅
- **Durata sbagliata (25s→16s)** → ora misuro le durate **reali** con ffprobe (non stime) → l'accumulo
  continua finché il video vero raggiunge il target. ✅
- **Audio "crash" alle giunzioni** → filtri audio per-segmento + crossfade triangolare. ✅
- **Moto che si ferma** → è la rampa I2V; leva: **Avanzate → Overlap/drop** più alto (es. 32-40)
  copre la rampa. Se non basta, abbassiamo la `strength` della guida latente.

---

## 🔧 Cosa fare al risveglio (in ordine)

1. **Riavvia l'app** (carica tutto).
2. **Video lungo**: rigenera (25s) → verifica durata corretta, niente nero, niente crash audio,
   giunzione coerente. Se il moto si ferma ancora, alza **Overlap** in Avanzate.
3. **ReTake**: tab Video → Editing video → ReTake → scegli un video → imposta Inizio/Fine → Genera.
4. **Talk**: idem in modalità Talk (per ora senza maschera bocca = Re-Imagine). Scrivi il dialogo nel
   prompt («…she says: "…", perfect lip-sync»).
5. Riportami i risultati: completo la maschera volto di Talk e affino ReTake/giunzioni.

## File toccati
- `core/template_engine.py` — apply_retake, apply_just_talk, _v2v_source_mask, dispatch V2V.
- `ui/widgets/v2v_panel.py` — NUOVO: pannello V2V (accordion+radio+slider+anteprima+validazione).
- `ui/tabs/video_tab.py` — innesto del pannello V2V.
- `ui/main_window.py` — orchestrazione V2V + fix video lunghi (durata reale, audio per-segmento) + mux audio ReTake.
- `scripts/ui2api.py`, `scripts/ui2api_fill.py` — convertitore UI→API (usato per analisi; i template
  convertiti in `templates/v2v/` restano come riferimento, NON usati a runtime: il backend innesta sul superset).
