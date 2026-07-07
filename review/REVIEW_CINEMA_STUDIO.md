# Review approfondita — AI Film Studio (Cinema Studio)

*Panel a 5 voci (Regista/DOP · Montatore/Post · Filosofo dello spirito artistico · UX Designer · Workflow-optimizer
+ Software Architect), verificato sul codice. La visione originaria (`docs/ai-film-studio-spec*.md`) è usata come
**fonte di idee, non vincolo**. GUI solo PySide6, 2 varianti (Classica + Pro).*

---

## Verdetto in una riga (per voce)
- **Regista/DOP:** ottimo *generatore di sequenze coerenti asset-driven*, ma **non parla ancora cinema** — la
  grammatica di ripresa esiste solo come stringhe morte, la coerenza si ferma a PuLID+prompt.
- **Montatore:** la "timeline" è una *barra di riordino*, il master un `concat` netto.
- **Filosofo:** avete costruito l'**officina**, manca il **teatro**: lo strumento chiede *"cosa produci?"* non
  *"cosa vuoi dire?"*. Mancano i verbi dell'autore — **vedere, scegliere, dirigere**.
- **UX:** è una *fabbrica di bottoni "Genera"*; il gesto più poetico (`@Nome` = chiamare in scena i personaggi)
  è oggi il più fastidioso (dialog modale + bug undo).
- **Architetto:** ben stratificato; 3 nodi critici risolvibili **senza refactor** (catena LoRA scollegata,
  scheduler fragile, hardcode modello = solo wiring).

**Da riconoscere (già eccellente):** migrazione a **backend NATIVO** (supera la visione ComfyUI), continuity
engine esplicito, PuLID + keyframe i2v, `series`-scope per asset/LoRA, doppiaggio + re-lip-sync, SFX MMAudio
per-azione, animatic con dialoghi, export OTIO/NLE, pattern **Pro-per-sottoclasse** (factory `_make_*_tab`).

---

## 1. Bug critici verificati (rompono funzioni già promesse)

| # | Bug | Dove | Effetto |
|---|---|---|---|
| **B1** | `#Variante` parsata ma **non usata** in generazione (query su `owner_kind='asset'`) | `core/film_studio/gen_service.py:34-53` | `@Marco#insanguinato` genera Marco **pulito** → TUTTO il sistema Variazioni/Abbigliamento non arriva al motore |
| **B2** | **LoRA video mai caricata**: `resolve_refs` prende solo `flavor='image'` anche per gli shot video | `gen_service.py:47` | le LoRA video addestrate non hanno effetto |
| **B3** | **Dialoghi multi-battuta rotti**: tutte le battute a `start_s=t` uguale; lip-sync usa solo la 1ª | `montage_tab.py:224-243`, `storyboard_tab.py:288-306` | campo/controcampo **impossibile** |
| **B4** | **Seed=7 per tutti gli shot** (batch non passa seed) | `gen_service.py:60`, `storyboard_tab.py:189` | nessun lock identità per-personaggio |
| **B5** | **Modello immagine hardcoded flux**: handler non passa `arch` (gen_image lo accetta, backend supporta flux/flux2/zimage/ernie) | `film_studio_window.py:28` | niente z-image/ernie |
| **B6** | **LoRA orfane**: i wizard copiano i file ma non fanno INSERT in tabella `lora` | `lora_video_tab.py:389-401` | binding manuale, catena rotta |
| **B7** | **Aspect ratio master = controllo morto** | `montage_tab.py:37`, `otio_export.render_master:31` | 2.39:1/9:16 ignorati |
| **B8** | **`continuity.warnings()` = codice morto** (mai chiamata) | `continuity.py:49-59` | script-supervisor anti-salto invisibile |
| **B9** | **Continuity per-asset mai popolata** (solo time/weather/light/mood) | `storyboard_tab.py:70-76` | metà continuity engine inattiva |
| **B10** | **`delete_project` cascade incompleta** (dimentica image/audioclip/artifact/job/lora); `duplicate_project` non copia montage/lora/image | `db.py:163-194` | righe orfane |
| **B11** | **Rigenerazione distruttiva** (sovrascrive) + keyframe **non visibili** nel tree | `storyboard_tab.py:188-195, 274, 50` | nessuna curatela |
| **B12** | **Bug undo `@Nome`**: gestore `#` fa `setPlainText` → azzera cursore/undo | `asset_dialog.py:94-107` | incidente su copioni lunghi |

---

## 2. Le 5 voci (sintesi)

### Regista/DOP — la grammatica cinematografica è scollegata
`SHOT_TYPES`/`CAMERA_MOVES`/`TRANSITIONS` (`style_catalog.py:25-33`) sono **costanti mai importate** fuori dal
file: il campo `camera` dello shot è testo libero dall'LLM, non un valore scelto/validato. `TRANSITIONS` è
folklore (nessun campo `transition` in DB, master = `ffmpeg concat` = solo stacchi). **Nessuna copertura**
(dialogo = 1 shot → shot-reverse-shot impossibile, B3), **niente eyeline/180°/screen-direction**, **niente
camera/lighting continuity** (lente/focale/temp.colore/LUT — `continuity` ha un solo enum `light`). Il video
**non usa PuLID** (identità dipende solo dal keyframe → deriva non controllata sui video lunghi).

### Montatore/Post — la timeline non è un editor
Clip video solo trascinabili in orizzontale per riordinare (`timeline_widget.py`); tracce audio non movibili.
**Mancano trim con maniglie in/out (clip sempre `in_s=0..durata`), ripple/roll, J/L-cut, snapping, marker,
multi-traccia video, zoom, waveform, player/scrubbing.** Transizioni reali: nessuna. Color/LUT: nessuno.
Ducking = volume statico (no sidechain/fade). NLE esporta **solo V1, senza trim né audio**. *Positivo:*
doppiaggio+re-lip-sync, SFX per-azione, animatic sono avanzati e ben pensati.

### Filosofo — manca l'anima autoriale
Lo strumento è costruito attorno al verbo **"Genera"**. Mancano i tre verbi dell'autore: **vedere, scegliere,
cambiare**. Quattro qualità assenti: (a) **l'intenzione persistente** (una *logline emotiva* iniettata in ogni
prompt — oggi solo stringhe forbidden/required); (b) **la voce autoriale come vincolo insistente** (stili come
tendina volatile, non *Style Bible* gravitazionale); (c) **la serendipità controllata** (il seed-lock c'è ed è
il gesto giusto, ma manca la *variante*: rigenerare **sovrascrive** → serendipità distruttiva); (d) **il "no"
creativo/direzione** ("questo, ma più solo" — oggi la rigenerazione è un rimescolamento neutro, non una
direzione). Principio-cardine: **ogni output deve essere un invito a rispondere, non un punto d'arrivo** →
triade *tieni / varia / dirigi* su ogni artefatto.

### UX — l'ergonomia dei momenti chiave
- **`@Nome` è un dialog modale bloccante** (`asset_dialog.py:88`) con doppio-click e liste testuali senza
  miniature; il gestore `#` **azzera l'undo** (B12); non si può scrivere `@` letterale. → **completer inline
  non-modale** (stile menzione Notion/Slack) con thumbnail, filtro live, varianti a 2° livello via
  `QTextCursor` (preserva undo), `@` letterale se non scelgo.
- **Modo "Interattivo" non implementato**: è solo un prefisso combobox + **1** chiamata LLM (`script_tab.py:14,
  95`). → tavolo di **co-scrittura** (bivi + beat + "Genera Racconto").
- **Stadio "Racconto cinematografico" fuso col copione** (`script_tab.py:116`, split sul primo `\n\n`). →
  stadio narrativo separato prima del breakdown tecnico.
- **"Rigenera segmento" contestuale assente**: `↻ Rigenera` rigenera **tutto** appendendo "(varia con un seed
  diverso)". → context-menu su selezione (*simile / cosa-cambiare…*) che sostituisce solo il segmento.
- **Nessuna galleria/accettazione set immagini**; **keyframe non visibili** nel tree (B11). → thumbnail + strip
  varianti + pin non-distruttivo.
- **Classica/Pro INVERTITE**: la Pro aggiunge *più guida* (empty-state, banner-flusso, `pro_tabs.py:79`) → chi
  inizia ne ha bisogno, il Pro la trova invadente. **Riallineare**: guida→Classica, densità/controllo→Pro.

### Workflow-optimizer + Architect — automazione con controllo
- **Catena LoRA img→video già abilitata** da `Subject.real_media` (le foto reali diventano clip i2v nel dataset
  video, `lora_video_service.py:51,336`) → serve solo un **orchestratore** con gate umani, riusando i signal
  `finished_lora` e l'idempotenza degli stadi. Chiude anche le **LoRA orfane** (INSERT DB oggi mancante, B6).
- **Scheduler fragile**: nessun `recover()` (job `running` orfani post-crash), nessun handler `lora_train`,
  `est` mai persistito, nessun retry.
- **Video subprocess-per-shot** (`gen_service.py:107`) ricarica LTX ~22B ad ogni ripresa = **collo di bottiglia
  enorme** in "Genera Tutti" → worker video **persistente** (scheduler seriale → 1 worker residente).
- **Data model**: niente FK/indici/CASCADE, `commit()` per-op (→ WAL), migrazioni ad-hoc (→ `user_version`),
  costanti duplicate (`NEG` divergente tra due file), **versioning schema inutilizzato** (shot sovrascrivono
  path fissi invece di `add_version()`).
- **Doppia GUI**: pattern corretto (Pro override solo le factory) — **mantenere**; unico punto fragile
  `_wrap_master_confirm` con `disconnect()` (meglio un hook virtuale `_before_master()`).

---

## 3. Use case (dove brilla / si inceppa)
- **A) Corto noir 3′, 2 attori ricorrenti** — *brilla* end-to-end (sheet→LoRA/PuLID→breakdown→Auto→master);
  *si inceppa*: `@Marco#insanguinato` esce pulito (B1), seed unico (B4), look noir senza LUT/lente.
- **B) Dialogo campo/controcampo (6 battute)** — **caso peggiore**: 1 shot/6 battute → 1 video, 1 battuta
  lip-syncata, le altre sovrapposte (B3). Serve la **copertura**.
- **C) Spot 30″ prop-hero** — *brilla* (turntable prop, SFX, ducking, FCPXML); *si inceppa*: niente trim per il
  ritmo, transizioni assenti, aspect ignorato (B7).
- **D) Serie con asset condivisi** — *brilla* (series-scope); *si inceppa*: continuity solo forward, niente
  bibbia cross-episodio, niente lock lente/luce di serie.

---

## 4. Roadmap prioritizzata

### P0 — SUBITO (sbloccano la coerenza già promessa)
1. **B1** — `resolve_refs` usa `#Variante`: query immagini+LoRA su `owner_kind='variation'`.
2. **B2** — carica LoRA del **flavor** giusto (image per keyframe, video per shot).
3. **B5** + selezione modello: `img_arch/img_model/vid_model` in `project.config`; dropdown dal detector; handler
   passano `arch/model`. Estrarre **`core/model_selection.py`** condiviso col Music Video Studio.
4. **B4** — seed per-shot/per-personaggio nel batch keyframe.
5. **B6** — REGISTER dei wizard → **INSERT DB `lora`** + binding asset; estendere `lora` con `weight`/`base_arch`.
6. **B10** — cascade `delete_project` completa + `duplicate_project` completo; **WAL** + `scheduler.recover()`.

### P1 — SUBITO/MEDIO (coerenza + curatela + montaggio base)
7. **Copertura minima dialoghi** (shot→N inquadrature, 1 battuta/inquadratura, lip-sync per battuta) → B3.
8. **Context-menu "Rigenera segmento"** (simile / cosa-cambiare) su editor narrativi (solo selezione, reversibile).
9. **Galleria set attore** (Accetta / Rigenera-angolo / Rigenera-tutto-seed) + **abbigliamento 6-angoli**.
10. **Thumbnail nel tree** + strip varianti + rigen **non distruttiva** (pin seed).
11. **@-completer inline** non-modale + fix undo `#` (B12) + `@` letterale.
12. Attivare **`continuity.warnings()`** + popolare continuity per-asset (B8/B9); **aspect reale** nel master (B7).

### MEDIO
13. **Orchestratore LoRA img→video** (`core/film_studio/lora_orchestrator.py` + tabella `lora_pipeline`) con
    GATE umani, riusando `Subject.real_media` + `finished_lora`.
14. **Worker video persistente** (elimina subprocess-per-shot); coda ordinata per (arch, set-LoRA).
15. **Style Bible / director's intent** persistente (logline + palette-lock + registi-faro) in ogni prompt.
16. **Stadio Racconto** separato + **co-scrittura interattiva** vera.
17. **Vista-pagina scena** immersiva + **riallineamento Classica/Pro** (guida→Classica, densità→Pro).
18. **Camera/lighting continuity** (lente/focale/temp/LUT); `transition` + `xfade/afade`; montaggio
    **trim/ripple/marker/snapping/waveform**; `in_s/out_s` reali fino all'NLE (+audio).
19. Storyboard usa **`add_version()`** (storia take); migrazioni versionate.

### MOONSHOT (visione autoriale/di frontiera)
- **Virtual script supervisor** LLM (rilegge il breakdown, segnala 180°/eyeline/oggetti/ora del giorno).
- **Copertura automatica** (LLM propone il découpage per stile).
- **Reference-chaining visivo** (ultimo frame N → conditioning keyframe N+1).
- **DoP-agente** (propone camera+lighting dalla Style Bible, dialoga).
- **Montaggio a ritmo emotivo** (curva di tensione → modula durate + musica).
- **"Il secondo sguardo"** (agente-critico post-master = il "no" creativo finale).
- **Serendipità dirigibile** (cursore "quanto sorprendimi" per scena).
- **Character-bible + continuity retroattiva**; auto-QA VLM closed-loop; multi-GPU per-GPU.

---

## 4-bis. Stato implementazione (branch `feature/cinema-studio-review`)
**✅ Fatto e committato** (compila + smoke test 14/14; B1/B6 verificati con test DB):
- **P0**: B1 (#Variante→variante), B2 (LoRA flavor), B4 (seed per-shot), B5 (modello da config + `core/
  model_selection.py` + dropdown storyboard), B10 (cascade+WAL+`recover()`+lora.weight/base_arch).
- **P1**: B3 (dialoghi in sequenza+overflow), B7 (aspect master letterbox), B8 (continuity warnings attivi),
  B11 (thumbnail keyframe nel tree), B12 (undo `@Nome#` via QTextCursor), context-menu «Rigenera segmento»
  (Copione+Scenografia).
- **Medio**: orchestratore LoRA img→video (`lora_orchestrator.py` + tabella `lora_pipeline` + `register_lora`
  chiude B6) con **driver GUI nel LoRA Studio** (gate galleria-foto); **Style Bible** (logline/palette/look/
  registi iniettata in ogni prompt).

**⏳ Rimane** (feature grandi, richiedono validazione live — training/rendering — o rework UI su 2 varianti):
- **B9** popolare continuity per-asset (wardrobe/condition/emotion dalla UI).
- **Copertura dialoghi** shot-reverse-shot (shot→N inquadrature) — il singolo intervento «cinema» maggiore.
- **Galleria set attore** (Accetta/rigen-angolo) + abbigliamento 6-angoli; **rigen non-distruttiva** (strip varianti).
- **@-completer inline** non-modale (oggi resta il dialog; l'undo però è già corretto, B12).
- **Worker video persistente** (elimina subprocess-per-shot); **transizioni** (campo+`xfade`); **timeline pro**
  (trim/ripple/marker); **camera/lighting continuity** (lente/LUT); **stadio Racconto** + co-scrittura;
  **riallineamento Classica/Pro**; storyboard→`add_version()`.
- **Moonshot** (virtual script supervisor, DoP-agente, montaggio a ritmo emotivo, «secondo sguardo», ecc.).
- **feature_matrix**: aggiungere le nuove sotto-capability (style_bible, lora_pipeline, model-select) al seed
  esterno con i18n_key — oggi funzionano perché dentro `cap.filmstudio` (già gated) e dev è permissivo.

## 5. Sintesi finale del panel
Avete un'**officina** eccellente; manca il **teatro**. Tre atti riportano l'anima — **intenzione persistente**
(Style Bible + logline che perseguitano ogni prompt), **curatela** (vedere/scegliere/scartare senza distruggere),
**dialogo** (LLM che rilancia invece di erogare) — e tre gesti li rendono ergonomici: **@-completer inline**,
**triade tieni/varia/dirigi**, **thumbnail + galleria varianti**. Sul piano tecnico i 3 sblocchi maggiori sono
**far arrivare `#Variante`/LoRA-video/seed al motore** (P0), l'**orchestratore LoRA con gate** e il **worker
video persistente**. Nessuno di questi duplica codice: si innestano sui pattern esistenti (`attach()`, factory
`_make_*_tab`, `shot.seed`, `Subject.real_media`, `finished_lora`).
