# Cast del Video Musicale — piano (ricicla Cinema Studio)

## Context
I video musicali devono poter avere **personaggi ricorrenti e riconoscibili** tra i clip: cantante/i,
musicisti (con **coerenza dello strumento**), comparse. L'utente vuole: generare **o** selezionare l'identità
di ciascuno, usarla nei vari clip, con **frequenza di apparizione** e **location** per clip — il tutto
**opzionale/attivabile** nella creazione delle descrizioni dei clip. La fondazione tecnica (coerenza volto
per-scena via **Krea 2 Edit** + `scene['char_ref']`) è **già** in codice; qui si aggiunge il sistema di casting.

Principio guida: **riusare i componenti di Cinema Studio (Film Studio)**, che ha già un sistema cast completo.

## Cosa Cinema Studio offre GIÀ (da riciclare, non riscrivere)
- **Modello dati** (`core/film_studio/db.py`): tabelle `asset` (character/prop/location) + `image` (foto di
  riferimento, `owner_kind='asset'|'variation'`) + `asset_variation` + `lora`.
- **Resolver** `gen_service.resolve_refs(text)` (`core/film_studio/gen_service.py:52`): estrae `@Nome[#Variante]`
  → `{ref_images, loras, trigger_map}` **per-scena** (ogni scena porta i riferimenti dei suoi personaggi).
- **Cast manager UI** `ui/film_studio/asset_manager_tab.py`: crea asset, **genera** (gen_service.gen_image con
  face_ref) o carica foto, gestisce varianti.
- **compose_shot_prompt** (`gen_service.py:113`): prompt shot = @ref risolti + camera/luce + continuity + Style Bible.
- **gen_service.gen_image** (già esteso): `coherence_engine='krea2_edit'` usa i riferimenti della scena
  (face_ref+ref_images, max 2) → volto coerente.
- **gen_scene_image** (music, già esteso): `scene['char_ref']` per-scena + `coherence_engine`.

## Design (nuovo = sottile strato sopra i componenti riciclati)

### 1. Il video musicale È un progetto con cast
- Il video musicale diventa un `project` `kind='music_video'` nel **db di Film Studio** → eredita `asset`/
  `image`/`variation`. NIENTE nuovo schema. Il cast (cantante/i, musicisti) sono `asset` character del progetto.

### 2. Cast manager (riciclato)
- Riusa `asset_manager_tab` (o una sua versione ridotta "Cast") aperta dal tab Musica: per ogni personaggio
  Nome + **Ruolo** (Cantante/Musicista/Comparsa) + **Identità** (genera via gen_service.gen_image t2i, oppure
  carica foto → salvata come `image` dell'asset).
- **Musicista → Strumento**: campo strumento con (a) **tag testuale costante** (es. "red Gibson Les Paul") SEMPRE
  iniettato nel prompt del clip (default, scala a N musicisti), e (b) **foto strumento opzionale** come 2°
  riferimento immagine per un "hero shot" (usa il 2° slot multi-ref di Krea 2 Edit: personaggio=frame1,
  strumento=frame2; quindi 1 personaggio + strumento per clip-hero).
- Aggiunge alla riga asset: **frequenza di apparizione** (% target sui clip) + ruolo. (nuovi campi in `fields`
  JSON dell'asset — nessuna migrazione schema.)

### 3. Storyboard con cast (estende l'LLM esistente)
- Estende il prompt dello storyboard musicale (`music_video_service`/LLM) per: (a) conoscere il **cast**
  (nomi+ruoli+strumenti), (b) inserire i tag `@Nome` nei clip rispettando la **frequenza** per-personaggio,
  (c) assegnare una **location** per clip. Output per scena: `chars` (lista @Nome), `location`, `img_prompt`.
- Il resolver `resolve_refs(clip.description)` (RICICLATO) trasforma i `@Nome` → `ref_images` → si popola
  `scene['char_ref']` (single) o multi-ref (personaggio+strumento). `location` → coda del prompt.

### 4. Generazione (già pronta)
- `gen_scene_image` instrada `scene['char_ref']` + `coherence_engine='krea2_edit'` → volto/strumento coerenti
  per clip. I clip che NON citano cast restano t2i puro. Tutto **opt-in**.

### 5. Controlli UI (nel tab Musica / Music Video Studio)
- Pannello **Cast** (apre l'asset_manager riciclato): definisci personaggi + identità + strumento + ruolo +
  frequenza.
- **Editor clip** dello storyboard: per-clip multiselect **personaggi** (@Nome), **location**, toggle on/off.
- Switch **Coerenza volto** (già aggiunto) globale.

## File (riusa ≫ nuovo)
**Riusati**: `core/film_studio/db.py` (asset/image/variation), `gen_service.resolve_refs`/`compose_shot_prompt`/
`gen_image`, `ui/film_studio/asset_manager_tab.py`, `gen_scene_image` (già esteso).
**Nuovi/estesi**: `core/music_video_service.py` (storyboard con cast+frequenza+location; risoluzione @Nome→
scene['char_ref']), `ui/tabs/music_tab.py` + `ui/music_video_studio.py` (pannello Cast + editor clip +
frequenza), i18n.

## Domanda aperta (per l'utente)
Strumenti: proposto **tag testuale costante (default, scala a più musicisti) + foto strumento opzionale come
2° riferimento per i clip-hero**. Confermare, o preferisci solo-immagine / solo-testo?

## Verifica
- Definisci 2-3 personaggi (1 cantante + 1 chitarrista con strumento) → genera storyboard → i clip con `@Cantante`
  mostrano lo stesso volto; il chitarrista ha lo stesso volto + stessa chitarra; la frequenza rispetta i target;
  le location cambiano per clip. Giudizio a occhio ([[test-loop-protocol]]).
