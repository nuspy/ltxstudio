# AI Film Studio — Specifica Funzionale & Architetturale del Modulo *Text-to-Film*

> **Destinatario:** Claude Code.
> **Natura del documento:** specifica funzionale orientata a user story, workflow, meccanismi, modello dati e regole. Non contiene esempi di codice di proposito: l'implementazione (framework, librerie, struttura file) è demandata a Claude Code, che deve poterla derivare da queste regole.
> **Cosa NON è:** non è un capitolato chiuso. Dove il comportamento non è specificato, applica le best practice di produzione cinematografica e di pipeline generative descritte qui.

---

## 0. Indice ragionato

1. Visione e principi guida
2. Integrazione con l'app madre (runtime condiviso)
3. Glossario e modello di dominio
4. I due percorsi di coerenza (Rapido vs LoRA) — selezionabili
5. Architettura dell'informazione e pattern UX trasversali
6. Modulo **Asset Manager** (Attori, Props, Ambienti, Architetture)
7. Modulo **LoRA Studio** (training, libreria condivisa, composizione)
8. Modulo **Voice Studio** (clonazione, generazione da descrizione, binding)
9. Modulo **Copione** (4 input → formato cinematografico)
10. Modulo **Scenografia** (Scene → Inquadrature → immagini → video)
11. Modulo **Montaggio / Master**
12. **Continuity Engine** (coerenza trasversale)
13. **Job Scheduler** multi-GPU
14. Modello dati (entità e relazioni)
15. Pipeline end-to-end
16. Use case professionali aggiuntivi
17. Decisioni da confermare

---

## 1. Visione e principi guida

Il modulo trasforma un **testo** (libro, racconto, mini-idea, tema, o costruzione interattiva con LLM) in un **film completo**: scenografia → scene → inquadrature → immagini di riferimento → video → montaggio. Si innesta su un'app esistente di generazione AI di immagini/video/audio, riusandone le capacità.

**Principi non negoziabili:**

- **Lineare ma flessibile.** Il percorso "felice" è un workflow a tab in successione (Input → Copione → Scenografia → Montaggio). Ma ogni stadio è ri-entrabile e ogni artefatto è rigenerabile singolarmente senza distruggere il resto.
- **Due velocità, una sola interfaccia.** Ogni progetto sceglie un **percorso di coerenza**: *Rapido* (senza training) o *LoRA* (alta coerenza). Vedi §4.
- **Non distruttivo.** Ogni rigenerazione produce una **versione**; nulla viene sovrascritto in modo irreversibile. L'utente può tornare a una versione precedente di qualsiasi artefatto (testo, immagine, video, voce).
- **Coerenza come stato esplicito, non come fortuna.** La coerenza tra inquadrature e tra scene è governata da un **Continuity State** persistente (§12), non lasciata al caso del prompt.
- **Human-in-the-loop configurabile.** Due modalità: **Supervisionata** (checkpoint con approvazione a ogni stadio — default professionale) e **Auto** (testo → master con revisione finale). Selezionabile a livello progetto.
- **Asset-centrico.** Attori, props, luoghi, architetture, LoRA e voci sono asset di prima classe, riusabili tra progetti e tra episodi di una serie.

---

## 2. Integrazione con l'app madre (runtime condiviso)

Il modulo si apre in una **finestra/contesto indipendente** ma **condivide il runtime** dell'app madre, così da invocarne direttamente le funzioni invece di reimplementarle.

**Capacità ereditate dall'app madre (da riusare, non riscrivere):**

| Capacità                                                                    | Uso nel modulo                                                    |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Generazione immagini (Flux + modelli intercambiabili: z-image, ernie, ecc.) | Character sheet asset, keyframe inquadrature                      |
| Flux Kontext                                                                | Generazione/variazione coerente per asset e dati di training LoRA |
| Generazione video (LTX 2.3)                                                 | Generazione inquadrature, clip di training video-LoRA             |
| Video lunghi senza limite                                                   | Inquadrature lunghe / scene continue                              |
| PuLID                                                                       | Coerenza volto nel **Percorso Rapido**                            |
| Multi-referenza su frame differenti                                         | Conditioning keyframe iniziale/intermedio/finale                  |
| Wizard creazione LoRA (immagini + video)                                    | Cuore del **Percorso LoRA** (§7)                                  |
| Clonazione voci                                                             | Base del Voice Studio (§8)                                        |

**Backend di generazione:** ComfyUI. Serve un **livello di astrazione "Generation Service"** che:

- costruisce/sottomette grafi ComfyUI parametrici;
- espone un workflow **Image-to-Image multi-referenza** che accetta *N immagini di riferimento + descrizione + modello selezionabile* e (nel Percorso LoRA) *uno o più LoRA con relativi trigger e pesi*;
- espone un workflow **Video** con conditioning su keyframe iniziale/finale + referenze + LoRA video;
- i modelli disponibili (immagine e video) e i loro parametri sono gestiti nei **Settings**.

**LLM providers:** riusa l'astrazione provider già esistente in `C:\Projects\ForexGPT\src\forex_diffusion\llm_providers`. Tutte le chiamate LLM (espansione storia, breakdown scene/inquadrature, rigenerazioni, captioning) passano per quel layer, con provider/modello configurabili nei Settings.

**Persistenza:** database relazionale per metadati ed entità; storage file per gli artefatti binari (immagini, video, audio, pesi LoRA). Ogni record asset/artefatto referenzia i propri file.

**Esecuzione:** un **Job Scheduler** (§13) media tutte le generazioni pesanti, sfruttando l'hardware multi-GPU disponibile.

---

## 3. Glossario e modello di dominio

Terminologia da usare in modo **coerente** in codice e UI.

- **Progetto.** Un film singolo *oppure* un episodio di una serie. Contiene input, copione, scenografia, montaggio.
- **Serie.** Contenitore di più Progetti/Episodi che condividono asset e LoRA (cast principale, luoghi ricorrenti).
- **Episodio.** Progetto appartenente a una Serie.
- **Asset.** Entità riutilizzabile e coerente tra scene/progetti. Quattro tipi: **Attore**, **Prop**, **Ambiente/Luogo**, **Architettura**.
- **Variazione.** Stato alternativo di un asset (es. età diversa, pettinatura, condizione "insanguinato/sporco/bagnato", oppure per i luoghi una "sub-area" interna).
- **Abbigliamento (solo Attori).** Set di immagini che vestono una variazione dell'attore (fronte, retro, lato dx/sx, 3/4 dx/sx).
- **Character Sheet / Asset Sheet.** Set canonico di immagini coerenti di un asset (turnaround multi-angolo, espressioni, posa a Y). È sia output per l'utente sia **materiale di training** per le LoRA.
- **LoRA.** Adattatore addestrato su un asset per coerenza forte. Due gusti per asset: **LoRA-immagine** e **LoRA-video**, addestrati dalla **stessa base immagini**. Ha un **trigger token** univoco. Ha uno **scope** (serie o episodio).
- **Voce.** Profilo vocale (clonato da referenza *oppure* generato da descrizione), **legabile a un Attore**.
- **Copione (formato cinematografico).** Racconto strutturato derivato dall'input, con riassunto trama iniziale.
- **Scena.** Porzione di video continuativa: stesso luogo, successione temporale ininterrotta. Composta da Inquadrature.
- **Inquadratura (Shot / Ripresa).** Il singolo video che compone una scena, con stacco, propria angolazione e durata. Unità atomica di generazione video. (*"Ripresa" e "inquadratura" sono sinonimi; in codice usa `Shot`.*)
- **Keyframe.** Immagine di riferimento di un'inquadratura: **iniziale** (sempre), **intermedi** (opzionali), **finale** (opzionale, per raccordo/transizione).
- **Continuity State.** Stato persistente che descrive luce, colore, mood, orario, meteo, stato degli asset (abbigliamento, condizione, emozione) lungo scene e inquadrature. Governa la coerenza (§12).
- **@Reference.** Riferimento testuale a un asset dentro il testo: `@Nome` (asset) o `@Nome#Variante` (variazione specifica). Risolve, a runtime, in referenze immagine (Percorso Rapido) o in LoRA+trigger (Percorso LoRA).

**Gerarchia:** `Serie → Episodio/Progetto → Copione → Scena → Inquadratura → {Keyframe, Video}`. Trasversali al progetto: `Asset → Variazione → {Abbigliamento, LoRA-img, LoRA-video}` e `Voce`.

---

## 4. I due percorsi di coerenza (selezionabili)

La scelta è esposta in UI **a livello di progetto** (con possibilità di override per singolo asset). È la decisione architetturale centrale del modulo.

### 4.1 Percorso Rapido (PuLID + multi-referenza)

- Nessun training. La coerenza si ottiene passando al workflow ComfyUI le **immagini di riferimento** dell'asset (dal Character Sheet) + PuLID per il volto + multi-ref per il conditioning dei keyframe.
- **Pro:** immediato, costo zero di training, ideale per bozze, animatic, asset secondari, progetti brevi.
- **Contro:** coerenza buona ma non massima su lunghe sequenze, identità che può "derivare", controllo limitato su pose/espressioni inedite.

### 4.2 Percorso LoRA (alta coerenza)

- Per ogni asset rilevante si addestra una **LoRA-immagine** e una **LoRA-video** (§7) dalla **stessa base immagini**.
- In generazione, l'`@Nome` viene risolto nel **trigger token** della LoRA, la LoRA viene caricata con un peso, e (opzionalmente) i keyframe restano come conditioning aggiuntivo.
- **Pro:** coerenza forte e ripetibile, riuso tra episodi/progetti, controllo su espressioni/condizioni inedite, qualità "broadcast".
- **Contro:** richiede una fase di training (tempo/GPU), va gestita la **composizione di più LoRA** (§7.5).

### 4.3 Tabella di scelta

| Criterio                | Rapido                              | LoRA                                   |
| ----------------------- | ----------------------------------- | -------------------------------------- |
| Setup                   | nessuno                             | training per asset                     |
| Coerenza su serie lunga | media                               | alta                                   |
| Riuso cross-episodio    | per immagini                        | nativo (LoRA condivisa)                |
| Costo prima resa        | basso                               | alto                                   |
| Costo per resa N-esima  | costante                            | ammortizzato                           |
| Uso consigliato         | bozze, asset minori, progetti brevi | cast principale, serie, qualità finale |

### 4.4 Modalità mista (raccomandata)

Il modulo deve permettere **mix per asset**: cast principale in LoRA, comparse in Rapido. La selezione di default è di progetto; ogni asset può dichiarare il proprio percorso. Il Generation Service deve gestire entrambi nello stesso job (alcuni soggetti via LoRA, altri via referenza/PuLID).

---

## 5. Architettura dell'informazione e pattern UX trasversali

### 5.1 Struttura a workflow (tab)

Tab principali, in ordine logico, con scorrimento libero:

1. **Input & Setup** — tipo film, durata, stili, percorso di coerenza, modalità (Auto/Supervisionata), input testo.
2. **Copione** — racconto in formato cinematografico, editing, rigenerazioni.
3. **Scenografia** — scene e inquadrature, con scorrimento tra scene; generazione immagini e video.
4. **Montaggio / Master** — timeline, audio, export.

Trasversali (sempre accessibili, anche da finestre/pannelli dedicati): **Asset Manager**, **LoRA Studio**, **Voice Studio**, **Coda Job**, **Lista Progetti**.

### 5.2 Lista Progetti

Vista d'ingresso: elenco progetti (e serie), con stato del workflow, anteprima, riapertura, duplicazione, archiviazione. Ogni stadio (Copione, Scenografia) ha salvataggio esplicito; il salvataggio persiste *tutto* lo stato corrente con le modifiche.

### 5.3 Sistema **@Reference** + **AssetDialog** (unificato, una sola implementazione)

Comportamento identico ovunque si scriva testo che può citare asset (input, descrizione scena, descrizione inquadratura):

- Digitando `@` si apre **AssetDialog**: ricerca con filtri per **tipo** e, in funzione del tipo, per proprietà specifiche (per Attori: sesso, età, etnia, ecc.). Ricerca **LIKE** con modificatori **AND/OR/NOT**.
- Doppio click su un'anteprima → dettaglio asset → navigazione tra **variazioni** → selezione (`@Nome#Variante`).
- Dal dialog si possono **creare** nuovi asset di qualsiasi tipo o nuove varianti di un asset esistente (apre il rispettivo workflow di creazione).
- **Editing intelligente del riferimento (regola NOTA A):**
  - Cancellando solo `#Variante` da `@Nome#Variante` → AssetDialog si apre **direttamente nella sezione di selezione della variante** di quell'asset.
  - Cancellando `@Nome` → AssetDialog si apre nella **sezione iniziale** di selezione asset.
- **Propagazione:** se nella *descrizione di scena* si sostituisce un asset/variante, **tutte le inquadrature** della scena aggiornano il riferimento corrispondente.

### 5.4 Pattern di **rigenerazione** (unificato)

Tre granularità, stesso comportamento in tutto il modulo:

- **Rigenerazione segmento di testo.** Selezione testo → tasto destro → "Rigenera segmento" → dialog con due opzioni: *(a)* "rigenera simile" e *(b)* "cosa cambiare / cosa deve avvenire". Vale per copione, descrizione scena, descrizione inquadratura.
- **Rigenerazione singolo artefatto** (una immagine, un video, una posa del set). Bottone sotto l'artefatto. **Vincolo critico:** la rigenerazione di una singola immagine deve **preservare angolo, soggetto e tipo di rappresentazione** della precedente (es. "viso 3/4", "figura intera fronte, posa a Y").
- **Rigenerazione totale** (intero set/scena/copione). Usa un **seed diverso** per ottenere un'alternativa.
  Tutte producono **versioni** (§5.5), mai sovrascrittura cieca.

### 5.5 Versioning non distruttivo

Ogni artefatto (testo, immagine, video, voce, LoRA) mantiene una **cronologia di versioni** con seed/parametri/prompt usati. UI con selettore versione e "ripristina". I **seed** sono sempre visibili e bloccabili per riproducibilità.

### 5.6 Modalità Auto vs Supervisionata

- **Supervisionata (default):** ogni stadio richiede approvazione esplicita prima di procedere (copione → scenografia → immagini → video → master).
- **Auto:** il sistema attraversa gli stadi con scelte di default, fermandosi solo a errori/ambiguità forti, e presenta il risultato per revisione finale. Anche in Auto, ogni artefatto resta rigenerabile.

---

## 6. Modulo **Asset Manager**

CRUD su DB per quattro tipi di asset, ciascuno con generazione, variazioni e ricerca. Obiettivo: **coerenza** di attori/props/luoghi/architetture tra scene, inquadrature e progetti, riusando asset esistenti.

### 6.1 Attori

**Campi:** Nome; **Descrizione** (testo libero da cui si genera, es. "donna 23 anni, capelli neri, occhi verdi, muscolatura femminile definita, magra…"); campi strutturati per **sesso, età, etnia, altezza, capelli, occhi, corpo, segni particolari**; **tag** di ricerca.

**Generazione Character Sheet (bottone "Genera"):**

- Genera un set coerente: **volto a espressione neutra da più angoli** + **corpo in posa Y da più angolazioni**, derivato dalla descrizione (e/o da una o più **foto base** caricate dall'utente come riferimento).
- Le immagini del set **devono essere coerenti tra loro** (stesso soggetto, illuminazione/skin uniformi).
- **Rigenerazione singola** ("Rigenera" sotto ogni immagine): preserva **angolo + soggetto + tipo di rappresentazione** della foto rimpiazzata.
- **Accetta** (salva il set) oppure **Rigenera tutto** (nuovo seed → set alternativo con apparenza diversa).
- Tecnicamente: usa il workflow Image-to-Image multi-ref (foto base) + Flux Kontext per la coerenza del soggetto attraverso gli angoli.

**6.1.1 Variazioni.** Dopo l'accettazione, l'utente può creare set di **variazioni** via descrizione (es. età diversa, pettinatura, condizione "insanguinato/sporco/bagnato"). Stesso workflow di generazione/rigenerazione del set base. Ogni variazione è uno stato selezionabile come `@Nome#Variante`.

**6.1.2 Abbigliamento.** Per la base e per ogni variazione, set di abiti via descrizione, reso su **fronte, retro, lato dx, lato sx, 3/4 dx, 3/4 sx**. Stesso pattern (rigenerazione singola o totale). Se accettato, salvato in una **lista abbigliamenti** legata all'attore. *(Il workflow di generazione abbigliamento è identico a quello dell'edit dell'Attore.)*

**6.1.3 Ricerca.** LIKE non esatta, con AND/OR/NOT, su tag e caratteristiche. Risultati a lista con: nome, foto **viso di fronte**, **figura intera di fronte**, e azioni **edit/delete**.

**User story attori:**

- *Come regista* voglio definire un attore da descrizione testuale e ottenere un character sheet multi-angolo coerente, così da riusarlo identico in tutte le scene.
- *Come regista* voglio partire da foto reali di una persona per ancorare l'aspetto, mantenendo controllo creativo sulle variazioni.
- *Come regista* voglio una variante "ferito" del protagonista per il terzo atto, con il suo guardaroba, senza ricreare l'attore da zero.

### 6.2 Props

Stesso scheletro CRUD/generazione/variazioni/ricerca degli Attori, **senza abbigliamento**, con **viste multi-angolo** dell'oggetto e variazioni (es. "spada pulita" / "spada insanguinata").

### 6.3 Ambienti / Luoghi

Come sopra, con **viste da angoli differenti** e **sub-luoghi interni** al posto dell'abbigliamento: es. Luogo "Foresta" → interno "Radura nella foresta". Le variazioni coprono condizioni (giorno/notte, stagione, meteo). Ogni sub-luogo è navigabile come variante.

### 6.4 Architetture

Come Luoghi, con **sub-luoghi interni** all'architettura: es. Architettura "Castello" → interno "Camera del Re". Multi-angolo + variazioni (es. "sala del trono integra" / "in rovina").

### 6.5 Generation Service per immagini (requisito condiviso)

Un unico workflow ComfyUI Image-to-Image che:

- accetta **N immagini di riferimento** + **descrizione**;
- consente di **cambiare il modello** AI (z-image, ernie, flux, …) da Settings;
- nel Percorso LoRA, accetta **uno o più LoRA** con trigger e pesi;
- preserva, in rigenerazione singola, i parametri di inquadratura/posa.

---

## 7. Modulo **LoRA Studio**

Trasforma i Character/Asset Sheet in coerenza forte e riusabile. È l'estensione chiave rispetto al testo originale.

### 7.1 Principio di bootstrapping (best practice)

Stessa **base immagini** addestra sia LoRA-immagine sia LoRA-video:

1. Dal Character/Asset Sheet (e variazioni/abbigliamenti) si compone un **dataset canonico**: turnaround multi-angolo, espressioni, condizioni, abiti, posa Y.
2. Si addestra la **LoRA-immagine** (Flux). Questa migliora la generazione di **ulteriore materiale di training** coerente (espressioni/pose inedite), che arricchisce il dataset.
3. Dal dataset (immagini + brevi clip generate con LTX 2.3) si addestra la **LoRA-video** (LTX). Le clip di training preservano identità e movimento (sfruttando preservazione pose/movimento già disponibile nell'app).
4. Le due LoRA condividono lo **stesso trigger token** dell'asset, così `@Nome` funziona identico in immagini e video.

### 7.2 Captioning / tagging (best practice)

- Caption **consistenti** durante il training: `<trigger> , <tipo> , <vista/angolo> , <espressione/emozione> , <abbigliamento> , <condizione>`.
- L'identità è sempre veicolata dal **trigger token** (non da parole comuni che causerebbero collisioni di concetto).
- Il `@Nome` dell'app è un **alias UI** che mappa sul trigger reale in fase di build del prompt.

### 7.3 Trigger token univoci

Ogni LoRA possiede un trigger **raro e univoco** per evitare interferenze. Il modulo genera e registra il trigger; l'utente vede `@Nome`, il Generation Service inietta il trigger.

### 7.4 Libreria LoRA condivisa + scoping (serie/episodio)

- Le LoRA sono asset di prima classe, **condivisibili tra progetti** e **tra episodi** di una serie.
- **Scope:**
  - **Serie:** cast principale, luoghi/props ricorrenti dell'intera serie.
  - **Episodio/Progetto:** personaggi/luoghi/oggetti specifici di un episodio.
- **Combinazioni richieste:** un personaggio "guest" di scope episodio può ricomparire in un episodio successivo (resta agganciabile), mentre nello stesso episodio convivono LoRA serie + LoRA episodio.
- **Versioning LoRA:** ogni LoRA è versionata e ri-addestrabile (miglioramento dataset → nuova versione) senza rompere i progetti che la usano (pin di versione per progetto).

### 7.5 Composizione multi-LoRA (best practice)

Quando un'inquadratura contiene più soggetti LoRA:

- **trigger distinti** (garantito da §7.3) per evitare "concept bleed";
- **pesi moderati e bilanciati**; il modulo espone pesi per-LoRA con default sensati e avverte sopra una soglia di LoRA simultanee;
- ove il backend lo consenta, **controllo regionale/attentivo** (ciascuna LoRA agisce sull'area/soggetto pertinente) per ridurre l'interferenza;
- limite consigliato e fallback: se troppe LoRA in una stessa resa, suggerire split dell'inquadratura o uso del Percorso Rapido per i soggetti minori.

### 7.6 Wizard di training (UX) — interamente supervisionato

Stadi guidati, **ognuno con gate manuale** (l'utente avanza allo step successivo solo quando approva): *seleziona asset/variazioni → costruisci dataset (auto + revisione/aggiunta/rimozione immagini) → captioning automatico (editabile) → parametri (con preset) → training (via Job Scheduler) → preview di validazione → accetta/ri-addestra → avanza*.

- La preview deve mostrare il soggetto in pose/espressioni **non** presenti nel dataset, per misurare la generalizzazione.
- **Controllo e rigenerazione:** l'utente può rigenerare i campioni di validazione, scartarli, modificare dataset/caption/parametri e **ri-addestrare** quante volte vuole prima di accettare. Nessun avanzamento automatico: il passaggio allo step successivo è sempre un'azione esplicita.
- Vale anche fuori dal wizard: ogni LoRA in libreria è ispezionabile, testabile (genera campioni col trigger) e ri-addestrabile in qualsiasi momento.

**User story LoRA:**

- *Come showrunner* voglio addestrare una volta il cast principale e riusarlo in tutta la serie con coerenza identica.
- *Come regista* voglio comporre nella stessa scena la LoRA della protagonista (serie) e quella dell'antagonista guest (episodio) senza che i volti si contaminino.
- *Come tecnico* voglio ri-addestrare una LoRA con dataset migliorato senza rompere gli episodi già renderizzati.

---

## 8. Modulo **Voice Studio**

Estende il modulo voce esistente (oggi solo clonazione) con due capacità nuove.

### 8.1 Clonazione (esistente)

Profilo vocale da **referenza reale** (riusa la funzione dell'app madre).

### 8.2 Generazione da descrizione (nuovo)

Creazione di voci **senza referenza reale**, da **descrizione testuale** (es. "uomo anziano, baritono, leggera raucedine, accento del nord, tono autorevole e stanco"). Parametri esposti dove sensato (genere percepito, età, timbro, ritmo, accento, emozione di base). Output: un **profilo vocale** salvato e riutilizzabile come quello clonato.

### 8.3 Binding voce ↔ personaggio

Ogni **Attore** può avere una **voce legata** (clonata o generata). In fase di generazione audio dell'inquadratura, i dialoghi attribuiti a `@Nome` usano automaticamente la sua voce.

### 8.4 Dialoghi per inquadratura + lip-sync

La descrizione dell'inquadratura specifica **chi parla** (`@Nome`) e cosa dice. Il modulo genera l'audio dei dialoghi con la voce legata e lo associa alla clip.

- **Lip-sync (core):** l'audio dei dialoghi viene **sincronizzato con il labiale** del personaggio nel video dell'inquadratura. Il lip-sync è uno step di post sulla clip generata (riusa/estende le capacità dell'app madre o un nodo ComfyUI dedicato) ed è rigenerabile singolarmente come ogni altro artefatto.
- **Timing dialoghi:** la durata del parlato deve stare dentro la durata dell'inquadratura; il sistema segnala se un dialogo eccede e propone di accorciare il testo o allungare l'inquadratura.

**User story voce/audio:**

- *Come regista* voglio dare a un personaggio inventato una voce credibile senza avere un doppiatore, partendo da una descrizione.
- *Come regista* voglio che tutti i dialoghi del protagonista usino sempre la stessa voce, automaticamente, in tutte le scene.
- *Come regista* voglio che il labiale dei personaggi sia sincronizzato con i dialoghi generati.
- *Come distributore* voglio rigenerare i dialoghi in un'altra lingua mantenendo la stessa voce del personaggio e ri-sincronizzando il labiale.

---

## 9. Modulo **Copione**

### 9.1 Setup (tab Input)

- **Durata** del film (minuti+secondi).
- **Stile** delle riprese: lista di **~50 stili** selezionabili (surrealista, minimalista, b/n, noir, cartoon, iperrealista, ecc.). *Claude Code popola e mantiene la lista in configurazione.*
- **Tipologia film**, eventualmente **multipla** (es. comico + horror): lista di **~20 generi** (fantascienza, comico, romantico, thriller, ecc.).
- **Percorso di coerenza** (Rapido/LoRA/Misto) e **modalità** (Auto/Supervisionata).

### 9.2 Quattro tipi di input

1. **Storia intera** (libro/racconto): la LLM la riduce a **formato cinematografico**.
2. **Mini-storia / idea**: la LLM la **espande** in formato cinematografico.
3. **Random da tema**: solo il tema → la LLM **genera ed espande**.
4. **Interattivo con LLM**: costruzione passo-passo, la LLM propone alternative ed esegue istruzioni dell'utente; bottone **"Genera Racconto"** consolida in formato cinematografico.

In **tutti** i casi: il racconto si apre con un **riassunto breve della trama**, rispetta **durata** e **genere(i)**, e supporta `@Reference` (§5.3) per asset ricorrenti.

### 9.3 Editing del copione

- Editing manuale libero.
- **Rigenerazione segmento** (pattern §5.4).
- **Rigenerazione completa** del copione (bottone), anch'essa sottoposta a editing.

**User story copione:**

- *Come autore* voglio incollare un racconto e ottenere un trattamento cinematografico della durata target.
- *Come autore* voglio costruire la storia conversando con l'AI, che mi propone biforcazioni.
- *Come autore* voglio rigenerare solo un paragrafo chiedendo "rendilo più teso" senza toccare il resto.

---

## 10. Modulo **Scenografia** (Scene → Inquadrature → Immagini → Video)

Dal copione accettato si genera lo **script/copione tecnico**: **Scene**, ciascuna divisa in **Inquadrature** in successione.

### 10.1 Breakdown scene

Per ogni **Scena** la LLM genera:

- **Descrizione** della scena;
- **Elenco asset**: attori, props, ambienti, architetture. Se ricorrenti/esistenti → `@Nome` (o `@Nome#Variante`); se nuovi → solo citati (e creabili al volo via AssetDialog);
- **Stile immagini**: eredita lo stile globale, con **override per scena** per ragioni artistiche (istruzione LLM);
- **Durata scena**: vincolo — la **somma delle durate scene = durata film** (istruzione LLM).

### 10.2 Breakdown inquadrature

Per ogni **Inquadratura**:

- **Durata in secondi** (vincolo: somma durate inquadrature = durata scena, coerente col contenuto);
- **Descrizione finalizzata al video**: movimenti camera, tipo di inquadratura, movimenti dei soggetti, espressioni, rumori/SFX, **dialoghi con attore parlante** (`@Nome`), colori, attori, props, luoghi, architetture; asset esistenti come `@Nome#Variante`;
- **Descrizioni dei keyframe**: immagine **di apertura** sempre; **riferimenti intermedi/finali** quando utili (movimenti complessi, raccordi);
- **Coerenza obbligatoria** con la scena: luci, colori, mood, meteo, orario, asset → tutti allineati al **Continuity State** (§12).
- Il **prompt alla LLM** deve essere "intelligente": esplicita la finalità (generare descrizioni atte alla resa video/immagine), preferibilmente con **output strutturato JSON** per parsing affidabile dei campi (durata, asset, dialoghi, keyframe, camera).

### 10.3 Editing scena/inquadratura

Stessi pattern di §5.4: rigenerazione segmento, singola inquadratura, intera scena. Ogni scena è presentata **interamente** (tutte le inquadrature e gli elementi) su una pagina, con scorrimento tra scene; a fondo pagina: **Salva** e **Genera Immagini**.

### 10.4 Generazione immagini di riferimento

"Genera Immagini" produce i **keyframe** descritti per **tutte** le inquadrature, inviando gli asset di riferimento al workflow **Image-to-Image** (Percorso Rapido) **oppure** caricando le **LoRA** corrispondenti agli `@Nome` (Percorso LoRA), con i keyframe come conditioning aggiuntivo.

- Ogni immagine è mostrata **dentro** la descrizione dell'inquadratura (keyframe iniziale + eventuali ref intermedi/finali + ref asset rigenerati).
- Bottone di **rigenerazione** sotto ciascuna (preserva inquadratura/posa, §5.4).
- **Coerenza tra immagini** della stessa inquadratura e della stessa scena: vincolo forte (stesso color grade, luce, identità asset).

**Strategia keyframe (best practice):** iniziale = obbligatorio (entrypoint del video); intermedi = on-demand del sistema quando il movimento/durata lo richiede; finale = quando serve raccordo con l'inquadratura successiva (continuità di posa/posizione). In Percorso LoRA i keyframe restano utili come guida composizione anche se l'identità è garantita dalla LoRA.

### 10.5 Generazione video

- **"Genera Video"** (per scena): genera i video di tutte le inquadrature della scena via LTX 2.3, con conditioning sui keyframe (iniziale/finale) e, in Percorso LoRA, con la **video-LoRA** dei soggetti (trigger da `@Nome`).
- **"Genera Tutti"** (a fine ultima scena): genera l'intero film. *(Dettagli di scheduling in §13.)*
- Coerenza intra/inter-scena governata da Continuity State.

### 10.6 Salvataggio e progetti

Salvataggio esplicito di copione e scenografia (persiste tutte le modifiche). Il workflow è a tab: **Input, Copione, Scenografia** (scorrimento scene), **Montaggio**. Lista progetti riapribile.

**User story scenografia:**

- *Come regista* voglio che, cambiando l'attrice nella descrizione di scena, tutte le inquadrature si aggiornino da sole.
- *Come regista* voglio rigenerare una sola inquadratura mantenendo identiche luce e palette delle vicine.
- *Come regista* voglio che le durate delle inquadrature sommino esattamente alla durata della scena e del film.

---

## 11. Modulo **Montaggio / Export**

Porta dalle clip generate a un progetto montabile. **Decisione confermata dall'utente:** l'output principale **non** è un master "cotto", ma l'**esportazione degli spezzoni come file progetto** apribile in software di editing esterni (Filmora e altri NLE), dove l'utente fa il finishing. Il master renderizzato resta un'opzione aggiuntiva.

### 11.1 Timeline interna

- **Timeline** automatica: scene e inquadrature ordinate, con durate dal breakdown; editing manuale di ordine, tagli e durate.
- **Transizioni** tra inquadrature/scene (cut, dissolvenze, ecc.) con default per stile/genere, modificabili.
- **Tracce:** video (clip per inquadratura) + audio multitraccia (vedi 11.2).

### 11.2 Audio (core, non più estensione)

Comparto audio completo, mixato in timeline:

- **Dialoghi** dal Voice Studio (voce legata al personaggio) con **lip-sync** sincronizzato (§8.4).
- **Musica:** generazione per scena/atto guidata dal mood del Continuity State (cue musicali con attacco/stacco, intensità).
- **Ambient / atmosfere:** letti di fondo coerenti col luogo e l'orario (pioggia, città, foresta…).
- **SFX / FX:** effetti puntuali ancorati ad azioni dell'inquadratura (passi, porte, colpi, ecc.), idealmente desumibili dai "rumori" già descritti nell'inquadratura (§10.2).
- Ogni elemento audio è un **artefatto versionato e rigenerabile** singolarmente, con volume/ducking gestibili in timeline (es. musica che abbassa sotto i dialoghi).

### 11.3 Anteprima / Animatic

Pass rapido a bassa risoluzione (keyframe in sequenza con i tempi e, se presente, audio di dialogo) prima della resa video piena, per validare ritmo e continuità a basso costo.

### 11.4 Export verso software di editing (requisito primario)

Esporta lo **spezzonato** + la timeline + le tracce audio in formati progetto/interscambio che i principali NLE sanno aprire, così l'utente rifinisce dove preferisce:

- **Filmora** (target citato esplicitamente): generare un **file progetto** importabile, oppure — se il formato nativo non è documentabile/stabile — un pacchetto **media + timeline** in un formato che Filmora importa (vedi sotto) così da ricostruire la sequenza.
- **Formati di interscambio standard da supportare:** **FCPXML** (Final Cut / DaVinci Resolve / Filmora), **AAF** e **OTIO** (OpenTimelineIO) come pivot neutro, **EDL** come fallback minimale. Premiere via XML/AAF.
- **Cosa contiene l'export:** clip video per inquadratura con in/out e ordine, marker di scena, tracce audio separate (dialoghi, musica, ambient, SFX) con i loro tempi, transizioni, frame-rate e aspect ratio del progetto, e i **media linkati** (cartella asset esportata accanto al file progetto, percorsi relativi).
- **Strategia consigliata a Claude Code:** usare **OpenTimelineIO** come modello interno della timeline e i suoi adapter per emettere i vari formati (FCPXML/AAF/EDL), riducendo il lavoro per-software. Verificare per ciascun target la compatibilità reale (alcuni NLE importano FCPXML, altri AAF) e mappare le feature non supportate con un fallback graceful (es. transizione non supportata → cut + marker).
- **Pacchetto di export auto-contenuto:** una cartella con `progetto.<fmt>` + `/media` + un manifest (mapping inquadratura→file, scene, durate) per portabilità e ri-link.

### 11.5 Export master (opzionale)

In aggiunta all'export progetto: render di un **file unico** nel formato/aspect ratio scelto (preset 16:9, 2.39:1, 9:16, ecc.), risoluzione/codec selezionabili, con l'audio già mixato. Utile per anteprime/condivisione, non sostituisce il progetto NLE.

**User story montaggio/export:**

- *Come regista* voglio esportare tutte le inquadrature come progetto Filmora con le tracce audio separate, per montare e rifinire lì.
- *Come regista* voglio un animatic in pochi minuti per giudicare il ritmo prima di spendere GPU sui video.
- *Come montatore* voglio che musica e ambient arrivino su tracce dedicate, già sincronizzate, così posso solo ritoccare i livelli.
- *Come regista* voglio, in alternativa, un master 2.39:1 già mixato per una rapida condivisione.

---

## 12. **Continuity Engine** (coerenza trasversale)

Componente che rende la coerenza uno **stato esplicito**, non un effetto collaterale del prompt.

- **Continuity State per scena/inquadratura:** orario, meteo, luce, color grade/mood; per ogni asset presente: variazione attiva, abbigliamento, **condizione** (insanguinato/sporco/bagnato…), emozione corrente.
- **Propagazione:** lo stato si propaga in avanti tra inquadrature e tra scene; le modifiche (es. l'attore si sporca in S2 e resta sporco fino al cambio) sono tracciate come transizioni.
- **Iniezione nei prompt:** ogni prompt immagine/video riceve lo stato pertinente, così i campi (luce, palette, condizione asset) sono allineati senza che l'utente li ripeta.
- **Guardrail & warning:** il motore segnala incoerenze (es. un'inquadratura chiede "notte" dentro una scena "giorno", o un asset con condizione che salta indietro senza motivo) e propone correzioni.
- **Elementi vietati/forzati per progetto:** liste di negative/forbidden e di must-have a livello progetto/scena per stabilità.

---

## 13. **Job Scheduler** multi-GPU

Tutte le generazioni pesanti (immagini, video, training LoRA) passano per una **coda**.

- **Coda con priorità e stato** (in attesa/in corso/fatto/errore), retry, log.
- **Stima costo/tempo** prima del lancio (per inquadratura/scena/film), così l'utente decide.
- **Distribuzione multi-GPU:** sfrutta l'hardware disponibile (es. workstation 4×GPU) parallelizzando job indipendenti (inquadrature diverse, training in background).
- **Batch "Genera Tutti":** pianifica l'intero film come insieme di job, con possibilità di pausa/ripresa e di rigenerare selettivamente i falliti.
- **Non blocca la UI:** l'utente continua a lavorare su copione/asset mentre la coda gira.

---

## 14. Modello dati (entità e relazioni)

Descrizione concettuale (Claude Code definisce schema/migrazioni). Campi chiave indicati; ogni entità ha id, timestamp, e **cronologia versioni** dove rilevante.

- **Series** (1—N Project).
- **Project/Episode** — riferimento a Series (nullable); input config (durata, generi, stili, percorso, modalità); riferimenti a Script, Scenes, Montage.
- **Asset** — tipo {actor|prop|location|architecture}; campi descrittivi/strutturati; tag; scope di riuso; (N—N con Project via referenze).
  - **AssetVariation** — appartiene a Asset; descrizione; tipo (variazione | sub-luogo); immagini set.
  - **Wardrobe** (solo actor variation) — set immagini multi-vista.
  - **Image** — appartiene a set; angolo/vista, tipo rappresentazione, seed, prompt, versione.
- **Lora** — appartiene a Asset(Variation); gusto {image|video}; trigger token; scope {series|episode}; versione; pesi file; dataset di riferimento.
- **Voice** — origine {cloned|described}; parametri; (1—1 opzionale con Actor).
- **Script** — appartiene a Project; testo formato cinematografico; riassunto; versioni.
- **Scene** — appartiene a Script; descrizione; stile (override); durata; ordine; Continuity State; asset referenziati (N—N).
- **Shot** — appartiene a Scene; descrizione video; durata; ordine; dialoghi (lista: attore→testo); keyframe (iniziale/intermedi/finale); riferimenti asset/LoRA risolti; Continuity State derivato.
- **Artifact** — generico per output generati (image/video/audio): tipo, file, parametri (seed/model/lora/pesi/prompt), versione, stato job.
- **Job** — tipo {image|video|lora_train|audio_voice|audio_music|audio_sfx|lipsync}; stato; priorità; GPU; stime; riferimento all'artefatto prodotto.
- **AudioClip** — tipo {dialogue|music|ambient|sfx}; appartiene a Shot/Scene; voce/sorgente; tempi (in/out); volume/ducking; versione.
- **MontageTimeline** — appartiene a Project; ordine clip, transizioni, **tracce audio separate** (dialoghi/musica/ambient/sfx), frame-rate, aspect ratio; **target di export** {filmora|fcpxml|aaf|otio|edl|master}; pacchetto media+manifest.

Relazioni notevoli: `@Reference` risolve `(Project, "@Nome#Variante") → AssetVariation → {Image refs | Lora(trigger)}` secondo il percorso di coerenza attivo.

---

## 15. Pipeline end-to-end (riepilogo)

```
Input (4 tipi) + Setup (durata, generi, stili, percorso, modalità)
        │  LLM (provider riusato)
        ▼
Copione cinematografico (riassunto + racconto)  ──editing/rigenerazione──┐
        │  LLM (breakdown, JSON)                                          │
        ▼                                                                 │
Scene (descr., asset @, stile, durata)  →  Inquadrature (durata, descr.   │
        video, dialoghi @, keyframe)  ──editing/rigenerazione────────────┘
        │
        ├─ (Asset Manager) character/asset sheet coerenti
        ├─ (LoRA Studio, se Percorso LoRA) training img+video, trigger
        ├─ (Voice Studio) voci legate agli attori
        │
        ▼  Continuity Engine inietta stato; Job Scheduler accoda
Keyframe (i2i multi-ref / LoRA)  →  Video inquadrature (LTX + keyframe/LoRA)
        │                          → Audio: dialoghi+lip-sync, musica, ambient, SFX
        ▼
Montaggio: timeline + audio multitraccia + transizioni
        ▼
Export progetto NLE (Filmora / FCPXML / AAF / OTIO / EDL) [+ master opz.]
```

---

## 16. Use case professionali aggiuntivi (estensioni proposte)

Pensati per coprire necessità di produzione reale; da prioritizzare con l'utente.

- **Serie & casting persistente:** libreria LoRA di serie; un episodio eredita il cast, aggiunge guest episodici.
- **Reshoot mirato:** rigenerare una singola inquadratura/sequenza senza toccare il montaggio del resto (non distruttivo).
- **Style override per scena/atto:** flashback in b/n dentro un film a colori, gestito dal Continuity Engine.
- **Storyboard / shot list export:** esportazione dei keyframe come storyboard e della shot list come documento di produzione (durate, asset, dialoghi).
- **Animatic preview:** pass rapido per validare ritmo prima della resa video (già in §11).
- **Linguaggio di camera (preset):** tipologie di inquadratura (campo lungo, totale, piano americano, primo piano, dettaglio…) e movimenti (dolly, pan, tilt, handheld) come tag suggeriti dalla LLM e selezionabili.
- **Multilingua / doppiaggio:** rigenerare i dialoghi in altre lingue mantenendo la voce legata, con ri-sincronizzazione del lip-sync.
- **Export NLE multi-target:** un solo modello timeline (OTIO) → più formati progetto (Filmora/FCPXML/AAF/EDL), così l'utente rifinisce nel software che preferisce.
- **Seed locking & reproducibility:** blocco seed per asset/inquadratura per risultati ripetibili.
- **Continuity warnings:** il motore segnala salti di luce/condizione/asset tra inquadrature.
- **Stime di costo/tempo e budget GPU:** preventivo prima di "Genera Tutti", con scheduling 4×GPU.
- **Import/export progetto** e **duplicazione** (template di serie).
- **Collaborazione/commenti** sugli artefatti (estensione, se utile).

---

## 17. Stato delle decisioni

### Confermate dall'utente (vincolanti)

1. **Continuity Engine / memoria narrativa (§12):** **incluso e centrale**.
2. **Intervento manuale a ogni step, LoRA incluse (§5.4, §5.6, §7.6):** ogni stadio è controllabile, gli artefatti (incluse le LoRA e i loro campioni) sono ispezionabili e rigenerabili, e l'avanzamento allo step successivo è **sempre manuale/esplicito**. Default operativo: **Supervisionata**.
3. **Audio completo come core (§8.4, §11.2):** dialoghi, **lip-sync**, **musica**, **ambient**, **SFX**, mixati in timeline. Non sono estensioni.
4. **Export verso software di editing (§11.4):** output primario = **file progetto** per NLE esterni (Filmora e altri), via formati di interscambio (Filmora / FCPXML / AAF / OTIO / EDL). Il master renderizzato è opzionale (§11.5).

### Ancora da confermare

5. **Modalità mista per-asset (§4.4):** assunta **sì** (cast in LoRA, comparse in Rapido nello stesso job). Confermare la fattibilità sul backend ComfyUI in uso.
6. **Liste stili (~50) e generi (~20):** Claude Code le popola in configurazione; confermare se preferisci fornirle tu.
7. **Limite pratico di LoRA simultanee per inquadratura (§7.5):** definire la soglia di warning e la strategia di fallback (split inquadratura / Percorso Rapido per i minori).
8. **Priorità target NLE per la v1 (§11.4):** Filmora è il target citato; confermare se in v1 basta Filmora + un pivot standard (FCPXML/OTIO) o se servono subito anche AAF/Premiere/Resolve.
