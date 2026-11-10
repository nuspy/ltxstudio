# LTX Studio — istruzioni di progetto

## Sistema di lavoro standard (deciso dall'utente il 14/10/2026)

Vale per ogni richiesta che non sia una modifica minima (vedi «Quando NON usare la catena»).

### Ruoli
| Ruolo | Modello | Compiti |
|---|---|---|
| **Pianificatore / revisore** | Opus 5.5 (sessione principale, parla con l'utente) | piani, strategia, visione d'insieme, decisioni critiche, dubbi, analisi di alta qualità; revisione finale dei report |
| **OrchestratorProgrammatore** | Sonnet (alias `sonnet`, sotto-agente, ID riusabile) | gestisce il progetto, lancia gli swarm di ricerca, verifica i risultati, implementa il codice (da solo o con sotto-agenti Sonnet), esegue i test, consegna il report |
| **Ricercatori** | Haiku (alias `haiku`, swarm) | estrazione e inventario: trovare file, elencare riferimenti, leggere header, riassumere codice — MAI decisioni |
| **Verificatore** | Sonnet | rifà una ricerca Haiku dubbia, informato di cosa cercare, della ricerca precedente e dei suoi errori |

(Modelli: alias `opus` / `sonnet` / `haiku` = ultima versione nota alla build di Claude Code; aggiornarlo per avere Sonnet 5.5 / Haiku 5.5.)

### Flusso
1. **Opus** scrive il piano (obiettivo, vincoli, criteri di accettazione, cosa NON fare) in un brief su file e avvia
   l'OrchestratorProgrammatore (o riprende quello già esistente sullo stesso argomento, per riusarne il contesto).
2. **Orchestrator**: swarm Haiku per la ricerca sul codice → verifica (grep, file:riga, header, confronto byte) →
   se una ricerca è sbagliata o incerta la rifà con un Sonnet informato.
3. **Escalation**: per decisioni critiche, dubbi o problemi importanti l'Orchestrator NON indovina: chiude il suo
   turno con una sezione «DECISIONE RICHIESTA» (domanda, opzioni, raccomandazione, prove). Opus risponde riprendendolo
   con SendMessage. Le decisioni che spettano all'utente (cancellazioni, scelte di prodotto, licenze) Opus le gira
   all'utente.
4. **Implementazione** da parte dell'Orchestrator (o di Sonnet da lui creati/riusati), coi test previsti.
5. **Report finale** a Opus, che verifica a campione (diff, test, file) e approva o rimanda indietro con correzioni.

### Regole per gli agenti (risparmio token e qualità)
- **Brief su file, non nel prompt**: contesto, regole e cartella di lavoro in un file .md; il prompt lo richiama.
- **Haiku solo con output strutturato** (JSON per voce, campo «confidenza») e su domande fattuali verificabili.
  Esperienza del 14/10: ~30% di errori nel decidere «questo file è usato?» → ogni affermazione che porta a
  cancellare o cambiare codice va verificata da Sonnet con prova (file:riga, grep, confronto byte per byte).
- **Ogni agente scrive SOLO nei suoi file** (nomi univoci o sottocartella propria): il 14/10 due agenti in
  parallelo si sono sovrascritti script e risultati nella scratchpad condivisa.
- **Dimensione degli swarm**: ≤10 Haiku per giro; al più un giro di verifica Sonnet per argomento prima di
  scalare a Opus.
- **Riuso del contesto**: stesso argomento → riprendere l'agente esistente (SendMessage), non crearne uno nuovo.
- **GPU una alla volta**: un solo test live per volta, controllare VRAM/RAM/commit prima, chiudere i processi di
  test dopo; mai test mentre l'app dell'utente è aperta.
- **Azioni distruttive** (cancellare pesi, file del repo, rami git): l'Orchestrator prepara lo script e l'elenco,
  Opus lo porta all'utente; non si cancella senza approvazione esplicita (il classificatore di sicurezza blocca
  comunque le cancellazioni: l'utente le lancia con `!`).
- **Report** standard: cosa è cambiato (file, commit), test eseguiti con esito (dal vivo dalla GUI quando serve),
  cosa resta aperto, domande. Corto: i dettagli stanno nei file citati.

### Quando NON usare la catena
Domande, spiegazioni, correzioni di una riga, verifiche puntuali: le fa Opus direttamente. La catena serve quando ci
sono ricerche ampie sul codice, implementazioni su più file o più passi indipendenti.

## Regole di progetto (dettagli nella memoria `~/.claude/projects/E--ai-ltxstudio/memory/`)
- Test sempre dalla GUI (widget veri, generazione reale); casi in `tests/gui_regression`.
- Ogni testo della GUI tradotto nelle 15 lingue (`scripts/i18n_add_keys.py`); ogni modifica UI anche nella variante Pro.
- Ogni modello/feature nuovo: catalogo pesi (`build/catalog_v2.py` → `data/weights_catalog.json`,
  `build/catalog_use_deps.py`) e feature matrix (`python -m feature_matrix sync` in `C:\Projects\feature_matrix`).
- Niente ComfyUI: pesi convertiti in formato nativo; preferenza fp8 / int4.
- File con modifiche non committate dell'utente: committare solo i propri hunk.
- Mai scegliere o sostituire un checkpoint in automatico; mai cancellare file senza chiedere.

## Metodo di test (deciso dall'utente il 30/10/2026 — vale per ogni agente, prima di ogni nuova direttiva)
- Prima si implementa nella GUI, poi si testa DALLA GUI (widget veri + bottone) o dal comando che la GUI lancia
  davvero, verificando che la funzione chiamata nel backend sia quella collegata al widget.
- I test girano sull'app REALMENTE avviata: una sola istanza, job in sequenza senza riavvii, memoria reale.
  Niente runner/worker isolati o offscreen come prova finale.
- Uno script unico esegue i test uno dopo l'altro; su errore si corregge e si rifà lo stesso test (nel giro
  «tutte le architetture × modelli» si annotano gli errori in un .md, si corregge tutto insieme, si rifà il giro).
- Niente agente per singolo test; riusare sempre lo stesso agente (SendMessage) per cache e contesto.
- A fine giro una pagina HTML di revisione: miniatura cliccabile, cosa è stato fatto e cosa verificare, prompt,
  modello, LoRA, originale affiancato per gli edit, un textbox per test e il bottone «Pubblica» che crea un .md con
  i test non superati. Textbox vuoto = superato. Lo script APRE la pagina nel browser a fine giro; il giudizio
  sugli output lo dà l'utente lì (niente token spesi a descriverli).
- Agenti: modello meno costoso che basti al compito (Sonnet/Haiku per eseguire, controllare e riportare l'errore);
  riusarli sempre con SendMessage, un agente nuovo solo per un argomento nuovo; script lunghi in background senza polling.
