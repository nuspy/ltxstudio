# Diario dei task — LTX Studio

Ultimo aggiornamento: 31/10/2026 mattina.

## In corso adesso
- Crash di Windows delle 05:51 del 31/10: l'utente dice che la RAM (4×32 GB DDR4-3200) è instabile già a 3000 e stabile a 2800, quindi la riporta a 2800. Probabile causa delle 7 schermate blu dal 17/10.
- 03/11: l'utente ha giudicato le 3 pagine di revisione (0543, 1810, 0510) e ha pubblicato i commenti. Le correzioni sono in coda.
- 02/11: si porta in LTX Studio tutto Openjutsu (cambio di sfondo, personaggi e oggetti nei video con MiniMax H3), in 5 fasi. L'agente lavora sul ramo nuovo feature/openjutsu. Per ogni azione un menu «Motore» con solo i motori che la sanno fare, scelto dall'utente. Brief: orch\BRIEF_openjutsu_02-11.md.
- 02/11: errore dell'inpaint video con GGUF LTX-2.5: causa trovata (la GUI usava l'ambiente Python di LTX 2.3 anche per i modelli 2.5), corretto nel codice, messaggio d'errore ora intero. Manca la prova dall'app vera.
- 02/11: fase 1 di Openjutsu scritta (menu «Motore», «Cambia sfondo», «Sostituisci personaggio» con H3 a pezzi uniti). Test senza GPU tutti verdi. 02/11 22:28-22:34: primo giro dall'app vera con il via di basalt. Tutti e 3 i casi falliti in un paio di minuti per errori di codice: il video di riferimento H3 e la LoRA di inpaint 2.3 sul modello 2.5. GPU restituita a basalt. Si correggono senza GPU, poi si richiede il via.
- 02/11 22:54-23:10: secondo giro. Gli errori di codice sono risolti, ma tutti e 3 i casi finiscono la memoria della scheda video: H3 non usa il piano di memoria già sistemato a ottobre, e il GGUF in inpaint resta tutto sulla GPU. GPU restituita a basalt. Si correggono senza GPU.
- 02/11 23:24-23:44: terzo giro. L'inpaint con LTX-2.5 (versione INT8) ora funziona dall'app vera. I 2 casi H3 vanno ancora in memoria piena, contro un nostro limite di memoria per processo. GPU restituita a basalt. Il GGUF Q8 del 2.5 non può passare a blocchi dalla RAM: limite della libreria del 2.5.
- 03/11 16:30-17:00: trovato un errore importante in H3 (cambio sfondo, sostituisci, Scambia), corretto nel commit eb679f33. Le maschere della persona erano sfasate nel tempo rispetto ai fotogrammi: alla fine del video fino a 9 fotogrammi avanti. È la causa principale dello sfondo vecchio che resta e cresce nel tempo, e forse dei bordi sbagliati segnalati in sostituisci. I casi H3 dei turni 7 e 8 vanno rifatti (turno 11, script pronto). La riprova «rifugio» del turno 10, con solo la prima correzione: sfondo vecchio intorno alla persona dal 22 al 16%, pennacchio sopra la testa sparito.
- 03/11: l'utente chiede come rendere H3 più veloce. Una prova da 12 s dura circa 22 minuti: 12,5 di generazione e circa 5 di codifica del testo, che è anomala (100 s per pezzo). L'ipotesi del limite di pixel sul video è stata provata sulla CPU e scartata; aggiunto un cronometro (commit cf3f9c4d). In attesa dell'utente: provare 4 passi (LoRA già sul disco) e l'attenzione sparsa Veda (già sul disco), con nuovi comandi in Modifica Video.
- 03/11 16:06-17:04: turno 10, 15 casi su 15 riusciti: la riprova «rifugio» e i 14 casi rimasti dei 31 (4 video con riferimenti, 10 immagini). Errore del lanciatore (un percorso in stile /c/ letto da Python per Windows dava la lista dei rimasti vuota), corretto durante il turno. I 31 casi del giudizio e i 14 a maschera larga sono tutti verdi per il controllo automatico. Il turno 11 (H3 con le maschere corrette) parte quando l'agente delle opzioni H3 ha committato: sta modificando gli stessi file.
- 03/11 17:00: download approvati in corso (turbo integre, int8 ufficiale, nvfp4 rockerBOO, LoRA di edit, circa 47 GB). L'utente ha detto che la connessione non è più a consumo.
- 03/11 15:11-16:05: turno 9, 31 casi tutti riusciti per il controllo automatico: i 14 di Integra con maschera larga (212, 387 e 837 px, e contesto manuale) e 17 dei 31 casi del giudizio.
- 03/11 15:10: trovata e corretta la causa dello sfondo vecchio rimasto intorno alla persona nel cambio sfondo H3 (commit 4cc3e29c). La zona tenuta dall'originale includeva uno spazio sopra la testa e la posa di ±2 fotogrammi, che servono solo a «Sostituisci personaggio»: lì restava il mercato, come un pennacchio sopra la testa. Da riprovare dal vivo.
- 03/11 14:18-15:10: turno 8, tutti e 3 i casi riusciti: H3 sostituisci col ritratto (normale e riquadro largo) e Scambia con 1 persona, sul video nuovo.
- 03/11 13:08-14:16: turno 7, 3 prove di cambio sfondo H3 sul video nuovo, tutte riuscite (22 min l'una). Lo sfondo vecchio residuo cala molto. Nella fascia intorno alla persona: 65% nel caso base, 44% con la sala da tè grigia, 22% con il rifugio e 35% con il rifugio a guida grigia. Lontano dalla persona: da 53% a 12-15%, cioè quasi tutto cambiato. Il migliore è il rifugio, con il riferimento normale. Resta un aumento nella seconda metà del video (6-12 s: fascia 33%).
- 03/11 11:59-12:24: turno 6 bis. La prova H3 sul video nuovo è riuscita (22 min, 3 pezzi, picco 24-26 GB): la causa della memoria piena era una perdita di memoria nel calcolo a blocchi, ora corretta. Però il nuovo sfondo assomiglia ancora troppo al mercato originale: circa il 65% della fascia intorno alla persona resta come prima. Si provano un luogo molto diverso e una guida dello sfondo tutta grigia. Errore del lanciatore: ha letto l'esito vecchio e si è fermato, e la GPU è rimasta 30 minuti presa senza lavoro. Corretto.
- 03/11 11:39-11:51: turno 6. Il caso di prova H3 sul video nuovo va ancora in memoria piena: la frammentazione è risolta, ma la memoria occupata davvero supera di circa 2,7 GB quella prevista. Si aggiunge un ripiego automatico (stesso pezzo rifatto con più modello in RAM) e la scomposizione della memoria a ogni errore, poi si riprova.
- 03/11 10:42-11:34: generato dall'app il nuovo video base (uomo in un mercato coperto, 12 s, 1280×704, con audio). I 5 casi H3/Scambia sul video nuovo sono tutti andati in memoria piena sulla GPU: la memoria si frammenta e il piano la sottostima. Si corregge, poi si riprova. «Zona di contesto» di Integra implementata (commit 5b64a268); si aspetta la conferma dell'utente sul significato della %.
- 03/11 10:01-10:21: video H3 n. 4 riuscito: dopo la generazione si liberano 13,3 GB di pesi rimasti sulla scheda. Tutti i 45 casi sono verdi: pagina unica di revisione aperta all'utente (orch\app_live\review_20261103_1021, su http://127.0.0.1:8790/). Si aspetta il suo giudizio.
- 03/11 05:47-06:44: turno corto. Klein 9B (realDream) riuscito: la memoria libera è rimasta sopra i 43 GB. Prima prova dal vivo del tab «Scambia» (1 persona) riuscita. Il video H3 n. 4 è fallito per memoria della GPU: la correzione è già scritta, va riprovata. Alle 09:20 il PC si è riavviato; dopo il riavvio è stato rilasciato il blocco della GPU e i lavori sono ripresi.
- 03/11 04:32-05:41: turno con i giri A3, C e B. H3 cambia sfondo e sostituisci col ritratto riusciti, con l'alone bianco corretto: nella fascia intorno alla persona i pixel bianchi passano dal 15-24% a meno dell'1%. Video del giro B: 7 su 7 ok, compreso quello da 40 s con LTX-2.5. Klein 9B (realDream) ancora chiuso per memoria. Il video H3 n. 4 è rinviato al turno dopo.
- 03/11 04:10-04:27: giro A2 (i 6 rossi del giro A). 5 su 6 ok: la correzione della memoria funziona per FLUX FP8 e Krea 2 Edit. Resta il Klein 9B (crash). Giro C (2 casi H3): fermato subito da un errore nel tracciamento SAM2, probabilmente introdotto dal lavoro sul menu Segmentazione. Si corregge, poi nuovo turno con il caso Klein, il giro C e il giro B.
- 03/11 01:30-02:53: giro A dei casi commentati (34 tra immagini e selezione): 28 ok, 6 rossi. Cinque sono chiusure dell'app per memoria, quasi tutti FLUX in FP8 e Krea 2 Edit, con il processo di generazione che arriva a 58-64 GB. Uno è il FLUX.2 di Integra, che non riceve l'architettura. Si correggono senza GPU, poi si rifanno insieme ai giri B (video) e C (H3). La pagina di revisione arriva all'utente solo quando tutto è verde.
- 03/11 00:33-00:53: quinto giro, i 2 casi H3 (cambia sfondo e sostituisci personaggio) RIUSCITI dall'app vera, con i video prodotti. Pagina di revisione: orch\app_live\review_20261103_0053. Giudizio dell'utente: nel cambio sfondo persona, gesti, labbra e voce sono rimasti originali e il luogo è nuovo, ma intorno alla persona c'è un alone bianco molto spesso. Nel sostituisci personaggio ci sono aloni bianchi ai bordi, e la prova va rifatta con un personaggio dal viso visibile, non un astronauta col casco. Si corregge, poi si riprova nel giro B.
- 02/11 23:51-03/11 00:10: quarto giro, solo i 2 casi H3. La memoria non finisce più, ma ci sono due errori nuovi: un errore nel calcolo a blocchi (cambia sfondo) e un job finito senza file di uscita (astronauta). GPU restituita a basalt. Si correggono senza GPU.

- 03/11 10:42-11:36: turno 1. Nuovo video di prova fatto dall'app (orchpp_liveixtures\persona_intera_12s.mp4, uomo in un mercato, 12 secondi, con voce): riuscito. I 5 casi H3/Scambia su quel video sono TUTTI falliti per memoria della scheda video piena (frammentata: 5 GB riservati ma inutilizzabili al passo 5 di 8). Correzione scritta (a inizio di ogni blocco si libera la cache), da riprovare con la GPU.
- 03/11: «Zona di contesto» di Integra (Automatica/Manuale, posizione rossa e contesto gialla sempre visibili, due cursori di trasparenza, scritta «Stai disegnando») scritta e unita nel ramo feature/openjutsu; test senza GPU verdi. Mancano i casi dal vivo (150% e manuale) nel giro delle maschere larghe.

- 03/11 (decisione dell'utente) ordine della GPU:
  1. prova breve di basalt (1-2 ore);
  2. i nostri turni 12-14 e la guida (4-5 ore);
  3. la corsa da 55 ore di basalt.
  Il turno 11 era già partito e finisce prima della prova di basalt.
- 03/11 17:14: turno 11 in corso: i casi H3 rifatti con le maschere corrette (sfondo «rifugio», sostituisci col ritratto, Scambia, poi il riquadro largo se c'è tempo).
- 03/11 sera: opzioni H3 in Modifica Video fatte (commit 37a509f9): menu «Modello H3», turbo con passi e shift, LoRA H3 filtrate, Veda, LightVAE, encoder del testo. Default invariati, test verdi. Le prove dal vivo sono nei turni 12-14.
- 03/11 sera (decisione dell'utente): guida di latenti allineata per H3, subito e in parallelo. Un agente nuovo lavora sul ramo feature/h3-guide, worktree .claude\worktrees\h3guide. Brief: orch\BRIEF_h3_guida_allineata_03-11.md.
- 03/11 sera: secondo gruppo di download approvato, in coda dopo il primo:
  - NSFW Unlocked V3;
  - Pika-H3-Flex a 6 passi;
  - Qwen-Image 2.1 Lightning v2, solo la LoRA a 4 passi;
  - 10Eros beta7 fix1;
  - DaSiWa v4 Turbo int8;
  - Singularity v2 int8.
  FLUX.3 Klein non si scarica: è già integrato (task 13). L'utente ha chiesto di cancellare i pesi vecchi sostituiti: l'elenco si prepara dopo aver verificato i nuovi.

- 03/11 sera (richiesta dell'utente), ramo feature/canvas-sam31 con un agente nuovo. Brief: orch\BRIEF_canvas_sam31_03-11.md.
  - Le due maschere (posizione/contesto) e i loro cursori si vedono solo con i modelli che le usano; i cursori a 5 px l'uno dall'altro.
  - «Scontorna» e «Sfondo» si vedono solo dove servono.
  - SAM 3.1 completato: avviso sull'approvazione di Meta, SAM2 nel frattempo, controllo automatico ogni 30 minuti e bottone per scaricare, caricatore con transformers.
  - 03/11 sera: errore mio, poi corretto. Avevo cancellato SAM 3.1 di Meta per sostituirlo con la copia di unsloth, che però fa solo immagini. L'utente ha chiesto di ripristinare Meta: unsloth cancellato, Meta in riscaricamento. Il file multiplex di Comfy-Org resta per il confronto; se è equivalente, si tengono tutti e due.
  - 03/11 sera: il codice ufficiale SAM 3 di Meta è revisionato (nessuna rete nell'inferenza con file locale). Si esegue solo dal lanciatore E:\AI\Tools\sam3_run.sh, con il permesso dato dall'utente. Le dipendenze mancanti sono installate in E:\AI\Tools\sam3_deps, senza toccare il venv condiviso.

- 05/11: TEST SOSPESI dall'utente («aspetta la mia autorizzazione per continuare i test»).
  - Il turno 45 è pronto ma fermo: Integra con volto protetto, MSR 2.3 con taglio della guida, MSR 2.5 con più forza, Krea con riquadro, rigiro di SAM 3.1.
  - Il server delle revisioni via Tailscale è stato chiuso da Claude Code per memoria scarsa; si rilancia solo su richiesta.
- 04/11 18:24: coda GPU di LTX finita; GPU passata a basalt (TUTTO FINITO).
  - Turni 31-36: molti casi bocciati dal controllo automatico perché il viso non somiglia al riferimento:
    - Integra 9 casi su 12;
    - MSR 2.5 e MSR 2.3 4 su 5 ciascuno;
    - Krea 3 su 3;
    - Integra dal canvas 4 su 5.
  - SAM 3.1 fallito in 3 casi: il menu non arriva al backend, errore di tipo bf16/float, multiplex con codice 1.
  - All'agente: correzioni senza GPU.
  - Revisioni da giudicare: 10, in un'unica pagina http://100.110.144.5:8790/tutte (commit 64a09b04).
- 04/11 16:55: turni 43 e 44 fatti, tutti i casi riusciti.
  - Turno 43:
    - Pika: 374 s su int8, 369 s su nvfp4;
    - DaSiWa Turbo: 315 s a 4 passi, 385 s a 8 passi.
  - Turno 44: con lo sfondo del riferimento grigio al 100 % lo sfondo cambia davvero (rifugio al posto del mercato). Persona riconoscibile, video intero coerente nelle 3 clip.
  - Da decidere con l'utente dopo la sua revisione: diventano il default del cambio sfondo la maschera come l'originale, il margine 4 px e il grigio al 100 %?
  - Partiti alle 16:57 i turni 31-36.
- 04/11 16:20: job della GUI fermi in coda dopo il riavvio del PC.
  - Causa: data\gpu.lock scritto pieno di zeri dal riavvio, mai giudicato abbandonato.
  - Corretto nel commit 22a20cef: il blocco illeggibile viene rimosso, la scrittura usa fsync e l'attesa compare nel log.
  - Verificato dal vivo alle 16:17: il blocco è stato rimosso e il job è partito.
  - Persa un'ora di GPU (14:15-16:20).
  - Aggiunto il server delle revisioni via Tailscale: http://100.110.144.5:8790/ (commit 6ceb37cf).
  - Singularity v2 convertito e messo nel menu (commit 57077b44), da provare dal vivo.
  - Visori 3D creati solo all'apertura della scheda, e WebGL spento nei test (commit d9b8e845), per il crash Qt all'avvio.
- 04/11 14:20: la sessione si era chiusa alle 07:27, durante il turno 43; ripresa ora.
  - Turno 43:
    - fatti Pika su int8 (374 s) e DaSiWa Turbo a 4 passi (315 s);
    - Pika su nvfp4 non è partito per un crash della GUI all'avvio (access violation in _build_ui, da indagare);
    - DaSiWa a 8 passi interrotto.
  - Ora girano i due casi mancanti del 43, poi il turno 44 (sfondo grigio 100 % con margine 4), poi i turni 31-36.
  - Ripreso il download di Singularity v2 (era all'87 %).
- 04/11 07:12: turno 42b fatto.
  - Moltiplicazioni int8: 529 s invece di 605. FP4: 516 s invece di 555.
  - Margine 5 px: viso riconoscibile e nessun doppione visibile. In tutti i casi lo sfondo resta il mercato.
- 04/11 06:30: turni 41 e 42 fatti.
  - Turno 41: i tre modelli H3 hanno velocità vicine. L'nvfp4 di Abiray sta tutto nella GPU. Qualità da giudicare in review_quant_1104_042235.
  - Turno 42, misure su clip 1 a 736p:
    - video di movimento a 512 px: 605 s invece di 1400;
    - attenzione int8: 439 s invece di 605;
    - decodifica fp16: nessuna differenza di tempo.
  - I casi con le moltiplicazioni int8 e FP4 hanno ripreso un clip già fatto (difetto della chiave): da rifare nel turno 42b.
  - DaSiWa Turbo riparato a blocchi: sha256 OK.
- 04/11 04:50: GPU, decisione dell'utente.
  - Basalt non alterna più: fa un test breve, poi lascia la GPU a LTX fino alla fine di tutti i turni, poi la usa lui.
  - Coda: 41 (in corso), 42, 40, 43, poi 31-36. Fine stimata verso le 14:00.
- 04/11 03:45: H3 più veloce come l'originale.
  - Turno 40 fallito per memoria esaurita nella compressione VAE a 736p.
  - Correzione nel commit b8e3f253: VAE solo encoder (0,7 GB invece di 9,7, stesso risultato bit a bit) e liberati ~12 GB del lettore di testo che restavano sulla scheda.
  - Approvate dall'utente 4 accelerazioni prese dall'originale:
    - video di movimento a 512 px, di default, con opzione per la risoluzione piena;
    - moltiplicazioni int8 vere;
    - attenzione int8, come casella spenta di default;
    - decodifica VAE a pezzi, senza rileggere il testo dopo un errore.
  - Le implementa l'agente, poi si misurano nel turno 42.
- 04/11 03:00: disco E: quasi pieno (21 GB liberi; l'utente ricorda più di 400 GB liberi ieri).
  - Download del secondo gruppo fermati: arrivati Pika fl2va e Qwen Lightning; interrotti Unlocked V3, Pika ref2va, 10Eros beta7 (21,6 su 22,9 GB) e DaSiWa (5,5 su 21 GB).
  - Cancellata la cache dequant di Klein (18 GB, ok dell'utente).
  - Si completa la beta7, poi si cancella la beta6 (21,7 GB, ok dell'utente).
  - Trovato cosa ha occupato lo spazio dal 02/11 (circa 375 GB):
    - modello di basalt, 94,5 GiB;
    - disco di Docker modificato, 75 GiB, che esiste da giugno;
    - pesi H3 di LTX, circa 100 GiB;
    - Openjutsu originale, 55 GiB;
    - pagefile su E:, 19,5 GiB;
    - E:\DUMP1e74.tmp, 9,3 GiB, del 02/11 01:10.
  - Decisioni dell'utente (04/11 03:20):
    - Openjutsu originale resta come esempio finché serve;
    - DUMP da cancellare (lo lancia l'utente, bloccato dal controllo);
    - beta6 e copia ComfyUI della beta7 da cancellare dopo il «verifica OK»;
    - tra rockerBOO e gli altri si decide dopo il turno 41;
    - DaSiWa Turbo da continuare e integrare;
    - le uscite dei test le pulisce l'utente;
    - basalt e Docker li gestisce l'utente nell'altra sessione.
  - Download in coda: DaSiWa e Pika ref2va, poi turbo ComfyUI e Unlocked V3; Singularity v2 solo con almeno 60 GB liberi.
  - Openjutsu originale: nessun doppione, ma ridisegna anche la persona (viso diverso, borsa sparita), 11,4 min per 12 s a 736p. Confronto in orch\app_live\CONFRONTO_originale_vs_nostro_0411.jpg/.mp4.
  - Proposti da cancellare: originale rockerBOO, DUMP, DaSiWa parziale; dopo le verifiche, sorgente beta7, beta6 e Openjutsu.
  - Openjutsu originale installato in E:\AI\Tools\openjutsu_orig (55 GB, ComfyUI 0.44 + 12 pacchetti di nodi + 48 GB di modelli), pronto per la prova GPU.
- 04/11 02:40: server che non parte RISOLTO nel codice (commit b1122f38).
  - Causa: sul PC un filtro di rete scarta le connessioni locali da certe porte sorgente. Le prove indicano NordVPN (driver NDivert) oppure filtri rimasti da Avast, che è disinstallato. Driver firmati, nessun indizio di malware.
  - Rimedio: le connessioni locali dell'app partono sempre da porte buone, con nuovo tentativo; il server viene rilanciato se resta fermo.
  - Misure: blocchi da 7,5 % a 0 su 120 prove, e da 2/30 a 0/90 avvii reali. Verifica mia: 0/40.
  - Da verificare dall'utente: pausa di NordVPN, poi rilanciare la prova delle porte (scratchpad\wfp\prova_porte_cattive.py).
- 04/11 02:05: causa trovata del server di inferenza che a volte non parte, difetto vero dell'app.
  - Il server resta bloccato per sempre nell'avvio di asyncio su Windows, nella self-pipe in loopback (accept senza connessione); dump con py-spy.
  - È successo 3 volte stanotte: ogni volta si perdono 15 minuti di turno.
  - Affidato all'agente di Integra: rilancio automatico del server se non risponde entro un minuto, più la causa di fondo.
  - Turno 16a chiuso alle 02:05 dallo stop-file, senza usare la GPU. GPU a basalt; noi in coda per il turno 40.
- 04/11 02:00: esito dell'ablazione H3 (turno 16a).
  - I prompt non contano. La persona cancellata dal riferimento aiuta poco.
  - Il responsabile principale è la zona tenuta attorno alla persona: con il margine di Sostituisci il fantasma passa da 26 a 12 fotogrammi su 34 e il ritardo quasi sparisce.
  - L'Openjutsu originale tiene un margine di 10/20/40 px con sfumatura. Noi l'avevamo tolto il 3/11 per l'alone bianco: è lì che è nato il doppione.
  - Decisione dell'utente: due cose in parallelo. Un turno col nostro porting allineato all'originale (margine 20 px sfumato, 736p, int8, DMAD a 6 passi), e l'installazione dell'Openjutsu originale con ComfyUI in E:\AI\Tools\openjutsu_orig, solo per il confronto, con la regola «niente ComfyUI» sospesa dall'utente. Brief: orch\BRIEF_openjutsu_originale_04-11.md.
  - Correzione: la differential diffusion riparata non è usata da nessuna parte (tolta il 13/07), quindi non spiega l'«incollato» di Integra.
- 04/11 01:45 (ok dell'utente): rami semplificati.
  - L'app gira da E:\ai\ltxstudio sul ramo `master`. Gli agenti lavorano sul ramo `sviluppo` (.claude\worktrees\openjutsu). I turni girano dal worktree pulito `prove`.
  - Cancellati 35 rami già uniti e 22 worktree (contenevano solo la cache model_detect.json).
  - Resta `task12-memoria`: superato, il suo contenuto è già nel codice in versione più recente. Si cancella con l'ok dell'utente.
  - Pulizia del merge (8c9d58f6):
    - differential diffusion di Integra/Componi riparata: mancava un import, e l'errore veniva ingoiato in silenzio;
    - protezione della memoria per Krea;
    - la musica usa il suo venv;
    - tenuto un solo approccio per le LoRA su fp8.
  - harness.py doppio rinominato (3afd0f0c).
- 04/11 01:30 (richiesta dell'utente): regole di codice pulito per tutti, salvate in memoria: nomi parlanti, niente errori ingoiati, pyflakes prima di ogni commit, funzioni brevi, una soluzione per problema.
- 04/11 01:20 (richiesta dell'utente): codice riunito. Il ramo di lavoro feature/editing-expansion (cartella principale) è avanzato a feature/openjutsu (35f96d6e, 245 commit): H3 edit, guida, ablazione, SAM 3.1, canvas, Integra, scenari e traduzione. Le modifiche non committate dell'utente sono salvate (stash@{0} e copia in scratchpad\merge_check\backup_utente_0411).
  - image_flux.py, krea2_infer.py e model_detect.json: modifiche riapplicate.
  - model_manager.py: la versione locale dell'utente e quella committata (LoRA su fp8 con hook) cambiano lo stesso punto in modi diversi. Nel ramo resta quella committata; quella dell'utente è nello stash, da fondere a mano insieme all'utente.
  - Trovato con pyflakes un errore vecchio: `Image` non importato in image_flux.py:1861/1870 (commit 515b111a). Da correggere nel ramo integrazione-30-10.
- 04/11 00:44: turno 16a ripartito. Il primo tentativo è fallito per un altro danno da merge: in workers/minimax_h3_infer.py la variabile `guide` era usata prima di essere definita. Corretto nel commit d408299a; con pyflakes corretti anche due nomi indefiniti vecchi (commit 35f96d6e).
  - Da indagare (difetto vero dell'app): l'avvio automatico del server di inferenza a volte non riesce. È successo due volte stasera (attesa di 905 s) e il 02/11 aveva impiegato 306 s.
- 04/11 00:16: turno 30 riuscito, 30 immagini dei 6 scenari in 20 minuti. L'utente le ha giudicate tutte buone. Dalle 00:17 gira il turno 16a (ablazione H3, 8 varianti); dopo vengono i turni 31-35 (Integra, inpaint, Krea, MSR) sugli scenari nuovi.
- 03/11 22:21-23:57: turno 30 fallito due volte. La prima il server di inferenza non è partito (causa non trovata: non era basalt). La seconda c'era un errore del driver dopo il merge di SAM 3.1 (vis_mism nella funzione sbagliata), corretto nel commit 7acc3810. Ripartito alle 23:56: la prima immagine è ok.
- 03/11 23:45 (decisione dell'utente): i campi di testo H3 si traducono E arricchiscono in inglese con l'enhancer esistente. Il soggetto per la segmentazione si traduce fedelmente, senza arricchirlo. Casella attiva di default. Assegnato all'agente H3, dopo l'ablazione.
- 03/11 23:35: ablazione aggiornata con l'utente. Tolto il lip sync (nel secondo pezzo c'è e funziona). Aggiunte la frase esplicita di allineamento fotogramma per fotogramma, e questa insieme alla frase del contesto. Da verificare: la descrizione scritta in italiano dall'utente finisce senza traduzione nello schema inglese del prompt H3.
- 03/11 23:25 (correzione dell'utente): niente rimedi costruiti prima di sapere quale parametro è responsabile. Turno 16a = ablazione un fattore alla volta, solo sul primo pezzo (7 varianti da ~6,5 min): prompt senza «exact look and performance», lip sync spento, frase del contesto, persona cancellata dal riferimento, maschera come in Sostituisci, prompt di Sostituisci. Va prima del turno 16.
- 03/11 23:15: analisi a passi delle due pipeline H3 (richiesta dell'utente).
  - Il ritardo del modello c'è in tutte e due: in Sostituisci si vede come lip sync in ritardo all'inizio.
  - Nel cambio sfondo il modello ridisegna ANCHE la persona, per tre motivi: il prompt chiede «keeps exact look and performance… lip-sync», il riferimento la mostra nitida, la maschera gli lascia testa e contorni.
  - Correzioni in arrivo: prompt «zona della persona vuota» insieme alla persona cancellata dal riferimento (turno 16b); campo ritratto che restava pieno da Sostituisci al cambio sfondo (turno 11).
- 03/11 23:00: l'utente osserva che Sostituisci e Scambia NON hanno il doppione, il cambio sfondo sì.
  - Parametri identici, tranne l'azione e il prompt.
  - Differenze: Sostituisci tiene fermo il 68-83 % di ogni fotogramma (tutto lo sfondo), il cambio sfondo il 2-10 %. Nel riferimento di Sostituisci la persona è un manichino grigio, nel cambio sfondo è nitida, e il modello la ridisegna col suo orologio.
  - Variante nuova: nel riferimento del cambio sfondo la persona viene cancellata (riempita con lo sfondo sfocato). Prova in testa al turno 16b.
- 03/11 22:55: causa precisa del doppione H3, dall'osservazione dell'utente («dal quinto secondo è perfetto»). Il primo pezzo non ha contesto: tiene solo il 2-10 % di ogni fotogramma, mai la testa. Il secondo parte da 39 fotogrammi interi già fatti (100 %). Rimedio nuovo: primo pezzo in due passate, cioè si rigenera con i suoi primi fotogrammi puliti come contesto. Si prova nel turno 16 accanto a guida e orologio. SAM 3.1: si unisce adesso al ramo delle prove; turno 36 per la prova dal vivo.
- 03/11 22:40: guida allineata unita in feature/openjutsu, più due opzioni nuove (spente di default):
  - «guida con la sola persona su fondo neutro»;
  - «orologio del riferimento allineato» (idea TARA senza LoRA).
  Test CPU verdi, 9/9 sul pipeline diffusers vero. Turni H3 riorganizzati:
  - 16 = guida + orologio;
  - 16b = guida con la sola persona + Sostituisci con guida;
  - 16c = i casi ghost rimasti.
- 03/11 22:20: misura del doppione. La persona ridisegnata è quella di 5-20 fotogrammi prima (mediana -12), a dente di sega. Gli ingressi sono allineati (riferimento a 24 fps regolare, nessun duplicato): lo sfasamento nasce nel modello, perché il riferimento nativo ha un orologio suo. Da provare nel turno 16: guida allineata, oppure orologio del riferimento riportato su quello del video generato (idea TARA senza LoRA).
- 03/11 22:00: l'utente ha trovato la causa del doppione H3: la persona ridisegnata dal modello è la stessa persona SFASATA NEL TEMPO di circa 10 fotogrammi. Va a destra e a sinistra col passo e ogni ~3 s si risincronizza. Matte e manichino non bastano. Rimedio probabile: la guida di latenti allineata, che copia i tempi fotogramma per fotogramma (il riferimento nativo no). La stessa causa spiega il lip sync in ritardo di Sostituisci e Scambia. Il ramo guida va unito in feature/openjutsu e provato in testa al turno 16.
- 03/11 20:24-21:30: turno 15, rimedi al doppione H3, 3 casi su 4 (Scambia passa al turno 16). Fedeltà della testa nei primi 6 s (0 = come l'originale): prima 0,124; matte 0,007; manichino 0,115; manichino + matte 0,007 con meno volti fantasma. Il rimedio che conta è il matte. La GPU è passata a basalt per il suo blocco; poi tocca al nostro turno 30.
- 03/11 21:05 (decisione dell'utente): sulla GPU si ALTERNA un turno LTX Studio e un blocco di basalt. I nostri turni in ordine: 15 (in corso) → 30 → 31 → 32 → 35 (Krea 2 Edit: prompt, passi e CFG) → 33 → 34 → 16 → 17 → 18 → 12 → 13 → 14.
- 03/11 21:00: cause vere trovate. Commit 0933ed83, 775669d8, b40de0fa, merge 28252364.
  - Integra Qwen: il composito e l'originale annotato fanno da riferimento, e il modello di edit li ricopia.
  - Integra Kontext: non usa Kontext ma un inpaint al 65 % senza identità, per questo la ragazza cambia.
  - Inpaint FLUX base: la maschera è troppo piccola, circa 65 token.
  - MSR LTX 2.5: la LoRA non veniva mai applicata (corretto).
  - Controlli automatici pronti: invariato, incollata, identità, oggetto, volto nei video. Tutti i 38 casi bocciati oggi risultano rossi.
  - Approcci nuovi da provare.
  - Turni pronti: 30 (le 30 immagini dei 6 scenari), 31 (Integra), 32 (inpaint), 33 (MSR 2.5), 34 (MSR 2.3).
- 03/11 20:24: turno 15 in corso: H3, rimedi al doppione e al lip sync del primo pezzo. Poi, in fila, i turni 12, 16, 13, 14 e 17 (velocità, rincorsa, quantizzazioni, turbo 4 passi, video base nuovi). Basalt ha finito la prova breve; la sua corsa da 55 ore aspetta il via dell'utente.
- 03/11 20:24: regola nuova: le prove girano dal worktree pulito .claude\worktrees\prove, portato all'ultimo commit di feature/openjutsu prima di ogni turno. Gli agenti lavorano in openjutsu e le loro modifiche a metà non entrano più nelle prove.
- 03/11 19:50: giudizio dell'utente sulla pagina dei 48 casi rifatti (docs\pubblicato_2026-11-03T18-47-37-019Z.md): quasi tutto bocciato.
  - 10 casi su 19 erano IDENTICI byte per byte al mattino: le «correzioni» erano solo varianti spente di default. Errore di metodo: d'ora in poi si controlla in automatico prima di ogni pagina.
  - Integra incolla ancora la ragazza: il modello di edit riceve la scena con la ragazza già incollata e la conserva.
  - L'inpaint FLUX non genera l'oggetto; i video con riferimenti perdono le identità.
  - Decisione dell'utente: soggetti nuovi (orsacchiotto astronauta al posto della bussola, ambiente più semplice, ragazza nuova).
  - Agente nuovo con brief orch\BRIEF_integra_inpaint_msr_v2_03-11.md: cause vere, controlli automatici («invariato», «incollata», oggetto presente, volto), esperimenti GPU mirati.
- 03/11 19:50: difetti H3 nuovi segnalati dall'utente: nel cambio sfondo, nei primi 5 secondi, due immagini della persona non sincronizzate; in Sostituisci e Scambia manca il lip sync nei primi secondi. Il resto va bene («buona sostituzione»). Passati all'agente H3.

## In coda — senza GPU
- 03/11 sera (decisione dell'utente): in Modifica Video, solo con il motore H3, quattro opzioni con i default invariati:
  - sezione LoRA H3, compresa la turbo a 4 passi;
  - casella «Attenzione sparsa Veda»;
  - casella «Decoder leggero (LightVAE)»;
  - menu «Modello H3» con i soli modelli che sanno fare l'editing, scaricati alla prima scelta o dal tab Pesi.
  Poi il confronto della qualità fra nvfp4, int8 e int4 sullo stesso caso. Il limite di pixel del text encoder non si applica (non serve). Brief: orch\BRIEF_h3_velocita_modelli_03-11.md (agente esistente).
- 03/11 sera: ricerca dei pesi H3 non ufficiali (distillazioni, quantizzazioni, ibridi, LoRA di edit). Prima dei download si mostrano all'utente la lista e i GB. Brief: orch\BRIEF_ricerca_pesi_h3_03-11.md.
- 03/11 (decisione dell'utente): in Integra, «Zona di contesto» con due modalità a scelta. Automatica: cursore e campo libero in %, anche oltre il 60%. Manuale: pennello, che parte dalla zona automatica. Sul canvas sempre visibili, insieme, la posizione in rosso e il contesto in giallo. Un selettore sceglie quale maschera si sta disegnando, e sopra il canvas una scritta lo dice nello stesso colore della maschera; e due cursori verticali, uno per maschera, regolano la trasparenza in %. Vale per tutti i motori che usano le due maschere. Classica + Pro, 15 lingue.
- 03/11 mattina: l'utente ha giudicato la pagina dei 45 casi: 22 da rifare, 23 superati. Problemi principali:
  - Integra incolla la persona scontornata invece di integrarla; con Kontext la persona entra nel tavolo e l'utente propone due maschere, una per la posizione e una per il contesto;
  - Krea 2 Edit non mette la ragazza;
  - l'inpaint FLUX mette macchie invece della bussola;
  - nei video con riferimenti le identità si perdono;
  - in H3 i primi 2 secondi hanno lo sfondo vecchio, ci sono macchie bianche ai bordi e contorni scuri.
  Brief: orch\BRIEF_giudizio_45casi_03-11.md.
- 03/11: correzione della memoria (opzione b) fatta nel codice. Misure sulla CPU: FLUX FP8 scende a 25,7 GB di picco; Klein 9B da 22,5 a 12 GB; per Krea 2 il caricamento va verificato sulla GPU. Restano da rifare dall'app vera i 5 casi rossi del giro A.
- 03/11 03:32: SAM 3.1 scaricato in E:\AI\Models\Segmentation\sam3.1 (3,5 GB, dimensione verificata). Si usa nella fase 2.
- 03/11: fase 2, prima parte fatta senza GPU: menu «Segmentazione» (SAM2 predefinito, SAM 3.1), finestra di aiuto, voce di catalogo, 15 lingue. SAM 3.1 per ora non si può scegliere: il suo formato è quello originale di Meta e serve un caricatore. Si usa il pacchetto ufficiale «sam3»: sorgente in E:\AI\Tools\sam3_src (commit f1647e8d0c) e due piccole dipendenze in E:\AI\Tools\sam3_deps, scaricate il 03/11 senza toccare gli ambienti Python esistenti. Da provare sulla CPU. Poi viene il tab «Scambia».
- 03/11: verificare due funzioni indicate dall'utente. (1) La sfumatura della maschera in H3, forse troppo larga e causa dell'alone bianco. (2) La correzione del tono vicino alle giunture (_seam_tone_match in image_flux.py): forse crea le cornici chiare in Integra e non sistema il salto di tono nell'outpaint.
- 03/11: correggere i commenti dell'utente alle 3 pagine di revisione. Tolti i doppioni: restano 43 casi (orch\app_live\commenti_revisioni_unificati_20261103.md). Problemi principali: Integra incolla la foto con una cornice bianca invece di integrarla; nell'inpaint la bussola non compare o vola, forse per la maschera del test messa nel posto sbagliato; job falliti. Viene prima della fase 2 di Openjutsu. Brief: orch\BRIEF_commenti_revisioni_03-11.md.
- Punto 16 della lista del 30/10 (finestra di anteprima in Modifica immagine, come in Crea): nei commit non risulta fatto, da verificare e fare.
- Unire il ramo integrazione-30-10 nel ramo principale, solo con i nostri pezzi, dopo l'ok dell'utente.

## In coda — con GPU
- Nessun lavoro in coda sulla GPU: si riparte dal giudizio dell'utente.

## Decisioni in attesa dell'utente
- Cosa fare per il crash di Windows: aggiornare il driver NVIDIA, provare la RAM, attivare il file di diagnosi del crash.
- Immagini d'esempio delle LoRA: trovate solo per 28 LoRA su 103, serve un altro metodo.

## Fatti
- 30/10: nuovo sistema di test sull'app vera, con pagina di revisione e bottone «Pubblica».
- 30-31/10: primo giro completo (88 casi). Trovati due errori del sistema di test: il canvas non si svuotava e la revisione mostrava file di altri casi. Corretti.
- 31/10: corretti molti dei 29 punti segnalati dall'utente (lista lavori, download in background, avanzamento, Componi, selezione AI, video lunghi per versione, import civitai.red).
- 31/10 03:15-05:43: giro mirato, 35 casi: 24 ok, 9 falliti, 2 chiusure dell'app per memoria.
- 31/10-01/11 (finito alle 05:09): corretti tutti i 16 casi rossi delle revisioni, riprovati dall'app vera. Tutti verdi: LoRA su fp8, Z-Image svdq, LTX Q6_K, MiniMax H3, FLUX.2 dev nvfp4, Krea 2 Edit, edit di un file video, voci Qwen 18+, polpo, lazo. Corretto anche il blocco dell'app all'avvio, dovuto all'elenco delle voci online.
- 31/10 14:09-18:10: giro completo finito (ripreso dal caso 5, RAM a 3000, nessuna schermata blu). 104 casi: 89 ok, 8 falliti (gli stessi del giro mirato), 5 errori del sistema di test, 2 chiusure dell'app per memoria.

## Decisioni già prese
- 03/11 (decisione dell'utente): la zona di contesto automatica si misura in PIXEL attorno alla sagoma vera della ragazza (non al rettangolo), allargata seguendo la sua forma. Accanto, solo come indicazione, la % sulla larghezza della ragazza. A 0 px la zona coincide con la sagoma. Il valore iniziale è vicino al comportamento di oggi.
- 02/11: Openjutsu. Il codice GPL dei nodi ComfyUI resta fuori: si rifà tutto dal codice MIT e dalle descrizioni.
- 02/11: SAM2 e SAM 3.1 tutti e due, scelti dall'utente in un menu. SAM2 resta il predefinito. Se SAM 3.1 manca, l'app spiega di usare SAM2 intanto e come chiedere l'accesso su HuggingFace, avvisando che l'approvazione richiede tempo.
- 02/11: SAM 3.1 sì. La licenza di Meta permette l'uso commerciale; noi diamo il link e l'utente lo scarica da HuggingFace.
- 02/11: MiniMax H3 e le sue LoRA: noi non distribuiamo i pesi, diamo solo il link; l'utente li scarica da sé.
- 02/11: in Openjutsu la capacità più importante per l'utente è «cambia sfondo» tenendo persona e movimento di camera: va nella prima fase.
- Modifica: slider «Fedeltà all'originale», 0 = cambia tutto.
- Riferimenti nel prompt: @image_N.
- civitai.red: import da pagina e procedura guidata per la chiave, abbonamento manuale.
- Immagini d'esempio delle LoRA: cercate online, mai generate in automatico.
- La sessione basalt ha la precedenza sulla GPU quando la chiede.
- 02/11: il motore H3 si aggiunge anche alle azioni che esistono già (oggetti, personaggio, forse modifica globale e lip-sync, se la prova va bene). Non si toglie nulla e i motori di oggi restano quelli predefiniti.
- 03/11: la GPU si divide a turni con basalt tramite il file C:\TEMP\gpu_lock.json (autorizzato dall'utente). Chi la vuole controlla il file: se è libera la prende, se è occupata si mette in attesa. Un turno è un giro intero di al massimo 70 minuti, poi si libera il file e si manda FINE. Sostituisce la regola del 02/11 (il sì di basalt ogni volta).

## Altri progetti aperti
- basalt (altra sessione): prove del motore sulla GPU, coordinate con noi.
