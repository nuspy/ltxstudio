# LTX Studio — Sistema di Licenze (Business + Legale)

> Documento non tecnico. Spiega il **modello di licenza** al cliente, fornisce la **bozza di EULA/Termini d'uso** e i **guardrail legali** non negoziabili.
> La documentazione tecnica del codice di licensing esiste separatamente.

---

## 1) Come funziona la licenza (spiegato al cliente, in parole semplici)

Compri il tier che vuoi (Demo, SFW, NSFW, Studio o Founder Lifetime) e ricevi **un codice/file di licenza personale**. Lo incolli una sola volta dentro l'app, dalla schermata "Licenza", e l'app lo verifica **direttamente sul tuo computer, offline**: non serve nessuna connessione a internet, nessun account, nessun server da contattare. Da quel momento l'app sblocca tutte le funzioni del tuo tier e continua a funzionare anche se sei completamente scollegato dalla rete. Il codice è **tuo** e resta valido secondo le regole di versione descritte qui sotto.

In pratica: **un acquisto → un codice → incolli → funziona, anche offline, per sempre nelle versioni a cui hai diritto.**

---

## 2) Upgrade e Downgrade (con esempio numerico)

Ogni versione dell'app ha un **numero di build crescente** (v100, v101, v102, …). La tua licenza contiene una o più **"finestre di versione"** del tipo `{da_build, a_build, tier}`: cioè *"da questa build a quest'altra build hai diritto a questo tier"*. Funziona come una **licenza perpetua con fallback** (lo stesso meccanismo delle licenze JetBrains): quello che hai sbloccato mentre pagavi un certo livello **resta tuo per sempre su quelle versioni**, anche se in futuro cambi piano.

### Upgrade (passi a un tier superiore)
Lo **stesso codice** sblocca subito il tier più alto, senza doverlo sostituire. Non devi reinstallare nulla: l'app riconosce il nuovo diritto e abilita immediatamente le funzioni aggiuntive.

### Downgrade (passi a un tier inferiore o annulli)
Le versioni dell'app **già rilasciate mentre avevi il tier superiore restano sbloccate a quel livello, per sempre**. Solo le versioni **rilasciate dopo** il downgrade ti daranno il tier inferiore.

### Esempio concreto e numerico

- **Giugno** — Compri **T3 (NSFW, 25€)**. Scarichi la **v100**, l'ultima disponibile. Tutto NSFW sbloccato.
- **Luglio / Agosto** — Escono **v110** e **v115**: le scarichi, sono tutte **NSFW** per te, perché rilasciate mentre eri T3.
- **Settembre** — Decidi di passare a **T2 (SFW, 15€)** — un downgrade.
- Risultato: **la v100, la v110, la v115 e qualsiasi versione fino a quella di settembre restano NSFW per te, per sempre.** Continui a usarle con tutte le funzioni del tier superiore.
- **Ottobre** — Esce la **v120**, rilasciata *dopo* il tuo downgrade. Quella versione per te sarà **solo SFW (T2)**. Se la vuoi NSFW, ti basta tornare a T3 e la sbloccherai (e tutte le successive finché resti T3).

**In una frase:** non perdi mai ciò che hai "maturato" — perdi solo l'accesso ai tier superiori **sulle versioni future** rilasciate dopo aver smesso di pagarli.

---

## 3) Perché diamo il compilato e non il sorgente

LTX Studio viene distribuito come **eseguibile compilato (Nuitka)**, non come codice sorgente. È una scelta di sostenibilità, non di sfiducia: il sorgente, una volta in circolazione, viene ridistribuito e il prodotto smette di poter esistere come progetto pagato che riceve aggiornamenti e supporto. Siamo onesti su un punto: **un eseguibile compilato è, in teoria, decompilabile** — non esiste protezione perfetta, e non la promettiamo. Il nostro obiettivo è scoraggiare il **95%** degli usi impropri, non l'1% di esperti determinati. La protezione reale non è l'offuscamento: è la **firma crittografica** delle licenze (nessuno può *emettere* codici validi senza la nostra chiave privata) e il flusso continuo di **aggiornamenti, modelli e supporto** che solo i clienti legittimi ricevono. Chi pirata congela una vecchia copia; chi paga migliora ogni mese.

---

## 4) FAQ Licenze

**1. Funziona senza internet?**
Sì. La verifica della licenza è **completamente offline**: avviene sul tuo PC tramite firma crittografica. Nessuna connessione obbligatoria, né all'attivazione né durante l'uso. Internet serve solo se vuoi scaricare aggiornamenti o nuovi modelli.

**2. Posso usarlo su 2 PC?**
La licenza è **personale e a uso singolo**. È pensata per un uso da parte di una sola persona; puoi installarla sulle **tue** macchine (es. desktop e portatile) purché l'uso non sia simultaneo e non condiviso con terzi. L'uso contemporaneo da parte di più persone o la condivisione del codice non sono consentiti.

**3. Cosa succede se annullo l'abbonamento / faccio downgrade?**
Mantieni **per sempre** il tier sulle versioni rilasciate mentre lo pagavi (vedi sezione 2). Le versioni rilasciate **dopo** l'annullamento/downgrade torneranno al tier inferiore o, se annulli del tutto un piano a pagamento, al livello Demo per le build future.

**4. Cosa succede agli aggiornamenti?**
Gli aggiornamenti pubblicati **mentre hai un tier attivo** restano sbloccati a quel tier per sempre. Gli aggiornamenti **futuri**, rilasciati dopo la scadenza/annullamento, seguono il tier che hai in quel momento. La **Founder Lifetime** sblocca il tier al massimo livello su tutte le build presenti e future incluse nel piano.

**5. Posso rivenderlo, regalarlo o trasferirlo?**
No. La licenza è **non trasferibile**: non può essere rivenduta, ceduta, prestata, sub-licenziata o regalata. Un codice è legato all'acquirente originale.

**6. Ho cambiato PC / formattato: perdo la licenza?**
No. Ti basta reinstallare l'app e reincollare il **tuo codice di licenza**: conservalo (es. nella mail d'acquisto). Non è legato all'hardware in modo bloccante.

**7. Che differenza c'è tra Demo e i tier a pagamento?**
La **Demo (T1)** è gratuita ma limitata e applica un **watermark**. I tier a pagamento rimuovono il watermark e sbloccano funzioni: **T2 SFW (15€)**, **T3 NSFW (25€)**, **T4 Studio (45€)** con import di LoRA/modelli personali e accesso API.

**8. La Founder Lifetime cosa include?**
Un pagamento unico (**199–299€**) che dà accesso a vita al tier al livello massimo previsto dal piano, comprese le build future incluse. È pensata per i primi sostenitori del progetto.

**9. Se trovo il mio codice condiviso/piratato online?**
Segnalacelo. Non possiamo "spegnere" da remoto una copia offline, ma **possiamo invalidare** un codice per gli aggiornamenti e il supporto futuri. Chi usa copie piratate **non riceve update, modelli nuovi né assistenza**.

**10. Posso usare l'app per lavoro / a fini commerciali?**
Sì, i contenuti che **tu generi** li usi come vuoi, nei limiti di legge e dei guardrail (sezione 6). La licenza copre l'uso del software da parte tua; **non** ti autorizza a rivendere o ridistribuire il software stesso.

**11. Rimborsi?**
Eventuali rimborsi seguono i termini di vendita indicati al checkout e la normativa applicabile [PAESE]. La Demo gratuita esiste proprio per provare il prodotto prima di acquistare.

---

## 5) Bozza EULA / Termini d'Uso

> Testo pronto all'uso, da rivedere con un legale prima della pubblicazione. I campi tra [parentesi] sono placeholder.

**CONTRATTO DI LICENZA CON L'UTENTE FINALE (EULA) — LTX Studio**
*Ultimo aggiornamento: [DATA]. Titolare: [RAGIONE SOCIALE / NOME], [INDIRIZZO], [CONTATTO].*

**1. Oggetto.** Il presente contratto regola la concessione in licenza d'uso del software "LTX Studio" (il "Software"), distribuito in forma di **eseguibile compilato**. Installando, attivando o utilizzando il Software, l'utente ("Licenziatario") accetta integralmente questi termini. In caso di disaccordo, non installare né utilizzare il Software.

**2. Concessione della licenza.** Il Titolare concede al Licenziatario una licenza **personale, non esclusiva, non trasferibile e revocabile** all'uso del Software secondo il tier acquistato. La licenza è concessa in uso, **non venduta**; ogni diritto non espressamente concesso resta del Titolare.

**3. Restrizioni.** È vietato: (a) effettuare **reverse engineering**, decompilazione, disassemblaggio o tentare di estrarre il codice sorgente, salvo nei limiti inderogabili di legge; (b) **copiare, ridistribuire, rivendere, cedere, sub-licenziare, noleggiare o pubblicare** il Software o il codice di licenza; (c) rimuovere o aggirare watermark, controlli di licenza o misure tecniche; (d) condividere il proprio codice di licenza con terzi o usarlo in modo simultaneo/condiviso.

**4. Tier e aggiornamenti.** Le funzionalità disponibili dipendono dal tier attivo. Gli aggiornamenti rilasciati mentre un tier è attivo restano fruibili a quel tier sulle relative versioni ("perpetual fallback"); le versioni rilasciate dopo un downgrade/annullamento seguono il tier vigente al momento del rilascio.

**5. Contenuti illegali — DIVIETO ASSOLUTO.** È **tassativamente vietato** usare il Software per produrre, modificare o distribuire: (a) **qualsiasi contenuto che ritragga minori** in contesti sessuali o di sfruttamento (condotta penalmente rilevante); (b) **deepfake o rappresentazioni non consensuali di persone reali identificabili**, in particolare a contenuto sessuale, diffamatorio o ingannevole; (c) contenuti che violino leggi applicabili, diritti di terzi (immagine, privacy, proprietà intellettuale) o l'ordine pubblico. La violazione comporta la **revoca immediata** della licenza, senza rimborso, e l'eventuale segnalazione alle autorità competenti.

**6. Contenuti per adulti (NSFW) — 18+.** Le funzioni NSFW sono riservate a **maggiorenni (18+)**. Attivandole, il Licenziatario dichiara di avere l'età legale nella propria giurisdizione e di assumersi ogni responsabilità sull'uso e sulla conservazione dei contenuti generati.

**7. Responsabilità sui contenuti.** Il Licenziatario è l'**unico responsabile** dei contenuti che genera, importa o distribuisce con il Software, e degli usi che ne fa. Il Titolare non controlla, non approva e non è responsabile di tali contenuti.

**8. Nessuna garanzia.** Il Software è fornito **"così com'è" e "come disponibile"**, senza garanzie di alcun tipo, esplicite o implicite, incluse idoneità a uno scopo specifico, assenza di errori o continuità del servizio. Nei limiti di legge, il Titolare **non è responsabile** per danni diretti, indiretti, incidentali o consequenziali derivanti dall'uso o dall'impossibilità d'uso del Software. Ove la responsabilità non possa essere esclusa, essa è limitata all'importo pagato dal Licenziatario per la licenza.

**9. Durata e risoluzione.** La licenza è efficace dall'attivazione. Il Titolare può risolverla in caso di violazione dei presenti termini; alla risoluzione cessa ogni diritto d'uso. Le clausole su divieti, responsabilità e legge applicabile sopravvivono alla risoluzione.

**10. Proprietà intellettuale.** Il Software, il marchio "LTX Studio" e i relativi diritti restano di esclusiva titolarità del Titolare e/o dei rispettivi licenzianti.

**11. Legge applicabile e foro.** Il presente contratto è regolato dalla legge di **[PAESE]**. Per ogni controversia è competente in via esclusiva il **foro di [CITTÀ/SEDE]**, salvo norme inderogabili a tutela del consumatore.

**12. Varie.** L'eventuale invalidità di una clausola non pregiudica le restanti. Il presente EULA costituisce l'intero accordo tra le parti in merito al Software.

*Continuando a usare il Software dichiari di aver letto e accettato il presente EULA.*

---

## 6) Guardrail legali operativi

### Cosa NON promettere MAI nel marketing
- Mai presentare o suggerire la generazione di **deepfake di persone reali** (celebrità, politici, persone identificabili) né "volto di chiunque".
- Mai accostare le funzioni NSFW a **minori**, "giovani", età ambigue, contesti scolastici o linguaggio che evochi minorenni. **Tolleranza zero.**
- Mai promettere **anonimato totale**, "impossibile da tracciare", "fai quello che vuoi senza conseguenze".
- Mai garantire **protezione anticopia totale**, "impossibile da piratare/decompilare". Restiamo onesti (vedi sezione 3).
- Mai promettere **risultati legali/professionali** specifici o uso del prodotto per finalità illecite, fraudolente o diffamatorie.

### Age-gate (controllo età)
- **Gate 18+ obbligatorio** prima di accedere a qualsiasi area o funzione NSFW: schermata di conferma esplicita "Dichiaro di essere maggiorenne (18+)" con accettazione attiva.
- Le pagine di marketing che mostrano o descrivono NSFW devono avere il gate **prima** del contenuto, non dopo.
- Conservare evidenza dell'accettazione dei termini e del gate (timestamp), ove tecnicamente possibile e nel rispetto della privacy.

### Disclaimer da mettere sulle pagine
- **Pagine NSFW:** *"Contenuto riservato a un pubblico adulto (18+). Vietato generare contenuti raffiguranti minori o deepfake non consensuali di persone reali: tali usi sono illegali e comportano la revoca della licenza e la segnalazione alle autorità."*
- **Pagina prezzi/licenze:** *"La licenza è personale e non trasferibile. La verifica avviene offline. Vedi EULA per i termini completi."*
- **Footer generale:** *"L'utente è l'unico responsabile dei contenuti generati. Il Software è fornito senza garanzie. © [ANNO] [TITOLARE]."*

### Principio guida
Fermi e inflessibili sui **divieti legali** (minori, deepfake non consensuali, 18+, responsabilità dell'utente). Rassicuranti, chiari e onesti con il **cliente corretto**: offline, licenza tua, nessun trucco, aggiornamenti veri.
