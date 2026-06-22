# LTX Studio — Target Audience & Buyer Personas

## 1. Sintesi: chi paga e perché

Pagano creator e power user che vogliono generare immagini, video e asset 3D in **locale**, sulla **loro** GPU NVIDIA, senza crediti, senza abbonamenti cloud che lievitano e senza ToS che li censura. Il cuore di cassa è il creator **adult** (T3) che ha bisogno di privacy assoluta e zero censura; il volume arriva dai creator/dev SFW (T2). LTX Studio vende una cosa sola, ripetuta in mille modi: **"i tuoi contenuti, sul tuo PC, senza nessuno che guardi o conti."**

---

## 2. Personas per tier

### T1 — "Marco il Curioso" (Demo gratis, funnel)
- **Chi è:** appassionato AI che ha già visto decine di video YouTube su Stable Diffusion/ComfyUI e vuole "provare prima di pagare".
- **Hardware già posseduto:** RTX 3060/3070 12GB o superiore, a volte una 4070. Ha già installato ComfyUI almeno una volta.
- **Dolore principale:** ha messo insieme 30 nodi a mano, si rompe a ogni aggiornamento, e i tutorial sono già obsoleti. Perde più tempo a far funzionare i workflow che a creare.
- **Cosa cerca:** un'app che "fa la cosa" senza dover cablare grafi, per capire se vale i soldi.
- **Perché paga (zero):** non paga — è il gancio. Prova la demo, vede che genera in 3 click, e capisce cosa gli manca.
- **Messaggio di vendita:** "Smetti di cablare nodi. Apri, scrivi, genera — gratis, sul tuo PC."
- **Dove lo trovi:** r/StableDiffusion, r/comfyui, commenti YouTube (Olivio Sarikas, Sebastian Kamph, Nerdy Rodent), Discord di ComfyUI e dei modelli LTX/FLUX.

### T2 — "Giulia la Produttrice" (SFW ~15€/mese, volume)
- **Chi è:** creator di contenuti video/immagini, indie game dev, marketer, animatore o montatore che produce asset ogni settimana per lavoro.
- **Hardware già posseduto:** RTX 4070 Ti / 4080 / 3090, workstation da lavoro. Spesso già abbonata a un tool cloud che le costa caro a volume.
- **Dolore principale:** i crediti cloud finiscono a metà progetto, il rendering di un video lungo costa una fortuna, e per un indie dev pagare mesh + rigging a parte è un salasso. La prolunga video sui SaaS è limitata a pochi secondi.
- **Cosa cerca:** pipeline completa locale — immagini, video con **prolunga illimitata**, mesh 3D + rigging per asset di gioco, restauro, ControlNet per il movimento — a costo fisso mensile.
- **Perché paga proprio QUESTO tier:** lavora SFW (clienti, store, YouTube monetizzabile), non gli serve l'adult, e a 15€/mese sostituisce 3-4 abbonamenti cloud che pagava a consumo.
- **Messaggio di vendita:** "Tutta la tua pipeline — video illimitati, 3D, rig, restauro — a costo fisso, senza crediti che finiscono."
- **Dove lo trovi:** indie game dev su r/gamedev, r/IndieDev, itch.io community, Discord di Godot/Unity/Unreal, gruppi di montatori e motion designer, YouTube creator economy, Patreon (campagna SFW).

### T3 — "Alex il Creator Adult" (NSFW ~25€/mese, **target più redditizio**)
- **Chi è:** creator adult su OnlyFans/Fansly, character/waifu artist, generatore di contenuti NSFW che oggi vive nel terrore del ban e dei costi a credito.
- **Hardware già posseduto:** RTX 3090/4090 (spesso comprata apposta), a volte doppia GPU. Tecnicamente già smaliziato.
- **Dolore principale:** i SaaS cloud lo **censurano**, gli **contano i crediti**, gli **ispezionano i contenuti** e possono bannarlo o leakare materiale sensibile. Ogni generazione passa per server di terzi: rischio reputazionale e legale. I generatori NSFW online sono lenti, costosi e poco privati.
- **Cosa cerca:** generare set, personaggi consistenti, video adult e voce/clone **interamente in locale**, senza che un byte esca dal suo PC, senza filtri di contenuto, senza limiti di crediti.
- **Perché paga proprio QUESTO tier:** privacy + zero censura + zero credit = valore esistenziale per il suo business. 25€/mese è nulla rispetto a un singolo set acquistato o a un ban che gli azzera l'account. È il tier con la **disponibilità a pagare più alta** e il churn più basso.
- **Messaggio di vendita:** "Nessun credito, nessun ToS, nessuno che guarda. I tuoi contenuti adult restano sul tuo PC — punto."
- **Dove lo trovi:** community NSFW di Civitai (modelli e tag adult), SubscribeStar.adult, subreddit creator adult (r/onlyfansadvice e simili), Discord di character/waifu artist, forum AI-NSFW, gruppi di "AI influencer" e virtual model.

### T4 — "Dario lo Studio" (Studio ~45€/mese)
- **Chi è:** power user/piccolo studio che **addestra e colleziona LoRA propri**, costruisce pipeline ripetibili e vuole automazione.
- **Hardware già posseduto:** RTX 4090 / multi-GPU, NAS pieno di checkpoint e LoRA, ambiente da pro.
- **Dolore principale:** ha centinaia di LoRA/modelli sparsi senza metadati, deve classificarli a mano, e nessun tool gli dà un'**API REST** per orchestrare la generazione da script o da un suo backend.
- **Cosa cerca:** **import** di LoRA/modelli custom + **classificazione automatica** + **API REST** per integrare LTX Studio nella propria pipeline e generare a batch.
- **Perché paga proprio QUESTO tier:** è l'unico che gli dà import + auto-catalogazione + API; per chi monetizza con pipeline industriali, 45€/mese è un costo operativo trascurabile.
- **Messaggio di vendita:** "Importa i tuoi LoRA, lascia che li classifichi da soli, e pilota tutto via API."
- **Dove lo trovi:** community di LoRA trainer (Civitai creator, Discord di kohya_ss/OneTrainer), r/StableDiffusion sezione training, gruppi di sviluppatori che costruiscono prodotti sopra ComfyUI, HuggingFace.

---

## 3. Anti-persona: a chi NON vendere

- **Chi non ha GPU adeguata.** Sotto una RTX 3090 (o equivalente VRAM) l'esperienza è frustrante o impossibile. Non spendere un euro di marketing su chi ha una 1660, una GPU integrata o solo un portatile da ufficio: genererà solo richieste di supporto e recensioni negative.
- **Chi vuole un'app cloud "dal telefono".** LTX Studio è desktop Windows, locale, gira sulla **loro** macchina. Chi cerca un'app mobile o un servizio web "apri e usa ovunque" è nel posto sbagliato — non è il nostro prodotto e non lo sarà.
- **Chi cerca il gratis a tutti i costi.** Chi vuole solo la demo per sempre, o pretende tutto sbloccato senza pagare, non si converte: assorbe supporto e zero ricavi. La demo (T1) serve come funnel, non come pubblico da coltivare all'infinito.
- **Chi non vuole installare ComfyUI / non è disposto a un setup tecnico.** Serve un minimo di volontà tecnica. Chi pretende "un click e tutto magicamente pronto senza prerequisiti" va gestito con aspettative chiare prima dell'acquisto.

Filtrare bene questi profili **fin dal copy** (requisiti hardware in evidenza, "100% locale Windows") riduce refund, chargeback e supporto.

---

## 4. Gerarchia di priorità per il lancio

**1° — T3 NSFW come traino di cassa.** È il segmento con la disponibilità a pagare più alta (25€), il churn più basso (il valore privacy/no-censura/no-credit è esistenziale per il loro business) e i meno sensibili al prezzo. La leva di vendita è netta e difendibile: i SaaS cloud **non possono** offrire ciò che offriamo (locale, privato, senza censura, senza crediti). Concentrare qui il messaggio principale su SubscribeStar.adult, Civitai NSFW e Discord di character artist massimizza il ricavo per euro di marketing speso.

**2° — T2 SFW come volume.** Più grande come bacino (indie dev, marketer, montatori, animatori), pubblicizzabile **apertamente** su Patreon, YouTube e itch.io senza vincoli di piattaforma adult. Prezzo d'ingresso basso (15€), valore evidente (sostituisce abbonamenti cloud a consumo). È la base larga che dà numeri e prova sociale.

**3° — T4 Studio come upsell ad alto margine.** Pochi clienti ma ad alto valore (45€) e churn bassissimo: chi integra l'API non se ne va più. Non è un segmento da inseguire a freddo al lancio — si coltiva **convertendo** i T3/T2 più tecnici che già usano il prodotto e chiedono import + API.

**4° — T1 Demo come motore del funnel.** Non genera ricavo diretto: il suo unico compito è alimentare i tre tier a pagamento. Va misurato sul tasso di conversione demo→pagante, non sul numero di download.

**In sintesi del lancio:** martellare T3 sui canali adult per fare cassa, far volume con T2 sui canali pubblici, lasciare che T4 emerga come upsell naturale, e usare T1 come imbuto sempre acceso.
