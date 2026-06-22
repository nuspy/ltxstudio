# 00 — Strategia Master (il quadro completo)

> Questo è il "perché" dietro tutto. **Non serve per agire** (per agire usa `00_START_QUI.md`). Leggilo quando hai energia, per avere il disegno completo in testa.

## Il problema che risolviamo
Sei senza lavoro con ~1 mese di liquidità. Hai un prodotto desktop AI maturo (LTX Studio) e altri componenti. Obiettivo: **reddito ricorrente il più in fretta possibile**, vendendo **solo il compilato** (no sorgente) con **licenze a tier**.

## La tesi in una frase
**Vendiamo un AI Studio desktop all-in-one, locale e senza censura, come app già pronta** — a un pubblico che oggi è costretto a scegliere tra tool gratis ma complicatissimi (SwarmUI/ComfyUI) e servizi cloud comodi ma censurati, a consumo e non privati. Noi diamo il meglio dei due: potenza completa + zero-setup + privacy totale.

## Le 3 leve che lavorano insieme
1. **Cassa immediata** — vendite *one-time* Founder Lifetime (199-299€, 30 posti) + vendite dirette su itch.io. È ciò che copre il mese critico.
2. **MRR ricorrente** — abbonamenti a 4 tier che crescono nel tempo (sostenibilità mese 2-6).
3. **Funnel di traffico** — YouTube (motore principale) + community (Reddit/Civitai/Discord) + demo gratuita come esca. Senza traffico, nessun tier converte.

## Architettura commerciale (decisa)
- **Patreon** = vetrina 100% SFW + community + funnel. Niente NSFW (lo vieta). Brand personale (volto + nome).
- **SubscribeStar.adult** = tier NSFW (T3) + download protetti. Handle separato anonimo.
- **itch.io + Stripe/sito** = vendite one-time (demo gratis + Founder).
- **Discord** = community, supporto, retention.
- **YouTube** = acquisizione principale (modello: SECourses).

## I pacchetti
T1 Demo (gratis, funnel) · T2 SFW 15€ · **T3 NSFW 25€ (target più redditizio)** · T4 Studio 45€ (LoRA + API) · **Founder Lifetime 199-299€** · early-bird −30%. Dettaglio e matrice funzioni in `03`.

## Il target
- **T3 NSFW** = traino di cassa: creator adult che vogliono generare in locale, privati, senza crediti né censura cloud.
- **T2 SFW** = volume: creator video/immagini, indie game dev (3D+rig), marketer.
- **T4 Studio** = upsell pro (modelli propri + API). Dettaglio in `02`.

## Il differenziatore difendibile vs concorrenza
- vs **tool gratis** (SwarmUI/Pinokio): integrazione profonda (video-extend illimitato + 3D+rig + voce in UNA GUI) + NSFW pronto + zero-setup compilato + supporto + update + catalogo curato.
- vs **cloud NSFW** (ZenCreator/Mage): locale, privato, niente crediti/limiti, censura zero, usi i tuoi modelli.
- vs **SECourses** (il modello da emulare): lui vende script attorno a tool altrui; noi vendiamo un'app integrata propria. Dettaglio in `01`.

## La protezione del business
- **Compilato (Nuitka)**, non sorgente → niente redistribuzione facile.
- **Licenze firmate Ed25519**, verifica offline, modello a "finestre di versione" (= perpetual fallback JetBrains) per upgrade/downgrade. Nessuno può *emettere* licenze senza la tua private key.
- **Update continui** = la vera difesa: chi pirata una versione non riceve miglioramenti e supporto. Dettaglio in `06`.

## La verità sul runway (onestà)
Gli abbonamenti rendono lentamente (primi ~100€ tra mese 6-9, churn 5-10%/mese). **La cassa che ti salva il primo mese viene dai Founder/one-time, non dall'MRR.** Per questo: posti Founder limitati + lancio veloce. **Checkpoint al giorno 21** per decidere con dati alla mano *prima* che la liquidità finisca. Numeri in `04`.

## I guardrail non negoziabili
NSFW solo fuori da Patreon/Gumroad. **Vietati deepfake non consensuali di persone reali e qualsiasi contenuto con minori** (penale, distrugge tutto). 18+ rigoroso, EULA, utente responsabile dei contenuti. In `06`.

## Sequenza di lancio (14 giorni)
Settimana 1 = fondamenta (account + community) + costruzione (video 1, pagine). Settimana 2 = lancio (pubblica video, apri Founder) + prime vendite. In parallelo io costruisco il sistema licenze + il compilato. Passo-passo in `00_START_QUI.md`.

## Cosa faccio io vs cosa fai tu
- **Io (già fatto):** ricerca di mercato, tutti i documenti di `marketing/`, i testi pronti, gli script video, il piano licenze.
- **Io (in corso):** codice del sistema licenze + gating per tier + build compilata di consegna.
- **Tu:** apri gli account, registra i video (script già pronti), pubblica i post (già scritti), rispondi alla community, esegui i build quando te li passo. Tutto guidato passo-passo in `00_START_QUI.md`.
