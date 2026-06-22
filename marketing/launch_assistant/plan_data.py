# -*- coding: utf-8 -*-
"""Dati del piano di lancio: fasi, passi (coordinati con dipendenze), premi, testi
pronti da incollare e messaggi motivazionali.

Pensato per chi è sotto forte stress: ogni passo è atomico, ha un detail breve,
un link diretto all'azione e un riferimento al documento in marketing/.
Tutto è guidato dall'app assistant.py — l'utente non legge questo file.
"""
from __future__ import annotations

# Cartella marketing (i documenti .md a cui i passi rimandano)
DOCS = {
    "start": "00_START_QUI.md",
    "strategia": "00_STRATEGIA_MASTER.md",
    "concorrenti": "01_ANALISI_MERCATO_CONCORRENTI.md",
    "personas": "02_TARGET_PERSONAS.md",
    "prezzi": "03_PACCHETTI_PREZZI_TIER.md",
    "finanze": "04_PIANO_FINANZIARIO.md",
    "setup": "05_PIATTAFORME_SETUP.md",
    "licenze": "06_LICENZE_SISTEMA.md",
    "youtube": "08_PIANO_CONTENUTI_YOUTUBE.md",
    "social": "09_PIANO_SOCIAL_POST.md",
    "copy": "10_COPY_PATREON_SUBSCRIBESTAR.md",
    "outreach": "11_OUTREACH_COMUNITA.md",
}

# Fasi (key, titolo, emoji, giorni)
PHASES = [
    ("prep",       "Preparazione",        "🎒", "Giorno 0"),
    ("fondamenta", "Fondamenta",          "🧱", "Giorno 1"),
    ("vetrina",    "Vetrina (Patreon)",   "🪟", "Giorno 2"),
    ("adult",      "Incasso (SubscribeStar/itch)", "💳", "Giorno 3"),
    ("produzione", "Produzione",          "🎬", "Giorni 4-7"),
    ("lancio",     "Lancio + Vendite",    "🚀", "Giorni 8-14"),
    ("tech",       "Cose che faccio io",  "🤖", "In parallelo"),
]

# ─────────────────────────────────────────────────────────────────────────────
# PASSI. who: "TU" guida il flusso; "IO" = lo fa Claude (mostrato a parte).
# kind: account|content|sales|community|checkpoint|selfcare|tech
# ─────────────────────────────────────────────────────────────────────────────
TASKS = [
    # —— PREP ——
    dict(id="prep_materials", day=0, phase="prep", who="TU", kind="account",
         title="Metti in una cartella i tuoi materiali",
         detail="Sul desktop crea una cartella 'LANCIO'. Dentro: una tua foto profilo, "
                "il nome pubblico (AI Tools Forge / il tuo nome), una email dedicata al progetto, "
                "e i dati di pagamento (PayPal o IBAN) per ricevere i soldi.",
         doc="setup", section="Sezione 0", deps=[], hermes=True),
    dict(id="prep_obs", day=0, phase="prep", who="TU", kind="content",
         title="Scarica OBS Studio (per registrare i video)",
         detail="OBS è gratis e registra lo schermo + microfono. Scaricalo e installalo. "
                "Lo configuriamo al Giorno 4.",
         url="https://obsproject.com/download", url_label="Scarica OBS",
         doc="youtube", deps=[], hermes=True),

    # —— FONDAMENTA (Giorno 1) ——
    dict(id="yt_create", day=1, phase="fondamenta", who="TU", kind="account",
         title="Crea il canale YouTube",
         detail="Accedi a YouTube Studio col tuo account Google e crea il canale col nome pubblico. "
                "È il motore principale che porta visite.",
         url="https://studio.youtube.com", url_label="Apri YouTube Studio",
         doc="setup", section="Sezione 2", deps=["prep_materials"], hermes=True),
    dict(id="yt_desc", day=1, phase="fondamenta", who="TU", kind="account",
         title="Incolla la descrizione del canale",
         detail="Copia il testo già pronto e incollalo nella descrizione del canale (Impostazioni → Personalizzazione → Informazioni di base).",
         doc="copy", section="Sezione 1", copy_key="yt_desc",
         deps=["yt_create"], hermes=False),
    dict(id="discord_create", day=1, phase="fondamenta", who="TU", kind="account",
         title="Crea il server Discord",
         detail="Apri Discord, clicca '+' a sinistra → 'Crea un mio server'. Chiamalo come il brand. "
                "È la tua community (= supporto e clienti che restano).",
         url="https://discord.com/app", url_label="Apri Discord",
         doc="setup", section="Sezione 3", deps=[], hermes=True),
    dict(id="discord_channels", day=1, phase="fondamenta", who="TU", kind="account",
         title="Crea i canali Discord",
         detail="Crea questi canali di testo: #annunci  #supporto  #showcase  #nsfw-gated  #bug. "
                "Il canale #nsfw-gated va marcato come 'NSFW/18+' nelle impostazioni del canale.",
         doc="setup", section="Sezione 3", deps=["discord_create"], hermes=True),
    dict(id="rest_day1", day=1, phase="fondamenta", who="TU", kind="selfcare",
         title="Stop per oggi. Riposati.",
         detail="Hai posato le due fondamenta (canale + community). È abbastanza per oggi. "
                "Spegni tutto e stacca: domani apriamo le piattaforme di incasso. Sei in anticipo, non in ritardo.",
         deps=["yt_create", "discord_create"], hermes=False),

    # —— VETRINA (Giorno 2) ——
    dict(id="patreon_create", day=2, phase="vetrina", who="TU", kind="account",
         title="Crea/configura la pagina Patreon",
         detail="È la tua vetrina pubblica SFW (niente contenuti espliciti qui: lo vieta Patreon). "
                "Usa il tuo profilo già esistente AI Tools Forge.",
         url="https://www.patreon.com/c/AIToolsForge", url_label="Apri il mio Patreon",
         doc="setup", section="Sezione 4", deps=[], hermes=True),
    dict(id="patreon_about", day=2, phase="vetrina", who="TU", kind="account",
         title="Incolla il testo 'About' della pagina",
         detail="Copia il testo pronto e incollalo nella sezione descrizione/About della pagina Patreon.",
         doc="copy", section="Sezione 2", copy_key="patreon_about",
         deps=["patreon_create"], hermes=False),
    dict(id="patreon_t1", day=2, phase="vetrina", who="TU", kind="sales",
         title="Crea il livello T1 — Demo (Gratis)",
         detail="Crea un tier gratuito ('free membership'). Incolla la descrizione pronta. Serve da imbuto.",
         doc="copy", section="Sezione 3", copy_key="tier_t1",
         deps=["patreon_create"], hermes=False),
    dict(id="patreon_t2", day=2, phase="vetrina", who="TU", kind="sales",
         title="Crea il livello T2 — SFW Full (15€)",
         detail="Crea il tier a 15€/mese con la descrizione pronta. È il tuo tier di volume.",
         doc="copy", section="Sezione 3", copy_key="tier_t2",
         deps=["patreon_create"], hermes=False),
    dict(id="patreon_t4", day=2, phase="vetrina", who="TU", kind="sales",
         title="Crea il livello T4 — Studio (45€)",
         detail="Crea il tier a 45€/mese con la descrizione pronta. È l'upsell 'pro' (modelli propri + API).",
         doc="copy", section="Sezione 3", copy_key="tier_t4",
         deps=["patreon_create"], hermes=False),
    dict(id="patreon_settings", day=2, phase="vetrina", who="TU", kind="sales",
         title="Attiva 'free membership' e Autopilot — NON marcare adult",
         detail="Nelle impostazioni Patreon: abilita la membership gratuita e Autopilot (converte i free in paganti). "
                "Verifica che la pagina NON sia marcata come 'adult' e che non ci sia nessun contenuto NSFW.",
         doc="setup", section="Sezione 4", deps=["patreon_create"], hermes=True),

    # —— INCASSO (Giorno 3) ——
    dict(id="ss_register", day=3, phase="adult", who="TU", kind="account",
         title="Registrati su SubscribeStar.adult (handle anonimo)",
         detail="Qui vive il tier NSFW. Usa un handle SEPARATO e anonimo (non il tuo nome reale). "
                "Completa la verifica 18+.",
         url="https://subscribestar.adult/", url_label="Apri SubscribeStar.adult",
         doc="setup", section="Sezione 5", deps=[], hermes=True),
    dict(id="ss_tier", day=3, phase="adult", who="TU", kind="sales",
         title="Crea il livello T3 — NSFW (25€)",
         detail="Crea il tier a 25€/mese con la descrizione pronta (sobria, niente esplicito nel testo). "
                "Attiva l'age-gate 18+. È il tuo target più redditizio.",
         doc="copy", section="Sezione 4", copy_key="tier_t3",
         deps=["ss_register"], hermes=False),
    dict(id="itch_create", day=3, phase="adult", who="TU", kind="account",
         title="Crea l'account itch.io",
         detail="itch.io serve per le vendite 'una tantum' (cassa immediata) e ammette contenuti per adulti.",
         url="https://itch.io/register", url_label="Registrati su itch.io",
         doc="setup", section="Sezione 6", deps=[], hermes=True),
    dict(id="itch_demo", day=3, phase="adult", who="TU", kind="sales",
         title="Crea il prodotto 'Demo gratuita'",
         detail="Nuovo progetto su itch.io: la demo gratis (esca del funnel). Incolla la descrizione pronta.",
         url="https://itch.io/game/new", url_label="Nuovo progetto itch.io",
         doc="copy", section="Sezione 5", copy_key="itch_demo",
         deps=["itch_create"], hermes=False),
    dict(id="itch_founder", day=3, phase="adult", who="TU", kind="sales",
         title="Crea il prodotto 'Founder Lifetime' (199-299€, 30 posti)",
         detail="Nuovo progetto: l'offerta Founder a vita, prezzo fisso, posti limitati a 30. "
                "È la PRINCIPALE fonte di cassa del primo mese. Incolla la descrizione pronta.",
         url="https://itch.io/game/new", url_label="Nuovo progetto itch.io",
         doc="copy", section="Sezione 5", copy_key="itch_founder",
         deps=["itch_create"], hermes=False),

    # —— PRODUZIONE (Giorni 4-7) ——
    dict(id="obs_scene", day=4, phase="produzione", who="TU", kind="content",
         title="Configura OBS (cattura schermo + microfono)",
         detail="In OBS aggiungi una sorgente 'Cattura schermo' e una 'Cattura audio in ingresso' (microfono). "
                "Fai una prova di 10 secondi e riascolta: l'audio deve essere chiaro.",
         doc="youtube", deps=["prep_obs"], hermes=True),
    dict(id="record_v1", day=4, phase="produzione", who="TU", kind="content",
         title="Registra il Video 1 (demo SFW)",
         detail="Apri lo script del Video 1 (è scritto parola per parola). Leggilo mentre registri lo schermo "
                "che mostra una generazione reale. Non puntare alla perfezione: punta a finirlo.",
         doc="youtube", section="Script Video 1",
         deps=["yt_create", "obs_scene"], hermes=True),
    dict(id="warmup_reddit", day=5, phase="produzione", who="TU", kind="community",
         title="Crea l'account Reddit e 'scaldalo'",
         detail="Crea un account Reddit. Commenta in modo UTILE (non promozionale) in 2-3 thread di r/StableDiffusion "
                "o r/comfyui. Serve a non sembrare spam quando pubblicherai.",
         url="https://www.reddit.com/r/StableDiffusion/", url_label="Apri r/StableDiffusion",
         doc="outreach", deps=[], hermes=True),
    dict(id="warmup_civitai", day=5, phase="produzione", who="TU", kind="community",
         title="Crea l'account Civitai",
         detail="Civitai è pieno del tuo pubblico (anche NSFW). Crea l'account e guarda come postano gli altri.",
         url="https://civitai.com/", url_label="Apri Civitai",
         doc="outreach", deps=[], hermes=True),
    dict(id="edit_v1", day=6, phase="produzione", who="TU", kind="content",
         title="Monta ed esporta il Video 1",
         detail="Taglia le pause lunghe, sistema l'audio, esporta. Va benissimo un montaggio semplice.",
         doc="youtube", deps=["record_v1"], hermes=True),
    dict(id="meta_v1", day=6, phase="produzione", who="TU", kind="content",
         title="Prepara titolo, descrizione e tag del Video 1",
         detail="Sono già pronti nel documento YouTube. Copia titolo, descrizione e i 5 tag.",
         doc="youtube", section="Script Video 1", deps=["edit_v1"], hermes=False),
    dict(id="checkpoint_w1", day=7, phase="produzione", who="TU", kind="checkpoint",
         title="Checkpoint settimana 1",
         detail="Guarda quanto hai fatto (l'app te lo mostra). Rifinisci la pagina Founder. "
                "Sei a metà strada: il grosso del setup è fatto.",
         doc="finanze", deps=[], hermes=True),
    dict(id="founder_polish", day=7, phase="produzione", who="TU", kind="sales",
         title="Rifinisci la pagina Founder (scarsità: 30 posti)",
         detail="Scrivi chiaro 'Solo 30 posti'. La scarsità deve essere VERA. Aggiungi un contatore se puoi.",
         doc="prezzi", deps=["itch_founder"], hermes=True),

    # —— LANCIO (Giorni 8-14) ——
    dict(id="publish_v1", day=8, phase="lancio", who="TU", kind="content",
         title="PUBBLICA il Video 1 su YouTube",
         detail="Carica il video con titolo/descrizione/tag pronti. Questo è il giorno in cui inizi a essere visibile. Respira e premi pubblica.",
         url="https://studio.youtube.com", url_label="Apri YouTube Studio",
         doc="youtube", deps=["edit_v1", "meta_v1", "yt_create"], hermes=True),
    dict(id="founder_waitlist", day=8, phase="lancio", who="TU", kind="sales",
         title="Apri la waitlist / pre-order Founder",
         detail="Metti online la pagina Founder su itch.io. Puoi già incassare i pre-order anche se il file finale "
                "lo consegni tra qualche giorno (te lo preparo io).",
         doc="prezzi", deps=["founder_polish"], hermes=True),
    dict(id="post_announce", day=8, phase="lancio", who="TU", kind="community",
         title="Pubblica il post di annuncio",
         detail="Copia il post 'annuncio lancio' già pronto e pubblicalo (X + Discord #annunci). Metti il link al video.",
         doc="social", section="Post: annuncio lancio", copy_key="post_announce",
         deps=["publish_v1"], hermes=True),
    dict(id="respond_comments", day=9, phase="lancio", who="TU", kind="community",
         title="Rispondi a TUTTI i commenti e messaggi",
         detail="Qui si convertono i clienti. Rispondi con gentilezza e velocità. Anche una sola risposta conta.",
         deps=["publish_v1"], hermes=True),
    dict(id="founder_open", day=10, phase="lancio", who="TU", kind="sales",
         title="Apri UFFICIALMENTE le vendite Founder",
         detail="Togli la waitlist, metti 'acquista ora'. Aggiungi il link Founder nel video (scheda/descrizione) e nei post. "
                "Primo incasso atteso. È il momento per cui hai lavorato.",
         doc="prezzi", deps=["founder_waitlist"], hermes=True),
    dict(id="record_v2", day=11, phase="lancio", who="TU", kind="content",
         title="Registra il Video 2",
         detail="Script già pronto (alternativa al cloud / foto→3D). Stesso metodo del Video 1, ora sai come si fa.",
         doc="youtube", section="Script Video 2", deps=["record_v1"], hermes=True),
    dict(id="publish_v2", day=11, phase="lancio", who="TU", kind="content",
         title="Pubblica il Video 2",
         detail="Monta veloce e pubblica. Due video in due settimane = canale che vive.",
         url="https://studio.youtube.com", url_label="Apri YouTube Studio",
         doc="youtube", deps=["record_v2"], hermes=False),
    dict(id="posts_round2", day=12, phase="lancio", who="TU", kind="community",
         title="Secondo giro di post nelle community",
         detail="Usa gli altri post pronti (showcase, mini-tutorial, confronto). Uno per canale, niente spam.",
         doc="social", deps=[], hermes=True),
    dict(id="earlybird", day=12, phase="lancio", who="TU", kind="sales",
         title="Attiva l'early-bird (−30% primi 3 mesi)",
         detail="Crea un'offerta a tempo per i primi 50 abbonati. Blocca presto le entrate ricorrenti.",
         doc="prezzi", deps=[], hermes=True),
    dict(id="outreach", day=13, phase="lancio", who="TU", kind="community",
         title="Outreach mirato (5-10 risposte di valore)",
         detail="Trova chi è frustrato dai limiti dei tool attuali e rispondi con valore, menzionando il prodotto "
                "solo dove pertinente. Il playbook ti dà le ricerche da fare.",
         doc="outreach", deps=[], hermes=True),
    dict(id="checkpoint_w2", day=14, phase="lancio", who="TU", kind="checkpoint",
         title="Checkpoint settimana 2",
         detail="Conta: Founder venduti, abbonati, iscritti Discord. Apri il piano finanziario e confronta con gli scenari. "
                "Decidiamo insieme i passi del mese 2.",
         doc="finanze", deps=[], hermes=True),

    # —— TECH (le faccio io) ——
    dict(id="tech_licensing", day=99, phase="tech", who="IO", kind="tech",
         title="Sistema di licenze (codici per tier + upgrade/downgrade)",
         detail="Lo scrivo e testo io. Genera i codici-licenza firmati che proteggono il prodotto. "
                "Pronto entro il giorno ~7-10. Tu non devi fare nulla.",
         doc="licenze", deps=[], hermes=False),
    dict(id="tech_build", day=99, phase="tech", who="IO", kind="tech",
         title="Compilato di consegna (l'app pronta da dare ai clienti)",
         detail="Preparo io l'eseguibile compilato (Nuitka) da consegnare. Tu non devi compilare nulla.",
         doc="licenze", deps=[], hermes=False),
    dict(id="tech_issue", day=99, phase="tech", who="IO", kind="tech",
         title="Primo codice-licenza emesso a un cliente",
         detail="Quando arriva la prima vendita, genero io il codice e ti do l'email pronta da inviare.",
         doc="copy", section="Sezione 7", deps=["tech_licensing", "tech_build"], hermes=False),
]

# ─────────────────────────────────────────────────────────────────────────────
# PREMI / TRAGUARDI (gamification). check: spec valutata dall'app sul set 'done'.
#   {"type":"task","id":...}            -> quel passo fatto
#   {"type":"all","ids":[...]}          -> tutti questi passi fatti
#   {"type":"count","n":N}              -> almeno N passi TU fatti
#   {"type":"streak","n":N}             -> streak di N giorni
#   {"type":"sales","n":N}              -> almeno N vendite registrate
# ─────────────────────────────────────────────────────────────────────────────
BADGES = [
    dict(id="b_start",     emoji="🚀", title="Si parte!",
         desc="Hai fatto il primo passo. Il più difficile.", check={"type": "count", "n": 1}),
    dict(id="b_fondamenta", emoji="🧱", title="Fondamenta gettate",
         desc="Canale + community creati.", check={"type": "all", "ids": ["yt_create", "discord_create"]}),
    dict(id="b_vetrina",   emoji="🪟", title="Vetrina pronta",
         desc="Patreon e i tier SFW sono online.",
         check={"type": "all", "ids": ["patreon_create", "patreon_t1", "patreon_t2", "patreon_t4"]}),
    dict(id="b_incasso",   emoji="💳", title="Pronto a incassare",
         desc="SubscribeStar e itch configurati.",
         check={"type": "all", "ids": ["ss_tier", "itch_founder"]}),
    dict(id="b_ciak",      emoji="🎬", title="Primo ciak",
         desc="Hai registrato il primo video. Sei un creator.", check={"type": "task", "id": "record_v1"}),
    dict(id="b_onair",     emoji="📡", title="Sei in onda",
         desc="Il primo video è pubblico.", check={"type": "task", "id": "publish_v1"}),
    dict(id="b_riflettori", emoji="✨", title="Sotto i riflettori",
         desc="Primo annuncio nelle community.", check={"type": "task", "id": "post_announce"}),
    dict(id="b_caccia",    emoji="🏹", title="Caccia aperta",
         desc="Le vendite Founder sono ufficialmente aperte.", check={"type": "task", "id": "founder_open"}),
    dict(id="b_streak3",   emoji="🔥", title="Costanza",
         desc="3 giorni di fila a lavorarci. La costanza batte il talento.", check={"type": "streak", "n": 3}),
    dict(id="b_firstsale", emoji="💶", title="PRIMO INCASSO",
         desc="La prima vendita. La prova che funziona. Festeggia davvero.", check={"type": "sales", "n": 1}),
    dict(id="b_five",      emoji="🌟", title="Si fa sul serio",
         desc="5 vendite. Non è fortuna, è un prodotto.", check={"type": "sales", "n": 5}),
    dict(id="b_finish",    emoji="🏁", title="Maratoneta",
         desc="Hai completato tutti i passi del lancio. Ce l'hai fatta.",
         check={"type": "all", "ids": [t["id"] for t in TASKS if t["who"] == "TU"]}),
]

# Messaggi motivazionali brevi (ruotati nell'header)
MOTIVATION = [
    "Un passo alla volta. Solo il prossimo. Nient'altro.",
    "Non devi farlo bene. Devi solo farlo.",
    "Sei in anticipo, non in ritardo. Stai costruendo.",
    "Ogni spunta è un mattone della tua libertà.",
    "Il momento più difficile è iniziare. Poi va in discesa.",
    "Non sei solo: io sono qui e seguiamo il piano insieme.",
    "Fatto è meglio che perfetto.",
    "Domani ti ringrazierai per quello che fai oggi.",
    "Piccoli passi, ogni giorno. Così si vince.",
    "Hai già costruito il prodotto. Questa è la parte facile.",
]

# Messaggi quando l'utente dice "mi sento bloccato"
STRUGGLE = [
    "Va bene fermarsi. Respira. Non sei in ritardo su niente.",
    "Facciamo il passo più piccolo possibile: anche solo aprire un link conta.",
    "Se sei stanco, riposa davvero 20 minuti. Tornare lucido vale più di forzare.",
    "Guarda quanto hai già fatto, non quanto manca. Sei più avanti di ieri.",
    "Non devi finire oggi. Devi solo non fermarti del tutto. Un micro-passo basta.",
    "Sei una persona in difficoltà che sta agendo lo stesso. È coraggio, non debolezza.",
]

# Nudge contro l'inattività (mostrati dopo qualche minuto fermo)
NUDGES = [
    "Sono ancora qui. Quando vuoi, c'è un piccolo passo che ti aspetta.",
    "Anche solo un micro-passo e poi stacchi: va benissimo.",
    "Nessuna fretta. Ma ricorda: il prossimo passo è piccolo e l'ho già preparato per te.",
    "Se ti senti bloccato, premi 'Mi sento bloccato': ti aiuto a sbloccarti.",
]


def task_by_id(tid):
    for t in TASKS:
        if t["id"] == tid:
            return t
    return None


def phase_title(key):
    for k, title, emoji, days in PHASES:
        if k == key:
            return f"{emoji} {title}"
    return key
