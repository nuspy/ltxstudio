# 03 — Pacchetti, prezzi e matrice funzioni

Questo file definisce **cosa vendi, a che prezzo, e cosa include ciascun livello**. I prezzi sono di partenza (ancorati al concorrente-modello SECourses ~12€ e ai SaaS NSFW 8-20$); si possono alzare dopo il lancio, mai abbassare.

## I 4 tier in una riga

| Tier | Nome | Prezzo | Dove si vende | A chi |
|---|---|---|---|---|
| **T1** | Idea / Demo | **Gratis** (o 5€ "Supporter") | Patreon (free) + itch.io | Curiosi, funnel di acquisizione |
| **T2** | SFW Full | **15€/mese** | Patreon + itch.io | Creator video/immagini, indie game dev, marketer |
| **T3** | NSFW | **25€/mese** | **SubscribeStar.adult** (mai Patreon) | Creator adult, community NSFW — **il target più redditizio** |
| **T4** | Studio | **45€/mese** | Patreon + SubscribeStar | Power user: LoRA propri, pipeline, API |

**Leve di cassa immediata (lancio):**
- 🔥 **Founder Lifetime** — **199-299€ una-tantum**, posti **limitati a 30**, dà il livello **T3 (NSFW) a vita** + tutti gli update futuri. Scarsità reale → urgenza. È la principale fonte di cassa del primo mese.
- ⏱️ **Early-bird** — **−30% sui primi 3 mesi** per i primi **50 abbonati** (qualsiasi tier ricorrente). Blocca presto l'MRR.

## Matrice funzioni (cosa sblocca ogni tier)

Basata sulle funzioni REALI e testate del prodotto. ✅ incluso · ❌ non incluso · ⚠️ limitato.

| Funzione | T1 Demo | T2 SFW | T3 NSFW | T4 Studio |
|---|:--:|:--:|:--:|:--:|
| Genera immagini (testo→immagine) | ⚠️ base, max 768px, watermark | ✅ | ✅ | ✅ |
| Numero generazioni | ⚠️ ~20 totali | ∞ | ∞ | ∞ |
| Inpaint / Outpaint / Recolor | ❌ | ✅ | ✅ | ✅ |
| Segmentazione oggetti (SAM) | ❌ | ✅ | ✅ | ✅ |
| Video testo→video e immagine→video | ❌ | ✅ | ✅ | ✅ |
| Prolunga video illimitata + ReTake | ❌ | ✅ | ✅ | ✅ |
| Audio nativo + TTS + voice clone | ❌ | ✅ | ✅ | ✅ |
| Restauro: super-risoluzione, +FPS, volti | ❌ | ✅ | ✅ | ✅ |
| ControlNet (canny/depth/pose) | ❌ | ✅ | ✅ | ✅ |
| 3D: immagine→mesh + rigging scheletro | ❌ | ✅ | ✅ | ✅ |
| **Modalità NSFW (encoder + inpaint senza censura)** | ❌ | ❌ | ✅ | ✅ |
| **Import LoRA/modelli custom + classificazione auto** | ❌ | ❌ | ❌ | ✅ |
| **API REST (automazione/pipeline)** | ❌ | ❌ | ❌ | ✅ |
| Update e nuove funzioni | ⚠️ limitati | ✅ | ✅ | ✅ |
| Supporto Discord | community | ✅ prioritario | ✅ prioritario | ✅ corsia dedicata |

> Nota tecnica: il confine vero tra SFW e NSFW (T2→T3) è un singolo interruttore nel motore (`core/template_engine.py`): a T2 la modalità NSFW è forzata a OFF anche se la UI provasse a chiederla. Questo rende il gating sicuro e a basso rischio di bug.

## Logica di prezzo (perché questi numeri)

- **T1 gratis = funnel, non reddito.** Serve a far provare il prodotto e a riempire la lista Patreon (poi Autopilot converte verso i pagamenti). Alternativa: 5€ "Supporter" per chi vuole "sostenere l'idea".
- **T2 a 15€** = poco sopra SECourses (~12€), giustificato dall'app integrata propria (non solo script). Bacino più largo → volume.
- **T3 a 25€** = sotto i SaaS NSFW cloud che costano 8-20$ **ma a consumo/crediti**; qui è illimitato, locale e privato → percepito come affare. **È il tier con la più alta disponibilità a pagare e il churn più basso.**
- **T4 a 45€** = prezzo "pro": chi vuole import modelli propri + API sta costruendo un business e paga volentieri.
- **Founder 199-299€** = ~10-12 mesi di T3 pagati subito, in cambio di "a vita". Conviene a te (cassa ora) e al cliente entusiasta (risparmio + status di fondatore).

## Strategia di retention (perché restano iscritti)

Lanci con **foto + video** già completi e prometti, come **update inclusi**, il resto che è già nel codice e va solo rifinito/sbloccato: 3D+rigging, voice clone avanzata, animazione, lip-sync. Ogni update mensile è un motivo per **non disdire**. Comunichi una **roadmap pubblica** (1 funzione nuova al mese) → l'iscritto che disdice perde gli update futuri.

## Regole d'oro sui prezzi

1. **Non scendere mai** sotto questi prezzi dopo il lancio (svaluta il prodotto e fa arrabbiare chi ha pagato).
2. **Alza** i prezzi per i nuovi iscritti quando aggiungi funzioni grosse; chi è già dentro mantiene il prezzo vecchio (lock-in psicologico → riduce churn).
3. **Founder e early-bird sono a tempo/posti limitati e devono finire davvero.** La scarsità falsa si nota e brucia la fiducia.
4. **Un solo upsell alla volta:** dal T2 spingi al T3 (NSFW) o al T4 (Studio) in base alla persona, non entrambi insieme.

→ I testi pronti per descrivere questi tier sulle pagine sono in **`10_COPY_PATREON_SUBSCRIBESTAR.md`**.
