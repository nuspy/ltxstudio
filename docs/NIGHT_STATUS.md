# Stato notturno — branch `feature/unified-model-manager` (per il risveglio)

Onestà sul vincolo: **non potevo validare la GUI a occhio mentre dormivi.** Per non lasciarti l'app rotta ho
fatto solo lavoro **verificabile al 100% via codice** e ho **diagnosticato** con precisione tutto il resto, che
richiede il tuo test. Qui sotto: cosa è FATTO+committato, lo stato stabile, e i punti pendenti con causa+fix.

## ✅ FATTO e committato (sicuro)
- **Colore bottoni Genera #3ECF8E uniforme** (`theme.style_generate`) su video_edit/world/music/lora/film. Il
  Genera principale (`gen_btn`, objectName `generate`) era già #3ECF8E → ora tutti coerenti.
- **PuLID su RTX 5090**: xformers `memory_efficient_attention` → torch **SDPA** (sm120 non ha il kernel). +
  clear_cache se PuLID fallisce (era leak VRAM → encode lentissimo).
- **FLUX schnell/distillati**: caricano nativi (config `guidance_embeds=False`; prima crash meta-tensor).
- **Registro footprint modelli** (`core/model_registry.py`, auto-popolato dtype-aware) + **eviction SIZE-AWARE**
  (ComponentStore: sfratta il 2GB non il 10GB) — la smart-memory che hai chiesto.
- **Timeline log dal click Genera** (ogni step: encode/denoise/VAE/save + VRAM ded/alloc/reserved + RAM).
- **comfy figlio persistente**: scritto ma **default OFF** (la 1ª versione con readline bloccante deadlockava il
  thread GUI). Comfy torna al **one-shot che funziona**.

## ✅ FATTO — 2ª tornata (fix concreti verificati)
- **integra immagini: `NameError 'cfg'`** in `_start_native_kontext` (integra mask-based) → legge
  `get_inpaint_config()`. RISOLTO+committato.
- **comfy residenza RIABILITATA** con **reader-thread non-bloccante** (validato con mock: no deadlock, riuso a
  caldo tra i job, free). Fallback one-shot se il figlio muore. `COMFY_PERSISTENT=0` per spegnere.
- **PuLID video: checkbox di attivazione** (opt-in esplicito; da spento campi+bottone disabilitati).
- **Aggiunta foto al canvas NON blocca più**: lo scontorno u2net girava SINCRONO sul thread GUI (freeze lungo al
  1° load ONNX) → ora **opt-in** (`_auto_cutout_on_add`, default OFF); aggiunta istantanea, scontorno manuale.

## Stato STABILE ora
- **Motore comfy (il tuo)**: one-shot, veloce, VRAM dinamica ComfyUI. **Funziona.**
- **Motore diffusers** (fallback PuLID/inpaint/kontext): figlio persistente ON ma **respawna** (vedi sotto) →
  ricarica. Non rompe (fallback), ma non ha ancora residenza vera.

## ⏳ PENDENTI — causa + fix (da fare col tuo test)

### A. Residenza robusta (memoria) — il cuore
- **comfy-persistente**: la mia IPC usava `readline()` bloccante sul thread GUI → deadlock se il figlio moriva in
  init. **Fix**: reader-thread non-bloccante + timeout (COMFY_PERSISTENT riacceso). *Testabile con mock.*
- **diffusers-respawn**: `[persistent_runner] pronto` ×8 = respawn ad ogni gen → cache vuota → reload. Causa: `/free`
  **KILLA** il figlio (server.py:217/341), e il motore **comfy chiama `/free` su :8189 prima di OGNI t2i**
  (img_router.py:145) → alternando comfy/diffusers ogni comfy ammazza il figlio diffusers. **Fix**: `/free` =
  **DEMOTE (svuota VRAM, tieni il processo)** per lo stesso motore; kill solo al vero cambio-modello. Usa il
  registro footprint per liberare il minimo.

### B. Estendere la memoria ai VIDEO (tutte le pipeline/modelli)
- Oggi solo LTX-2 :8190 è residente (e solo in Cinema Studio). self_forcing/krea_rt/LingBot = subprocess per-clip.
- **Fix**: i worker/server video registrano il footprint (model_registry) + residenza + eviction size-aware, e
  cablare :8192 (krea_rt/self_forcing) come :8190. Additivo, ma va testato per-arch.

### C. Bug tab Video
1. **#4 audio non generato** — REGRESSIONE. Serve sapere **con quale arch/funzione** generavi audio (LTX-2 audio
   nativo? canta/lip-sync a2v? MultiTalk?). LingBot NON ha audio. Da lì trovo il param/toggle perso.
2. **#5 PuLID video volto da file** — probabile già risolto dal fix xformers di stanotte; possibile cache volto
   (PuLID cache-a per id transformer). **Ritesta**; se il volto nuovo non entra, aggiungo l'invalidazione cache.
3. **"Modifica video AI" — einops `b c (d p1)(h p2)(w p3)` + maschera vuota**: le dimensioni del video sorgente
   NON sono divisibili per i fattori patch del VAE LTX → l'encode fallisce. **Fix**: resize/pad del video (frame,
   H, W) ai multipli richiesti prima dell'encode, in `workers/ltx2_infer.py` (path v2v/edit). Serve un video di
   test per validare le dimensioni.
4. **#2 UX v2v**: campo "video di partenza" compare solo in Talk/ReTake (non in Normal); il Genera è quello
   condiviso; PuLID è il bottone "🧑 Primo frame dal volto", non una checkbox. Da chiarire/riprogettare la UX.

### D. Bottoni (oltre al colore, già fatto)
- **Spostare Genera/Interrompi video** dal workspace sinistro (canvas, main_window:756-765) alla sezione
  generazione video. È un cambio di layout → serve il tuo occhio per non rompere.
- **Bottoni Genera "mancanti"**: il Genera è UNO condiviso (immagine+video). I singoli tab non hanno un proprio
  Genera → è un design da decidere (uno per sezione vs. condiviso).

## Ordine consigliato al risveglio (ognuno testato dalla GUI prima del successivo)
1. **#4 audio** (dimmi l'arch) — è una regressione che ti blocca.
2. **A residenza robusta** (demote-non-kill + reader-thread) — sblocca il <10s a caldo su entrambi i motori.
3. **C3 einops video-edit** (con un video di test).
4. **B memoria video**.
5. **#5 pulid**, **#2/D UX+bottoni**.
