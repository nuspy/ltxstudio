# Gestore Unificato dei Modelli — piano (branch `feature/unified-model-manager`)

## Obiettivo (richiesta utente)
1. **Caricare direct-to-VRAM** (niente staging RAM inutile) **E** tenere i **modelli residenti** tra le generazioni.
2. **Non scaricare a ogni gen**: un modello resta in memoria finché (a) cambia pipeline e non serve più, oppure (b) serve spazio.
3. Quando serve liberare spazio con **più modelli caricati di misure diverse**, scegliere **QUALE** sfrattare in modo intelligente (size/cost-aware), non a caso.
4. Vale per **tutte** le pipeline immagine e video, **tutti** i modelli, **tutte** le funzioni.

## 1. Mappa attuale — dove si carica e se resta residente

### Immagine — server nativo `:8189` (`inference_server`) + Krea2 subprocess
- Default: ogni job gira in un **figlio `job_runner` che muore** → **reload ad ogni gen**. `ModelManager`+`ComponentStore` (smart) vivono nel figlio e **spariscono con lui**.
- Residenza SOLO con `NATIVE_PERSISTENT_CHILD=1` (**OFF**, fragile) o `placement_mode=sticky` (solo dentro un batch).
- **Krea 2**: subprocess one-shot (`img_gen_job`) → reload; server `:8191` esiste ma **OFF**. L'**edit** deve restare subprocess fresco (coerenza identità).
- Backend coperti (`image_flux.run`): t2i FLUX.1/FLUX.2/Z-Image/Ernie, inpaint/outpaint/Fill, Kontext/"integra"/edit-instruct, FLUX.2 Klein, PuLID, compose/multiref, upscale (`image_upscale`), segment (`image_segment`).

### Video — dispatch centrale `core/ltx_video_job._ARCH_RUNNER`
- Default: **subprocess one-shot per-clip** → reload ad ogni clip.
- Unico residente di default: **server LTX-2 `:8190`**, ma usato **solo** da Cinema Studio/Lucky Director.
- `krea_rt`/`self_forcing`/LingBot/REST/extend/lip-sync: **sempre reload**. Server `:8192` esiste ma **non cablato**.

### Sistema memoria — due piani poco connessi
- **Intra-processo** (nel figlio torch): `ComponentStore` (refcount, demote CPU, footprint **dtype-aware**) + `ModelManager` (`ensure`/`make_room`/`unload`). Smart ma **muore col figlio** nel default.
- **Cross-processo** (GUI, senza torch): `gpu_registry` (model-aware, `before_job`) + `mem_coordinator` (coarse per-categoria) → parlano ai server via HTTP `/free` (hard-free) + `/system_stats`; VRAM via `gpu_mem` (NVML).

## 2. Com'era lo "smart" e cosa si è disconnesso
- Commit chiave: `40f5029`/`39d5b99` (footprint + **planner COMBINATORIO** `core/vram_calibration.plan_placement`), `515b111` (misura reale fp8), `67a47fb`+`419e3aa` (`gpu_registry`), `dac60de` (persistent child).
- **Problemi**:
  1. Il **planner combinatorio** (l'unico pezzo davvero "smart" per scegliere cosa tenere/scaricare minimizzando l'offload) **NON è cablato a runtime** — è sepolto nel dialog "Avanzate → Gestione VRAM".
  2. L'eviction runtime (`make_room`, `evict_unused`, `gpu_registry.plan`) è **LRU/categoria cieca → NON size-aware**.
  3. Il `ComponentStore` (il "cervello") **muore col figlio per-job** → nessuna residenza tra gen nel default.
  4. `/free` cross-server **distrugge tutto** il server (poi paga reload pieno), invece di sfrattare il singolo modello meno utile.
  5. Contabilità VRAM doppia (torch intra vs NVML cross), non sincronizzata: `gpu_registry` non conosce il contenuto del `ComponentStore`.

## 3. Design del gestore unificato

**Principio**: un unico **Registro di Residenza** (estensione di `gpu_registry`) che conosce OGNI modello residente (immagine+video) col suo **footprint dtype-aware**, e decide load/reuse/evict in modo **size-aware**; ogni motore (server :8189/:8190/:8191/:8192) diventa **residente e robusto** ed espone un `/free` **selettivo**.

### 3a. Load direct-to-VRAM + robusto (foundation)
- Caricare i pesi evitando il path fragile `meta`+`dispatch_model` (causa dei crash meta-tensor di oggi e del pianto del persistent child). Dove possibile `make_room` PRIMA → assegnazione diretta su `cuda` (streaming), niente buffer RAM.
- Elimina insieme: latenza RAM-staging, picco RAM, e la fragilità del 2° load in-process.

### 3b. Residenza robusta per-motore
- **Immagine `:8189`**: `NATIVE_PERSISTENT_CHILD` **ON di default**, reso robusto dal load 3a (respawn solo al cambio-modello che il registro decide).
- **Krea2 `:8191`**: server persistente ON (solo t2i; edit resta subprocess fresco).
- **Video**: server `:8190` (LTX-2) esteso a **tutti** i percorsi video LTX (non solo Cinema Studio); `:8192` cablato per krea_rt/self_forcing con autostart.
- Tutti espongono `/system_stats` con `model_key` + **footprint per-modello dtype-aware**.

### 3c. Registro di Residenza size-aware (il cuore)
- Stato di verità unico della VRAM: elenco `{server, model_key, footprint_mb, last_used, category}` letto dai `/system_stats`.
- `plan(want, need_mb)`:
  - se `want` già residente → **reuse** (0 reload);
  - se serve spazio → **eviction size/cost-aware**: scegli il sottoinsieme di residenti da sfrattare che libera `≥ need_mb` **minimizzando il costo di reload** (best-fit su footprint, tie-break LRU), **mai** "scarica tutto". Riusa la logica di `plan_placement` (planner combinatorio già scritto).
  - **policy**: un residente si tocca SOLO se (i) è di categoria/pipeline non più usata dal nuovo job, o (ii) serve il suo spazio. Stessa pipeline/modello ancora utile → **resta**.
- `/free` **selettivo**: `POST /free {model_key|need_mb}` → il server sfratta *quei* modelli (o quanto basta), non azzera la cache.

### 3d. Coordinamento immagine↔video
- Al cambio categoria, il registro non fa hard-free cieco: calcola quanto serve al nuovo motore e sfratta **selettivamente** dai residenti di altra categoria il minimo necessario (size-aware), tenendo il resto caldo se la VRAM basta.

## 4. Piano implementativo (un commit per fase; test SOLO dall'app completa via `tests/app_env`)
- **F1 — Load direct-to-VRAM robusto** (image_flux + estendibile): make_room-first + assign su cuda, no meta/dispatch. Verifica: 0 meta, VRAM diretta, RAM piatta.
- **F2 — Persistent child immagine ON + robusto**: residenza tra gen stesso-modello; respawn deciso dal registro. Verifica: gen1 load, gen2 reuse caldo.
- **F3 — Residenza Krea2 `:8191` + video `:8190`/`:8192` cablati** ovunque (non solo Cinema Studio). Verifica: 2ª clip/gen riusa.
- **F4 — Registro size-aware + `/free` selettivo**: footprint dtype-aware per-modello nei `/system_stats`; eviction best-fit; policy keep-until-change/space. Verifica: 2 modelli di misure diverse → sfratta quello giusto.
- **F5 — Cabla il planner combinatorio** (`plan_placement`) nella decisione di placement a runtime.
- **F6 — Verifica end-to-end dall'app** (harness): matrice foto↔video, multi-modello, no reload spurio; giudizio memoria in Task Manager + `/system_stats`.

## 4bis. Requisito SMART-MEMORY (direttiva utente, esplicita) + apprendimenti dal debug live
**Registro VRAM + eviction combinatoria size-aware** (il cuore del gestore):
- Tenere un **registro della VRAM occupata** per-componente (footprint reale dtype-aware, categoria, last_used, "serve ancora nel job?").
- Quando serve spazio: **calcolo combinatorio** di *cosa* scaricare e *quando*, per **minimizzare tempo e carico/scarico**. Principio: **più economico sfrattare un modello da 2GB che uno da 10GB** se il buco richiesto è piccolo (best-fit su footprint, non LRU cieco). Riusa `core/vram_calibration.plan_placement`.
- Scaricare un componente **appena non serve più** nel job (es. **T5/CLIP dopo l'encode del prompt** → STICKY) o al **cambio pipeline** se il nuovo non lo usa. Mai a ogni gen.

**Apprendimenti dai log reali (t2i FLUX schnell + PuLID, 32GB):**
1. **fp8 layerwise casting NON funziona su CPU** (float8 è dtype GPU): in CPU-load il transformer resta **bf16 (24GB)** → va caricato **direct-to-cuda** perché il cast fp8 riesca (12GB). [FATTO]
2. **Windows: l'allocatore CUDA non rilascia il transiente** bf16 del direct-load (`expandable_segments` non supportato) → `reserved` (17GB) > `allocated` (12GB): il transformer "pesa" 17GB in VRAM. → il **vero fix è un loader fp8 STREAMING** (assegna i tensori fp8 direttamente, **senza** il transiente bf16, stile `flux2_transformer.load_flux2_transformer`). [DA FARE — F1b]
3. **placement RESIDENT-AWARE**: `_place` deve sottrarre ciò che è **già su cuda** dal fabbisogno, o va in offload a torto. [FATTO]
4. **T5-XXL bf16 = 9GB**; in fp8 ~4.5GB. **Tutti i componenti fp8 di default** (direttiva utente) — audit: FLUX.1 T5/CLIP e i VAE oggi bf16, da valutare fp8 (eccezione VAE fp32 anti-macchie [[flux-mottling-sub-1mp]]). [DA FARE]
5. **STICKY quando PuLID/tight**: T5/CLIP offload dopo l'encode → libera ~10GB per denoise+PuLID. [FATTO tattico]
6. **PuLID su RTX 5090**: xformers `memory_efficient_attention` non ha kernel sm120 → SDPA. [FATTO]

## 5. Rischi / vincoli
- Il persistent child era fragile (2° load in-process → segfault Windows): la F1 (load robusto) è **prerequisito** della F2.
- L'edit Krea2 NON può essere residente (drift identità) — resta subprocess fresco.
- Test SOLO con app completa su (mem_coordinator/ComponentStore reali): i probe isolati sono invalidi/distruttivi.
- fp8 ovunque (direttiva utente) mantenuto; footprint calcolato dtype-aware.
