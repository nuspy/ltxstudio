# Review — Gestione memoria/VRAM · Colori dropdown · Macchie inpaint/outpaint
_Data: 2026-07-10 · branch `feature/cinema-studio-review` · analisi su codice reale (+ sorgente ComfyUI in
`E:\AI\Comfy_Nexus.bak`, solo analisi). Test di GENERAZIONE reale rimandati (ottimizzazione ForexGPT occupa
CPU/RAM, fragile). B e C: causa trovata e **fix già applicato** (validazione a video/live rimandata)._

---

## A) Sistema di gestione memoria — review + roadmap

### Architettura attuale (multi-processo, un server per famiglia; nessun gestore VRAM globale unico)
- **App PyQt** — non tocca mai `torch.cuda.*` (creerebbe un contesto CUDA → crash subprocess 3D Blackwell);
  VRAM via NVML/nvidia-smi (`core/gpu_mem.py:16`).
- **Server immagini :8189** (`inference_server/server.py`) — coda seriale, `ModelManager` condiviso,
  `/free`→`MM.clear()` (`:147-150`).
- **Server video LTX-2.3 :8190** (`workers/ltx2_server.py`, venv separato) — cache propria `_PIPE_CACHE`,
  **non** usa ModelManager/ComponentStore, **nessun `/free`** (`:55-73`).
- **Subprocess one-shot 3D/World/Rig** — GPU pulita ogni run, chiamano `/free` di :8189.
- **ModelManager** (`inference_server/model_manager.py`): LRU di "shell" pipeline (`ensure` key `{arch}:{model}`
  `:65`), `make_room(keep=…)` `:96`, `unload` con **demote→CPU-RAM** `:122-163`, `evict_others` al cambio
  modello `:165-177`, `_cpu_cache`+trim LRU `:179-206`, `place` gpu/offload a soglia `:240-256`,
  `apply_loras` hot-swap `:260-307`, `clear` totale `:208-231`, `flush`=empty_cache incondizionato `:233`.
- **ComponentStore** (`inference_server/component_store.py`): cache **refcount** di T5-XXL/CLIP-L/VAE condivisi
  FLUX.1 (`acquire/retain/release` `:62-87`), `evict_unused` con demote→CPU/destroy `:132-169`, budget RAM
  `_ram_ok`/`_shrink_cpu_lru` `:104-130`, `placement_epoch` per rebind `:172`.
- **Placement per-job epoch-based** (`image_flux.py:_ensure_placement:148`): ri-piazza solo se l'epoch è
  avanzato, altrimenti no-op → **rigenerazione stesso modello NON ricarica né ri-piazza** (requisito centrato).
  `_place` a 3 livelli gpu / model_cpu_offload / sequential_cpu_offload `:60-114`; anti spill Windows `:100`.
- **fp8 layerwise** (`image_flux.py:859`) → FLUX.1 in ~11GB GPU-resident, niente offload.
- **block_swap** FLUX.2/video (`flux2_transformer.py:enable_blockswap:61`) + **GGUF dequant on-the-fly** video
  (`ltx2_infer.py:_auto_gguf:472`, Q4 ~13GB / 21× speedup per `docs/STATO_NOTTE_GGUF.md`).
- **Auto-calibrazione HW** (`core/hw_profiles.py`) profili 8→48GB → margini/floor/offload/format-hint.

### Cosa è già ottimo
- ComponentStore refcount (T5/CLIP/VAE caricati una volta e condivisi); demote→CPU invece di destroy;
  epoch "no-op se stessa pipe" (rigenerazioni gratis); 3 fasce offload + block-swap; fp8 layerwise; GGUF
  dequant; app senza contesto CUDA; monitor live RAM/VRAM.

### Gap/criticità verificati
- **G1 — Unload NON pipeline-aware (criticità centrale).** Nessuna nozione di "categoria/pipeline successiva".
  Due soli estremi: `/free` = clear TOTALE (`server.py:149`→`clear:208`) oppure `evict_others` a grana
  "modello" (`model_manager.py:165`). Manca il livello "per tipo": immagine→inpaint stesso modello è salvato
  solo di striscio dal ComponentStore, non da una policy esplicita.
- **G2 — Coordinamento unidirezionale.** Il video libera l'immagine, ma **niente libera il video**: :8190 non
  ha `/free` (`ltx2_server.py:55-73`) → job immagine dopo un video trova VRAM/RAM ancora occupata.
- **G3 — `empty_cache()` incondizionato** (`model_manager.py:233`, `component_store.py:163`); ComfyUI lo fa
  solo se il reserved-libero supera ~25% del free (`…/model_management.py:822-826`).
- **G4 — Unload "tutto o niente" per modello, non per-byte** (`model_manager.py:106`); ComfyUI fa partial
  (`model_unload(memory_to_free)`, `lowvram_model_memory`).
- **G5 — Il server video ignora ComponentStore/MM**: `_PIPE_CACHE.clear()` a ogni cambio tipo
  (`ltx2_infer.py:467`) → gemma/VAE ricaricati passando tra ti2v/a2v/iclora/retake/inpaint.
- **G6 — Spill shared-VRAM non contabilizzato** nella soglia `_place` (`image_flux.py:96`; `_WORK_GB=8` fisso
  in `ltx2_infer.py:414`).
- Tecniche ComfyUI non replicate: `VRAMState` enum (LOW/NORMAL/HIGH/SHARED), `dynamic_pins`/soft-pin.

### Roadmap M1–M6
| # | Intervento | Impatto/costo | Dove | Tecnica ComfyUI |
|---|---|---|---|---|
| **M1** | **Unload PIPELINE-AWARE**: enum categoria (image/video/3d/world/music) + `/free?keep=<cat>` / `/prepare?for=<cat>`; stessa categoria→no-op, cambio→scarica solo il non-necessario | **ALTO / basso** | `server.py:147`, `model_manager.py:208`(`clear(keep_category)`)+`:96`(`make_room(keep)`), chiamanti `ltx2_infer.py:918`/`ltx2_service.py:56`/subprocess 3D | `free_memory(mem, keep_loaded=[...])` `model_management.py:789` |
| **M2** | **`/free` sul server video + coordinamento bidirezionale** (fix G2) | ALTO / basso | nuovo endpoint in `ltx2_server.py`; chiamarlo prima dei job immagine e dal pulsante GUI `main_window.py:995` | — |
| **M3** | **`soft_empty_cache` euristico** (fix G3): empty_cache solo se reserved-free > 25% | medio / basso | `model_manager.py:233`, `component_store.py:163` | `soft_empty_cache` `:1944` |
| **M4** | **Partial load per-byte** (fix G4) — budget dinamico di blocchi residenti, generalizza il block-swap FLUX.2 | ALTO / **MAJOR** | `flux2_transformer.py:61`, `image_flux.py:_place:60`, `ltx2_infer.py` | `model_load(lowvram_model_memory)` `:708` + `load_models_gpu:911` |
| **M5** | **ComponentStore/MM anche al server video** (fix G5) — refcount/demote per gemma+VAE condivisi | ALTO / **MAJOR** | `ltx2_infer.py:467` | — |
| **M6** | **Contabilizzare spill shared-VRAM** nella soglia (fix G6); `_WORK_GB` dinamico; `VRAMState` per fascia | medio / medio | `image_flux.py:100`, `ltx2_infer.py:414` | `VRAMState` enum |

**Raccomandazione:** batch **M1+M2+M3** (alto valore, basso rischio, centra l'obiettivo "mantenimento
intelligente pipeline-aware"). **M4, M5** sono MAJOR → discutere prima. M6 opzionale successivo.

---

## B) Dropdown senza colori VRAM verde/giallo/rosso — CAUSA CERTA + FIX APPLICATO
- **Causa:** i colori sono **calcolati e assegnati correttamente** (`core/model_precision.classify` +
  `ui/widgets/precision_combo.populate`→`setItemData(..., Qt.ForegroundRole)`; VRAM reale 31.8 GB). **Non si
  vedevano** perché `ui/theme.py` impostava `color:` nella regola QSS `QComboBox QAbstractItemView`: con un
  `color` sul popup, `QStyleSheetStyle` **ignora il `ForegroundRole` per-item**. Regressione da tema.
- **Fix applicato:** rimosso `color` dalla sola view del combo (sfondo/selezione restano; default eredita da
  `QWidget`, per-item lo sovrascrive). *(commit fatto; conferma a video rimandata)*
- **Gap collaterali (roadmap, no bug):** `compatible_components` (`core/model_detect.py:339`) non copre
  `dreamo`/`qwen_image`/`hidream`; i default video `_RECOMMENDED` (`core/model_catalog.py:25`: distilled 8/0.9,
  22B 30/3.0, eros 32/1.0, Q4 6/0.9) finiscono **solo nel tooltip**, non applicati ai widget al cambio modello
  (sull'immagine `ARCH_DEFAULTS` li applica: FLUX 20/3.5, Z-Image 8/1.0, Ernie 20/4.0).

## C) Macchie + linea/stacco di tono inpaint/outpaint — CAUSA + FIX APPLICATO
- **Causa (`inference_server/backends/image_flux.py`):** il color-match al bordo era **solo per outpaint**.
  Sull'inpaint c'era solo il feather (4–14px) che *sfoca* la linea ma non corregge il **salto di tono**: su aree
  ampie/lisce il generato resta di luminosità/saturazione/contrasto diversi → linea + stacco. Le **macchie** =
  (1) dominante: quel mismatch di tono; (2) rumore VAE a bassa frequenza sulle aree piatte. I **"quadrati"** =
  **VAE tiling**, che è **OFF di default** (`:869`): se ricompaiono, in ambiente c'è `NATIVE_VAE_TILING=1`
  (log `[image] VAE tiling=...`).
- **Fix applicato:** helper condiviso `_seam_tone_match` (offset per-canale + **gain std** seam-localizzato con
  falloff distance-transform) su **inpaint E outpaint**. Micro-test sintetico: salto al confine **23.2→4.2
  (−82%)**, tono riallineato verso la foto; seam-localizzato → non ridipinge oggetti lontani dal bordo. Env
  `NATIVE_INPAINT_COLORMATCH=0` per disattivare. *(commit fatto; validazione live rimandata)*
