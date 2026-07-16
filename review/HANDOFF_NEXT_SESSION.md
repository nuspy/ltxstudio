# HANDOFF — Nuova sessione: test estesi in fp8 + task residui

> **Scopo di questo file.** Trasferire lo stato a una sessione fresca. Il TaskList è in-memory
> (non persiste tra sessioni): questo documento è la **fonte di verità** dei task ancora aperti
> e degli obiettivi dichiarati. Aggiornato: **2026-07-11**, branch `feature/cinema-studio-review`.
>
> **Prima cosa da fare nella nuova sessione:** ricreare i 13 task qui sotto con `TaskCreate`
> (mantenendo gli stessi numeri #NN nel titolo per tracciabilità) e ripartire dai **test estesi in fp8**.

---

## 0. DIRETTIVA CENTRALE — «tutto su fp8/fp4» (memoria [[fp8-fp4-default-everywhere]])

Direttiva utente (10/07, confermata): **il DEFAULT di ogni modulo deve preferire i pesi fp8/fp4
(nvfp4/GGUF Q4), evitando fp16/bf16 a piena precisione**. Motivo: i pesi bf16 non entrano nei 32GB
(RTX 5090) → offload CPU↔GPU = lentezza estrema. fp8/fp4 restano residenti in VRAM → veloci.
Vale SOLO per il default automatico; la scelta ESPLICITA dell'utente nel dropdown va rispettata
(regola [[no-auto-model-selection]]).

### Stato fp8 già raggiunto in questa sessione (committato `90c3bd9`)
| Modulo | Stato |
|---|---|
| **FLUX.1** (Dev/Krea/Kontext/Fill) | ✅ **CORRETTO** — `enable_layerwise_casting(fp8_e4m3)` applicato a TUTTI i transformer non-gguf (prima solo ai già-fp8). MISURATO **45s** vs >10min in offload. `image_flux.py` `_load_flux_tr`~444 + branch checkpoint ~854; `_precise_vram_mb`~343 conta i bf16 grandi come 1B se storage fp8 |
| **FLUX.2** (Dev/Klein) | ✅ **già fp8** — loader custom `flux2_transformer.py` (Fp8Linear: pesi in `float8_e4m3fn`, dequant on-the-fly per-layer). Nessuna modifica necessaria |
| **Video LTX-2** | ✅ già GGUF Q4 (`ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf`) |
| **SeedVR2** | ha già peso fp8 (`seedvr2_ema_3b_fp8_e4m3fn`) — ma NON ancora cablato come default (vedi #156) |
| **FBCache** | ✅ **OFF di default** (`native_launcher.py:28` `NATIVE_FBCACHE=0`) → elimina le **macchie chiaro-scure** su FLUX (era la causa confermata: `01` macchiata vs `01b` pulita). Opt-in con `NATIVE_FBCACHE=1` |

### fp8 ANCORA da verificare/estendere nei test estesi
- **FlashVSR / SeedVR2**: oggi usano `enable_cpu_offload()` sempre-on + scale 4.0 → lentissimi (#156).
  Applicare la stessa lezione: peso fp8 **residente** (no offload) + scala default 2×.
- **Voce/Musica** (F5/XTTS/ACE-Step/HeartMuLa): verificare che i default non siano fp16 pieni.
- **3D** (TRELLIS.2): verificare precisione di default (attenzione: `precision=bf16` con autocast
  rompeva i kernel sparse-conv Triton — già fix rimuovendo autocast, `scripts/trellis_generate.py`).
- Auto-select immagini: confermare che a parità di modello scelga la variante fp8/GGUF quando esiste.

---

## 1. OBIETTIVI DICHIARATI (da riprendere)

1. **Sessione di test ESTESA e autonoma, ora in fp8** — rigenerare tutta la galleria/matrice di test
   con i default fp8 attivi, per confermare che qualità (no macchie) e velocità (no offload) siano
   corrette su TUTTI i moduli. Modalità: **autonoma senza collaborazione**; l'utente ispeziona alla
   fine e segnala solo gli artefatti difettosi. **NON cancellare gli output** (restano in `output/`).
   Meccanismo: `tests/gui_regression/run_real.py` (RealSpy esegue i worker in modo sincrono = gen reale,
   VRAM-safe via `/free`). Prompt sempre CREATIVI/vari, mai gattini/cani ([[test-prompts-creative]]).
2. **#155 Lucky Director** — cortometraggio 5 min (vedi task sotto), ora che le immagini sono pulite/veloci.
3. **#108 Asset marketing** — completare i 12 asset (3/12 fatti).
4. **Chiudere i difetti di qualità reali** segnalati dall'utente: ACE-Step rumore grigio (#... vedi sotto),
   FlashVSR/SeedVR2 (#156), t2v perf (#158), faceswap foto reale (#157).

---

## 2. TASK IN PROGRESS (3) — da ricreare e continuare

### #51 — Eliminare TUTTE le dipendenze ComfyUI residue
`in_progress`. **Runtime già a ZERO chiamate ComfyUI** (ultima rimossa commit `3c8a855`). Residuo =
solo DEAD CODE dietro `is_native()=False` (template_engine, *_pipeline.py, comfy_driver, video_ltx).
PURGE FISICO = pass dedicato POST-LANCIO con re-test di ogni feature (refactor accoppiato, rischio alto)
→ NON nel loop autonomo. Vedi `docs/COMFY_AUDIT.md`.

### #61 — Consolidare i venv (14→7)
`in_progress`. Consolidamento FATTO (commit `37d4ba2`): Host A=`inference_server_venv` (assorbiti
ACE-Step, XTTS, heartlib, FlashVSR, SkinTokens), Host B=`F5-TTS/venv` (assorbito SeedVR2). Import-smoke
tutti OK. **RESTA:** (1) **live-test a GPU libera** di musica/voce/SeedVR2/FlashVSR/rig dall'host
consolidato; (2) SOLO dopo conferma → cancellare i donor (~30GB) previo backup NAS. Distruttivo → non
in autonomia. Copre la validazione #61 dentro i test estesi.

### #108 — Media marketing Patreon (docs/media/, 12 asset)
`in_progress`. 3/12 fatti (commit `46abce2`: hero_banner, images_hero_grid, reddit_thumb). RESTANO 9:
alcuni via app+GPU locale (intro_flow_60s screen-rec, images_edit_demo, studio_lora_from_one_photo,
filmstudio UI screenshot, video_character_replace, 3d_turntable). Poi rigenerare `patreon_0001.pdf`.

---

## 3. TASK APERTI (10) — da ricreare

| # | Titolo | Nota chiave |
|---|---|---|
| **#155** | Test finale **Lucky Director** 5 min | `core/film_studio/lucky_director.py`. Param: 480p, **3:2**, scene cross-fade, take **hard-cut**, coerenza **PuLID**. Prompt: ragazzo si innamora della gelataia, la invita in modo divertente, date perfetto, lei gli offre un gelato ma lui «odia i gelati» perché ne ha mangiati ogni giorno per mesi per vederla, ridono e si baciano. Output in `output/`, non cancellare. **Dipende da immagini pulite/veloci (fp8) — ora pronte** |
| **#156** | Fix **FlashVSR/SeedVR2** super-res (hang/>30min su 2s) | `core/flashvsr_service.py`, `core/seedvr2_service.py`. Causa diagnosticata: NON hang ma offload sempre-on + scale 4.0. Fix fp8: peso residente + scala default 2× + timeout |
| **#157** | Validare **face-swap** con foto reale | delega a `inference_server_venv` OK; detector antelopev2 non rileva volti FLUX stilizzati. Testare con foto vera; valutare det_size/soglia più permissivi |
| **#158** | Perf **t2v 704×1280×8s** (>10min) | verificare chunking video lunghi + block_swap + chunk size. Nota: già validati t2v 512² e 704×1280×49f reali |
| **#100** | Test live **arricchimento prompt** (Gemma) | validare checkbox on/off su immagini/video/edit/musica + fallback LLM-assente non rompe il flusso |
| **#101** | Test live **PuLID potenziato** | volto reale (Martina): media multi-foto, start_step 1-4, true_cfg>1, strength≤3, campo volto in kontext, one-click volto→primo frame→i2v. Tarare i default |
| **#94** | Test GUI reale con **qtbot** (click/type/select) | F1 inventario `docs/GUI_TEST_MATRIX.md`, F2 upgrade harness qtbot mantenendo Spy, F3 PoC su video_tab. pytest-qt 4.5 già presente |
| **#50** | Multi-GPU reale + pausa/ripresa coda | `scheduler.py` predisposto (seriale). Richiede **HW multi-GPU** → non testabile ora |
| **#72** | **Paints-Undo** making-of (immagine→video disegno) | MAGGIORE, nuovo tab+venv. **Attende conferma utente sul piano** prima di implementare |
| **#75** | Wheel pre-compilati per CUDA arch | Script `scripts/install_accelerators.py` FATTO. **Azione utente**: scaricare wheel torch211/sm120 in `E:\AI\Tools\precompiled_wheels` |

---

## 4. DIFETTO DI QUALITÀ APERTO (non ancora un task) — ACE-Step rumore grigio

`21_music_acestep.wav` = **solo rumore grigio** (HeartMuLa OK). Usa ACE-Step-v1-3.5B. Bug reale del
generatore (pesi/param/pipeline), non del venv (gira dall'host consolidato). **Da aprire come task e
investigare** nella nuova sessione. Vedi `review/LIVE_TEST_SESSION_REPORT.md` §2.

---

## 5. REGOLE/VINCOLI PERMANENTI (CLAUDE.md + memoria)

- **Rispondere in italiano.** Commit terminano con `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **UI su ENTRAMBE le varianti** (Classica V1 + Pro V2) — [[ui-changes-both-variants]].
- **i18n obbligatorio in tutte le 15 lingue**, mai stringhe hardcoded — [[gui-integration-organic]].
- **NO auto model selection** ([[no-auto-model-selection]]): mai scegliere/sostituire il checkpoint in
  automatico; unica eccezione = gestione memoria dopo dialog di conferma. La direttiva fp8 vale SOLO per
  il DEFAULT quando l'utente non sceglie.
- **Ogni feature nel feature_matrix** ([[always-add-to-matrix-builder]]) e integrata nella GUI nel punto
  pertinente ([[gui-integration-organic]]).
- **NON toccare** `Comfy_Nexus.bak` né i venv (solo analisi per replicare come Comfy risolve — [[study-comfyui-solutions]]).
- **App 100% locale** ([[local-only-target]]); HF token solo per download una-tantum pesi gated.
- **GPU/torch.cuda**: il processo GUI non chiama mai `torch.cuda.*` — VRAM via `core/gpu_mem.py` ([[app-no-cuda-context]]).
- **Download discipline**: killare zombie prima di rilanciare; connessione metered ([[download-discipline]]).
- **Prompt di test creativi/vari**, mai gattini/cani.

---

## 6. AMBIENTE / COMANDI SERVER

- InferenceServer immagini `:8189` → `python -m inference_server.server` (venv
  `E:\AI\Tools\inference_server_venv\Scripts\python.exe`). Env: `NATIVE_ATTENTION=auto`,
  `NATIVE_FBCACHE=0` (default, no macchie), `NATIVE_FLUX1_FP8=1` (fp8 residente).
- REST API `:8200` → `python -m api.app` (venv GUI `Comfy_Nexus.bak/ComfyUI/venv`). Coda seriale:
  **non riavviare i server a metà run** (ingolfa la coda → falsi timeout).
- Worker video (ltx2_venv), 3D/voce/musica: spawnati per-job dai service, nessun server persistente.
- Harness: `QT_QPA_PLATFORM=offscreen <inference_server_venv python> tests/gui_regression/run_real.py --filter <tab>`.
  Esiti in `output/tests/REAL/REAL_SUMMARY.txt` + `.txt` per-caso. Ripartenza granulare per categoria.

---

## 7. RIFERIMENTI

- `review/LIVE_TEST_SESSION_REPORT.md` — report sessione live precedente (8 bug corretti + difetti aperti).
- `docs/COMFY_AUDIT.md` — audit dipendenze ComfyUI (#51).
- `C:\Users\atain\.claude\projects\E--ai-ltxstudio\memory\` — memoria (MEMORY.md è l'indice).
- Piano test esaustivo originale: `C:\Users\atain\.claude\plans\joyful-twirling-parrot.md`.
