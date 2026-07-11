# Report sessione test LIVE (autonoma) — 2026-07-11

Sessione di validazione con **generazione reale su GPU** (RTX 5090) di tutte le feature validate finora solo
"a secco", + i task-test pendenti. Meccanismi: harness GUI reale (`tests/gui_regression/run_real.py`, RealSpy),
REST API :8200, galleria curata. **Nessun output cancellato** — tutti i file sono in `output/` per l'ispezione.

## 🐛 Bug REALI trovati e corretti (8 — tutti committati)
Tutti latenti perché i rispettivi percorsi non erano mai stati eseguiti live end-to-end.

| # | Bug | File | Commit |
|---|---|---|---|
| 1 | Path worker `dirname³` → `E:\ai\workers\…` (manca `ltxstudio`); rompeva TUTTI i modi di Modifica Video AI | ui/main_window.py `_VideoEditAIWorker` | 76d7ee2 |
| 2 | `KeyError: 'name'` — LoRA passati con chiave `path`, worker legge `name` | workers/ltx2_infer.py + main_window | 76d7ee2 |
| 3 | Edit prendeva dim/durata dal tab t2v (704×1280/8s) invece che dal **video sorgente** → output distorto | main_window `_gen_video_edit_ai` (`_probe_video_dims`) | 76d7ee2 |
| 4 | **Rumore grigio audio** su i2v/ti2v/iclora (LTX-2 genera audio-rumore non richiesto) | workers/ltx2_infer.py | b6073a9 |
| 5 | Crash Triton su TRELLIS `precision=bf16` (autocast rompe i kernel sparse-conv) | scripts/trellis_generate.py | b6073a9 |
| 6 | Default upscale `4xUltraSharp.pth` **inesistente** (reale: `4x-UltraSharpV2.safetensors`) | core/restore_pipeline.py | 0fbd3ce |
| 7 | **Face-swap rotto**: insightface fuori dal venv GUI → ora **delegato** all'inference_server_venv | core/faceswap.py + scripts/faceswap_run.py | 0fbd3ce |
| 8 | Formato **3:2** mancante nel Lucky Director | core/film_studio/lucky_director.py | (+3:2) |

## ✅ Feature validate LIVE con artefatti reali
- **#102 replace_char** (SAM2 + composite per-frame, audio del sorgente preservato) — 2/2 PASS harness.
- **TRELLIS 9 parametri** — GLB reale `output/3d/trellis__portraitA_7/model.glb` (tutti i 9 flag attivi).
- **Galleria** `output/review_gallery/` (13 OK): t2i FLUX.1/Krea/FLUX.2, edit-kontext, recolor, segment,
  t2v, i2v, a2v-lipsync, controlnet-canny, musica HeartMuLa, musica ACE-Step.

## ⚠️ DIFETTI DI QUALITÀ segnalati dall'utente (REALI — da risolvere)
1. **Macchie chiaro-scure su immagini FLUX** (t2i FLUX.1/FLUX.2, edit-kontext). Comune ai 3 arch → stadio
   condiviso. **Ipotesi forte: FBCache** (`NATIVE_FBCACHE=1`, cache dei blocchi transformer per velocità) —
   il commento in `image_flux.py:889` conferma "FBCache rompe gli arch image-conditioned (Kontext/inpaint)".
   Test di conferma (rigenerazione con FBCache OFF) in corso. **Azione probabile**: FBCache OFF di default
   per le immagini (o soglia più conservativa), oppure escluderlo dagli arch image-conditioned.
2. **Musica ACE-Step = solo rumore grigio** (`21_music_acestep.wav`). HeartMuLa OK. Bug reale del generatore
   ACE-Step v1-3.5B → da investigare (pesi/parametri/pipeline).
3. **FlashVSR + SeedVR2** (super-risoluzione video): **hang / >30 min** su una clip di 2s → da investigare.
4. **t2v 704×1280×8s** (default video_tab): molto lento (>10 min) — performance.

## ℹ️ NON bug (artefatti di test/fixture)
- **a2v "tuuuu"**: la fixture audio era un **tono sine 440Hz**, non voce → niente parlato da sincronizzare.
  Ri-testare con una clip **TTS/voce reale**.
- **faceswap galleria fallito**: (a) server API stale caricato prima del fix, (b) i ritratti **FLUX stilizzati**
  non hanno un volto rilevabile dal detector. Con **foto reali** funziona (già validato #31/#68).
- **sonify/TTS/3D "timeout" nella galleria**: causati dall'intervento a metà run (kill SeedVR2 + restart :8189)
  che ha **ingolfato la coda seriale di :8200** → i job non sono partiti. NON bug del prodotto.
- **upscale galleria fallito**: nome modello errato nel MIO script di galleria (l'app è corretta, vedi #6).

## Consolidamento venv #61 (validazione parziale)
Musica (HeartMuLa/ACE-Step) girano dall'host consolidato `inference_server_venv` (OK per HeartMuLa; ACE-Step
produce rumore ma **gira**, quindi il venv non è la causa). TTS/enhance non conclusi per il blocco coda.

## Prossimi passi
1. Confermare/risolvere FBCache (macchie FLUX) — priorità alta (impatta tutte le immagini + i keyframe del film).
2. ACE-Step rumore, FlashVSR/SeedVR2 hang.
3. **Lucky Director #155**: cortometraggio 5 min (3:2, 480p, cross-fade scene, hard-cut riprese, PuLID) — pronto
   ma dipende dalla qualità immagine (FBCache) → conviene farlo dopo il fix #1.
