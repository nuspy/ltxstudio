# Review MODULI RESTANTI — Musica/Audio, 3D, Cinema, Convertitore, Pesi, Avanzate (2026-07-07)

3 reviewer paralleli. Bug utente diagnosticati e FIXATI (vedi commit).

## BUG UTENTE — root cause VERE (verificate da me, non solo dai reviewer)
| Bug | Root cause REALE | Fix |
|---|---|---|
| «Genera musica non fa nulla» | `_gen_song` → `if self._busy(): return` SILENZIOSO; `_busy()` include `_install_proc` che resta settato PER SEMPRE se l'installer non parte (errorOccurred non gestito) | busy ora VISIBILE (status+log col motivo) + errorOccurred azzera _install_proc. NB: la tesi del reviewer (signal da thread daemon) è FALSA — i signal cross-thread sono queued/legali |
| «Music video non genera immagini» | gen_scene_image inviava job SENZA 'model' → KeyError sul server; in più il batch (_next_batch) IGNORAVA res['error'] e proseguiva "completato" senza nulla | default _default_model(arch) sul server + _on_batch_item raccoglie errori → dialog finale con l'elenco + progress N/M |
| KeyError 'model' su «rigenera» | idem sopra | fixato (commit precedente) |

## MUSICA — ACE-Step 1 vs 1.5 (mappa per l'estensione richiesta)
v1 ha e v1.5 NON espone: scheduler (euler/heun/pingpong), cfg_type (apg/cfg/cfg_star), omega,
guidance_interval/decay/min_guidance, ERG (tag/lyric/diff), doppia guida gs_text/gs_lyric.
v1.5: parametri STRUTTURATI (bpm/keyscale/timesignature/lingua), planner 5Hz "think", steps fissi 8 turbo,
task futuri LOCKED (extract/lego/complete/cover/remix in _LOCKED_15).
PIANO estensione (contenitori avanzati): (1) verificare nell'API 1.5 quali parametri esistono davvero;
(2) esporli in MusicAdvanced con gating engine=="ACE-Step 1.5" (🧪 beta se instabili); (3) _LOCKED_15
dinamico via acestep15_service.supports_task(). — DA IMPLEMENTARE (prossimo lotto, serve verifica API 1.5).

## 3D (Modello/Rig/Animazione/Mondo)
- TRELLIS ✓ validato · RIG ✓ validato · MOCAP cattura ✓ validato · RETARGET ✗ MAI testato live (da fare).
- Catena Trellis→Rig→Anim collegata; **WorldGen era un'ISOLA** → aggiunto ponte mesh_ready→rig_tab (fix oggi).
- MediaPipe pinnato 0.10.18 (API legacy); Blender auto-install ok; fix WorldGen (nunchaku/pano_depth/pt3d/DA-2)
  tutti ancora presenti in install_world.py ✓.

## CINEMA STUDIO
- Leak FD in gen_service (json.dump(open)) → FIXATO oggi.
- Scheduler senza recovery post-crash (job 'running' per sempre; coda in-memory) → DA FARE (timeout+stale).
- Path hardcoded LTX2_PY/WORKER/GGUF; timeline drag&drop da verificare live; entrambe le varianti riviste.

## CONVERTITORE
- convert_tab = conversioni 3D (glb→fbx…), NON quantizzazione modelli. dequant_comfy: fallback SILENZIOSO
  se venv ComfyUI assente. Mancano: conversione inversa, batch, wizard quantizzazione GGUF.

## PESI/AGGIORNA MODELLI
- weights_catalog.json: 15+ path Windows hardcoded (portabilità); DEFAULT_BASE hardcoded in 3 moduli.
- model_detect cache NON thread-safe (Lock mancante). fast_downloader: solido (resume/multi-chunk).

## AVANZATE
- advanced_tab hub: verificare connect() di open_workflow/analyzer/navigator in MainWindow.
- import_models: _SearchWorker forse mai istanziato (codice morto?). weights_tab: nessuna validazione base_dir.

## TeaCache — calibrazione live (parte 1)
1° giro: `[teacache] attivo su 0 stage` → output BIT-IDENTICI (PSNR 108dB) e tempi uguali = l'hook per-stage
NON trovava mai il modello (transformer costruito LAZY dentro la run). FIXATO con patch di CLASSE su
X0Model.forward (riuso della velocity + to_denoised sui latenti correnti — corretto da paper). 2° giro in corso.
