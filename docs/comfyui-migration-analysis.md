# Migrazione da ComfyUI a backend proprietario — Analisi (2026-06-22)

Obiettivo: sostituire ComfyUI (eseguito in background) con un backend nostro/integrato per
alleggerire il sistema, eliminare le dipendenze fragili (versioni ComfyUI, custom node, controllo
limitato), in cambio di dover gestire noi la VRAM nei passaggi tra moduli.

---

## A) Tutto ciò che usa ComfyUI nel progetto

**Strato di interfaccia (4 file)**
- `core/comfy_driver.py` — client REST+WebSocket: `/prompt`, `/history`, `/object_info`, `/view`, `/upload/image`, `/system_stats`, `/free`, WS preview/progress/interrupt.
- `server/runner.py` — submit+poll+download output; path fissi (input ComfyUI, output, _uploads).
- `core/comfy_launcher.py` — avvia `run_nvidia.bat` in console separata.
- `core/comfy_compat.py` + `comfy_compat/ltxstudio_comfy_compat/patches.py` — **shim con 6 monkey-patch** su core ComfyUI per LTX-2.3 (supported_models, av_model×2, text_encoders, latent_preview, nodes_textgen) + `prompt_unlock`.

**12 pipeline che generano grafi** (`core/*_pipeline.py`): gen, upscale, inpaint, kontext, integrate, extend, segment, restore, video_enhance, face_restore, trellis, rig. Più `template_engine.py` (superset video) e `flux2_encoder.py` (risoluzione encoder FLUX.2).

**Nodi**: ~75 class_type unici = 50 core ComfyUI + **25 custom node da 12 repo**:
`comfyui_segment_anything`, `impact-pack`, `facerestore_cf`, `ComfyUI-TRELLIS2`, `ComfyUI-SkinToken`, `ComfyUI-SeedVR2`, `comfyui-frame-interpolation` (RIFE), `rgthree-comfy`, `AILab` (Qwen prompt enhancer), nodi LTXV/IAMCCS (extend), GroundingDINO, TTS nodes.

**Template superset video** `pluto_superset.api.json`: ~210 nodi / 60 class_type unici, con `disable_modules` (rgthree Context/Switch) per attivare/disattivare rami (enhancer, controlnet, refine/upscale, audio, model selector).

**14 architetture**: FLUX.1, FLUX.2 (+Klein), Z-Image, Ernie, FLUX Checkpoint, FLUX Fill, FLUX Kontext (immagini); LTX-2.3 (video+audio); SeedVR2, RIFE (restauro video); GroundingDINO+SAM (segmentazione); CodeFormer/GFPGAN (volti); TRELLIS.2, SkinTokens (3D); ESRGAN (upscale); Qwen3 (prompt enhance); TTS/voice.

**47 file** toccano ComfyUI (driver/runner/launcher/compat, 12 pipeline, template_engine, flux2_encoder, UI binding, catalog, installer, shim, build).

---

## B) Fattibilità della migrazione

**Verdetto: ALTA per ~80% dei moduli, MEDIA per il video LTX, grazie al fatto che quasi tutti i modelli hanno oggi un percorso standalone/diffusers.**

| Modello | Percorso nativo disponibile | Fattibilità |
|---|---|---|
| FLUX.1 / Fill / Kontext | `diffusers` (FluxPipeline/FluxFill/FluxKontext) | ★★★★★ |
| FLUX.2 (+Klein) | `diffusers` (Flux2Pipeline) | ★★★★☆ (encoder per-variante già risolto da `flux2_encoder.py`) |
| Z-Image / Turbo | `diffusers` ZImagePipeline | ★★★★★ |
| ERNIE-Image | `diffusers` ErnieImagePipeline | ★★★★★ |
| **LTX-2.3 (video+audio)** | **pacchetto ufficiale `Lightricks/LTX-2` (ltx-pipelines/ltx-core) + diffusers LTX2** | ★★★☆☆ (extend autoregressivo è custom nostro) |
| SeedVR2 | repo standalone ByteDance | ★★★★☆ |
| RIFE | Practical-RIFE standalone | ★★★★★ |
| GroundingDINO + SAM-HQ | `segment-anything` / `GroundingDINO` standalone | ★★★★☆ |
| CodeFormer / GFPGAN | repo standalone | ★★★★★ |
| ESRGAN upscale | `spandrel`/`basicsr` standalone | ★★★★★ |
| TRELLIS.2 / SkinTokens | già in ambiente isolato (pixi); repo originali standalone | ★★★☆☆ |
| Qwen3 prompt enhancer | `transformers` | ★★★★★ |
| TTS / voice clone | provider standalone (XTTS/F5/CosyVoice) | ★★★☆☆ |

**Punti a favore**: i grafi sono generati come dict in Python (UI disaccoppiata); l'interfaccia driver è minima (≈7 endpoint + WS) → si può creare un `Backend` astratto con due implementazioni (ComfyUI / nativo) e migrare un modulo alla volta. `flux2_encoder.py` è già indipendente.

**Punti contro**: lo **shim LTX-2.3** (6 monkey-patch) indica che il video è la parte più "viva"/instabile; il **template superset** (rgthree Context, switch, timeline prompt) è fortemente accoppiato a ComfyUI e va riscritto come logica Python; la **gestione VRAM** oggi è gratis (ComfyUI) e diventa codice nostro.

---

## C) Vantaggi per modulo

- **Immagini (FLUX/FLUX.2/Z-Image/Ernie/Kontext/inpaint/integrate/upscale/restore)**: controllo totale del sampling, niente più rotture da update ComfyUI/custom node, avvio più rapido, meno RAM/processi, error handling diretto, hot-swap LoRA gestito da noi. Parità facile (diffusers espone scheduler/guidance/lora).
- **Video LTX-2.3**: usando il pacchetto ufficiale Lightricks si elimina lo **shim/monkey-patch** (la fonte di fragilità maggiore) e si segue direttamente l'API ufficiale (anche audio sincronizzato e LoRA). Versioni controllate da noi.
- **Video enhance (SeedVR2/RIFE)**: dipendenze isolate, niente custom node ComfyUI da mantenere allineati.
- **Volti (CodeFormer/GFPGAN)**: libreria diretta, niente `facerestore_cf` clone fragile.
- **Segmentazione (SAM/GroundingDINO)**: API dirette, niente impact-pack/segment_anything node.
- **3D (TRELLIS/SkinTokens)**: già semi-isolati; passandoli ai repo originali in venv dedicati escono del tutto dal grafo ComfyUI.
- **Trasversale**: un solo runtime nostro → binario più piccolo (niente ComfyUI+frontend), nessun server web esterno, dipendenze pinnate, licenza/telemetria sotto controllo, debugging serio (stack trace Python diretti).

---

## D) Problemi per modulo

- **Immagini**: riprodurre 1:1 dettagli come `LTXVScheduler`/`ModelSamplingAuraFlow(shift)`, `ConditioningZeroOut` (Z-Image), `DifferentialDiffusion` + `InpaintModelConditioning` (inpaint), `ReferenceLatent`/`FluxKontextImageScale` (Kontext/Klein). Sono tutti riproducibili in diffusers ma richiedono cura per non perdere qualità.
- **Video LTX-2.3**: l'**extend autoregressivo memory-bounded** (ContextLatent + ExtendSampler + decode per-segmento + stitch overlap + audio per-segmento) è logica NOSTRA costruita su nodi LTXV/IAMCCS: va riscritta sopra `ltx-core` (rischio più alto). NAG, soft-ref, CFG/step di default da replicare.
- **Video enhance**: SeedVR2 ha tuning VRAM/tiling/device-swap non banale (oggi lo fa il custom node); RIFE è semplice.
- **Segmentazione**: due percorsi (GroundingDINO+SAM-HQ e CLIPSeg legacy) + region click/box → riscrivere il routing maschere.
- **3D/TTS**: molti pesi, ambienti pesanti; rischio di duplicare ciò che già funziona via custom node (ROI bassa nel migrarli subito).
- **Template superset**: la composizione dinamica (disable_modules, switch, timeline-prompt, controlnet preprocessors AIO/DW/Canny) non ha equivalente diretto → diventa codice di branching Python.
- **Trasversale**: perdita del "gratis" di ComfyUI (preview live, gestione code, smart-memory, vasto ecosistema di nodi futuri). Ogni nuovo modello andrà integrato a mano (non più "arriva il custom node").

---

## E) Rischi e lavoro stimato (programmazione agentica)

**Rischi principali**
1. **Parità di qualità** (alto): differenze sottili di scheduler/seed/precisione tra ComfyUI e diffusers → serve un set di test di regressione visiva per modulo.
2. **Extend video autoregressivo** (alto): la parte più custom; rischio artefatti alle giunzioni/audio.
3. **Gestione VRAM** (alto): OOM nei passaggi modulo→modulo, leak, lentezza da reload continui.
4. **Manutenzione futura** (medio): ogni nuovo modello = lavoro nostro, non più "plug a custom node".
5. **Regressione feature** (medio): controlnet, GGUF, quantizzazioni fp8, LoRA multiple.
6. **3D/TTS** (basso): conviene lasciarli sugli ambienti isolati esistenti a lungo.

**Stima (sessioni-agente; 1 sessione ≈ mezza giornata di lavoro guidato)**

| Blocco | Lavoro | Stima |
|---|---|---|
| Infrastruttura: `Backend` astratto + `InferenceServer` persistente + `ModelManager` (VRAM) + job queue + progress/preview | fondazione | 6–10 |
| Immagini FLUX.1/FLUX.2 (gen) + LoRA + encoder | primo modulo nativo | 4–6 |
| Z-Image, Ernie, Kontext, integrate, inpaint/outpaint, upscale, restore | resto immagini | 6–9 |
| Segmentazione (SAM/GroundingDINO) + face restore | standalone | 3–5 |
| Video enhance (SeedVR2 + RIFE) | standalone | 3–5 |
| Video LTX-2.3 base (t2v/i2v + audio) su pacchetto ufficiale | video core | 6–10 |
| Extend autoregressivo + ReTake/Talk | video avanzato | 5–8 |
| Migrazione 3D (TRELLIS/SkinTokens) e TTS *(opzionale/dopo)* | code | 4–7 |
| Test di regressione, parità qualità, hardening VRAM | qualità | 5–8 |
| **Totale** (senza 3D/TTS) | | **≈ 38–61 sessioni** |

Ordine consigliato (strangler): infrastruttura → immagini → enhance/face/segment → video → extend → (3D/TTS per ultimi o mai). ComfyUI resta attivo finché ogni modulo non è migrato e validato dietro feature-flag.

---

## F) Strategia gestione memoria (il cuore del backend proprietario)

Oggi ComfyUI fa smart-memory automatica. Sostituirla con un **`ModelManager`** centrale dentro un **`InferenceServer`** persistente (processo unico in background che tiene i modelli e riceve job — NON un processo per job, altrimenti reload continui).

Componenti:
1. **Registry + caricamento lazy**: i modelli (UNet/DiT, VAE, text encoder, LoRA, upscaler…) si caricano on-demand e si tracciano per VRAM occupata e ultimo uso.
2. **Unload LRU + budget VRAM**: prima di caricare il modello del modulo richiesto, se il budget (es. 90% di `system_stats`) non basta, si scaricano i modelli meno usati. È esattamente lo scenario "da foto a video / cambio LoRA o modello".
3. **Offload diffusers**: `enable_model_cpu_offload` (default), `enable_sequential_cpu_offload` (low-VRAM), group/leaf offload, `vae.enable_tiling()/slicing()` per i video/4K. Esposti come "qualità vs VRAM".
4. **Quantizzazione**: fp8/bnb/optimum-quanto per i modelli grandi (FLUX.2 dev, LTX) come già si fa.
5. **LoRA hot-swap**: `load_lora_weights`/`set_adapters`/`unload_lora_weights`/`fuse` senza ricaricare il modello base; cache per set di LoRA attive.
6. **Job queue seriale** (come ComfyUI): un job alla volta, con `interrupt`/cancel e callback di progresso/preview → stessa UX delle tab attuali.
7. **Stesse API del driver**: `InferenceServer` espone `submit/poll/progress/interrupt` con la **stessa firma** del `comfy_driver`, così le tab non cambiano (il `Backend` astratto sceglie ComfyUI o nativo per-modulo via feature-flag).
8. **Reset pulito**: `torch.cuda.empty_cache()` + `gc` ai cambi di modulo; watchdog OOM che scarica tutto e ritenta una volta.

Politica pratica per i passaggi tipici:
- **foto → video**: scarica i pesi immagine (FLUX) prima di caricare LTX (entrambi grandi, raramente coesistono in 32GB senza offload).
- **cambio LoRA**: hot-swap senza toccare il modello base.
- **cambio modello stessa famiglia**: unload del precedente + load nuovo; tieni VAE/text-encoder condivisi in cache se compatibili.
- **moduli leggeri** (SAM, RIFE, face): caricabili insieme a un modello principale entro budget.

---

## Raccomandazione

Migrazione **fattibile e sensata**, ma è un progetto importante (≈40–60 sessioni-agente per il grosso). Approccio **incrementale strangler** dietro un `Backend` astratto: si parte dall'infrastruttura (InferenceServer + ModelManager) e dal modulo immagini (massimo uso, rischio minimo via diffusers), si valida la parità, poi si avanza verso il video; ComfyUI resta come fallback finché ogni modulo non è migrato. I moduli 3D/TTS (già isolati) si lasciano per ultimi o si mantengono separati.

### Fonti
- [LTX-2 — pacchetto ufficiale Python (Lightricks)](https://github.com/Lightricks/LTX-2)
- [LTX-2 in diffusers](https://huggingface.co/docs/diffusers/api/pipelines/ltx2)
- [Z-Image — diffusers (ZImagePipeline)](https://huggingface.co/docs/diffusers/api/pipelines/z_image)
- [ERNIE-Image — diffusers (ErnieImagePipeline)](https://huggingface.co/docs/diffusers/main/api/pipelines/ernie_image)
