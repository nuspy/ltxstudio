# Review modelli installati — luglio 2026 (bozza in lavorazione)

> Campagna 22/07: test live via GUI + review qualitativa (video+immagini) + FP4.
> Criterio VRAM: uso entro 32GB (RTX 5090); fp16/bf16 oltre 32GB = dannoso → sostituire con quantizzato.
> Stato: BOZZA — le sezioni test-live e raccomandazioni finali si completano a matrice conclusa.

## Pulizia già eseguita (Fase 0, approvata)
- **80.5 GB liberati**: `Checkpoints\LTX\ltx-2.3-22b-dev-fp8.safetensors` (27.1GB, sha256-identico alla copia in diffusion_models), `diffusers\ltx2\ltx-2.3-22b-distilled.safetensors` (43GB, identico alla copia in Checkpoints), `ltx-2-3-22b-dev-Q3_K_M.gguf.CORROTTO` (10.4GB).
- Fix collaterali: tag NSFW corretti (dev-fp8/distilled → SFW, eros → NSFW), default SeedVR2 → 3B fp8 esistente.

## Reputazione qualitativa (ricerca online 22/07, fonti 2025-26)

### Video
| Modello | Giudizio | Sintesi |
|---|---|---|
| LTX-2.3 22B dev / distilled 1.1 | **ECCELLENTE** | Miglior compromesso open su singola GPU; distilled = default community (8 step), dev = massimo dettaglio. Difetti: VRAM alta, occasionali incoerenze temporali. |
| HunyuanVideo 720 cfgdistill | **MEDIOCRE/OBSOLETO** | Re del 2025, superato da Wan/LTX-2; Tencent stessa l'ha rimpiazzato (1.5). Senso solo per LoRA legacy/FramePack. |
| FramePack I2V + F1 | **BUONO in declino** | Low-VRAM anti-drift, ma F1 ha flickering noto su pelle/superfici chiare; estetica Hunyuan datata. |
| Wan 2.1 14B + MultiTalk | **ECCELLENTE (lip-sync)** | MultiTalk tuttora SOTA open per lip-sync multi-speaker (anche canto). Limiti: 720p, degrada su rotazioni/occlusioni. |
| Krea Realtime 14B | **BUONO (nicchia)** | Autoregressivo interattivo (restyle live); non realtime su consumer, qualità sotto Wan/LTX. Solo sperimentazione. |
| LingBot-Video 1.3B/30B MoE | **BUONO, troppo nuovo** | MoE video (3B attivi), fisica/embodied top su RBench; max 480p, non compete per estetica. |
| Self-Forcing DMD 1.3B | **BUONO per preview** | Velocissimo, ma 1.3B: dettaglio limitato. Superato da Krea RT 14B. |
| SeedVR2 3B/7B + FlashVSR | **ECCELLENTE (coppia)** | SeedVR2 = qualità max (input soft/AI), FlashVSR = ~2× più veloce, meglio su footage reale/lungo. Sopra Topaz (SECourses). FlashVSR: nodo NON installato. |
| ltx2310eros (tenstrip) | **ECCELLENTE (nicchia NSFW)** | Standard NSFW per LTX-2.3 (~1200×5★). Difetto noto: volti degradano → esiste "Fix Face Workflow". |

### Immagini
| Modello | Giudizio | Sintesi |
|---|---|---|
| FLUX.1 Dev / Krea Dev / Fill | **BUONO** | Dev superato da FLUX.2 ma ecosistema LoRA enorme; Krea Dev ottimo anti-"AI look"; Fill resta il riferimento inpaint open. |
| FLUX.2 Dev + Klein | **ECCELLENTE (pesante)** | Miglior fotorealismo open (4MP, multi-reference); 32B → su 32GB serve fp8/quant+offload. Klein 9B = compromesso pratico. |
| Z-Image / Turbo | **ECCELLENTE (qualità/costo)** | 1024² in ~2s; adozione esplosiva; sotto FLUX.2 solo su dettaglio fine/prompt complessi. Cavallo da tiro quotidiano. |
| Ernie-Image 8B | **BUONO, poco adottato** | Forte su testo-in-immagine e layout (poster/fumetti); community occidentale quasi nulla. Nicchia. |
| Krea 2 Turbo/Raw | **ECCELLENTE (estetica)** | #1 lab indipendenti; 2K in ~2s a 8 step guidance 0 (conferma i nostri parametri). Si guida con reference, non prompt lunghi. |
| getphat Reality / fuxCapacity | **BUONO/ECC. (nicchia NSFW FLUX.1)** | Longevi e 5★; getphat post-v11 su Civitai è solo softcore. Base FLUX.1 = generazione precedente. |
| GGUF LTX QuantStack | **BUONO** | Q4_K_M = punto di equilibrio (Q2/Q3 degradano); fp8-scaled resta il riferimento qualità/velocità su Blackwell. |

## Inventario e misure (DB calibrazione)
_(da completare con la tabella finale: file, GB, quant, footprint misurato, fit-32GB, SFW/NSFW, cablato-in-GUI, esito test live, raccomandazione)_

Fatti già acquisiti:
- **fp16/bf16 oltre 32GB (dannosi per il criterio)**: `ltx-2.3-22b-distilled.safetensors` bf16 43GB (44GB misurati — copia unica rimasta in Checkpoints, il distilled fp8 24GB la sostituisce integralmente); LingBot MoE 30B bf16 ~60GB (+refiner ~55GB) e la variante SDNQ uint4 ~55GB; Wan2.1-T2V-14B-Diffusers ~80GB tot (transformer misurato 15.1GB — i pesi extra sono TE/VAE duplicabili); FramePack ×2 (~48GB, Hunyuan-based, non cablato).
- **Non cablati ad alcuna arch GUI** (inutilizzabili oggi dall'app): HunyuanVideo 720 fp8 (12.6GB), FramePack I2V+F1, Wan2.1-T2V-14B-Diffusers (usato solo come base MultiTalk? da verificare).
- **Incompleti**: `seedvr2_ema_7b_fp16.safetensors.download` (9.3GB parziali; esistono 3B fp8 + 7B fp8 completi).
- LoRA: catalogo LTX ok (incl. NSFW); IC-LoRA su disco ma NON mappate nel dict (detailer, outpaint, refocus, uncompress, deinterlace, motiondeblur, lipdub, restyle, union/motion-track): candidate all'aggiunta come control_type.

## Test live (matrice) — esiti 22-23/07
Tutti i casi generano DALLA GUI (report in output/tests/REAL_VIDEO/); qualità verificata sul frame.

| Caso | Esito | Note |
|---|---|---|
| distilled fp8 (t2v, i2v, i2v@768, audio, LoRA) | ✅ 5/5 | Richiede il merge aux-bundle (checkpoint parziale ComfyUI) — implementato; 104-199s a VRAM libera; frame fotorealistici verificati |
| dev fp8 t2v/i2v | ✅ | 101-125s |
| dev nvfp4 t2v/i2v (dequant→fp8) | ✅ | 101-105s |
| GGUF Q4_K_M t2v/i2v | ✅ | 170-185s; qualità buona (spike: 3.8 s/step) |
| GGUF Q6_K t2v/i2v | ⚠️ | Genera ma ~17 MIN a clip anche a VRAM libera + puntinatura diffusa; il run t2v ha contribuito a saturare il sistema (percepito come crash PC). **Verdetto: sconsigliato — usare Q4_K_M distilled**; niente ulteriori stress-test |
| eros (aux-bundle merge) | ✅ | Risolti i NaN (checkpoint parziale: video/audio-VAE assenti venivano zero-riempiti); ricetta 32 step/cfg 1.0; transformer 28GB → percorso streaming pesante |
| extend | ✅ | 92s |
| Self-Forcing (continuo) | ✅ | 76s, il più veloce |
| LingBot dense t2v | ⚠️ risolto in parte | Due root-cause trovate e corrette: (1) default per-arch mancanti (8 step/cfg 1 ereditati da LTX → ora 40/3.0 al cambio arch), (2) rewriter proprietario che falliva IN SILENZIO tornando prosa nuda → ora fallisce forte e scatta il composer. Con caption ricca+40 step l'output è cinematografico (verificato 832×480 E 704×448). Residuo: riparare il worker del rewriter (esce a vuoto rc=0) e portare il composer Gemma a subprocess |
| FP4 nativo (nvfp4_scaled_mm) | ✅ con guardia | Motore Blackwell completo (kernel 2.9-3.4 s/it + modalità dequant). Trovato e MISURATO: fondere LoRA (inclusa la distilled) impone il requant FP4 → perso il QAT → nubi di punti (in entrambe le modalità; decodifica scale corretta, cos 0.9954). Guardia attiva: FP4 residente solo per checkpoint distilled-fp4 senza LoRA (es. FP4ME in lista); i dev-nvfp4 con LoRA vanno in dequant→fp8 (qualità piena, già verde in matrice) |
| Video corrotti segnalati dall'utente | spiegati | «Quadrati di rumore» = generazioni avvenute con la VRAM occupata da altri programmi; rigenerati puliti a VRAM libera. Aggiunta la verifica del CONTENUTO ai criteri PASS |

## Lista FP4/quant da aggiungere (ricerca online 22/07 — download SOLO dopo approvazione, connessione metered)

Ordine di valore suggerito per la 5090 (dimensioni verificate su HF):

| # | File | Repo | GB | Quant | Perché |
|---|---|---|---|---|---|
| 1 | `flux2-dev-nvfp4-mixed.safetensors` | black-forest-labs/FLUX.2-dev-NVFP4 | 22.8 | NVFP4 ufficiale | FLUX.2 32B finalmente comodo su 32GB coi kernel fp4 Blackwell (rilascio BFL+NVIDIA) |
| 1b | `mistral_3_small_flux2_fp4_mixed.safetensors` | Comfy-Org/flux2-dev | 12.3 | fp4 TE | Text-encoder Mistral in fp4 (vs 18GB fp8): TE+DiT insieme in VRAM |
| 1c | `Flux2TurboComfyv2.safetensors` | Comfy-Org/flux2-dev | 2.8 | LoRA turbo | FLUX.2 dev in 8 step |
| 2 | `LTX-2.3-22B-Distilled-FP4ME.safetensors` | MrReclusive/LTX-2.3-FP4 | 15.2 | NVFP4 community | CORREZIONE 24/07: NON difettoso (format-identico all'ufficiale; migliaia lo usano). Nel NOSTRO loader nativo perde il condizionamento testuale perché ha piu cross-attention in FP4 (230 vs 220) e richiede i nodi 'Comfy Bathroom' col controllo di precisione PER-BLOCCO che non replichiamo (loading uniforme). Il MOTORE FP4 nostro rispetta i prompt (provato sull'nvfp4 UFFICIALE: polpo+palazzo). Per il FP4-residente pulito su distilled: convertire noi il distilled-fp8 tenendo attn2 a precisione alta |
| 3 | `ltx-2.3-22b-distilled-1.1-UD-Q4_K_M.gguf` | unsloth/LTX-2.3-GGUF | 16.4 | GGUF UD | Migliore del nostro Q4_K_M piatto (dynamic quant: layer critici upcastati) |
| 4 | Wan2.2 I2V A14B `Q8_0` High+Low | QuantStack/Wan2.2-I2V-A14B-GGUF | 15.4×2 | GGUF | Upgrade Wan 2.1→2.2 (Wan 2.5+ NON ha pesi aperti); con LoRA Lightning 4-step (lightx2v, ~0.6GB) |
| 5 | HunyuanVideo **1.5** 720p I2V `Q8_0` | jayn7/HunyuanVideo-1.5_I2V_720p-GGUF | 9.0 | GGUF | Sostituisce il nostro Hunyuan 720 13B vecchia gen (meglio con 1/3 dei parametri) |
| 6 | `svdq-fp4_r128-z-image-turbo.safetensors` | nunchaku-ai/nunchaku-z-image-turbo | 4.2 | SVDQ-FP4 | Z-Image in 4-bit quasi lossless (richiede stack nunchaku; wheel Windows dev) |
| 7 | `svdq-fp4_r32-flux.1-dev/kontext/krea/fill` | nunchaku-ai/* | ~7 cad. | SVDQ-FP4 | FLUX.1 3× più veloce ≈ qualità 16-bit (kernel nunchaku, solo Blackwell) |
| 8 | `svdq-fp4_r128-qwen-image-edit-2509-lightning-8steps` | nunchaku-ai/nunchaku-qwen-image-edit-2509 | 13.1 | SVDQ-FP4+Lightning | Miglior editing per istruzione in 4-bit (modello NUOVO, non installato) |
| 9 | `svdq-fp4_r32-FLUX.2-klein-9B` | tonera/FLUX.2-klein-9B-Nunchaku | 5.8 | SVDQ-FP4 | SCARICATO ma IN ATTESA: né la stabile 1.2.1 né la nightly nunchaku hanno classi FLUX.2 (il port community gira solo via nodo ComfyUI custom) — si integra quando upstream le pubblica |

Note: NVFP4 nativo richiede torch cu128/sm_120 (ok sulla 5090; il nostro loader NVFP4 dequant-a-fp8 li carica comunque). nunchaku = motore separato (wheel+integrazione: lavoro di porting nell'app, non solo download). Wan 2.5/2.6 "open" NON esistono (solo API).
