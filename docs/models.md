# LTX Studio — Catalogo & Audit modelli/LoRA (esaustivo, SFW + NSFW)

> Ricerca community giugno 2026 (Civitai, Reddit, HuggingFace, forum) + audit inventario locale (`E:\AI\Models`).
> **Stato**: ✅ presente · ⬇️ consigliato da aggiungere · 🗑️ cancellare (duplicato/non valido).
> **Formati**: **FP8** (più usato) · **NVFP4/FP4** (RTX 5090, ~2-3× più veloce) · FP16/BF16 (qualità max) · GGUF (VRAM bassa).
> Struttura: CHECKPOINT → LoRA (per arch, SFW/NSFW) → DA AGGIUNGERE → AZIONI OPERATIVE.

---

# 1) CHECKPOINT

## 1.1 IMMAGINE

### FLUX.1 [dev] e fine-tune
| Stato | Modello | Peso | Fmt | Cosa fa / Validità | Decisione |
|---|---|---|---|---|---|
| ✅ | **FLUX.1 dev** (base) | 22GB | fp16/fp8/gguf | Base 12B di riferimento, aderenza prompt | **TIENI** (base di tutto) |
| ✅ | **FLUX.1 Krea dev** | 22GB | fp16 | Estetica foto-realistica, meno "look AI" | **TIENI** |
| ✅ | **FLUX_DEV_Fill** | 22GB | fp16 | Inpaint/outpaint ufficiale BFL, gold standard | **TIENI** |
| ✅ | FLUX_Kontext_Dev | 22GB | fp16 | Editing istruito | **TIENI** |
| ✅ | **getphat Reality v23 Soft / v30 Hardcore** | 11GB×2 | fp8 | Fine-tune realismo NSFW di punta (Civitai 861840, top scaricati). NB: le LoRA FLUX-Dev "bruciano" su questo ckpt | **TIENI entrambe** (top realismo NSFW) |
| ✅ | **fuxCapacity NSFW Porn v20** | 11GB | fp8 | Fine-tune hardcore esplicito (Civitai 679262, 5★/188) | **TIENI** (n.2; ridondante con getphat v30 se serve spazio) |
| ✅ | **fluxunchained-fill-hyp8 Q5_K_S** | 7.7GB | gguf | Inpaint NSFW accelerato 8-step Hyper, leggero (Civitai 645943) | **TIENI** (fill NSFW low-VRAM) |

### FLUX.2
| Stato | Modello | Peso | Fmt | Cosa fa / Validità | Decisione |
|---|---|---|---|---|---|
| ✅ | **FLUX.2 Dev FP8 Mixed Scaled** | 33GB | fp8 | DiT 32B, top prompt-adherence (in uso nativo) | **TIENI** |
| ✅ | **miraclein flux2klein NSFW v2** | 8.5GB | fp8 | Unico NSFW serio su FLUX.2 Klein (Civitai 2453960, 5★/180), gen+edit, già cablato inpaint | **TIENI** |
| ⬇️ | **FLUX.2 klein 9B fp8** (base) | ~9GB | fp8/gguf | Distillato veloce, unica via su 12-16GB, ottimo per LoRA | aggiungere |
| ⬇️ | **FLUX.2 dev nvfp4** | ~15GB | **nvfp4** | RTX 5090: 2-3× più veloce | aggiungere (5090) |

### Z-Image (Lumina 6B)
| Stato | Modello | Peso | Fmt | Cosa fa / Validità | Decisione |
|---|---|---|---|---|---|
| ✅ | **Z_Image BF16** (base) | 11.5GB | bf16 | Base 6B, #1 open-source (Tongyi), hand-built type=lumina2 | **TIENI** |
| ✅ | **Z_Image Turbo** | 11.5GB | bf16 | 8-step sub-secondo, entra in 16GB | **TIENI** (workhorse) |
| ✅ | **zimageTurbo NSFW (ZiT)** | 11.5GB | bf16 | Fine-tune NSFW fotoreale (Civitai 2237711) | **TIENI** |
| ⬇️ | Z-Image Turbo int4 | 3-4GB | svdq-int4 | gira su 6-8GB | aggiungere (fascia bassa) |

### Altre architetture immagine
| Stato | Modello | Peso | Cosa fa / Validità | Decisione |
|---|---|---|---|---|
| ✅ | **Ernie / ernie-image** (8B) | 15GB | Miglior text-rendering open (Latino+CJK), Apache-2.0, già nativo | **TIENI** (nicchia "testo in immagine") |
| ✅ | **DreamO** (adapter FLUX.1) | 456MB | Framework identity/style/try-on (ByteDance, SIGGRAPH'25) — non checkpoint | **TIENI** (usa con FLUX.1 + quality LoRA) |
| ✅ | HiDream-I1 (17B) | 9GB | Qualità SOTA ma ecosistema piccolo, superato (HiDream-O1/FLUX.2) | ⚠️ **VALUTA/CANCELLA** se serve spazio e non usato |
| ⬇️ | UltraReal / Jib Mix Flux (finetune SFW) | 11-22GB | Realismo affidabile, pelle/luce | aggiungere (SFW realismo) |
| ⬇️ | Qwen-Image base + Jib Mix Qwen | ~19GB | Ottima aderenza+testo (arch NON ancora nel backend) | aggiungere se si abilita Qwen |

## 1.2 VIDEO

### LTX-2.3 (Lightricks 22B) — motore principale
| Stato | Modello | Peso | Fmt | Cosa fa | Decisione |
|---|---|---|---|---|---|
| ✅ | **LTX-2.3 dev fp8** | 27GB | fp8 | qualità piena, entra in 32GB | **TIENI** |
| ✅ | **distilled 1.1 fp8** | 23.5GB | fp8 | veloce 8-step CFG=1 | **TIENI** |
| ✅ | GGUF Q6/Q4/Q3 | 10-17GB | gguf | fallback VRAM | **TIENI** (almeno Q4+Q6) |
| ✅ | **ltx2310eros_v1** | 27.5GB | bf16 | merge NSFW | **TIENI** (NSFW) |
| ⬇️ | **LTX-2.3 dev nvfp4** (ufficiale) | **21.7GB** | **nvfp4** | RTX 5090: ~2× fp8 in T2V (⚠️ I2V perde fedeltà) | aggiungere (5090) |
| ⬇️ | **Sulphur-2-base** (NSFW) | 9.5GB | bf16/fp8/gguf | NSFW base attivo, T2V+I2V, prompt-enhancer (Civitai 2601098) | aggiungere (alt. eros, più leggero) |
| ⬇️ | Spatial/Temporal Upscaler x2 | 0.26-1GB | bf16 | qualità+FPS finali | aggiungere |

---

# 2) LoRA — AUDIT COMPLETO

## 2.1 IMMAGINE — FLUX.1

### SFW / tecnici (TIENI)
| LoRA | Cosa fa | Validità | Decisione |
|---|---|---|---|
| **hourglassv32** (292MB) | forma corpo a clessidra | "Hourglass\|olaz" 5★/476+ | **TIENI** (top) |
| **Migration_Lora_cloth** (164MB) | virtual try-on abiti (ttplanet) | Civitai 950736 | **TIENI** |
| **figures_TTP_Migration** (328MB) | 2D→figurine 3D (ttplanet) | Civitai 966113 | **TIENI** |
| **OiledSkin** (328MB) | pelle oliata/lucida | molto diffusa | **TIENI** |
| **sxz-Dark-Fantasy** (19MB) | stile dark fantasy | Civitai 678853 | **TIENI** |
| **Zombie_Flux** (19MB) | stile horror/zombie | RalFinger | **TIENI** |
| Moody (292MB) | stile fotografico cupo | export epoch (meglio release) | **TIENI** (cautela) |
| Photoreal_illux (18MB) | enhancer realismo **Illustrious/SDXL** | — | **TIENI** solo se supporti SDXL, altrimenti 🗑️ |

### NSFW (TIENI — concept/azioni utili, leggere)
| LoRA | Cosa fa | Decisione |
|---|---|---|
| tan-lines-01(513MB), RealUnderboob(164MB), Mature_Women(18MB), SteepSlopeTiddies(64MB), micro-thong(38MB), est3tic_lingerie(38MB), blacked(19MB) | concept anatomici/abbigliamento | **TIENI** |
| NSFW_POV_Deepthroat, Nsfw_Handjob, Nsfw_Balls_Sucking (18MB), PoV_Fist(292MB, Civitai 1481174) | azioni esplicite non coperte dal base | **TIENI** (niche) |
| Sexy_Festival_Goers(584MB) | scene festival sexy | **TIENI con riserva** (pesante) |
| lips2(18MB) | labbra carnose | ⚠️ bassa priorità (prompt-replaceable) |

### 🗑️ CANCELLA (FLUX.1, recupero ~11GB)
| LoRA | Perché |
|---|---|
| **Serie "FLUX Reality"** (almu/apho/blle/caza/dvel/jpav/lfal/lmil/mawi/mnra/mohr/ricc/sgmk/vmor/yanj Soft+Hard/ysch — ~10GB) | LoRA-persona di **individui sconosciuti**, nessun valore per prodotto generico |
| MarynAI, OlgAI, Olya (16-18MB) | persone sconosciute, non recensite |
| FluxCreative (584MB) | non verificabile online, origine ignota |
| MysticXXX-V3 (149MB) | reputazione non confermabile |
| underland-alice (38MB) | stile fuori tema |

### Acceleratore FLUX.2
| ✅ **Flux_2_Turbo_LoRA** (2.6GB) | distill DMD2 → 8 step (~6×), qualità ~invariata (fal ufficiale). **TIENI** ⚠️ licenza FLUX non-commerciale |

## 2.2 IMMAGINE — altre arch
| LoRA | arch | Cosa fa | Decisione |
|---|---|---|---|
| ✅ **Qwen-Image-Lightning 8-step** (1.6GB) | Qwen | acceleratore 12-25× (lightx2v) | **TIENI** se usi Qwen |
| ✅ Qwen_Real_Nud3s (281MB) | Qwen | NSFW fotoreale | TIENI se NSFW Qwen, altrimenti 🗑️ |
| ✅ **ZITnsfwLoRAv3** (14MB) | Z-Image | NSFW Z-Image Turbo (v3, 14/10 sostituisce la v2) | **TIENI** (costo zero) |
| ✅ **DreamO quality_lora pos/neg** | FLUX.1 | anti-plastic (coppia) | **TIENI** (necessarie a DreamO) |
| ✅ **MagicQuill** (color/edge/local/puzzle/removal) | editing | editing interattivo locale/removal (SOTA) | **TIENI** (set, insieme) |
| 🗑️ **CineStyle5, MoXinV1, blindbox** | SD1.5 | stili SD1.5 legacy | **CANCELLA** (fuori stack) |
| ⚠️ intrinsic_lora (albedo/depth/normal/shading) | SD1.5 | mappe inverse-render | TIENI solo se 3D le usa, altrimenti 🗑️ |
| 🔧 **gemma-3-12b abliterated_lora** (599MB) | **LTX-2.3 TE** (mal-classificato SDXL) | uncensor del text-encoder Gemma (no "Barbie&Ken" su NSFW) | **TIENI** + riclassifica |
| ⚠️ NSFW_ZI_V01(871MB), DR34ML4Y_LTXXX(1.1GB) | da verificare (hash) | etichette SDXL sospette (probabile Z-Image o LoRA video LTX-2.3) | **VERIFICA hash** (skill ai-catalog) |

## 2.3 VIDEO — LTX-2.3 (quasi tutte TIENI — infrastruttura)
| Gruppo | LoRA | Decisione |
|---|---|---|
| **Controllo ufficiali** | IC-LoRA canny/depth/pose/union/motion-track/detailer/hdr/uncompress/deinterlace/motiondeblur/refocus/outpaint/lipdub, CameraControls, dolly/static, Cameraman | **TIENI tutte** (controllo/qualità) |
| **Acceleratori** | distilled-1.1 rank 111/159/208, distilled-dynamic | **TIENI** (rank384 solo per t2v puro, può nuocere con input condizionati) |
| **Stile/qualità** | ReStyle IC-LoRA, Crisp_Enhance, Fantasy_Realism, edit_anything, inpaint_masked, anime2real/real2anime | **TIENI** |
| **Identità** | ID-LoRA-TalkVid-3K | **TIENI**; CelebVHQ-3K ridondante → 🗑️ se serve spazio |
| **NSFW** | Better Female Nudity v2, NSFWMOTION, Penile_Praxis_V4, ltxdeepthroat | **TIENI** se ramo NSFW attivo (molto sovrapposte → tienine poche) |
| 🗑️ | demopose (doppione di pose-control), fightV3 (niche) | CANCELLA se non usate |

## 2.4 VIDEO — altri motori (LoRA)
| LoRA | Decisione |
|---|---|
| LongCat cfg_step/refinement (5.3GB) | 🗑️ se non usi LongCat (il motore è LTX-2.3) |

---

# 3) DA AGGIUNGERE (consigliati, SFW + NSFW)

## Priorità 1 — NVFP4/FP4 (RTX 5090, massimo boost velocità)
| Modello | Peso | Perché |
|---|---|---|
| **LTX-2.3 dev nvfp4** (Lightricks ufficiale) | 21.7GB | ~2× fp8 in T2V su 5090 |
| **Nunchaku/SVDQuant FLUX.1-dev fp4** | ~6GB | ~3× più veloce di bf16, qualità ≈16-bit, ⅓ VRAM |
| **FLUX.2 dev nvfp4** | ~15GB | 2-3× su Blackwell |

## Priorità 2 — Acceleratori immagine
| Modello | Peso | Perché |
|---|---|---|
| **Hyper-SD FLUX 8-step** (ByteDance) | 1.4GB | acceleratore standard (scale 0.125) |
| **aidmaNSFWunlock** | 19MB | "unlock" NSFW più affidabile (130k+ DL) |
| FLUX.1-Turbo-Alpha 8-step | ~1GB | alternativa, dettaglio sotto 8 step |

## Priorità 3 — Realismo immagine (SFW + NSFW)
| Modello | Tipo | Perché |
|---|---|---|
| UltraReal / Jib Mix Flux | checkpoint SFW | realismo affidabile |
| **Photorealistic Skin – No plastic**, Flux Skin Texture v2 | LoRA SFW | elimina effetto plastica (top community) |
| Lustly Uncensored / Flux Anatomy Realism | LoRA NSFW | anatomia M/F bilanciata |

## Priorità 4 — Video NSFW base
| Modello | Perché |
|---|---|
| **Sulphur-2-base** (LTX-2.3) | NSFW base attivo, più leggero di eros, tutti i formati |

---

# 4) AZIONI OPERATIVE (per la distribuzione)

### 🗑️ Cancellazioni consigliate (recupero ~40GB+)
1. **FLUX Reality serie** (~10GB) + MarynAI/OlgAI/Olya + FluxCreative + MysticXXX + underland-alice → LoRA-persona/non-validate.
2. **SD1.5 legacy** (CineStyle5, MoXin, blindbox) se non usi SD1.5.
3. **HiDream** (9GB) se non usato (superato).
4. **LongCat** (5.3GB) se il motore resta LTX-2.3.

### 🔧 Misclassificazioni da correggere (skill `ai-catalog`)
- `gemma-3-12b abliterated_lora` → è LoRA **text-encoder LTX-2.3**, non SDXL.
- `DR34ML4Y_LTXXX`, `NSFW_ZI_V01` → verificare arch reale via hash Civitai.

### Verdetto generale
L'app ha **già ottima copertura FP8** (LTX/FLUX/Z-Image) + una **libreria LoRA enorme** (SFW+NSFW, immagini e video). I gap reali sono:
1. **FP4/NVFP4** (velocità RTX 5090) — il guadagno maggiore.
2. **Acceleratori immagine FLUX** (Hyper-SD) + **finetune realismo** (UltraReal/Jib Mix) + LoRA skin.
3. **Pulizia**: ~40GB recuperabili da duplicati/LoRA-persona inutili.
