# LTX Studio — Catalogo modelli & LoRA (consigliati, SFW + NSFW)

> Ricerca community giugno 2026 (Civitai, Reddit, HuggingFace, forum) + inventario locale.
> **Stato**: ✅ presente in `E:\AI\Models` · ⬇️ consigliato da aggiungere.
> **Priorità formati**: **FP8** (più usato) · **NVFP4/FP4** (RTX 5090, ~2-3× più veloce) · FP16/BF16 (qualità max) · GGUF (VRAM bassa).
> Organizzazione: **IMMAGINE / VIDEO → Checkpoint / LoRA → architettura → SFW / NSFW**.

---

# 🖼️ IMMAGINE — CHECKPOINT

## FLUX.1 [dev] — SFW
| Stato | Modello | Peso | Formati | Perché |
|---|---|---|---|---|
| ✅ | **FLUX.1 dev** (base) | 22GB | fp16/fp8/gguf | Base 12B di riferimento, ottima aderenza prompt |
| ✅ | **FLUX.1 Krea dev** | 22GB | fp16 | Estetica foto-realistica, meno "look AI" |
| ✅ | FLUX.1 Kontext / Fill | 22GB×2 | fp16 | Editing istruito / inpaint |
| ⬇️ | **Nunchaku SVDQuant fp4** (dev) | ~6GB | **nvfp4** | **Cruciale 5090**: ~3× più veloce di bf16, qualità ≈16-bit, ⅓ VRAM |
| ⬇️ | UltraReal Fine-Tune v4 | 11-22GB | fp8/fp16/gguf | Realismo affidabile, pelle/luce naturali |
| ⬇️ | Jib Mix Flux v12 (SRPO) | 11-22GB | fp8/fp16 | Look pulito, top per ritratti |

## FLUX.1 — NSFW
| Stato | Modello | Peso | Formati | Perché |
|---|---|---|---|---|
| ✅ | **fuxCapacity NSFW Porn** | 11GB | fp8 | NSFW esplicito già presente |
| ✅ | **getphat Reality v23/v30** (Soft/Hardcore) | 11GB×2 | fp8 | Fotoreale soft/hardcore |
| ⬇️ | Fluxed Up v10.2 | 8-23GB | fp16/gguf | "Standard" NSFW Flux community |
| ⬇️ | CyberRealistic Flux | 11-22GB | fp8/fp16/gguf | Merge fotoreale NSFW |
| 💡 | *(tendenza 2026: base dev + LoRA-unlock invece di checkpoint NSFW dedicati)* | | | |

## FLUX.2 [dev]
| Stato | Modello | Peso | Formati | Perché |
|---|---|---|---|---|
| ✅ | **FLUX.2 Dev FP8 Mixed Scaled** | 33GB | fp8 | DiT 32B, top prompt-adherence (in uso nativo) |
| ⬇️ | **FLUX.2 klein 9B fp8** | ~9GB | fp8/gguf | Distillato veloce, **unica via su 12-16GB**, ottimo per LoRA |
| ⬇️ | **FLUX.2 dev nvfp4** | ~15GB | **nvfp4** | **5090**: 2-3× più veloce (Tensor Core 5ª gen) |
| ✅ | miraclein flux2klein NSFW v2 | 8.5GB | fp8 | NSFW klein (in uso per inpaint) |

## Z-Image (Lumina 6B) — SFW + NSFW
| Stato | Modello | Peso | Formati | Perché |
|---|---|---|---|---|
| ✅ | **Z-Image BF16 / Turbo** | 11.5GB×2 | bf16 | DiT 6B, 8-step, base permissiva |
| ✅ | **zimageTurbo NSFW** | 11.5GB | bf16 | NSFW nativo (in uso) |
| ⬇️ | Z-Image Turbo int4 (low-VRAM) | 3-4GB | svdq-int4 | Gira su 6-8GB |

## Qwen-Image (DiT 20B) — *architettura NON ancora nel backend*
| Stato | Modello | Peso | Formati | Perché |
|---|---|---|---|---|
| ⬇️ | Qwen-Image base + Edit 2511 | ~19GB | fp8 | Ottima aderenza + testo; ecosistema LoRA maturo |
| ⬇️ | Jib Mix Qwen v6 | ~19GB | fp8/bf16 | Top realismo (per NSFW pieno + LoRA) |

## Ernie (DiT 8B)
| Stato | Modello | Peso | Formati | Perché |
|---|---|---|---|---|
| ✅ | **ernie-image** (base) | 15GB | bf16 | In uso nativo (text rendering ottimo) |
| ⬇️ | RedCraft / ERNIE RedMIX | varie | — | Merge NSFW (780+ review); ecosistema giovane |

---

# 🖼️ IMMAGINE — LoRA

## FLUX.1 — SFW (realismo + acceleratori)
| Stato | LoRA | Peso | Perché |
|---|---|---|---|
| ✅ | FluxCreative, Moody, Photoreal_illux | 18-584MB | stili/realismo presenti |
| ⬇️ | **Photorealistic Skin – No plastic** | ~250MB | elimina effetto plastica (top community) |
| ⬇️ | Flux Skin Texture v2 / Beauty Skin | 150-600MB | pori/texture beauty-shot |
| ⬇️ | **Hyper-SD FLUX 8-step** (ByteDance) | 1.4GB | **acceleratore** standard (scale 0.125) |
| ⬇️ | FLUX.1-Turbo-Alpha 8-step | ~1GB | acceleratore, dettaglio migliore sotto 8 step |

## FLUX.1 — NSFW
| Stato | LoRA | Peso | Perché |
|---|---|---|---|
| ✅ | NSFW_POV/Handjob, blacked, OiledSkin, RealUnderboob, tan-lines, +molte | 18-584MB | libreria NSFW già ricca |
| ⬇️ | **aidmaNSFWunlock** | 19MB | "unlock" più affidabile (130k+ DL), str. 0.5-0.7 |
| ⬇️ | Lustly Uncensored / Flux Anatomy Realism | 150-344MB | anatomia M/F bilanciata |

## FLUX.2 — LoRA *(ecosistema in crescita, meno maturo di FLUX.1)*
| Stato | LoRA | Peso | Perché |
|---|---|---|---|
| ✅ | Flux_2_Turbo_LoRA | 2.6GB | acceleratore FLUX.2 presente |
| ⬇️ | aidmaNSFWunlock (F2) + InstaPic UltraReal | 19-200MB | unlock NSFW + realismo social |

## Z-Image / Qwen — LoRA
| Stato | LoRA | Peso | Perché |
|---|---|---|---|
| ✅ | ZITnsfwLoRAv2 (Z-Image), Qwen Lightning 8-step, Qwen_Real_Nud3s | 15MB-1.6GB | NSFW + acceleratore presenti |
| ⬇️ | Realistic Snapshot v5 / Radiant Realism (Z-Image) | 50-200MB | top realismo ZIT |

---

# 🎬 VIDEO — CHECKPOINT

## LTX-2.3 (Lightricks 22B) — motore principale — SFW
| Stato | Modello | Peso | Formati | Perché |
|---|---|---|---|---|
| ✅ | **LTX-2.3 dev fp8** | 27GB | fp8 | qualità piena, entra in 32GB |
| ✅ | **LTX-2.3 distilled 1.1 fp8** | 23.5GB | fp8 | veloce (8 step CFG=1) |
| ✅ | GGUF Q6 / Q4 / Q3 | 10-17GB | gguf | fallback VRAM bassa |
| ⬇️ | **LTX-2.3 dev nvfp4** (ufficiale) | **21.7GB** | **nvfp4** | **5090**: ~2× fp8 in T2V. ⚠️ I2V perde fedeltà |
| ⬇️ | Spatial/Temporal Upscaler x2 | 0.26-1GB | bf16 | qualità+FPS finali senza ricomputo |

## LTX-2.3 — NSFW
| Stato | Modello | Peso | Formati | Perché |
|---|---|---|---|---|
| ✅ | **ltx2310eros_v1** | 27.5GB | bf16 | merge NSFW (già presente) |
| ⬇️ | **Sulphur-2-base** | bf16/fp8/gguf 9.5GB | tutti | NSFW base più attivo, T2V+I2V, prompt-enhancer |
| ⬇️ | LTX2.3-10Eros (ufficiale TenStrip) | varie | merge | miglior movimento corpo NSFW I2V |

## Wan 2.1/2.2 (extra) — SFW
| Stato | Modello | Peso | Perché |
|---|---|---|---|
| ✅ | Wan2.2 i2v/t2v/s2v high+low fp8, Wan2.1 FusionX, VACE | 13-32GB | i2v/t2v/audio-to-video alternativi |
| ⬇️ | Wan2.2 TI2V-5B fp8 | ~5GB | denso, gira su consumer, ottimo VRAM/velocità |
| 🗑️ | *duplicati kijai/SwarmUI Wan2.1* | — | ridondanti, da rimuovere |

## HunyuanVideo (extra)
| Stato | Modello | Peso | Perché |
|---|---|---|---|
| ✅ | HunyuanVideo 720 fp8 | 12.3GB | presente |
| ⬇️ | HunyuanVideo-1.5 fp8 | ~8GB | SOTA leggero 2026, più efficiente |

---

# 🎬 VIDEO — LoRA

## LTX-2.3 — acceleratori + controllo (SFW)
| Stato | LoRA | Peso | Perché |
|---|---|---|---|
| ✅ | **distilled-lora 384/dynamic**, IC-LoRA canny/depth/pose/detailer/outpaint/hdr/lipdub, camera-control, Crisp_Enhance, ID-LoRA | 0.3-7GB | **libreria LTX completissima già presente** |
| ⬇️ | distilled LoRA rank 159/208 (drbaph) | 4-5GB | acceleratore più leggero, ottimo 5090 |
| ⬇️ | IC-LoRA Union-Control (canny+depth+pose 1 file) | — | meno VRAM, 1 controllo per volta |

## LTX-2.3 — NSFW
| Stato | LoRA | Peso | Perché |
|---|---|---|---|
| ✅ | **Better Female Nudity v2, NSFWMOTION, Penile_Praxis, ltxdeepthroat** | 1-1.2GB | NSFW motion presente |
| 💡 | *ecosistema LoRA NSFW LTX limitato → assorbito nei checkpoint NSFW (Sulphur/10Eros)* | | |

## Wan — acceleratori (cruciali: 30→4-8 step)
| Stato | LoRA | Peso | Perché |
|---|---|---|---|
| ✅ | **CausVid, Self-Forcing, FusionX t2v/i2v, Seko 4-step (Wan2_2), lightx2v 4-step** | 0.2-1.1GB | **acceleratori presenti** (riducono i passi ~20×) |
| ⬇️ | LightX2V Wan2.2-Lightning Seko V1 rank64 ultimo | ~2.5GB | acceleratore 4-step di riferimento 2026 |

## Wan — NSFW + Hunyuan
| Stato | LoRA | Peso | Perché |
|---|---|---|---|
| ✅ | **~60 LoRA Wan NSFW** (cumshot, deepthroat, anal, nipple, fingering, dr34mj0b, nsfw-e14…) + Hunyuan Titfuck | 150-585MB | **libreria NSFW Wan vastissima già presente** |
| ⬇️ | WAN General NSFW (NSFW-22 H/L) | 150-300MB | "base NSFW" generalista a 2 stadi |

---

## 📌 Priorità download per la distribuzione (focus fp8 + fp4)
1. **NVFP4 per RTX 5090** (mancano tutti): LTX-2.3 dev-nvfp4 (21.7GB) · FLUX.1 Nunchaku fp4 (6GB) · FLUX.2 dev-nvfp4. → il guadagno di velocità maggiore.
2. **Acceleratori immagine**: Hyper-SD 8-step FLUX (1.4GB) · aidmaNSFWunlock (19MB).
3. **Finetune realismo immagine**: UltraReal / Jib Mix · LoRA Photorealistic Skin.
4. **NSFW video base**: Sulphur-2-base (alternativa a eros, più leggero/attivo).
5. **Pulizia**: rimuovere i duplicati Wan2.1 kijai/SwarmUI.

> Coperture: l'app ha GIÀ ottima copertura FP8 (LTX/Wan/FLUX) + libreria LoRA enorme (SFW+NSFW). Il gap principale è **FP4/NVFP4** (velocità 5090) e qualche **finetune realismo immagine** + **acceleratori FLUX**.
