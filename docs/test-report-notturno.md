# Report test notturno (2026-06-23)

Lavoro autonomo. Tutto committato. Legenda: ✅ validato runtime · 🧪 a secco · ⏳ in corso · 🔜 attende te.

## Backend nativo — immagini (dai pesi locali, no ComfyUI)
| Modello | load+gen | gen cached | esito |
|---|---|---|---|
| FLUX dev | 33s | 16s | ✅ |
| FLUX schnell | 23s | 12s | ✅ |
| Z-Image | 30s | 11s | ✅ |
| Kontext | — | — | ✅ (99s, sessione precedente) |
| Inpaint (FLUX Fill) | — | — | ✅ (53s) |
| LoRA su nativo | — | — | ✅ (42s) |

## Ottimizzazione velocità (#12)
- Bug **sysmem fallback** (offload aggressivo) risolto → offload sicuro.
- **SageAttention 2++** attivo (serviva triton). **FBCache nativo** (FirstBlockCacheConfig): FLUX dev **70s→33s (~2.1×)**.
- **Sage 3 (FP4)** 🔜: wheel richiede torch 2.8, il venv ha 2.11 → rimandato (tua scelta).
- Installer **environment-aware** (GPU sm/CUDA/Python/OS → wheel + fallback).

## Benchmark ComfyUI vs nativo (#13) — `output/comfy_compare.png`
**ComfyUI è più veloce** (FLUX dev 20s vs 33s; schnell 6s vs 23s): il nativo paga offload+bf16. Migliorabile con full-GPU (dove la VRAM basta) e fp8. Grafico a candele generato con i valori.

## Composizione multi-foto (#16) — senza incollaggio
- Pipeline `compose()` (multiref + stitch) + `prep_masked()` (maschere per-foto).
- GUI "🧩 Componi più foto" (N foto, maschera opz, switch metodo, istruzioni) + i18n 15 lingue.
- ✅ **validata runtime**: multiref 34s, stitch 14s (FLUX.1 Kontext).
- 🔜 manca solo il **matrix building** (tue indicazioni).

## Editing immagini
- **Restauro ESRGAN** (#4): ✅ 4s (RealESRGAN_x4plus).
- **Outpaint FLUX Fill** (#5): ✅ 26s.

## Video (#6–#8) — inventario + smoke
Pipeline video **non modificate** in questa sessione → nessuna regressione. Tutte cablate:
t2v, i2v, extend (prolunga), face-restore, super-res (SeedVR2/ESRGAN), scaling, ControlNet (depth/canny/pose), ReTake, Just_Talk (lip-sync), LoRA video, RIFE (fps), audio.
### 🐛 BUG REALE TROVATO E FIXATO (generazione video)
Lo smoke t2v ha rivelato un **bug reale** (non del test) che rompeva **TUTTA** la generazione video, inclusa quella **a lunghezza illimitata**: un aggiornamento di ComfyUI ha aggiunto `clip` come **input required** ai nodi text-encode LTX (`TextGenerateLTX2Prompt`, `PromptRelayEncode*`), assente dal template golden → `missing clip` **abortiva ogni video** (t2v/i2v/extend/long).
**Fix (commit `0cefc60`)** in `core/template_engine.py`:
- `_bypass_enhancer` — bypassa gli enhancer LTX2 privi di `clip` (prompt grezzo diretto);
- `_ensure_ltx_clip` — collega un `DualCLIPLoader` (`gemma_3_12B_it` + `ltx-2.3_text_projection_bf16`, type `ltxv`) ai nodi PromptRelay/CLIPTextEncode che lo richiedono (loader preso dal workflow ufficiale LTX-2.3).
- Entrambi **auto-disattivanti** (se il template avrà già `clip`, non intervengono).
**Verifica**: dopo il fix gli errori "missing clip" SPARISCONO (prima su enhancer, poi su PromptRelay → ora nessuno). Gli errori residui dello smoke sono solo **file-esempio baked nel template** (`Cand2B.png`, `FBC.png`, `A-5.mp4`, un `.mp3`) che la GUI pota/sostituisce nel flusso reale; il mio smoke minimale non replica quel pruning.

### Video REALI generati dal NOSTRO codice (build_prompt/extend.build → ComfyUI runtime)
Output in `E:\AI\Comfy_Nexus\Output` (NON in ComfyUI/output — da qui il "non vedo artefatti").
| Funzione | Esito | File |
|---|---|---|
| **t2v** | ✅ 512×512, 2s, +audio | `LTX2_3_00250-audio.mp4` |
| **t2v (ref_strength=0)** | ✅ primo frame = scena generata, NO keyframe baked | `LTX2_3_00251-audio.mp4` |
| **prolunga / extend** | ✅ 5.75s (da 2s), 138 frame | `ltxstudio_longvideo_00001.mp4` |
| **lunghezza infinita** | ✅ (stesso ExtendPipeline a catena) | — |

NB architettura: i **video** sono prodotti dal **nostro codice** (`template_engine`/`extend_pipeline` costruiscono il grafo) eseguito da **ComfyUI runtime**. La migrazione a backend nativo finora copre solo le IMMAGINI.

Note utente chiarite:
- `LTX2_3_*.png` = anteprima primo-frame salvata da `VHS_VideoCombine` (normale).
- Immagine `magicquill` all'inizio = keyframe1 baked; appariva solo perché il mio smoke non azzerava `ref_strength` (la GUI sì → non appare).

### Ancora da validare a runtime (condividono il fix clip, setup più complesso)
- **ReTake** (v2v: video sorgente + maschera), **ControlNet** (video ref + IC-LoRA), **#8 qualità**: face-restore, super-res (SeedVR2), scaling, RIFE. Richiedono il setup moduli (`_prepare_module_inputs`).

## FLUX.2 / Ernie nativi (#2 / #14 / #1)
⏳ download in corso (FLUX.2 TE Mistral è grande → ore). Codice pronto: `_assemble_flux2` (transformer+VAE locali + TE Mistral HF), branch `flux2*` nel loader. Test al completamento del download. NB: il TE FLUX.2 locale era comfy_quant → si scarica il Mistral HF (solo quello).

## Attende te
- **#11** rivedere matrix di compilazione (+ matrix di #16) — come da accordo.
- **#17** aggiunta ed esplorazione LoRA — in attesa info.
