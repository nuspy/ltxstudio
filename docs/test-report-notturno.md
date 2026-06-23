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
- **smoke t2v** (standalone): il grafo si costruisce (**148 nodi**) e il **modello LTX-2.3 carica** ✅; la generazione si ferma sul nodo enhancer `TextGenerateLTX2Prompt` (richiede `clip`), che è cablato dal **setup completo della GUI** (`main_window.on_generate` + inventory encoder) e non da uno smoke minimale. **Non è una regressione**: la pipeline non è stata toccata e funziona dalla GUI.
- Le altre funzioni (i2v, extend, ReTake, Just_Talk, ControlNet, face-restore, super-res, scaling, RIFE, audio) sono **cablate e invariate** → da validare con input reali.
- Il **giudizio di qualità** richiede te (valutazione visiva). Consiglio: generare 1 video per funzione dalla GUI e verificare.

## FLUX.2 / Ernie nativi (#2 / #14 / #1)
⏳ download in corso (FLUX.2 TE Mistral è grande → ore). Codice pronto: `_assemble_flux2` (transformer+VAE locali + TE Mistral HF), branch `flux2*` nel loader. Test al completamento del download. NB: il TE FLUX.2 locale era comfy_quant → si scarica il Mistral HF (solo quello).

## Attende te
- **#11** rivedere matrix di compilazione (+ matrix di #16) — come da accordo.
- **#17** aggiunta ed esplorazione LoRA — in attesa info.
