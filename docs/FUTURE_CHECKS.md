# Controlli futuri — ricontrollare periodicamente (upstream in movimento)

Cose che OGGI (luglio 2026) non funzionano o mancano per limiti UPSTREAM, non nostri.
Ricontrollare ogni ~1-2 mesi: se la community/il vendor ha risolto, integrare.

## 1. FP4 distilled LTX-2.3 QAT ufficiale
- **Stato oggi**: esiste solo `Lightricks/LTX-2.3-nvfp4` che è un **DEV** (richiede distilled-LoRA → in FP4 la fusione dà artefatti). NON esiste un *distilled*-nvfp4 QAT ufficiale.
- **Perché conta**: sbloccherebbe il FP4-residente pulito quotidiano (~11GB, motore già pronto in `nvfp4_scaled_mm.py`).
- **Le conversioni community NON valgono** (verificato 24/07): `MrReclusive/LTX-2.3-FP4` (FP4ME) è PTQ naïve → prompt-cieco (i pesi caricano ok, cos 0.995, ma la quantizzazione post-hoc rompe la cross-attention del testo). NON fare noi una PTQ naïve del distilled-fp8: stesso difetto.
- **Da ricontrollare**: HF `Lightricks/*` per un `distilled*-nvfp4` con metadati QAT (`_quantization_metadata`/wandb, NON `converted_by`). Anche il paper NVIDIA **SOAR** (arxiv 2605.12245) — se esce un quantizzatore NVFP4 che preserva la ricostruzione, la conversione in casa tornerebbe fattibile.
- Link: https://huggingface.co/Lightricks/LTX-2.3-nvfp4 · https://huggingface.co/MrReclusive/LTX-2.3-FP4

## 2. FP4 + LoRA su LTX (issue ComfyUI aperte)
- **Stato oggi**: fondere LoRA su un checkpoint FP4 rompe il soggetto (requant FP4 = perde il QAT). Confermato dalle issue ufficiali **ComfyUI #11908** e **#12000**; workaround community = nodi "Comfy Bathroom" (curve LoRA per-blocco).
- **Nostra guardia attuale**: FP4 residente SOLO senza LoRA; coi dev (LoRA obbligatoria) → dequant→fp8.
- **Da ricontrollare**: se ComfyUI/nunchaku implementano il LoRA-su-FP4 nativo (o se "Comfy Bathroom" diventa una libreria richiamabile), replicarne la logica per-blocco → sblocca LoRA + FP4 insieme.
- Link: https://github.com/Comfy-Org/ComfyUI/issues/11908 · https://github.com/Comfy-Org/ComfyUI/issues/12000

## 3. Nunchaku FLUX.2 / Klein (classe modello assente)
- **Stato oggi**: la wheel nunchaku (stabile 1.2.1 e nightly 1.3.0dev) NON ha classi FLUX.2 — solo Flux(.1)/Qwen/Sana/ZImage. Il Klein SVDQ (`tonera/FLUX.2-klein-9B-Nunchaku`, 5.8GB già scaricato) gira solo dentro un nodo ComfyUI custom.
- **Da ricontrollare**: release nunchaku per `NunchakuFlux2*` / supporto FLUX.2. Allora il Klein svdq si integra come i FLUX.1 svdq.
- Link: https://github.com/nunchaku-ai/nunchaku/releases

## 4. Nunchaku video (Wan)
- **Stato oggi**: nunchaku è solo immagini; il supporto video (Wan) è in un feature-branch upstream, non rilasciato. Per LTX non è in roadmap.
- **Da ricontrollare**: se esce nunchaku-video, valutare per Wan; per LTX resta il nostro motore FP4 nativo.

## 5. FlashVSR (upscaler video)
- **Stato oggi**: pesi presenti ma il **nodo ComfyUI FlashVSR non è installato** → la voce dà errore. SeedVR2 copre l'upscaling.
- **Da ricontrollare**: installare il nodo/porting nativo quando serve.

## 6. Klein FLUX.2 come inpaint evoluto (deferred storico)
- Vedi memoria `flux2-klein-inpaint` — resta in attesa di classi upstream stabili.
