# SINTESI Review completa — VRAM, Immagini, Video/Audio, LoRA (2026-07-07)

4 reviewer paralleli (report dettagliati: REVIEW_VRAM.md, REVIEW_IMMAGINI.md, REVIEW_VIDEO_AUDIO.md, REVIEW_LORA.md).

## ✅ FIX GIÀ IMPLEMENTATI (commit di oggi)
1. **Anti-OOM video**: worker ltx2 chiama /free del server immagini prima di partire (finding ALTA).
2. **Stima GGUF corretta** (niente floor gonfiato) + fp8 ×2 + margine 1.12 + log numerici placement.
3. **Inpaint SENZA maschera → modifica INTERA foto da descrizione** (Kontext, guidance 2.5) — richiesta utente.
4. **LoRA low_vram esteso a Klein** (anti-OOM training 32GB).
5. **LoRA video SBLOCCATO**: ltx_trainer importabile (era rotto → trainer_installed()=False) via .pth zero-download.
6. (Sessione precedente) sfratto LRU quando serve spazio per modello DIVERSO; stesso modello resta cached;
   model_cpu_offload al posto di to("cuda") che sforava col T5; guidance auto per modello; segment multimask.

## ❌ FALSI POSITIVI verificati e scartati
- _PIPE_CACHE.clear() "aggressivo" nel worker video: c'è l'early-return sulla chiave → 2° video identico riusa la pipe.
- Recovery bottoni LoRA: l'except del worker emette finished_lora → l'handler riabilita la UI.
- "Chiave cache → usare stessa chiave per arch diversi": NAIVE — pipeline class diverse (FluxPipeline vs Fill);
  la soluzione vera è la condivisione componenti (vedi MAJOR-1), eventualmente via Pipe.from_pipe.

## 🔶 PROPOSTE MAJOR (attendono OK utente)
- **MAJOR-1 Component Store (ROI massimo)**: cache con refcount per T5-XXL 9.5GB + CLIP-L 1.2GB + VAE 0.5GB
  condivisi tra TUTTE le pipeline FLUX.1 (base/Fill/Kontext/Krea/getphat) + riuso pipeline via from_pipe per lo
  stesso file modello con arch diverso. Risparmio: ~11GB di I/O per switch, −30-50s per cambio task. 2-4h.
- **MAJOR-2 Cache CPU-RAM**: evict → RAM (budget es. 40GB su 128) invece di delete → ricarico 10× più veloce. 3-4h.
- **MAJOR-3 SAM2 tracking video**: completare l'orchestrazione (GUI pronta). 4h.
- **MAJOR-4 HDR/EXR export video**. 8h.
- **MAJOR-5 fp8 nativo senza upcast** per FLUX.1 base (11GB reali in VRAM invece di 22): il 5090 ha tensor-core fp8.

## 🔧 MEDIUM rimanenti (minori, si possono fare senza discussione)
- GUI video: esporre mask_dilate/mask_blur (inpaint video) e conditioning_attention_strength (ControlNet).
- Modalità Fast/Pro/HQ esplicita nel tab Video (preset steps+sampler).
- Log visibile quando negative_prompt è ignorato dall'arch (Z-Image) e quando la guidance viene clampata.
- Preview sample images nel training LoRA + log non troncato a 160 char.
- Gating feature_matrix in GUI per cap.lora_* (upsell come le altre feature).
- Auto-registrazione LoRA in DB a fine training.
- Bottone "maschera foto" nel Componi senza callback (implementare o nascondere).
- JoyCaption: documentare/installare per caption VLM (venv15 assente).
- Tooltip: fidelity restauro volti, audio_scale V2V, strength→denoise mapping inpaint.

## Salute complessiva
- Video/Audio: 🟢 92% — solido, gap solo su tracking SAM2, HDR/EXR, preview visiva.
- Immagini: 🟢 buona — gap risolti oggi (no-mask, guidance); restano doc/tooltip e ControlNet immagini (assente by design?).
- VRAM: 🟡 → 🟢 dopo i fix di oggi; il salto vero è MAJOR-1+2 (obiettivo dichiarato dell'utente).
- LoRA: 🟡 — sbloccato il trainer video; restano resume/checkpoint, sample preview, gating GUI.
