# Review GESTIONE VRAM/RAM e caricamento modelli — LTX Studio (2026-07-07)

Reviewer: agente dedicato (very thorough). Obiettivo primario: minimizzare offload → massima velocità.
Obiettivo secondario: scaricare meno componenti possibile finché non cambia pipeline.

## BUG CONCRETI (prioritizzati)

| # | Bug | File:Riga | Severità | Impatto |
|---|-----|-----------|----------|---------|
| 1 | Chiave cache include arch: `f"{arch}:{model}"` → t2i e kontext/inpaint dello STESSO file = 2-3 caricamenti completi | image_flux.py:618 | 🔴 ALTA | +30-50s per switch task stesso modello |
| 2 | NESSUN coordinamento: worker video LTX-2 non chiama /free del server immagini :8189 → il modello immagine (22GB) resta in VRAM mentre il video ne chiede 33 → OOM/spill | workers/ltx2_infer.py | 🔴 ALTA | OOM, crash, spill lentissimo |
| 3 | Componenti condivisi NON riusati: T5-XXL 9.5GB + CLIP-L 1.2GB + VAE 0.5GB identici per TUTTI i FLUX.1 (base/Fill/Kontext/Krea/getphat) ma ricaricati per ogni pipeline | image_flux.py:205-248 (_assemble_flux) | 🔴 ALTA | 10-20GB I/O inutile per workflow, 5-8 min persi |
| 4 | GGUF non riconosciuto nella stima (solo "fp8" nel nome): Q4 stimato come bf16 pieno | image_flux.py:583 | 🟡 MEDIA | placement subottimale |
| 5 | Margine GPU 1.10 al limite: attivazioni grandi spillano in shared | image_flux.py:89 | 🟡 MEDIA | spill occasionale |
| 6 | _PIPE_CACHE.clear() nel worker video TROPPO aggressivo: cancella anche quando la chiave è la stessa → ricarica pipeline identica al 2° video | workers/ltx2_infer.py:417 | 🟡 MEDIA | +30-50s per video consecutivo |
| 7 | sticky flag del ModelManager mai usato | model_manager.py:74 | 🟡 MEDIA | — |
| 8 | Nessuna cache CPU-RAM dei modelli sfrattati (sempre ricarica da SSD; 128GB RAM inutilizzata) | model_manager.py:93-105 | 🟢 BASSA | ricarico 10× più lento del necessario |
| 9 | NATIVE_ACT_HEADROOM_MB citato nel docstring ma non più letto | image_flux.py:73 | 🟢 BASSA | — |
| 10 | Fallback sequential→offload silenzioso (difficile debug) | image_flux.py:601-606 | 🟢 BASSA | — |

## STIME VRAM — accuratezza
- FLUX.1 bf16: ok. fp8: ora ×2 (upcast) ok, lieve sovrastima.
- GGUF Q4: sovrastimato ~10GB (non riconosciuto → placement sbagliato).
- Overhead fisso 11000 (T5/CLIP/VAE/working); zimage 4000 (Qwen3) ok.
- Lo spill in shared-VRAM non è contabilizzato nella decisione.

## PIANO (dal reviewer, confermo l'ordine)
### Quick win (fatti subito)
1. /free del server immagini prima del worker video (10 min) — CRITICO anti-OOM
2. Chiave cache famiglia: `flux_family:{model}` per flux/kontext/inpaint (5 min)
3. Margine 1.10→1.12 + riconoscere .gguf nella stima (15 min)
4. _PIPE_CACHE: non fare clear se la chiave è identica (20 min)
5. Log numerici della decisione di placement (10 min)
### Strutturali (proposti, MAJOR — richiedono ok)
6. **Component Store con refcount** per T5/CLIP/VAE condivisi tra pipeline FLUX.1 (2-4h) — ROI massimo: −10-20GB I/O
7. Cache CPU-RAM (evict→RAM con budget, es. 40GB su 128) invece di delete (3-4h) — ricarico 10× più veloce

## Architetture valutate e scartate
- device_map balanced (accelerate): incompatibile fp8, Windows imprevedibile.
- torch_dtype=float8 nativo in diffusers: sperimentale, rischioso oggi (rivalutare per FLUX.2).
- Multi-GPU: non applicabile (single 5090).
