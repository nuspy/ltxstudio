# Valutazioni upgrade — #103 / #104 / #105

Analisi (senza test live: GPU occupata) dei componenti emersi dallo studio di **SECourses ComfyUI Auto 2.9**.
Stack attuale: **Windows, Python 3.12, torch 2.8 (runtime LTX-2) / 2.11 (inference_server), CUDA 12.8, RTX 5090 (sm_120)**.

---

## #103 — Migrazione Torch 2.9/2.11 + CUDA 13.1 + Python 3.11 [MAGGIORE]

**Cosa sbloccherebbe**: i wheel pre-compilati SECourses (flash_attn 2.8.4 sm120, xformers 0.35, cumesh) + kernel
Blackwell più recenti (potenziale +10–20% su alcuni path).

**Perché è rischioso**: i wheel sono **cp311 + cu131**; noi siamo **py312 + cu128**. Migrare significa **ricostruire
tutti i venv isolati** (LTX-2, inference_server, ltx2, venv15/ACE-Step, ai-toolkit, worldgen, voce…) su Python 3.11
+ CUDA 13.1, riscaricando torch/deps (**connessione metered**) e ri-testando OGNI modulo. Le nostre wheel attuali
(SageAttention, torchcodec/FFmpeg [[torchcodec-ffmpeg-windows]]) sono tarate su cu128 → andrebbero ri-trovate per cu131.

**Reversibilità (predisposta)**: codice = branch git; venv = **nuovi a fianco** (non tocco i vecchi) + **backup NAS**
(`scripts/backup_venvs_to_nas.sh` / `restore_venvs_from_nas.sh`, 14 TB liberi).

**Raccomandazione**: **NON urgente**. flash_attn (l'unico assente) non è critico — `triton`+`sageattention`+`xformers`
coprono già l'attention. Fare la migrazione solo come progetto dedicato, prima su UN venv di prova (inference_server),
misurando il delta perf reale prima di propagarla. **Go/no-go rimandato a decisione utente + GPU libera.**

---

## #104 — TeaCache (velocità) + QuantOps INT8 (memoria)

### TeaCache — ⭐⭐⭐ CONSIGLIATO (per le modalità HQ/Pro)
Cache training-free: salta i forward del transformer quando l'embedding di timestep cambia poco (distanza L1
accumulata). Esiste **TeaCache4LTX-Video** (ali-vilab). Speedup misurati: `rel_l1_thresh` 0.03 → **1.6×**, 0.05 → **2.1×**.
- **Applicabilità nostra**: sì — monkey-patch sul forward del transformer LTX in `ltx_core` (non serve ComfyUI).
  I coefficienti di rescaling per LTX-Video sono pubblicati; per LTX **2.3** (22B) andrebbero **ri-calibrati** (poca spesa).
- **Dove conviene**: modalità **HQ/Pro a molti step (30–40)** → 1.6–2.1×. Sulla distilled 8-step il guadagno è minore
  (pochi step da cachare) → toggle opzionale, on di default solo in HQ.
- **Sforzo**: medio (hook forward + soglia + calibrazione). **Richiede test live** per tarare thresh vs qualità.

### QuantOps INT8 block-wise — ⭐ marginale
Riduzione memoria via INT8 a blocchi. **Ma abbiamo già** GGUF (Q3/Q4/Q8 dequant on-the-fly [[native-video-engine]])
+ fp8 scaled + block-swap dinamico [[dynamic-vram-tuning]]. Il delta su ciò che già facciamo è piccolo. **Bassa priorità.**

**Prossimo passo #104**: implementare TeaCache come toggle "Cache veloce (TeaCache)" nelle modalità video HQ, con
calibrazione live della soglia. QuantOps: skip (coperto).

---

## #105 — FaceID / IP-Adapter (alternativa/complemento a PuLID)

- **IP-Adapter FaceID**: identità volto per **SDXL/SD1.5**. Noi usiamo **PuLID-FLUX** (superiore su FLUX, appena
  potenziato: multi-foto/start_step/true_cfg [[face-adapter-pulid]]). Utile SOLO se aggiungessimo modelli non-FLUX.
- **IP-Adapter (generico)**: image-prompt (stile/soggetto da foto). **Overlap** con kontext + multi-reference
  [[compose-multireference]] che già abbiamo.
- **InsightFace / ReActor**: **già integrati** (#31/#68/#73, face-swap post-process).

**Raccomandazione**: **bassa priorità / skip**. Per FLUX PuLID è meglio; per stile/soggetto abbiamo kontext.
Riconsiderare solo se introduciamo una pipeline SDXL/SD1.5.

---

## Sintesi operativa
| Task | Verdetto | Azione |
|---|---|---|
| #103 migrazione CUDA13 | rischioso, non urgente | backup NAS pronto; go/no-go all'utente |
| #104 TeaCache | **fare** (HQ video 1.6–2.1×) | implementare toggle + calibrazione live |
| #104 QuantOps | coperto da GGUF/fp8 | skip |
| #105 FaceID/IP-Adapter | ridondante su FLUX | skip salvo modelli SDXL |

Fonti: [TeaCache](https://github.com/ali-vilab/TeaCache) · [TeaCache4LTX-Video](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4LTX-Video/README.md) · [paper arXiv:2411.19108](https://arxiv.org/abs/2411.19108)
