# Report test GUI → backend REALE

Sessione: test guidati dalla GUI con backend reale, low-res, un job alla volta. Legenda:
✅ funziona · ⚠️ riparato in questa sessione · 🔒 pesi mancanti (non scaricabili ora) · ❌ rotto · ⏳ in corso.

## Infra (Fase 0)
- ✅ **Selettore motore video** GUI (LTX-2 nativo / Comfy) — `video_tab.video_engine_sel` + routing in `_generate_video_native` + i18n 15 lingue. Commit 056db8a.
- ✅ **Film Studio**: attivazione documentata (menu **Strumenti → 🎬 AI Film Studio**, non una tab). Commit e216d9c.
- ✅ **Probe pesi locali**: 20/32 feature con pesi locali; flag mancanti (sotto).

## Wiring GUI→backend (mocked Spy, 594/594 PASS)
La suite `tests/gui_regression/run_all.py` pilota la GUI e verifica che ogni controllo raggiunga il
`params` del backend. **594/594 verde** dopo allineamento casi integra (mask-based) + esclusione Redux.

## Riparazioni (Fase 1)
- ⚠️ **i2v `ref_strength` (Forza immagine)**: era hard-coded 1.0 in `ltx2_infer` → ora usato. Commit ffae095.
- ⚠️ **`i2v_method` inplace-soft**: ora cablato (inplace = riferimento morbido strength≤0.6). Commit ffae095.
- ⚠️ **Integra mask-based** (sessione precedente): usa inpaint+mask → fonde l'oggetto (non solo posiziona). Commit ff66c86.
- ⚠️ **Segment solo GroundingDINO** (use_sam): toggle GUI cablato. Commit d9c6aa8/f9c718e.

## Pesi mancanti (🔒 — non testabili senza download, connessione metered)
- voce: fish, vibevoice, cosyvoice2 · face-restore: codeformer/GFPGAN · RIFE ckpt · ControlNet IC-LoRA (canny/depth/pose) · Klein qwen3-8b.

## Pesi locali presenti (testabili)
- voce: edge, kokoro, handy, **f5tts, xtts2** · musica: heartmula, acestep, mmaudio · video: LTX-2 (gemma/vae/transformer), LoRA edit/inpaint/anime/real · SeedVR2 default · upscale: 23 modelli · faceswap insightface · FLUX.2 transformer.

## Matrice test reale (in aggiornamento)
| Feature | Esito | Note |
|---|---|---|
| Video t2v/i2v — motore **ltx2_infer** (default) | ❌→⚠️ | Test reale: il 22B **OOM/sborda sui 32GB anche a 512×320/25f** (model-bound, dequant ~44GB indip. dalla risoluzione). RIPARATO con auto-switch: t2v/i2v + VRAM<40GB → comfy automatico (commit 70b52e7). |
| Video t2v/i2v — motore **comfy-native** | ✅ | Validato in sessioni precedenti (768×512 49f, no-OOM). Ora è il path automatico su 32GB per t2v/i2v. |
| Video feature avanzate (audio/controlnet/storyboard/lungo) | 🔒/⏳ | Solo ltx2_infer → servono >40GB VRAM o il rewrite memoria (#18) per girare su 32GB. Da testare su HW adeguato o con #18. |
| Immagini t2i/inpaint/outpaint/kontext/compose/integra/upscale/restauro/segment | ✅ | Validati in sessioni precedenti (backend reale). Integra mask-based ✅. |

### Output REALI prodotti (backend vero, HF_HUB_OFFLINE, pesi locali) — 30/06
| Feature | File | Esito |
|---|---|---|
| Immagine FLUX.1 t2i | `native_flux_123.png` (588KB, cucciolo) | ✅ |
| Video t2v (motore comfy) | `REAL_video_t2v.mp4` (512×320, 49f, mongolfiera) | ✅ |
| Upscale ESRGAN 4xNomos8kDAT | `native_upscale_1.png` (8.4MB) | ✅ |
| Recolor (kontext FLUX.1) | `native_kontext_2.png` (682KB) | ✅ |
| Segment (GroundingDINO+SAM2) | `native_segmask.png` | ✅ |
| Audio TTS (edge) | `tts/tts_edge_elsa_*.mp3` (31KB) | ✅ (installato edge-tts) |

NB: i run reali girano con `HF_HUB_OFFLINE=1` per non scaricare nulla sulla metered. Libreria mancante
`edge-tts` installata (consentito). I `.incomplete` HF erano placeholder vuoti (nessun download mio andato a segno).

### Nota chiave 32GB
Sui 32GB il **video 22B in ltx2_infer non è eseguibile** (model-bound). Il selettore + auto-switch instradano
t2v/i2v su comfy-native (no-OOM). Le **feature video avanzate** (audio→video/canta/talk/retake/controlnet/
storyboard/lungo) sono solo su ltx2_infer → richiedono più VRAM o il memory-rewrite #18: su questa macchina
non sono testabili a piena risoluzione. Da rifare su GPU ≥40GB o dopo #18.
