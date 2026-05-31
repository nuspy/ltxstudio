# LTX Studio — Roadmap prossimi step (ordinata)

Stato a fine sessione: app T2V funzionante (prompt, video settings, LoRA con preview/trigger,
conflitti, preview live, canvas inpaint/outpaint), modelli 1.1 installati. Mancano il wiring
dei moduli e i pezzi che dipendono da catture golden + RAM.

Legenda: 🧑 = serve azione tua (browser/ComfyUI) · 💻 = codice (lo faccio io) · ⏳ = dipende da RAM

---

## STEP 0 — Preparazione (domani mattina)
1. ⏳ Installa i **128 GB RAM**, riavvia, avvia ComfyUI (`run_nvidia.bat`).
2. 💻 Pulisci catalogo dalle entry fantasma dei file 1.0 cancellati: `py -3 E:\AI\dictionary\tools\scan.py`.
3. 🧑 (facoltativo) Verifica se esiste un **gemma abliterated** su disco (per NSFW); altrimenti scaricalo.

## STEP 1 — Smoke test T2V reale (prova che tutto gira) ⏳
- Lancia `run_ltxstudio.bat`, scrivi un prompt, premi **Genera**.
- Atteso: preview live + video in `ltxstudio/output/`. Conferma che l'OOM è risolto coi 128 GB.
- Se ok → la pipeline è validata end-to-end.

## STEP 2 — Catture golden dei moduli 🧑 (sblocca tutto il resto)
Con Pluto caricato, per ognuno: attiva il modulo → console snippet → salva in `templates/`:
- `pluto_i2v.api.json`  (Keyframe 1 / Reference image ON)
- `pluto_controlnet.api.json` (Controlnet ON + ref video)
- `pluto_audio_clone.api.json` (ID LoRA voice ON)
- `pluto_audio_user.api.json` (Custom audio ON)
- `txt2img_zimage.api.json` (workflow Z-Image turbo, Export API)

## STEP 3 — Switch modello + NSFW 💻 (non serve cattura: 5751/6402 già nel template T2V)
- Implementare nel `template_engine` il rewire STANDARD ↔ 10EROS ↔ GGUF.
- Collegare il toggle NSFW → 10EROS + gemma abliterated nel grafo.

## STEP 4 — I2V reale 💻 (dopo cattura I2V)
- Caricare `pluto_i2v.api.json` come template alternativo; collegare export del canvas
  (immagine + posizione) → `/upload/image` → nodo immagine; binding ref_strength/size.

## STEP 5 — Tab Audio/Voce 💻 (dopo catture audio)
- Tab con scelta singola (clone/native/utente) + **pulsante Registra microfono** (audio_recorder già pronto).
- Clone: upload campione/registrazione → VHS_LoadAudioUpload + ID-LoRA.

## STEP 6 — Tab ControlNet 💻 (dopo cattura controlnet)
- Tipo (canny/depth/pose) + ref video + enable. Usa Union-Control IC-LoRA + DWPose già installati.

## STEP 7 — Inpaint/outpaint reale 💻 (serve mini-pipeline)
- Collegare la maschera del canvas a FLUX_DEV_Fill/MagicQuill (inpaint) e riempimento outpaint.

## STEP 8 — FrameGenerator 💻 (dopo cattura txt2img)
- Genera il frame iniziale da prompt (Z-Image/Flux) → lo carica nel canvas.

## STEP 9 — Rifiniture 💻
- Multi-keyframe (2-8), bottone "Inserisci trigger nel prompt", preset risoluzione ÷32,
  Detailer/Upscaler/Prompt-relay come tab.

## STEP 10 — Packaging 💻
- `build_exe.bat` → `.exe`; test avvio standalone.

---

### Dipendenze chiave
- STEP 4/5/6/8 dipendono dalle catture dello STEP 2.
- STEP 1 dipende dalla RAM (STEP 0).
- STEP 3 è fattibile subito (template già contiene i loader).
