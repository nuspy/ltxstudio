# ⟫ AGGIORNAMENTO 29/06 — RUNTIME = ZERO COMFYUI (confermato live)

- **Avvio**: `MainWindow` chiama SOLO `_maybe_autostart_native()`; `_maybe_autostart_comfyui()` è
  definito ma **NON chiamato** (vedi commento ~riga 436). `backend_config.json` = tutto `native`.
- **Validati live (nessun ComfyUI)**: immagini (gen FLUX, inpaint/outpaint FLUX Fill, integra/Kontext,
  upscale, segment SAM2), video (t2v/i2v/a2v/ReTake/ControlNet/RIFE/SeedVR2/face-restore via
  `workers/ltx2_infer.py` + `inference_server`), 3D (TRELLIS.2), voce (F5/XTTS/CosyVoice nativi).
- **Punto A "voice via comfy" del 27/06 = RISOLTO**: voice cloning ora NATIVO (#69), non più node-based.
- **Residuo comfy** = solo fallback GUARDATO (mai eseguito in nativo) ancora istanziato in main_window
  (`ComfyDriver`, `TemplateEngine`, `*_pipeline.py`) + `inference_server/backends/video_ltx.py` (path
  diffusers video MORTO, GUI usa ltx2_infer) + ramo `tts_engine._synthesize_comfy` + `import_models comfy_compat`.
- **Raccomandazione**: la rimozione FISICA è un refactor coupled (driver+engine+~12 pipeline istanziati);
  va fatta come pass dedicato con re-test live di OGNI feature, NON la sera prima della distribuzione.
  → **#51: obiettivo runtime RAGGIUNTO**; purge fisico pianificato post-lancio.

---

# Audit dipendenze ComfyUI (27/06) — cosa è ancora legato a ComfyUI

Grep `comfy|custom_node|object_info|driver.submit|_comfy_node_present|template_engine|.api.json`:
249 occorrenze in 37 file. Classificazione per criticità nel RUNTIME NATIVO (backend_config = tutto native):

## A) DIPENDENZA ATTIVA nel runtime nativo (DA SISTEMARE) — priorità
1. **`core/tts_engine.py`** — i motori di **voice cloning** CosyVoice2/F5-TTS/XTTS-2/VibeVoice sono
   rilevati/eseguiti via **custom node ComfyUI** (`_comfy_node_present`, `_scan_comfy_nodes`). Se l'utente
   sceglie uno di questi engine, l'app dipende da ComfyUI. → **Da rendere nativi** (Handy-API cloning
   Chatterbox/ZONOS2 + cosyvoice_service venv) e/o rimuovere i node-based. Edge-tts è già nativo (pip).
   (In corso: Handy come backend + scelta GUI.)

## B) FALLBACK ComfyUI legacy (NON eseguito ora — backend native; da RIMUOVERE nel prodotto)
Questi moduli COSTRUISCONO workflow ComfyUI ma sono raggiunti SOLO se `backend.is_native(...)`=False.
Con backend_config tutto "native" NON sono nel percorso runtime, ma restano nel codice:
- `core/template_engine.py` (golden workflow ComfyUI), `core/gen_pipeline.py`, `core/integrate_pipeline.py`,
  `core/kontext_pipeline.py`, `core/upscale_pipeline.py`, `core/inpaint_pipeline.py`,
  `core/frame_generator.py`, `core/segment_pipeline.py`, `core/face_restore_pipeline.py`,
  `core/video_enhance_pipeline.py`, `core/extend_pipeline.py`.
- `ui/main_window.py`: 78 occorrenze = i rami `else` (comfy) DOPO `if backend.is_native()`. Ogni op ha già
  il ramo nativo davanti; i rami comfy sono fallback morti con backend native.
→ Azione: a migrazione validata (#43), rimuovere i rami/ moduli comfy o metterli dietro un flag legacy.

## C) MODULI ComfyUI stessi (infrastruttura legacy, da togliere a fine)
`core/comfy_driver.py`, `core/comfy_compat.py`, `core/comfy_launcher.py`, e i riferimenti in
`ui/settings_dialog.py`/`ui/tabs/advanced_tab.py`/`import_models_tab.py` (toggle/patch ComfyUI).

## D) MENZIONANO comfy ma NON lo eseguono (OK, compatibilità/lettura)
- `core/offline_inventory.py` (emula object_info da filesystem — serve PROPRIO a non usare ComfyUI),
  `core/dequant_comfy.py` (dequant pesi comfy-quant, usa venv proprio), `core/flux2_encoder.py`
  (euristica nomi), `core/controlnet_preprocess.py` (commento), `workers/ltx_comfy_native.py`,
  `scripts/*` (conversione/migrazione one-shot).

## Conclusione
- **Unica dipendenza ComfyUI ATTIVA nel runtime nativo: i motori cloning voce in `tts_engine`** (A1).
  Tutto il resto è fallback/infra legacy (B/C) non eseguito con backend native, o semplice
  lettura/compat (D).
- Perché non scoperto prima: la migrazione è stata fatta per-feature con rami `if is_native` additivi,
  lasciando i rami comfy come fallback; non era stato fatto un grep trasversale. Fatto ora.

## Task generati
- Rendere nativo il voice cloning (A1) → in corso (Handy + scelta GUI).
- Pulizia finale rami/moduli comfy legacy (B/C) dopo validazione live #43 → Task dedicato.
