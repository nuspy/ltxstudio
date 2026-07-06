# Review TRAINING LoRA (immagini/video + wizard) — LTX Studio (2026-07-07)

Reviewer: agente dedicato (very thorough).

## Stato per area

| Area | Stato | Note chiave |
|---|---|---|
| (A) LoRA immagini (ai-toolkit) | ⚠️ Gap + fondamenta solide | dataset PuLID/Kontext/t2i ok; caption procedurale/VLM; NO stop/cancel da GUI; NO resume; sample images non mostrate; low_vram hardcoded solo flux2-dev (Klein a rischio OOM) |
| (B) LoRA video (LTX-2.3) | ⚠️ MAI testato live | architettura ok; **trainer_installed()=False nel setup attuale** (ltx_trainer non importabile dal venv ltx2) → training BLOCCATO; multi-keyframe rotation ben fatta; no resume |
| (C) Wizard immagini | ✅ Completo | coverage best-practice (18 casi person/12 place/9 object), 144 combo outfit×sfondo; JoyCaption NON installato (venv15 assente) → caption VLM indisponibile |
| (D) Wizard video | ✅ come (B) | dataset auto-generato clip i2v + rotation; stesso blocco trainer di (B) |

## Findings prioritari
| # | File:Line | Sev | Problema |
|---|---|---|---|
| 1 | lora_image_service.py:138 | ALTA | low_vram solo per flux2-dev → Klein non quantizzato (OOM 32GB) |
| 2 | lora_video_service.py:77-81 | ALTA | trainer_installed() False → verificare install_ltx_trainer.py |
| 3 | lora_*_tab.py:154 | MEDIA | thread muore → bottoni restano disabilitati (no recovery) |
| 4 | lora_image_service.py:149 | MEDIA | log troncato a 160 char (perde errori) |
| 5 | training_service.py:162 | MEDIA | sample images generate ma non visualizzate in GUI |
| 6 | — | MEDIA | batch/lr/save_every hardcoded (no expert mode) |
| 7 | — | MEDIA | nessun resume/checkpoint |
| 8 | api/gate.py | MEDIA | gating cap.lora_* solo in API, zero GUI (no upsell) |
| 9 | — | MEDIA | JoyCaption dipendenza non installata/documentata |
| 10 | lora_studio_tab.py:60 | BASSA | LoRA non auto-registrata in DB dopo il training |

## i18n: completo (15 lingue, tlora.*/ilora.*). Feature matrix: presente in API, ASSENTE in GUI.
