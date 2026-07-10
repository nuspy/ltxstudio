# Consolidamento venv — stato reale e piano di chiusura (#61)
_Data: 2026-07-10 · censimento STATICO (nessuna esecuzione GPU: processi pesanti attivi). Versioni lette dai
metadati pip (`pyvenv.cfg`, `torch/version.py`, `*.dist-info`), zero import._

## Censimento flotta (15 venv su disco)
| venv | Python | torch | CUDA | GB | ruolo |
|---|---|---|---|---|---|
| Comfy_Nexus.bak/ComfyUI/venv | 3.12.10 | 2.8.0 | cu128 | 11.6 | **app/GUI** |
| Tools/ltx2_venv | 3.10.11 | 2.8.0 | cu128 | 8.8 | **video LTX-2** |
| Tools/inference_server_venv | 3.12.10 | 2.11.0 | cu128 | 8.1 | **HOST immagini + 7 tool consolidati** |
| Tools/inference_server_venv_py311 | 3.11.9 | 2.11.0 | cu130 | 3.9 | base py3.11 per install_trellis (non runtime) |
| Tools/WorldGen/venv | 3.12.10 | **2.7.1** | cu128 | 7.3 | Mondo 3D (torch diverso → separato) |
| Tools/TRELLIS2/venv | 3.11.9 | 2.9.1 | cu130 | 6.2 | 3D (py+cuda diversi → separato) |
| **Tools/MMAudio/venv** | 3.10.11 | 2.11.0 | cpu | 1.6 | **DONOR** (sonify → host `ltx2_venv`, py3.10) |
| **Tools/SkinTokens/venv** | 3.12.10 | 2.11.0 | cu128 | 4.8 | **ORFANO** (rig gira sul venv ComfyUI) |
| **Tools/ACE-Step/venv** | 3.12.10 | 2.11.0 | cu128 | 6.1 | **DONOR** (acestep → host) |
| **Tools/ACE-Step/venv15** | 3.12.10 | 2.11.0 | cu128 | 6.4 | **DONOR** (acestep15 → host) |
| **Tools/F5-TTS/venv** | 3.12.10 | 2.11.0 | cu128 | 6.0 | **DONOR** (f5tts → host) |
| **Tools/heartlib/venv** | 3.12.10 | 2.11.0 | cu128 | 5.4 | **DONOR** (music/heartlib → host) |
| **Tools/XTTS/venv** | 3.12.10 | 2.11.0 | cu128 | 5.2 | **DONOR** (xtts → host) |
| **Tools/FlashVSR/venv** | 3.12.10 | 2.11.0 | cu128 | 5.0 | **DONOR** (flashvsr → host) |
| **Tools/SeedVR2/venv** | 3.12.10 | 2.11.0 | cu128 | 5.0 | **DONOR** (seedvr2 → host) |
## La consolidazione è GIÀ CABLATA nel codice
7 service selezionano l'host consolidato con fallback automatico:
```python
_CONS = Path(r"E:\AI\Tools\inference_server_venv\Scripts\python.exe")   # venv consolidato (#61)
XT_VENV_PY = _CONS if _CONS.exists() else <venv dedicato del tool>       # host se esiste, else fallback
```
Service consolidati (verificato con grep): `acestep_service`, `acestep15_service`, `f5tts_service`,
`flashvsr_service`, `music_service` (heartlib), `seedvr2_service`, `xtts_service`.

Poiché `inference_server_venv` **esiste**, questi 7 tool **già girano dall'host** — i loro venv dedicati sono
**donor inerti** (fallback che non scatta mai). `consolidate_venvs.py --dry` conferma che l'host è **superset
per nome** di TUTTI e 7 (0 pacchetti da copiare: F5-TTS, SeedVR2, ACE-Step, ACE-Step15, XTTS, heartlib, FlashVSR).

### Nota compatibilità (perché funziona con transformers 4.57 dell'host)
I venv donor pinnavano transformers diversi (4.46/4.50/**5.12**) e huggingface-hub 0.36 vs **1.21**. L'host ha
transformers **4.57.6** / hf-hub 0.36. F5-TTS e SeedVR2 sono stati **validati sul 4.57 dell'host** (commento in
`seedvr2_service.py`). ⚠️ Il consolidamento tiene la versione dell'HOST: la conferma finale che tutti e 7 girano
sul 4.57/0.36 va rifatta al live-test (per heartlib con `datasets` 5.0→3.4 e FlashVSR con transformers 4.46→4.57
il rischio è più alto — testare per primi).

Anche **MMAudio** è consolidato: `music_service.py` punta `MMAUDIO_VENV_PY` a **`ltx2_venv`** (stesso py3.10,
`_CONS_MM`), superset per nome (0 da copiare) → `MMAudio/venv` è un donor inerte. **SkinTokens/venv** invece è
**orfano**: nessun service lo referenzia (il rig gira sul venv ComfyUI).

## Cosa resta da fare (SOLO al live-test — è distruttivo)
La consolidazione runtime è completa; manca il **recupero disco**: cancellare gli 8 donor inerti (~40 GB) +
l'orfano SkinTokens (4.8 GB) = **~45 GB**.
Non eseguito ora: (a) è irreversibile, (b) va prima confermato il live-test di ciascun service dall'host,
(c) va fatto il backup NAS.

Playbook (in ordine):
1. `bash scripts/backup_venvs_to_nas.sh`  — snapshot dei donor sul NAS (ripristinabile con restore_venvs_from_nas.sh).
2. Live-test di ciascun service dall'host (voce F5/XTTS, canzone ACE-Step/heartlib, super-res SeedVR2/FlashVSR).
   Priorità ai rischiosi: **heartlib** (datasets 5.0→3.4), **FlashVSR** (transformers 4.46→4.57).
3. `python scripts/prune_donor_venvs.py --apply`  — cancella SOLO i donor il cui service usa `_CONS` e per cui
   l'host è superset (guardie integrate; default = dry-run).

## Venv che restano (giustificati, NON consolidabili)
- **app/GUI** (Comfy venv, torch 2.8) e **ltx2_venv** (py3.10, video+MMAudio): core, ABI diverse dall'host.
- **WorldGen** (torch 2.7.1), **TRELLIS2** (py3.11+cu130): Python/torch/CUDA diversi → la copia RECORD
  romperebbe l'ABI. Restano isolati.
- **inference_server_venv_py311**: interprete base py3.11 per creare i venv 3D, non un runtime.

## Bilancio
- **Runtime venv attivi**: da ~15 a **6** (app, ltx2[+MMAudio], inference_server[+7 tool], WorldGen, TRELLIS2,
  inference_server_py311). Consolidamento transformers-ecosystem **7→1** + MMAudio→ltx2 ✅ (già attivo in codice).
- **Recupero disco pendente**: **~45 GB** — 8 donor consolidati + orfano SkinTokens (gated dal live-test).
- **Playbook di cancellazione**: `scripts/prune_donor_venvs.py` (dry-run di default; `--apply` dopo backup+test).
