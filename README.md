# LTX Studio

Applicazione desktop (PySide6) per generare video e media con i modelli LTX, appoggiata a ComfyUI: prompt, impostazioni
video, LoRA con anteprima e parola di attivazione, anteprima dal vivo, canvas per inpaint e outpaint, e moduli aggiuntivi
(immagine in video, controllo con video di riferimento, voce clonata, 3D). Espone anche una REST API locale
(`http://127.0.0.1:8200`) e pacchetti per agenti (vedi [`agents/README.md`](agents/README.md)).

## Avvio

L'app va installata nell'ambiente Python di ComfyUI:

```
pip install -r requirements.txt
python app.py
```

oppure `run_ltxstudio.bat`. Lo script `build_exe.bat` prepara la versione impacchettata (serve PyInstaller).
Gli script `install_accelerators.bat` e `install_3d_features.bat` installano le parti facoltative.

## Cartelle principali

| Cartella | Contenuto |
| --- | --- |
| `ui`, `apps`, `core` | interfaccia e logica dell'applicazione |
| `api`, `server`, `inference_server`, `workers` | REST API locale e processi di generazione |
| `comfy_compat`, `templates` | collegamento a ComfyUI e grafi di esempio |
| `agents` | pacchetti per Hermes, OpenClaw, Claude Code, Antigravity, OpenAI Codex e client generici |
| `skills`, `tools`, `scripts` | competenze per gli agenti e strumenti di manutenzione |
| `tests` | test dell'applicazione |
| `docs`, `ROADMAP.md` | documentazione e prossimi passi |

## Note

- I modelli, i file generati (`output/`) e i media non fanno parte del repository.
- Le capacità «Cambia sfondo», «Sostituisci personaggio» e la catena di clip sono scritte da zero a partire da progetti con
  licenza MIT: vedi [`NOTICE-openjutsu.md`](NOTICE-openjutsu.md).
