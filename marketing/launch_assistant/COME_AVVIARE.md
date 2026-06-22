# Come avviare il Copilota di Lancio

## Modo più semplice
1. Apri PowerShell nella cartella `E:\AI\ltxstudio\marketing`
2. Scrivi:  `.\marketing.ps1`  e premi Invio.

> Se PowerShell blocca lo script, una volta sola esegui:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`  (poi conferma con S).

## Modo alternativo (sempre funziona)
Lancia direttamente con il Python che ha PySide6:
```
& "E:\AI\Comfy_Nexus\ComfyUI\venv\Scripts\python.exe" "E:\AI\ltxstudio\marketing\launch_assistant\assistant.py"
```

## Cosa fa
- Ti mostra **un solo prossimo passo** alla volta (scheda ⭐ Adesso).
- Dettagli a scomparsa, **bottoni-link** che aprono la pagina giusta, **Copia testo** pronto.
- **✨ Procedi**: fa l'azione per te dove possibile (crea cartelle, apre la pagina, copia il testo) e ti mostra il risultato da confermare.
- **🏆 Premi**: traguardi che sblocchi man mano.
- **🗺️ Piano**: la linea del tempo dei progressi e tutti i passi.
- **💬 Compagno (Hermes)**: spiega e incoraggia. Configura l'endpoint in ⚙️ Impostazioni
  (default `http://localhost:11434/v1`, modello `hermes3` — cambialo se il tuo Hermes in WSL usa altro).
- **💶 Ho fatto una vendita!**: premi questo bottone quando incassi: è una gratificazione che ti meriti.

I progressi si salvano da soli in `progress.json`. Puoi chiudere e riaprire quando vuoi: riprendi da dove eri.
