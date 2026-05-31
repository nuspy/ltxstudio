@echo off
REM Avvia il server REST API di LTX Studio (richiede ComfyUI attivo su :8188)
cd /d E:\AI\ltxstudio
"E:\AI\Comfy_Nexus\ComfyUI\venv\Scripts\python.exe" -m uvicorn server.api:app --host 0.0.0.0 --port 8200
