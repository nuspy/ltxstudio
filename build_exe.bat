@echo off
REM Packaging LTX Studio in un .exe standalone (Fase 4).
REM Richiede pyinstaller nel venv: pip install pyinstaller
cd /d "%~dp0"
set PY="E:\AI\Comfy_Nexus\ComfyUI\venv\Scripts\python.exe"

%PY% -m pip install pyinstaller >nul 2>&1

%PY% -m PyInstaller --noconfirm --windowed --name LTXStudio ^
  --add-data "core/bindings.json;core" ^
  --add-data "templates;templates" ^
  --collect-submodules PySide6 ^
  app.py

echo.
echo Eseguibile in: dist\LTXStudio\LTXStudio.exe
echo NB: il catalogo LoRA (E:\AI\dictionary\data\catalog.json) e i modelli restano esterni;
echo     ComfyUI deve essere in esecuzione per generare.
pause
