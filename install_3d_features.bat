@echo off
title LTX Studio - Installazione feature 3D (TRELLIS.2 + SkinTokens)
cd /d "%~dp0"
REM Usa il Python del venv di ComfyUI (stesso di run_ltxstudio.bat)
"E:\AI\Comfy_Nexus\ComfyUI\venv\Scripts\python.exe" scripts\install_3d_features.py
echo.
echo Installazione terminata.
pause
