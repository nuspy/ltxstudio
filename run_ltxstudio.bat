@echo off
title LTX Studio
cd /d "%~dp0"
REM Venv con PySide6 + requests + websocket-client (backend nativo). Cambia il path se necessario.
set "PYEXE=E:\AI\Comfy_Nexus.bak\ComfyUI\venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"
"%PYEXE%" app.py
if errorlevel 1 pause
