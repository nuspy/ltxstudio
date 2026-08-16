@echo off
title LTX Studio
cd /d "%~dp0"
REM Python con PySide6 + requests + websocket-client (backend nativo). Override: set LTX_PYTHON=...
REM (il vecchio default puntava al venv ComfyUI legacy, cartella rinominata: rimosso 17/08)
set "PYEXE=%LTX_PYTHON%"
if "%PYEXE%"=="" set "PYEXE=python"
"%PYEXE%" app.py
if errorlevel 1 pause
