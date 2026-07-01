@echo off
REM Auto-installazione dello skill LTX Studio per tutti gli agenti (Windows).
REM Uso:  install_skill.bat            (tutti)
REM       install_skill.bat claude_code
setlocal
set "PY=E:\AI\Comfy_Nexus.bak\ComfyUI\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
cd /d "%~dp0\.."
if "%~1"=="" (
  "%PY%" -m agents.install_skill
) else (
  "%PY%" -m agents.install_skill --agent %1
)
endlocal
