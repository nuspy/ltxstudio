# LTX Studio — avvia l'app principale (GUI).
# Python: override con $env:LTX_PYTHON, altrimenti "python" di sistema (serve PySide6).
$py = if ($env:LTX_PYTHON) { $env:LTX_PYTHON } else { "python" }
Set-Location $PSScriptRoot
& $py app.py
if ($LASTEXITCODE -ne 0) { Read-Host "Uscita con errore ($LASTEXITCODE) — premi Invio per chiudere" }
