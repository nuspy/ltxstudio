# LTX Studio — apre il BUILDER (build/build_gui.py): build per edizione × OS con manifest
# bakeato dalla feature matrix, vista funzioni × edizione, lancio editor matrix integrato.
$py = if ($env:LTX_PYTHON) { $env:LTX_PYTHON } else { "python" }
Set-Location $PSScriptRoot
& $py build\build_gui.py
if ($LASTEXITCODE -ne 0) { Read-Host "Uscita con errore ($LASTEXITCODE) — premi Invio per chiudere" }
