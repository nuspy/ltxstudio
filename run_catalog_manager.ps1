# LTX Studio — apre il gestore del REGISTRO MODELLI (CRUD di data/weights_catalog.json):
# voci per categoria/arch, tag d'uso + anteprima tooltip, deps, edizioni dalla feature matrix.
$py = if ($env:LTX_PYTHON) { $env:LTX_PYTHON } else { "python" }
Set-Location $PSScriptRoot
& $py tools\catalog_manager.py
if ($LASTEXITCODE -ne 0) { Read-Host "Uscita con errore ($LASTEXITCODE) — premi Invio per chiudere" }
