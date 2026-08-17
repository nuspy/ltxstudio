# LTX Studio — apre l'EDITOR della feature matrix (funzioni × edizioni, upsell, link Patreon).
# Modifica C:\Projects\feature_matrix\projects\ltxstudio\{matrix.json, ...}.
$py = if ($env:LTX_PYTHON) { $env:LTX_PYTHON } else { "python" }
Set-Location "C:\Projects\feature_matrix"
& $py -m feature_matrix.editor "C:\Projects\feature_matrix\projects\ltxstudio\project.json"
if ($LASTEXITCODE -ne 0) { Read-Host "Uscita con errore ($LASTEXITCODE) — premi Invio per chiudere" }
