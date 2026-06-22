# marketing.ps1 — avvia il Copilota di Lancio (AI Tools Forge)
# Doppio clic NON funziona per i .ps1: aprilo da PowerShell, oppure tasto destro -> "Esegui con PowerShell".
# In alternativa, lancia direttamente:  pwsh -File marketing.ps1

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$app  = Join-Path $here "launch_assistant\assistant.py"

if (-not (Test-Path $app)) {
    Write-Host "Errore: non trovo $app" -ForegroundColor Red
    exit 1
}

# Candidati Python in ordine di preferenza (il venv di ComfyUI ha già PySide6)
$candidates = @(
    "E:\AI\Comfy_Nexus\ComfyUI\venv\Scripts\python.exe",
    "E:\AI\ltxstudio\venv\Scripts\python.exe"
)

function Has-PySide6($py) {
    try {
        & $py -c "import PySide6" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}

$python = $null
foreach ($c in $candidates) {
    if ((Test-Path $c) -and (Has-PySide6 $c)) { $python = $c; break }
}
if (-not $python) {
    foreach ($name in @("python", "py")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and (Has-PySide6 $cmd.Source)) { $python = $cmd.Source; break }
    }
}

if (-not $python) {
    Write-Host "Non ho trovato un Python con PySide6." -ForegroundColor Yellow
    Write-Host "Installa con:  pip install PySide6   (consigliato anche PySide6-Addons per il browser interno)" -ForegroundColor Yellow
    exit 1
}

Write-Host "Avvio il Copilota di Lancio con: $python" -ForegroundColor Green
& $python $app
