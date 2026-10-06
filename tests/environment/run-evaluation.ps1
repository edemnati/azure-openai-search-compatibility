$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "load-env.ps1")

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $projectRoot
try {
    python scripts\evaluate_current_vs_target.py
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
