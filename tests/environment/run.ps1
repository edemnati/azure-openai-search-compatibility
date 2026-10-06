$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "load-env.ps1")

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Push-Location $projectRoot
try {
    python -m pytest tests\environment -m environment -q
    if ($LASTEXITCODE -ne 0) {
        throw "Environment tests failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
