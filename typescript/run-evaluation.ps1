$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "..\tests\environment\load-env.ps1")

Push-Location $PSScriptRoot
try {
    npm run evaluate
    if ($LASTEXITCODE -ne 0) {
        throw "TypeScript evaluation failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
