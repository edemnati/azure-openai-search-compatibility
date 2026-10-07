$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "..\tests\environment\load-env.ps1")

Push-Location $PSScriptRoot
try {
    npm run test:live
}
finally {
    Pop-Location
}
