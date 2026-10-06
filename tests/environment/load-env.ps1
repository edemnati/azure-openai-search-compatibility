$ErrorActionPreference = "Stop"

$environmentFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path -LiteralPath $environmentFile)) {
    throw "Create tests\environment\.env from .env.example before running environment tests."
}

foreach ($line in Get-Content -LiteralPath $environmentFile) {
    $trimmed = $line.Trim()
    if (-not $trimmed -or $trimmed.StartsWith("#")) {
        continue
    }
    $parts = $trimmed -split "=", 2
    if ($parts.Count -ne 2) {
        throw "Invalid environment line: $line"
    }
    $name = $parts[0].Trim()
    $value = $parts[1].Trim().Trim('"').Trim("'")
    [Environment]::SetEnvironmentVariable($name, $value, "Process")
}

if ($env:AZURE_SEARCH_USE_ADMIN_KEY -eq "true") {
    if (-not $env:AZURE_SEARCH_SERVICE_NAME -or -not $env:AZURE_SEARCH_RESOURCE_GROUP) {
        throw "Set AZURE_SEARCH_SERVICE_NAME and AZURE_SEARCH_RESOURCE_GROUP."
    }
    $key = az search admin-key show `
        --service-name $env:AZURE_SEARCH_SERVICE_NAME `
        --resource-group $env:AZURE_SEARCH_RESOURCE_GROUP `
        --query primaryKey `
        --output tsv
    if (-not $key) {
        throw "Could not retrieve the Azure AI Search admin key."
    }
    [Environment]::SetEnvironmentVariable(
        "AZURE_SEARCH_API_KEY",
        $key,
        "Process"
    )
}
