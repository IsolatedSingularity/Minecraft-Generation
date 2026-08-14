[CmdletBinding()]
param(
    [string]$HostAddress = '127.0.0.1',
    [int]$Port = 5173
)

$ErrorActionPreference = 'Stop'
$viewerRoot = $PSScriptRoot

if (-not (Test-Path -LiteralPath (Join-Path $viewerRoot 'node_modules'))) {
    Push-Location $viewerRoot
    try {
        & npm.cmd install
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    finally {
        Pop-Location
    }
}

Write-Host "Seed map:      http://${HostAddress}:$Port/seed-map.html"
Write-Host "3D structures: http://${HostAddress}:$Port/local-loader.html"
Push-Location $viewerRoot
try {
    & npm.cmd run dev -- --host $HostAddress --port $Port
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
