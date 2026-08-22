param(
    [string[]]$Roots
)

$ErrorActionPreference = "Stop"

if (-not $Roots -or $Roots.Count -eq 0) {
    $candidateRoots = @(
        "$env:APPDATA\PrismLauncher\instances",
        "$env:APPDATA\PrismLauncher",
        "$env:LOCALAPPDATA\PrismLauncher\instances",
        "$env:USERPROFILE\AppData\Roaming\PrismLauncher\instances",
        "$env:APPDATA\.minecraft\mods"
    )
    $Roots = $candidateRoots | Where-Object { Test-Path -LiteralPath $_ }
}

if (-not $Roots) {
    Write-Host "No common Prism/Minecraft roots were found. Pass -Roots explicitly."
    exit 1
}

$results = foreach ($root in $Roots) {
    Write-Host "Scanning $root ..."
    Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Extension -eq ".jar" -and
            $_.Name -match "(?i)xaero.*world.*map|xaerosworldmap|xaeroworldmap"
        } |
        Select-Object FullName, Length, LastWriteTime
}

$results | Sort-Object FullName -Unique | Format-Table -AutoSize
