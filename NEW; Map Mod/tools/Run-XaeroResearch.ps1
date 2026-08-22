param(
    [string]$ModernJar,
    [string]$HistoricalJar,
    [string]$OutputRoot = "..\outputs",
    [switch]$Decompile
)

$ErrorActionPreference = "Stop"
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

function Inspect-One([string]$jar, [string]$name) {
    if (-not $jar) { return }

    $inspection = Join-Path $OutputRoot "inspection-$name"

    & (Join-Path $PSScriptRoot "Inspect-XaeroJar.ps1") `
        -Jar $jar `
        -OutputDirectory $inspection `
        -ExtractAll `
        -DisassembleAll `
        -ExtractStrings

    $python = Get-Command python -ErrorAction Stop
    & $python.Source `
        (Join-Path $PSScriptRoot "index_javap.py") `
        (Join-Path $inspection "javap") `
        (Join-Path $inspection "class_methods.jsonl")

    if ($Decompile) {
        & (Join-Path $PSScriptRoot "Decompile-XaeroJar.ps1") `
            -Jar $jar `
            -OutputDirectory (Join-Path $OutputRoot "decompiled-$name")
    }
}

Inspect-One $ModernJar "modern"
Inspect-One $HistoricalJar "1.16.1"

if ($ModernJar -and $HistoricalJar) {
    $python = Get-Command python -ErrorAction Stop
    & $python.Source `
        (Join-Path $PSScriptRoot "compare_inspections.py") `
        (Join-Path $OutputRoot "inspection-modern") `
        (Join-Path $OutputRoot "inspection-1.16.1") `
        (Join-Path $OutputRoot "comparison-modern-vs-1.16.1.md")
}

Write-Host "Research pass complete: $OutputRoot"
