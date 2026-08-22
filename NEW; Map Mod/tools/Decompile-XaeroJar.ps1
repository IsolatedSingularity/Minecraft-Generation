param(
    [Parameter(Mandatory = $true)]
    [string]$Jar,

    [string]$OutputDirectory = ".\xaero-decompiled",
    [string]$CfrVersion = "0.152",
    [string]$ToolsDirectory = ".\.third_party_tools"
)

$ErrorActionPreference = "Stop"

$Jar = (Resolve-Path -LiteralPath $Jar).Path
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
$ToolsDirectory = [System.IO.Path]::GetFullPath($ToolsDirectory)

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
New-Item -ItemType Directory -Force -Path $ToolsDirectory | Out-Null

$java = Get-Command java -ErrorAction Stop
$cfr = Join-Path $ToolsDirectory "cfr-$CfrVersion.jar"

if (-not (Test-Path $cfr)) {
    $url = "https://repo1.maven.org/maven2/org/benf/cfr/$CfrVersion/cfr-$CfrVersion.jar"
    Write-Host "Downloading CFR $CfrVersion from Maven Central..."
    Invoke-WebRequest -Uri $url -OutFile $cfr
}

Write-Host "Decompiling locally..."
& $java.Source -jar $cfr $Jar `
    --outputdir $OutputDirectory `
    --silent true `
    --caseinsensitivefs true

Write-Host "Decompiled output:"
Write-Host "  $OutputDirectory"
