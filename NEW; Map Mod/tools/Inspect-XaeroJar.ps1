param(
    [Parameter(Mandatory = $true)]
    [string]$Jar,

    [string]$OutputDirectory = ".\xaero-inspection",

    [switch]$ExtractAll,
    [switch]$DisassembleAll,
    [switch]$ExtractStrings
)

$ErrorActionPreference = "Stop"

$Jar = (Resolve-Path -LiteralPath $Jar).Path
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$jarTool = Get-Command jar -ErrorAction Stop
$javap = Get-Command javap -ErrorAction Stop

# Metadata
Get-FileHash -Algorithm SHA256 -LiteralPath $Jar |
    Format-List |
    Out-File -Encoding utf8 (Join-Path $OutputDirectory "sha256.txt")

Get-Item -LiteralPath $Jar |
    Format-List FullName,Length,CreationTimeUtc,LastWriteTimeUtc |
    Out-File -Encoding utf8 (Join-Path $OutputDirectory "file_metadata.txt")

# Full JAR inventory
$listPath = Join-Path $OutputDirectory "jar_entries.txt"
& $jarTool.Source tf $Jar | Tee-Object -FilePath $listPath | Out-Null

$classes = Get-Content $listPath |
    Where-Object { $_ -match "^xaero/.+\.class$" -and $_ -notmatch "module-info\.class$" } |
    ForEach-Object { $_ -replace "/", "." -replace "\.class$", "" }

$classes | Set-Content -Encoding utf8 (Join-Path $OutputDirectory "xaero_classes.txt")

# A fast targeted list
$keywords = "MapWriter|MapPixel|MapProcessor|MapRegion|MapTile|Overlay|Cave|Biome|Color|Colour|Light|Height|Texture|Cache|WorldData|RegionDetection|GuiMap|Export"
$classes |
    Where-Object { $_ -match $keywords } |
    Set-Content -Encoding utf8 (Join-Path $OutputDirectory "priority_classes.txt")

if ($ExtractAll) {
    $extract = Join-Path $OutputDirectory "jar_extract"
    New-Item -ItemType Directory -Force -Path $extract | Out-Null
    Push-Location $extract
    try {
        & $jarTool.Source xf $Jar
    } finally {
        Pop-Location
    }
}

if ($DisassembleAll) {
    $javapDir = Join-Path $OutputDirectory "javap"
    New-Item -ItemType Directory -Force -Path $javapDir | Out-Null

    $i = 0
    foreach ($className in $classes) {
        $i++
        if (($i % 100) -eq 0) {
            Write-Host "javap $i / $($classes.Count)"
        }

        $safe = $className -replace '[<>:"/\\|?*]', "_"
        $out = Join-Path $javapDir "$safe.txt"
        try {
            & $javap.Source -classpath $Jar -p -c -s -constants $className 2>&1 |
                Out-File -Encoding utf8 $out
        } catch {
            "FAILED: $className`n$($_.Exception.Message)" |
                Out-File -Encoding utf8 $out
        }
    }
}

if ($ExtractStrings) {
    $script = Join-Path $PSScriptRoot "class_strings.py"
    $python = Get-Command python -ErrorAction Stop
    & $python.Source $script $Jar (Join-Path $OutputDirectory "class_strings.jsonl")
}

Write-Host ""
Write-Host "Inspection complete:"
Write-Host "  $OutputDirectory"
Write-Host ""
Write-Host "Start with priority_classes.txt and javap output for MapWriter / MapPixel."
