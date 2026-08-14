param(
    [string]$ClientJar = "",
    [string]$Output = ""
)

$ErrorActionPreference = 'Stop'
$viewerRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $viewerRoot
$baseline = Join-Path $repoRoot 'Assets\minecraft_1_16_1\viewer\client_structure_assets.zip'
if (-not $Output) {
    $Output = Join-Path $repoRoot 'Assets\minecraft_later_versions\viewer\structure_assets.zip'
}

if (-not $ClientJar) {
    $manifest = Invoke-RestMethod 'https://piston-meta.mojang.com/mc/game/version_manifest_v2.json'
    $versionUrl = ($manifest.versions | Where-Object id -eq '1.21' | Select-Object -First 1).url
    if (-not $versionUrl) { throw 'Minecraft Java 1.21 is missing from the official manifest' }
    $version = Invoke-RestMethod $versionUrl
    $ClientJar = Join-Path ([System.IO.Path]::GetTempPath()) 'minecraft-1.21-client.jar'
    Invoke-WebRequest -Uri $version.downloads.client.url -OutFile $ClientJar
    $actualHash = (Get-FileHash -Algorithm SHA1 $ClientJar).Hash.ToLowerInvariant()
    if ($actualHash -ne $version.downloads.client.sha1) { throw 'Minecraft Java 1.21 client hash mismatch' }
}

if (-not (Test-Path -LiteralPath $ClientJar)) { throw "Client archive not found: $ClientJar" }
if (-not (Test-Path -LiteralPath $baseline)) { throw "Baseline asset bundle not found: $baseline" }

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$baselineStream = [System.IO.File]::OpenRead($baseline)
$baselineZip = [System.IO.Compression.ZipArchive]::new($baselineStream, 'Read')
$baselinePaths = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
foreach ($entry in $baselineZip.Entries) { [void]$baselinePaths.Add($entry.FullName) }

$sourceStream = [System.IO.File]::OpenRead($ClientJar)
$sourceZip = [System.IO.Compression.ZipArchive]::new($sourceStream, 'Read')
$outputDirectory = Split-Path -Parent $Output
[System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
$outputStream = [System.IO.File]::Create($Output)
$outputZip = [System.IO.Compression.ZipArchive]::new($outputStream, 'Create')

function Test-IncludedPath([string]$path) {
    if ($path -match '^data/minecraft/structure/(ancient_city|trial_chambers)/') { return $true }
    if ($path -match '^data/minecraft/worldgen/template_pool/(ancient_city|trial_chambers)/') { return $true }
    if ($path -match '^data/minecraft/worldgen/processor_list/(ancient_city|trial_chambers)') { return $true }
    if ($path -match '^data/minecraft/worldgen/structure/(ancient_city|trial_chambers)\.json$') { return $true }
    if ($path -match '^data/minecraft/trial_spawner/') { return $true }
    if ($path -match '^data/minecraft/loot_table/chests/(ancient_city|trial_chambers)') { return $true }
    if ($baselinePaths.Contains($path)) { return $false }
    return $path -match '^assets/minecraft/(blockstates/|models/block/|textures/block/|textures/colormap/|textures/entity/decorated_pot/)'
}

$copied = 0
$structureTemplates = 0
try {
    foreach ($entry in $sourceZip.Entries) {
        if (-not $entry.Name -or -not (Test-IncludedPath $entry.FullName)) { continue }
        $destination = $outputZip.CreateEntry($entry.FullName, 'Optimal')
        $input = $entry.Open()
        $outputEntry = $destination.Open()
        try { $input.CopyTo($outputEntry) } finally { $outputEntry.Dispose(); $input.Dispose() }
        $copied++
        if ($entry.FullName -match '^data/minecraft/structure/(ancient_city|trial_chambers)/.+\.nbt$') {
            $structureTemplates++
        }
    }

    $manifestEntry = $outputZip.CreateEntry('viewer/later_structure_manifest.json', 'Optimal')
    $manifestWriter = [System.IO.StreamWriter]::new($manifestEntry.Open())
    try {
        $manifestWriter.Write((@{
            sourceVersion = 'Minecraft Java 1.21'
            families = @('ancient_city', 'trial_chambers')
            structureTemplates = $structureTemplates
        } | ConvertTo-Json -Depth 3))
    } finally { $manifestWriter.Dispose() }
} finally {
    $outputZip.Dispose()
    $outputStream.Dispose()
    $sourceZip.Dispose()
    $sourceStream.Dispose()
    $baselineZip.Dispose()
    $baselineStream.Dispose()
}

Write-Host "Built $Output"
Write-Host "$structureTemplates structure templates; $copied selected source entries"
