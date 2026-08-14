[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Jar,

    [string]$Out = (Join-Path $PWD 'mc-1.16.1-extracted'),

    [switch]$AllMinecraftAssets
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

Add-Type -AssemblyName System.IO.Compression.FileSystem

$jarPath = [IO.Path]::GetFullPath($Jar)
$outPath = [IO.Path]::GetFullPath($Out)
if (-not (Test-Path -LiteralPath $jarPath -PathType Leaf)) {
    throw "Jar not found: $jarPath"
}

$prefixes = if ($AllMinecraftAssets) {
    @('assets/minecraft/', 'data/minecraft/')
} else {
    @(
        'data/minecraft/structure/',
        'data/minecraft/structures/',
        'data/minecraft/worldgen/',
        'assets/minecraft/blockstates/',
        'assets/minecraft/models/block/',
        'assets/minecraft/textures/block/',
        'assets/minecraft/textures/colormap/'
    )
}

if (Test-Path $outPath) { Remove-Item $outPath -Recurse -Force }
New-Item $outPath -ItemType Directory -Force | Out-Null

$zip = [IO.Compression.ZipFile]::OpenRead($jarPath)
try {
    $copied = 0
    foreach ($entry in $zip.Entries) {
        if ([string]::IsNullOrEmpty($entry.Name)) { continue }
        $name = $entry.FullName.Replace('\', '/')
        $match = $false
        foreach ($prefix in $prefixes) {
            if ($name.StartsWith($prefix, [StringComparison]::Ordinal)) { $match = $true; break }
        }
        if (-not $match) { continue }

        $dest = Join-Path $outPath ($name.Replace('/', [IO.Path]::DirectorySeparatorChar))
        $parent = Split-Path $dest -Parent
        if (-not (Test-Path $parent)) { New-Item $parent -ItemType Directory -Force | Out-Null }
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
        $copied++
    }
}
finally {
    $zip.Dispose()
}

$manifest = Join-Path $outPath 'MANIFEST.sha256.tsv'
$rows = foreach ($f in Get-ChildItem $outPath -Recurse -File | Where-Object { $_.FullName -ne $manifest } | Sort-Object FullName) {
    $rel = $f.FullName.Substring($outPath.Length).TrimStart('\','/').Replace('\','/')
    $sha = (Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$sha`t$($f.Length)`t$rel"
}
@("sha256`tsize_bytes`tpath") + $rows | Set-Content -LiteralPath $manifest -Encoding UTF8

$structureFiles = @(Get-ChildItem $outPath -Recurse -File -Filter '*.nbt' | Where-Object {
    $_.FullName.Replace('\','/') -match '/data/minecraft/structures?/'
})

Write-Host "Extracted $copied files to $outPath" -ForegroundColor Green
Write-Host "Structure NBT files: $($structureFiles.Count)"
Write-Host "Manifest: $manifest"

# Print the actual path convention found in this jar. This is intentionally
# detected instead of assumed because legacy/modern jars differ on structure(s).
$roots = $structureFiles | ForEach-Object {
    $r = $_.FullName.Substring($outPath.Length).TrimStart('\','/').Replace('\','/')
    if ($r -match '^(data/minecraft/structures?)/') { $Matches[1] }
} | Sort-Object -Unique
if ($roots) { Write-Host ('Structure root(s): ' + ($roots -join ', ')) }
