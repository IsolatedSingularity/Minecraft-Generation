[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$viewerRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $viewerRoot
$clientRoot = Join-Path $repoRoot 'Game Reference\02_jar_extracted\client'
$output = Join-Path $repoRoot 'Assets\minecraft_1_16_1\viewer\client_structure_assets.zip'
$prefixes = @(
    'assets\minecraft\blockstates',
    'assets\minecraft\models',
    'assets\minecraft\textures',
    'data\minecraft\structures'
)

if (-not (Test-Path -LiteralPath $clientRoot)) {
    throw "Minecraft 1.16.1 extracted client was not found: $clientRoot"
}

$outputDirectory = Split-Path -Parent $output
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
$temporary = "$output.tmp"
if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }

$stream = [System.IO.File]::Open($temporary, [System.IO.FileMode]::CreateNew)
try {
    $archive = [System.IO.Compression.ZipArchive]::new(
        $stream,
        [System.IO.Compression.ZipArchiveMode]::Create,
        $false
    )
    try {
        $files = foreach ($prefix in $prefixes) {
            Get-ChildItem -LiteralPath (Join-Path $clientRoot $prefix) -Recurse -File
        }
        $packMeta = Join-Path $clientRoot 'pack.mcmeta'
        if (Test-Path -LiteralPath $packMeta) { $files = @($files) + (Get-Item -LiteralPath $packMeta) }

        foreach ($file in ($files | Sort-Object FullName)) {
            $relative = $file.FullName.Substring($clientRoot.Length + 1).Replace('\', '/')
            $entry = $archive.CreateEntry($relative, [System.IO.Compression.CompressionLevel]::Optimal)
            $entry.LastWriteTime = [DateTimeOffset]::new(1980, 1, 1, 0, 0, 0, [TimeSpan]::Zero)
            $input = [System.IO.File]::OpenRead($file.FullName)
            $destination = $entry.Open()
            try { $input.CopyTo($destination) }
            finally { $destination.Dispose(); $input.Dispose() }
        }
    } finally {
        $archive.Dispose()
    }
} finally {
    $stream.Dispose()
}

Move-Item -LiteralPath $temporary -Destination $output -Force
$hash = (Get-FileHash -LiteralPath $output -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "Built $output"
Write-Host "SHA256 $hash"
