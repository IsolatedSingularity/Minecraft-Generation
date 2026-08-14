[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$viewerRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $viewerRoot
$generated = Join-Path $viewerRoot 'src\seed-map\generated'

$environmentCandidate = if ($env:EMSDK) {
    Join-Path $env:EMSDK 'upstream\emscripten\emcc.exe'
} else {
    $null
}
$pathCandidate = Get-Command emcc.exe -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue
$candidates = @(@(
        (Join-Path $repoRoot '.oracle-bin\emsdk\upstream\emscripten\emcc.exe')
        $environmentCandidate
        $pathCandidate
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) })

if (-not $candidates) {
    throw 'Emscripten was not found. Install emsdk or place it in .oracle-bin/emsdk.'
}

$emcc = $candidates[0]
New-Item -ItemType Directory -Path $generated -Force | Out-Null

$vendor = Join-Path $viewerRoot 'vendor\cubiomes'
$wrapper = Join-Path $viewerRoot 'cubiomes\mc1161_wasm.c'
$output = Join-Path $generated 'mc1161.js'
$sources = @(
    'finders.c',
    'generator.c',
    'layers.c',
    'biomenoise.c',
    'biomes.c',
    'noise.c',
    'util.c',
    'quadbase.c'
) | ForEach-Object { Join-Path $vendor $_ }

$arguments = @(
    $wrapper
) + $sources + @(
    '-I', $vendor,
    '-O3',
    '-std=c11',
    '-fwrapv',
    '-s', 'MODULARIZE=1',
    '-s', 'EXPORT_ES6=1',
    '-s', 'ENVIRONMENT=worker',
    '-s', 'ALLOW_MEMORY_GROWTH=1',
    '-s', 'FILESYSTEM=0',
    '-s', 'EXPORTED_FUNCTIONS=["_malloc","_free","_mc_create","_mc_destroy","_mc_biome_tile","_mc_height_tile","_mc_structures","_mc_structure_stride"]',
    '-s', 'EXPORTED_RUNTIME_METHODS=["cwrap","HEAP32","HEAPF32","HEAPU8"]',
    '-o', $output
)

& $emcc @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Built $output"
