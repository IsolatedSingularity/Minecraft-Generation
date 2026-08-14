[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$viewerRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $viewerRoot
$testRoot = Join-Path $repoRoot '.oracle-bin\viewer-tests'

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
$vendor = Join-Path $viewerRoot 'vendor\cubiomes'
$testSource = Join-Path $viewerRoot 'tests\mc1161-smoke.c'
$output = Join-Path $testRoot 'mc1161-smoke.js'
New-Item -ItemType Directory -Path $testRoot -Force | Out-Null

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
    $testSource
) + $sources + @(
    '-I', $vendor,
    '-O2',
    '-std=c11',
    '-fwrapv',
    '-s', 'ENVIRONMENT=node',
    '-s', 'EXIT_RUNTIME=1',
    '-o', $output
)

& $emcc @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& node $output
exit $LASTEXITCODE
