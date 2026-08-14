[CmdletBinding()]
param(
    [string]$Target = (Join-Path $PWD 'minecraft-structure-viewer-local'),
    [switch]$KeepModernBuiltinBundles,
    [switch]$BuildOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Require-Cmd([string]$Name) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH."
    }
}

Require-Cmd git
Require-Cmd npm

$repo = 'https://github.com/ewanhowell5195/minecraft-structure-viewer.git'
$targetPath = [IO.Path]::GetFullPath($Target)

if (-not (Test-Path $targetPath)) {
    Write-Host "Cloning public reference viewer into $targetPath"
    git clone --depth 1 $repo $targetPath
    if ($LASTEXITCODE -ne 0) { throw 'git clone failed' }
}
elseif (-not (Test-Path (Join-Path $targetPath '.git'))) {
    throw "Target exists but is not a git clone: $targetPath"
}
else {
    Write-Host "Using existing clone: $targetPath"
}

Push-Location $targetPath
try {
    Write-Host 'Installing locked npm dependencies...'
    npm ci
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }

    # Remove the current runtime CDN import. The package is already a devDependency
    # in the upstream project, so Vite can bundle it into the local build.
    $lib = Join-Path $targetPath 'src\lib.js'
    $libBackup = Join-Path $targetPath 'src\lib.js.upstream-backup'
    if (-not (Test-Path $libBackup)) { Copy-Item $lib $libBackup }

    @'
import * as THREE from "three"
import * as renderer from "block-model-renderer"

let configured = false

export function loadLibrary() {
  if (!configured) {
    renderer.configure({ three: THREE })
    configured = true
  }
  return Promise.resolve(renderer)
}

export { THREE }
'@ | Set-Content -LiteralPath $lib -Encoding UTF8

    # The upstream viewer ships supplemental structures/features generated for its
    # current target version. They are useful normally, but mixing them into a
    # 1.16.1 base silently contaminates a version-locked corpus. Pure mode disables
    # them until you regenerate equivalent bundles from Minecraft 1.16.1.
    if (-not $KeepModernBuiltinBundles) {
        foreach ($name in @('builtin.zip', 'features.zip')) {
            $p = Join-Path $targetPath ("public\" + $name)
            $disabled = $p + '.disabled-for-pure-1.16.1'
            if ((Test-Path $p) -and -not (Test-Path $disabled)) {
                Move-Item $p $disabled
                Write-Host "Disabled modern supplemental bundle: $name"
            }
        }
    }

    $loaderSrc = Join-Path $PSScriptRoot 'local-loader.html'
    if (-not (Test-Path $loaderSrc)) { throw "Missing companion file: $loaderSrc" }
    Copy-Item $loaderSrc (Join-Path $targetPath 'public\local-loader.html') -Force

    Write-Host 'Building local viewer...'
    npm run build
    if ($LASTEXITCODE -ne 0) { throw 'npm run build failed' }

    Write-Host ''
    Write-Host 'Build complete.' -ForegroundColor Green
    Write-Host 'Local loader path: /local-loader.html'
    Write-Host 'Choose your actual Minecraft 1.16.1 client jar as the base file.'
    Write-Host ''

    if (-not $BuildOnly) {
        Write-Host 'Starting Vite dev server on loopback. Stop with Ctrl+C.'
        npm run dev -- --host 127.0.0.1
    }
}
finally {
    Pop-Location
}
