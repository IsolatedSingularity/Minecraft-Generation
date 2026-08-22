param(
    [string]$OutputRoot = "..\outputs",
    [string]$ModernMinecraft = "1.21.11",
    [string]$ModernXaero = "1.44.2",
    [switch]$DownloadModern,
    [switch]$ClonePublicRefs
)

$ErrorActionPreference = "Stop"
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

if ($DownloadModern) {
    $artifactDir = Join-Path $OutputRoot "artifacts"
    New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null

    $artifact = "xaeroworldmap-common-$ModernMinecraft-$ModernXaero.jar"
    $base = "https://chocolateminecraft.com/maven/xaero/map/xaeroworldmap-common-$ModernMinecraft/$ModernXaero"
    $url = "$base/$artifact"
    $dest = Join-Path $artifactDir $artifact

    Write-Host "Downloading official Xaero common artifact:"
    Write-Host "  $url"
    Invoke-WebRequest -Uri $url -OutFile $dest

    foreach ($suffix in @("sha256", "sha1", "md5")) {
        try {
            Invoke-WebRequest -Uri "$url.$suffix" -OutFile "$dest.$suffix"
        } catch {
            Write-Warning "Checksum .$suffix was not downloaded: $($_.Exception.Message)"
        }
    }

    Get-FileHash -Algorithm SHA256 -LiteralPath $dest |
        Format-List |
        Out-File -Encoding utf8 (Join-Path $artifactDir "$artifact.local-sha256.txt")

    Write-Host "Saved $dest"
}

if ($ClonePublicRefs) {
    $git = Get-Command git -ErrorAction Stop
    $refs = Join-Path $OutputRoot "public_refs"
    New-Item -ItemType Directory -Force -Path $refs | Out-Null

    $repos = @(
        "https://github.com/rfresh2/XaeroPlus.git",
        "https://github.com/DanDucky/XaerosMapFormat.git",
        "https://github.com/talamus/kotlin-xaero-mapmerger.git",
        "https://github.com/billstark001/xaero-world-map-bridge.git",
        "https://github.com/RuoChennn/MapSyncer-for-XaeroWorldmap.git",
        "https://github.com/Entropy5/JMtoXaero.git",
        "https://github.com/Gjum/voxelmap-cache.git"
    )

    foreach ($repo in $repos) {
        $name = [System.IO.Path]::GetFileNameWithoutExtension($repo)
        $dest = Join-Path $refs $name
        if (Test-Path $dest) {
            Write-Host "Already present: $name"
        } else {
            Write-Host "Cloning $repo"
            & $git.Source clone --depth 1 $repo $dest
        }
    }
}
