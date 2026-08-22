param(
    [Parameter(Mandatory = $true)]
    [string]$InspectionRoot,

    [string]$Pattern = "cave|fullCave|topY|height|overlay|biome|light|opacity|transparent|texture|fluid"
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path -LiteralPath $InspectionRoot).Path

Get-ChildItem -LiteralPath $root -Recurse -File |
    Where-Object { $_.Extension -in @(".txt", ".java", ".jsonl") } |
    Select-String -Pattern $Pattern -CaseSensitive:$false |
    Select-Object Path, LineNumber, Line |
    Format-Table -Wrap
