# Tools

## Requirements

Minimum:

- Windows PowerShell 5.1 or PowerShell 7
- a JDK containing `jar`, `java`, `javap`
- Python 3

Optional:

- Git, for cloning public references
- internet access, for downloading the official modern Maven artifact and CFR

## Workflow

### Find local Xaero JARs

```powershell
.\Find-XaeroJars.ps1
```

### Acquire current common artifact and public references

```powershell
.\Acquire-XaeroResearchInputs.ps1 -DownloadModern -ClonePublicRefs
```

### Full local inspection

```powershell
.\Inspect-XaeroJar.ps1 `
  -Jar "C:\path\xaeroworldmap.jar" `
  -OutputDirectory "..\outputs\inspection" `
  -ExtractAll -DisassembleAll -ExtractStrings
```

### Local decompile

```powershell
.\Decompile-XaeroJar.ps1 `
  -Jar "C:\path\xaeroworldmap.jar" `
  -OutputDirectory "..\outputs\decompiled"
```

### Search

```powershell
.\search_research.ps1 `
  -InspectionRoot "..\outputs\inspection" `
  -Pattern "fullCave|caveStart|caveDepth|topY"
```

## Why both javap and decompilation?

`javap -c -p` is useful because it shows the actual JVM call/field structure even when a decompiler reconstructs awkward control flow. CFR is easier to read. Use both for important methods.
