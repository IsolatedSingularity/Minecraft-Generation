# Xaero's World Map Rendering Research Pack

Purpose: a source-backed research package for reproducing useful Xaero's World Map behavior in a version-independent web seed viewer, with modern Minecraft as the behavioral reference and adapters for Minecraft 1.16.1 and later versions.

## Bottom line

The screenshot is a very strong match for **Xaero's World Map**. The most distinctive evidence is the combination of:

- fullscreen world map layout;
- coordinate + biome text at the top;
- compass and vertical toolbar on the right;
- bottom-center zoom readout;
- **`Top Y`** cave control;
- Nether rendering with the roof hidden and internal cave surfaces visible.

Xaero's own documentation describes "Cave Mode Top Y", Full and Layered cave modes, and "Legible Cave Maps", which matches the screenshot's behavior.

## What this pack contains

This bundle does **not** redistribute Xaero's proprietary JAR or bulk decompiled Xaero source. Instead it contains:

1. A technical architecture reconstructed from public documentation and public interoperability/add-on projects.
2. A map-save format summary and region hierarchy.
3. A cave-mode and pixel-rendering model.
4. A clean implementation specification for an online seed viewer.
5. A version bridge: modern behavior as the target, Minecraft 1.16.1 as one world-generation adapter.
6. PowerShell/Python tools that can inspect your **licensed local Xaero JARs** much more deeply:
   - hash and inventory the JAR;
   - extract resources;
   - disassemble every Xaero class with `javap`;
   - optionally decompile locally with CFR;
   - extract printable class strings;
   - build a method/class index;
   - compare two Xaero versions;
   - clone the best public interoperability projects.
7. A source manifest with official pages, Maven coordinates, historical 1.16.1 releases and third-party reverse-engineering references.
8. The screenshot you supplied as reference evidence.

## Recommended path

On Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass

cd .\tools

# 1. Acquire the current common JAR from Xaero's official Maven and clone public references.
.\Acquire-XaeroResearchInputs.ps1 -OutputRoot ..\outputs -DownloadModern -ClonePublicRefs

# 2. If you have a licensed historical/current JAR locally, inspect it:
.\Inspect-XaeroJar.ps1 `
    -Jar "C:\path\to\your\xaeroworldmap.jar" `
    -OutputDirectory "..\outputs\inspection-modern" `
    -ExtractAll `
    -DisassembleAll `
    -ExtractStrings

# 3. Optional: decompile the local JAR with CFR.
.\Decompile-XaeroJar.ps1 `
    -Jar "C:\path\to\your\xaeroworldmap.jar" `
    -OutputDirectory "..\outputs\decompiled-modern"

# 4. Build a searchable JSONL method index from javap output.
python .\index_javap.py "..\outputs\inspection-modern\javap" `
    "..\outputs\inspection-modern\class_methods.jsonl"

# 5. Repeat for the 1.16.1 JAR, then compare.
python .\compare_inspections.py `
    "..\outputs\inspection-modern" `
    "..\outputs\inspection-1.16.1" `
    "..\outputs\comparison-modern-vs-1.16.1.md"
```

If you have the Minecraft 1.16.1 Xaero JAR in a Prism instance, use:

```powershell
.\Find-XaeroJars.ps1
```

and it will scan common Prism Launcher locations.

## Where to start reading

1. `architecture/rendering_pipeline.md`
2. `architecture/cave_modes.md`
3. `seed_viewer/implementation_spec.md`
4. `seed_viewer/parity_tests.md`
5. `notes/what_to_inspect_in_the_jar.md`

## Important distinction

There are three levels of confidence in this pack:

- **Observed/documented**: stated by Xaero or directly exposed by a public integration surface.
- **Reverse-engineered by public tools**: independently decoded file layouts or class interactions.
- **Inferred**: a likely reconstruction of internal sequencing. These items are marked and should be confirmed against your licensed JAR before attempting exact pixel parity.

For your seed viewer, exact compatibility with Xaero's private storage format is not necessary unless you want import/export. The useful part is the rendering behavior: block-column selection, transparency, biome tint, height/slope shading, cave layer selection, and map tiling.
