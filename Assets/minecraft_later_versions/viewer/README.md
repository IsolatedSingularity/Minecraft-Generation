# Later-version structure viewer assets

`structure_assets.zip` is a deliberately narrow Minecraft Java 1.21 overlay
used only for the viewer's **Later Versions** group. It contains the canonical
Ancient City and Trial Chamber templates, their template pools and processors,
and only the modern block-rendering assets absent from the Java 1.16.1 bundle.

Rebuild it from an official Java 1.21 client archive with:

```powershell
powershell -ExecutionPolicy Bypass -File Viewer/scripts/build-later-structure-assets.ps1 -ClientJar path/to/1.21.jar
```

These two assemblies are intentionally excluded from claims about the
repository's primary Minecraft Java 1.16.1 target.
