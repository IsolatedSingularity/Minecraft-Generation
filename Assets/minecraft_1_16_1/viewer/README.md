# Minecraft Java 1.16.1 viewer assets

`client_structure_assets.zip` is a deterministic, browser-ready subset of the
version-locked client extraction in `Game Reference/02_jar_extracted/client`.
It preserves these original paths:

- `assets/minecraft/blockstates/`
- `assets/minecraft/models/`
- `assets/minecraft/textures/`
- `data/minecraft/structures/`
- `pack.mcmeta`

This is the minimum shared base needed for the repository's public structure
viewer to render the complete client-template catalog without asking visitors
to select a local JAR. Rebuild it with:

```powershell
.\Viewer\scripts\build-1161-client-assets.ps1
```

`worldgen_registry.zip` is generated from the mapped Java 1.16.1 village,
outpost, and bastion pool declarations. It contains 128 template-pool JSON
entries, 10 assembly-start descriptors, and a provenance manifest. Rebuild it
with `npm run build:worldgen-registry` from `Viewer/`.

Do not copy the complete client JAR into the repository.

Current deterministic inventory: 6,200 entries. Current SHA-256:
`4c3cef0beef889ed97253e8a41714bc11a9f41311705861f92710700212cbe47`.
