# A ready-made local seed-map baseline: Cubiomes Viewer

Before rebuilding ezseed's 2D half from scratch, use **Cubiomes Viewer** as a correctness and UX baseline:

- repository: `https://github.com/Cubitect/cubiomes-viewer`
- engine: Cubiomes
- desktop GUI: Qt
- map viewer: biomes plus supported structure overlays
- seed finder: hierarchical conditions, location searches, analysis tools

The source currently contains explicit `MC_1_16_1` handling, so Java 1.16.1 is not merely being collapsed into a generic 1.16 preset.

## Why this matters

For your project, there are really two separate deliverables:

1. a seed map that tells you *where* candidate/viable structures and biomes are;
2. a 3D structure/world viewer that tells you *what the blocks/piece graph look like*.

Cubiomes Viewer already solves much of (1) locally and gives you an independent reference for validating a custom OpenLayers/WASM frontend. Ewan's project is a much stronger reference for (2).

## Recommended use

Keep Cubiomes Viewer installed as a validation oracle while agents build your own frontend:

- enter the same signed 64-bit seed;
- select Java 1.16.1;
- compare structure coordinates from your local Cubiomes worker;
- compare biome samples/tiles at fixed coordinates;
- record mismatches as bugs before touching rendering.

Do **not** use it as a block-level world oracle. Cubiomes intentionally focuses on biome/feature/structure calculations rather than complete Minecraft block terrain generation.

## Architecture shortcut

If the goal is functionality rather than copying ezseed's visual shell, the fastest robust path is:

```text
Cubiomes native worker / WASM
        |
        +--> biome tiles
        +--> structure attempts + viability
        +--> strongholds / mineshafts / slime chunks
        |
custom local map UI (OpenLayers or Canvas)
        |
click structure marker
        v
Ewan-style local structure/world renderer
        |
        +--> raw 1.16.1 templates
        +--> exact ported assembly when validated
        +--> actual generated chunk/world oracle
```

This avoids reverse-engineering site-specific UI code that contributes nothing to generation fidelity.
