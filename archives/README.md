# Minecraft 1.16.1 Exact Local Viewer Research Bundle

Research snapshot: 2026-08-13

## Goal

Build a local Minecraft Java 1.16.1 seed + structure viewer that is source-backed and reproducible, not a visual approximation.

## Integration scope and fidelity target

This is **an additive module for the existing repository**, not a replacement for the repository's current Minecraft corpus, indexes, query tools, generation-library work, or other functionality. Agents must first inspect the existing repo and integrate this capability into the smallest sensible subtree/API boundary. Do not reorganize or take over unrelated systems.

The implementation target is **reference parity first**: reproduce the observable behavior, data flow, rendering semantics, controls, and outputs of the reference tools as closely as practical **locally and version-locked**, then layer project-specific UX or abstractions on top. Do not replace a reference behavior with a simpler approximation merely because it is easier. In particular, a Python dictionary of blocks or a cube-per-block renderer is useful for inspection/tests, but it is not a substitute for Ewan-style rendering parity.

Treat public/open source as the implementation reference where available, especially Ewan's repository and Cubiomes. For closed/private frontend code, reproduce behavior from public assets and observations rather than assuming internals. Preserve licensing/attribution boundaries and do not vendor unlicensed third-party source wholesale.

The central conclusion is that the three referenced sites solve different layers of the problem:

1. **MinecraftMaps Structure Viewer** is mostly a static NBT template inspector. It reads vanilla structure-template NBT and renders non-air blocks. This is useful, but it is not a seed/world generator.
2. **Ewan Howell's Structure Viewer** is the strongest reference for local structure rendering and assembly. It loads the vanilla client jar, parses NBT and world saves, renders real block models/textures, supports jigsaw pools and several procedural structures, and has extraction tools that run Minecraft's own server-side generation code to capture hardcoded pieces.
3. **ezseed** is a seed-map frontend. For Java seed calculations it credits **cubiomes compiled to WebAssembly**, with OpenLayers for the 2D map. It is the right reference for seed -> biome/structure-position logic, not for reconstructing every block of a naturally generated structure.

## The important exactness trap

Ewan's interactive `?seed=` is **not a Minecraft Java world seed**. Its current source uses a 32-bit Mulberry32 generator and stores an 8-hex-digit session seed. The viewer's algorithms are excellent references for piece assembly and rendering, but the interactive re-roll value is a viewer session RNG, not the 48-bit Java `Random` state derived from a 64-bit world seed.

Therefore, for an exact 1.16.1 seed viewer:

- use **cubiomes `MC_1_16_1`** for biome and structure-position calculations;
- use the **actual 1.16.1 jar** for templates, blockstates, models, textures, template pools, processor lists, and other packaged data;
- use **Ewan's architecture** as the rendering/assembly reference;
- use an **actual Minecraft 1.16.1-generated world/chunk** as the ground-truth oracle whenever terrain or full natural structure block placement must be exact.

## Recommended build order

### Phase A: exact local structure browser

This is the fastest win.

1. Clone `ewanhowell5195/minecraft-structure-viewer` locally.
2. Replace its CDN renderer adapter with the bundled npm dependency so it can run without jsDelivr at runtime.
3. Start it in `?manual` mode and provide your local `1.16.1.jar` as the base pack bytes.
4. Browse the jar's `data/minecraft/structures/...` files and render them using the jar's own blockstates/models/textures.
5. Validate NBT size/palette/block counts against your local extracted corpus.

See `docs/01_EWAN_STRUCTURE_VIEWER.md` and `scripts/bootstrap-ewan-local.ps1`.

### Phase B: exact 1.16.1 seed map

1. Build `Cubitect/cubiomes`.
2. Use `MC_1_16_1`, not the generic `MC_1_16` alias, because cubiomes distinguishes 1.16.1 and 1.16.5.
3. Compute structure attempts with `getStructurePos()` and validate with `isViableStructurePos()`.
4. Use `StrongholdIter` for strongholds and `getMineshafts()` for mineshafts.
5. Render results with any local 2D map layer. OpenLayers is what ezseed credits, but the calculation layer should remain separate from the UI.

See `docs/02_EZSEED_CUBIOMES.md` and `cubiomes/mc1161_seed_probe.c`.

### Phase C: exact naturally generated structure instances

Do **not** assume that a modern structure-assembly port plus a world seed is automatically version-exact.

For each structure family, choose one of:

- **Oracle-first:** let actual Minecraft 1.16.1 generate the relevant chunks, parse the resulting `.mca`, and render those blocks. This is the strongest correctness path.
- **Port-and-diff:** port the exact 1.16.1 generator/RNG path from your mapped/decompiled source, then differential-test it against generated chunks over many seeds.

For speedrunning-oriented seed scouting, Cubiomes positions plus a real-world oracle for selected locations is usually the best engineering tradeoff.

## Files in this bundle

- `docs/00_FINDINGS_MATRIX.md`: quick comparison of source availability, role, and exactness.
- `docs/01_EWAN_STRUCTURE_VIEWER.md`: source map, architecture, offline conversion, and 1.16.1 caveats.
- `docs/02_EZSEED_CUBIOMES.md`: what ezseed exposes, Cubiomes APIs, limitations, and recommended local seed-map layer.
- `docs/03_MINECRAFTMAPS_VIEWER.md`: the simpler NBT-template approach and where it falls short.
- `docs/04_MC_1_16_1_EXACT_ARCHITECTURE.md`: exactness tiers, pipeline, data model, and differential tests.
- `docs/05_CUBIOMES_VIEWER_READYMADE.md`: ready-made local 1.16.1 seed-map baseline and validation oracle.
- `docs/06_SECOND_PASS_CAVEATS.md`: Gemini-note triage, corrections, remaining blind spots, and what is/isn't sufficient for 1:1 parity.
- `AGENT_HANDOFF.md`: implementation instructions for local coding agents.
- `scripts/bootstrap-ewan-local.ps1`: clones the public Ewan repo, installs dependencies, removes the runtime jsDelivr dependency, adds a local-jar loader, builds, and optionally starts Vite.
- `scripts/local-loader.html`: browser wrapper that feeds your own jar/pack bytes through Ewan's documented embed API.
- `scripts/extract-mc1161-assets.ps1`: deterministic extractor + SHA-256 manifest for relevant data/assets from your 1.16.1 jar.
- `cubiomes/mc1161_seed_probe.c`: small Cubiomes-based JSONL structure-position probe for Java 1.16.1.
- `SOURCES.md`: primary/reference source inventory.

## Licensing note

At the research snapshot, GitHub reports `license: null` for `ewanhowell5195/minecraft-structure-viewer`. The repository is public, but that is not the same as an explicit redistribution license. This bundle therefore does **not** copy the repository source. The bootstrap script clones it from GitHub for your local use and applies a small local adapter. `block-model-renderer` is published as MIT, and cubiomes is MIT.

## Sources

Primary/reference URLs are collected in `SOURCES.md`.
