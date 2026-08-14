# Second-Pass Notes: Useful Gemini Ideas, Corrections, and Remaining Blind Spots

## Bottom line

The second research pass contains two useful implementation ideas:

1. **Run Cubiomes natively** for local/agent seed queries instead of requiring a browser/WASM layer.
2. **Expose parsed structure NBT as a lean Python/native query representation** for agents and tests.

Keep both. Neither should replace the high-fidelity viewer path.

## What is correct and worth incorporating

### Native Cubiomes service

For Java 1.16.1, a native Cubiomes executable/library is an excellent local computation layer. Prefer a stable JSON/JSONL or local RPC boundary so Python agents do not need to understand C ABI details.

Pin `MC_1_16_1`. Do not use the generic `MC_1_16` alias, which maps to a later 1.16 patch in current Cubiomes.

A Python wrapper is optional. A tiny native process with canonical JSON input/output is easier to version, test, and call from any agent language.

### Python/NBT helper representation

`nbtlib` or an equivalent parser is useful for:

- corpus validation;
- exact template size/palette/block counts;
- coordinate queries;
- test fixtures;
- indexing block types and block entities;
- exporting a structure to a simple sparse grid for agent reasoning.

If a Python grid is used, do **not** reduce palette entries to block IDs only. Preserve at least:

```text
Name
Properties
block position
block NBT / block entity data
entity data
source template id
```

Dropping `Properties` loses facing, half, shape, waterlogged state, rail orientation, stair shape, fence connectivity, etc.

## Corrections to the simplified description

### 1. Ezseed and Ewan do not share one WASM/C architecture

Ezseed publicly credits Cubiomes compiled to WASM for Java biome/structure calculations and OpenLayers for map rendering. That is the seed-map side.

The preferred Ewan structure viewer is a JavaScript/Vue/Three.js application using `block-model-renderer` plus its own NBT/world/assembly code. Do not assume its structure pipeline is Cubiomes/WASM.

### 2. Current Ewan NBT parsing is not described accurately as `pako`-based

The current public source uses browser `DecompressionStream("gzip")` in its NBT parser. More importantly, decompression is a small implementation detail. The hard part is all the behavior after parsing.

### 3. Ewan is not a cube-per-block texture viewer

A simple voxel cube renderer can reproduce MinecraftMaps' basic template visualization style, but it does not reproduce Ewan's preferred viewer.

The Ewan project uses real blockstate/model/texture assets and advertises greedy meshing, face culling, animated fluids/fire, live doors/trapdoors/gates, entity/block-entity interactions, jigsaw assembly, procedural structures, world loading/streaming, and export/walk modes.

Therefore: **do not replace Ewan parity with PyVista/Ursina cubes or OBJ cubes.** Those are optional debug/export tools only.

### 4. A structure-template NBT is not the same thing as a naturally generated structure instance

Many structures are assembled from multiple templates, procedural pieces, hardcoded generation logic, processors, markers, rotations, mirrors, terrain adaptation, and RNG. A raw `data/minecraft/structures/*.nbt` parser only solves the template layer.

For Java 1.16.1 natural structures, keep three separate concepts:

```text
template
piece/assembly prediction
final generated world blocks
```

### 5. Cubiomes should not be advertised internally as universally "100% final-world exact"

Cubiomes is the correct high-performance seed/biome/structure-location reference for many supported 1.16.1 calculations, but a structure attempt/viability result is not automatically a complete final block-level reconstruction. Terrain-dependent generation and per-structure details may require additional logic or an actual game oracle.

Ezseed's own current documentation similarly says structures that depend on terrain it does not fully simulate can be marked as candidates.

## 1:1 parity target by reference

### Ewan Structure Viewer

**Confidence: high for local application parity.**

Why:

- the current application source is public;
- its loader/embed protocol is documented;
- the renderer dependency and source architecture are visible;
- Minecraft jar/pack inputs can be supplied locally;
- world/NBT/assembly code is inspectable.

Caveat: the current repo targets modern Minecraft and its viewer re-roll/session RNG is not a Java world seed. For *1.16.1 natural-instance parity*, version-specific generator behavior still has to be ported/differential-tested or delegated to a real 1.16.1 generation oracle.

### MinecraftMaps Structure Viewer

**Confidence: very high for observable template-viewer parity.**

Its public description is intentionally simple: parse template NBT, skip air-like states, and draw colored cubes. This is easy to reproduce locally, including its limitations. It is not the renderer we should use as the final quality target.

### Ezseed 2D seed map

**Confidence: high for core Java 1.16.1 calculation parity, medium-high for pixel/UI parity.**

Cubiomes gives us the important Java worldgen calculation layer, and OpenLayers is publicly credited. The private/site-specific worker scheduling, caching, tile formats, UX, filters, and current frontend code may require further asset/network inspection if the goal becomes literal UI duplication rather than equivalent functionality.

## Remaining blind spots worth attacking

### A. Exact 1.16.1 natural assembly seeds and RNG call order

This is the biggest technical gap.

For Bastions, Fortresses, Strongholds, villages, ruined portals, etc., determine the exact 1.16.1 RNG entry point and call ordering that produces the piece graph and processors. Modern reference algorithms are not sufficient evidence by themselves.

### B. Terrain adaptation and final placement

Some structures depend on heightmaps, surrounding blocks, carving, liquids, processor rules, or chunk-generation stage. A standalone template/assembly renderer can be geometrically right while final world blocks differ.

Use actual generated 1.16.1 chunks as the oracle.

### C. Renderer details

For Ewan-level visual parity, test:

- blockstate variants and multipart rules;
- parent model inheritance;
- UV rotations/cullface/tint indices;
- transparent/cutout/translucent materials;
- biome tinting;
- connected-looking blocks caused by block states;
- waterlogged blocks and fluids;
- animated textures;
- block entities and entities;
- missing-model/texture behavior;
- resource-pack precedence.

### D. Version asset contamination

Do not mix modern Ewan supplemental bundles with a pure 1.16.1 source set unless each added asset has been shown to be valid for 1.16.1. Generate/version supplemental assets separately.

### E. "All Minecraft files" does not necessarily mean all worldgen behavior is data-driven

Java 1.16.1 has significant hardcoded generation logic. Some structures/features cannot be reconstructed from the client jar's NBT/JSON data alone. The mapped/decompiled 1.16.1 code or an actual server/game oracle is required for those paths.

## Would a deeper scrape still help?

Yes, but it is now a **second-stage optimization**, not a prerequisite to begin implementation.

A deeper browser/network/assets pass is most useful for:

1. matching ezseed's exact tile/worker/cache protocol and UI behavior;
2. finding any undocumented frontend constants or batching strategies;
3. comparing rendered output and interactions against Ewan at a fixed commit;
4. building automated screenshot/data parity fixtures across many seeds/structures;
5. tracking changes if the live sites evolve.

It is less valuable than version-specific 1.16.1 generator tracing if the primary goal is *correct Minecraft results*. For that goal, spend the next deep-research effort on the exact 1.16.1 structure-generation/RNG paths and oracle validation first.

## Agent rule of thumb

If an implementation proposal says "just parse NBT and render cubes" or "Cubiomes means every final block is exact," reject it as incomplete for this project.
