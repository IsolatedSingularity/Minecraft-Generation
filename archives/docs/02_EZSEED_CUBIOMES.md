# ezseed and Cubiomes: Seed-Map Reverse Engineering

Site: `https://ezseed.net/seed-map/`
Cubiomes: `https://github.com/Cubitect/cubiomes`
Cubiomes Viewer: `https://github.com/Cubitect/cubiomes-viewer`

## 1. What ezseed publicly discloses

ezseed's own site states that:

- Java biome/structure calculations use **cubiomes**;
- cubiomes is compiled to **WebAssembly** for browser execution;
- seed calculations run locally in the browser;
- the 2D map uses **OpenLayers**;
- its custom finder/analyzer/nearest/cluster UI combines JavaScript and WASM;
- structure positions are ports/reference implementations and terrain-dependent cases may be candidates rather than guaranteed placements.

Its terms also explicitly say results can be approximate or wrong for some edge cases and that it uses open-source libraries, WebAssembly, JavaScript ports, research notes and its own implementation work.

This means the clean-room local equivalent is straightforward at the architecture level even if ezseed's custom frontend source itself is not public:

```
world seed + version + dimension
        |
        v
   cubiomes native/WASM
        |
        +--> biome tiles
        +--> structure attempts
        +--> biome viability
        +--> strongholds
        +--> slime chunks / special finders
        |
        v
   typed binary/JSON results
        |
        v
 OpenLayers / Canvas / WebGL UI
```

## 2. Cubiomes explicitly supports Java 1.16.1

Current `biomes.h` has separate version constants:

- `MC_1_16_1`
- `MC_1_16_5`
- `MC_1_16 = MC_1_16_5`

For your project, always use **`MC_1_16_1`**.

The generator structure comments also identify the layered biome generator as the path for Minecraft 1.0 through 1.17, with separate Nether noise introduced for 1.16.

## 3. Core Cubiomes API for your seed map

### Biomes

Typical sequence:

```c
Generator g;
setupGenerator(&g, MC_1_16_1, 0);
applySeed(&g, DIM_OVERWORLD, seed);
```

Then use either:

- `getBiomeAt()` for point queries;
- `genBiomes()` for rectangular areas.

For 1.16.1, batch generation benefits from the layered generator and is ideal for map tiles.

### Structure generation attempts

Cubiomes documents the usual structure placement as a two-stage process:

1. compute the deterministic generation attempt for a structure region;
2. test whether biome requirements allow it to generate.

The API is:

- `getStructureConfig(structureType, mc, &config)`
- `getStructurePos(structureType, mc, seed, regX, regZ, &pos)`
- `isViableStructurePos(structureType, &generator, blockX, blockZ, flags)`

The structure-position routine depends on the structure type, region coordinates and lower 48 bits of the world seed for the normal region-placement families.

### Strongholds

Use:

- `initFirstStronghold()`
- repeated `nextStronghold()` calls

The iterator tracks ring number/index and returns biome-adjusted accurate positions.

### Mineshafts

Use the dedicated `getMineshafts()` area function rather than forcing mineshafts through the normal region-placement abstraction.

### Slime chunks

Cubiomes exposes the familiar exact Java slime-chunk test as `isSlimeChunk()`.

## 4. What Cubiomes does NOT give you

Cubiomes is a biome/feature/structure-position library, not a complete Java chunk generator.

Its own README warns that terrain-dependent structures can produce false positives in modern versions because it does not perform full block-level overworld generation. For 1.16.1, many classic structure location checks are substantially simpler than 1.18+, but the general rule remains:

> Cubiomes is the right source for seed-map coordinates and biome logic, not the final source for every block in terrain or every assembled structure piece.

Do not ask cubiomes to replace a real chunk generator.

## 5. Recommended local seed-map service

Keep calculation and visualization separate.

### Native worker

Build a small executable or library around cubiomes that accepts requests such as:

```json
{
  "version": "1.16.1",
  "seed": "6090144754301628691",
  "dimension": "overworld",
  "bbox": [-4096, -4096, 4096, 4096],
  "layers": ["biomes", "village", "fortress", "stronghold", "slime"]
}
```

Return deterministic JSON/MessagePack/binary tile data.

For a desktop-local project, native C is simplest and fastest. For an entirely browser-local project, compile the same wrapper to WASM with Emscripten.

### Map UI

OpenLayers is a proven fit and is what ezseed publicly credits. Suggested layers:

- raster/typed-array biome tile layer;
- vector structure marker layer;
- chunk grid layer;
- stronghold-ring layer;
- optional slime-chunk overlay;
- query/cursor overlay.

Persist only URL/view state in the frontend. Keep worldgen calculations in workers/native code.

## 6. 1.16.1 structure types to prioritize for speedrunning

For a 1.16.1 speedrun-oriented stack, prioritize:

- Village
- Ruined Portal
- Desert Pyramid
- Shipwreck
- Buried Treasure
- Ocean Ruin
- Nether Fortress
- Bastion Remnant
- Stronghold

Then add:

- Monument
- Mansion
- Outpost
- temples/huts/igloos
- mineshafts

Cubiomes exposes the relevant structure enums, including Fortress, Bastion, Ruined_Portal, Village, Shipwreck, Treasure and others.

## 7. Exactness flags in your own API

Every returned result should carry a confidence/origin field. Example:

```json
{
  "type": "village",
  "x": 1232,
  "z": -688,
  "status": "viable",
  "source": "cubiomes:MC_1_16_1",
  "block_exact": false
}
```

Recommended statuses:

- `attempt`: region RNG says a generation attempt exists;
- `viable`: Cubiomes biome viability passes;
- `oracle_confirmed`: actual 1.16.1 chunk NBT contains the structure start;
- `block_exact`: block volume was read from the actual generated chunk(s).

That prevents your UI/agents from silently conflating a seed-position prediction with a generated block model.

## 8. Why this is better than scraping ezseed's minified frontend

The valuable seed math is already in an MIT-licensed upstream library with a documented C API. Copying/minifying/reversing ezseed's custom UI would give you more maintenance burden and less correctness provenance.

Use ezseed as a behavioral/UI reference, Cubiomes as the calculation dependency, and your own local frontend.
