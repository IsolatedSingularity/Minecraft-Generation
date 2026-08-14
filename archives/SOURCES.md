# Sources

Research snapshot: 2026-08-13.

## Ewan Howell Structure Viewer

- Site: https://structure-viewer.ewanhowell.com/
- Repository: https://github.com/ewanhowell5195/minecraft-structure-viewer
- README: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/README.md
- package.json: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/package.json
- NBT: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/src/nbt.js
- Jar loading: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/src/mojang.js
- Pack layering: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/src/composables/usePacks.js
- Structure discovery: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/src/composables/useStructures.js
- Session dispatch: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/src/composables/useSession.js
- Jigsaw: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/src/jigsaw.js
- Transforms/RNG: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/src/transforms.js
- World reader: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/src/world.js
- Procedural registry: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/src/proc.js
- Generator directory: https://github.com/ewanhowell5195/minecraft-structure-viewer/tree/main/src/generators
- Builtin capture tool: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/tools/builtin/BuiltinExtract.java
- Builtin extractor: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/tools/builtin/extract.js
- Extraction common code: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/tools/builtin/common.js
- Feature registry extractor: https://github.com/ewanhowell5195/minecraft-structure-viewer/blob/main/tools/features/FeatureExtract.java

At the snapshot GitHub API repository metadata reported `license: null`.

## ezseed

- Home: https://ezseed.net/
- Seed map: https://ezseed.net/seed-map/
- Terms: https://ezseed.net/terms/
- Publicly credited calculation stack: Cubiomes compiled to WASM; OpenLayers for map rendering; custom JS/WASM tooling for site-specific finders/analyzers.

## Cubiomes

- Repository: https://github.com/Cubitect/cubiomes
- Viewer: https://github.com/Cubitect/cubiomes-viewer (desktop Qt seed/map GUI; source contains explicit `MC_1_16_1` handling)
- `biomes.h`: explicit `MC_1_16_1` version enum
- `generator.h`: generator setup / biome APIs
- `finders.h`: structure configs/positions, viability, strongholds, mineshafts, slime chunks and variants
- License: MIT

## MinecraftMaps

- Structure Viewer: https://www.minecraftmaps.com/tools/structure-viewer
- Public description covers extraction of vanilla NBT templates, `size`/`palette`/`blocks` parsing, air filtering, cube rendering, map-color fallback and template download.

## block-model-renderer

- npm package: https://www.npmjs.com/package/block-model-renderer
- jsDelivr package page: https://www.jsdelivr.com/package/npm/block-model-renderer
- Public package metadata describes it as a Minecraft block/item model renderer and reports an MIT license.
