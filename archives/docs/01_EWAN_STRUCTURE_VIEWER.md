# Ewan Howell Minecraft Structure Viewer: Reverse-Engineering Notes

Repository: `https://github.com/ewanhowell5195/minecraft-structure-viewer`
Site: `https://structure-viewer.ewanhowell.com/`

## 1. Why this is the best structure-viewer reference

The current public repository describes itself as a Vue 3 + Vite rewrite of an older structure viewer. Its advertised capabilities include:

- structure enumeration from a vanilla client jar;
- arbitrary resource/data packs and mod jars layered on top;
- `.nbt`, `.litematic`, `.schem`, and `.mcstructure` loading;
- greedy-meshed Minecraft block rendering using actual pack assets;
- jigsaw assembly using template pools, fallbacks, joints and collision rules;
- procedural assembly for structures whose layouts are not simply one NBT template;
- world-save / `.mca` reading and streaming;
- GLB/OBJ export.

This makes it much closer to what you want than a cube-per-block template viewer.

## 2. Current top-level stack

From `package.json`:

- Vue 3
- Vite
- Three.js
- `block-model-renderer`

The repo's `src/lib.js` dynamically imports `block-model-renderer` from jsDelivr at runtime, then injects its Three.js instance. For a self-contained local build, replace this CDN import with a normal package import. `block-model-renderer` is already present in the package dependency graph.

## 3. Jar acquisition and pack layering

`src/mojang.js`:

- reads Mojang's version manifest;
- chooses a release/snapshot or pinned version;
- downloads the client jar;
- caches it in the browser Cache API;
- normally uses the author's CORS proxy because Mojang's hosts do not expose browser-friendly CORS headers.

For local 1.16.1 this network path is unnecessary. The documented embed API accepts **raw jar/zip bytes** as `loadPacks({ base: <bytes> })`. That is the clean way to feed your own Prism/Minecraft jar.

`src/composables/usePacks.js` builds an ordered source stack and calls the renderer library's asset preparation function. Higher-priority packs override lower-priority sources. It also appends two generated bundles:

- `builtin.zip`: hardcoded/code-built structures captured from Minecraft generation code;
- `features.zip`: configured features serialized from Minecraft's registries.

## 4. Structure discovery

`src/composables/useStructures.js` uses a regex that intentionally accepts **both**:

- `data/<namespace>/structure/...`
- `data/<namespace>/structures/...`

That legacy plural compatibility is useful for older jars such as 1.16.1.

It scans zip entries directly and builds a `resource-relative-name -> zip-path` map. It can also scan worldgen template pools to determine likely starter pieces and standalones.

### 1.16.1 warning

Modern Minecraft moved much more worldgen configuration into datapack JSON. The current viewer's automatic discovery of jigsaw depth/radius reads modern `data/<ns>/worldgen/structure/*.json` records. If 1.16.1 does not ship equivalent data for a given structure family, the generic jigsaw engine may still work, but the viewer cannot infer all starting metadata from the jar alone.

For 1.16.1, obtain missing start-pool/size/distance metadata from your mapped source or from a version-specific registry extraction step, not from modern defaults.

## 5. NBT parser

`src/nbt.js` is a browser-native NBT reader. Important behavior:

- supports all standard NBT scalar, list, compound and array tags;
- auto-detects gzip by the `1f 8b` header;
- uses browser `DecompressionStream("gzip")`;
- reads Java big-endian NBT by default and can switch to little-endian;
- structure parsing reads `size`, `palette` or first entry of plural `palettes`, `blocks`, and entities;
- preserves per-block NBT data.

This is the correct level of parsing for vanilla structure templates. You do not need to convert NBT into an approximate intermediate format.

## 6. Rendering

The public README states that the renderer uses:

- greedy meshing;
- texture atlases;
- face culling;
- animated water/lava/fire;
- interactive doors, trapdoors, and gates.

This is a major difference from MinecraftMaps' simpler cube renderer. The important architectural choice is that the viewer loads **blockstates, block models and textures from the same pack/jar source stack** as the structure data. That is how you avoid hard-coded substitute colors/shapes.

For your local 1.16.1 project, keep the renderer version-agnostic and feed it the exact 1.16.1 assets.

## 7. Jigsaw assembly

`src/jigsaw.js` implements a generic template-pool assembly loop. It performs, at a high level:

1. Start with a template and its transformed bounding box.
2. Enumerate jigsaw blocks in the source piece.
3. Load the referenced pool.
4. Weighted-shuffle candidate elements.
5. Append fallback-pool candidates when configured.
6. For each candidate, try rotations and matching jigsaw connectors.
7. Require opposite front directions.
8. Respect rollable/aligned joint semantics.
9. Match target/name IDs with namespace normalization.
10. Compute child offset from transformed jigsaw coordinates.
11. Reject overlap or out-of-radius placements.
12. Repeat level by level.

The implementation also handles feature-pool elements and empty-pool elements.

### Critical RNG warning

The current viewer's generic random source in `src/transforms.js` is **Mulberry32**. Session seeds are 32-bit; `useSession.js` accepts only 1-8 hexadecimal digits in the URL and mixes those values to seed each generation level.

That means:

> Ewan's re-roll mode is deterministic for the viewer, but it is not a direct reproduction of the Java 1.16.1 world-seed RNG stream.

Do not wire a 64-bit world seed directly into `?seed=` and call the result exact.

For exact natural generation you need the 1.16.1 RNG seeding/order from Minecraft itself.

## 8. Procedural structures

`src/proc.js` and `src/generators/index.js` expose dedicated generators for major nontrivial families, including:

- igloo;
- end city;
- woodland mansion;
- jungle temple;
- desert pyramid;
- dungeon variants;
- Nether fortress;
- End spikes;
- stronghold;
- normal/mesa mineshafts and corridor variants;
- ocean monument.

The stronghold source, for example, contains explicit piece weights, placement caps, depth constraints, bounding-box tests and random per-piece variants. It is a valuable readable port of Minecraft's structure logic.

Again, audit version differences and RNG order before using it as a 1.16.1 world-seed oracle.

## 9. Hardcoded structure extraction: the most interesting part

`tools/builtin/BuiltinExtract.java` is particularly useful for your local knowledge library.

Its stated approach is:

- compile against an unobfuscated Minecraft server classpath;
- instantiate the game's real hardcoded structure pieces/features;
- run their generation code against a **capturing `WorldGenLevel`**;
- serialize the captured block output as ordinary structure NBT;
- use controlled random values so the capture is canonical/repeatable;
- run divergent random captures to discover cells controlled by random selectors.

This is exactly the kind of technique you should adapt to 1.16.1 when the game does not ship an NBT template for a structure piece.

`tools/builtin/extract.js` orchestrates compilation and execution. `tools/builtin/common.js` obtains the server jar/classpath for current modern versions.

### 1.16.1 adaptation caveat

Modern server jars use a bundler layout containing the real server jar and libraries. 1.16.1 packaging differs, so `prepareVersion()/extractBundler()` will need a compatibility branch. Your existing mapped 1.16.1 corpus is likely an easier classpath source.

## 10. Feature extraction

`tools/features/FeatureExtract.java` takes a complementary approach for configured features:

- boots Minecraft registries;
- iterates the feature registry;
- serializes each feature with Minecraft's own codec to datapack-style JSON.

This is excellent for modern versions whose configured features exist in code but not as JSON files.

For 1.16.1, registry/codec class names and APIs are version-specific. Use the idea, not the current Java source verbatim.

## 11. World reading

`src/world.js` contains a substantial in-browser world reader:

- reads world ZIP central directories without loading multi-GB ZIPs fully into memory;
- supports ZIP64 central directory handling;
- discovers dimensions;
- finds region files and entity region files;
- scans `.mca` location tables;
- decompresses chunk payloads;
- parses NBT;
- caches inflated region files in an LRU-like byte-bounded cache;
- assembles chunk grids into renderable tiles.

For exactness, this is arguably more important than any ported generator: if the actual 1.16.1 game generated the chunk, parsing that chunk gives you the real blocks.

## 12. Embed API: ideal for your local harness

The viewer documents a `postMessage` API. Useful commands include:

- `loadPacks({base, packs})`
- `loadStructure({data,name})` or `loadStructure({path})`
- `listStructures({filter})`
- `loadWorld({data,name,dimension,chunks,y,force})`

`base` can be a version string, raw jar/zip bytes, or a virtual source. For you, raw bytes are the important case.

The included `scripts/local-loader.html` uses this API so you can select a local `1.16.1.jar` and drive the viewer without its Mojang/CORS-Proxy path.

## 13. What to reuse vs. what to rewrite

### Reuse/reference heavily

- pack overlay model;
- NBT/world readers;
- transforms and blockstate normalization;
- actual block-model rendering strategy;
- template discovery;
- world-save mode;
- generic jigsaw connector/collision logic;
- hardcoded-piece capture technique.

### Reimplement/version-lock for 1.16.1

- RNG primitives and RNG call order for natural worldgen;
- world-seed -> structure-start seed derivation;
- structure-specific version rules;
- 1.16.1 jigsaw start-pool/depth/radius metadata where not present as JSON;
- processor behavior that changed across versions;
- any hardcoded generator whose modern implementation diverged from 1.16.1.

## 14. License status

GitHub repository metadata currently reports no repository license. Do not treat public readability as an automatic redistribution/modification grant. For personal research, clone the repository yourself. If you later publish derived code, resolve licensing with the author or build a clean-room implementation from the documented behavior and Minecraft's own legally available data/mappings.
