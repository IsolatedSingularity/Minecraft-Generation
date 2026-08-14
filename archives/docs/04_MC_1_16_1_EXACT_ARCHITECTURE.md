# Recommended Architecture for an Exact Java 1.16.1 Local Viewer

## 1. Define exactness explicitly

There are four different things people casually call a "seed viewer". Keep them separate.

### Level 1: exact saved template

Input: one NBT template from the 1.16.1 jar.

Exact means:

- every palette state is preserved;
- every block coordinate and block NBT is preserved;
- rendering resolves the 1.16.1 blockstate/model/texture assets.

This is straightforward.

### Level 2: exact structure-start coordinate

Input: world seed + version + dimension.

Exact means the same chunk/block start that Minecraft 1.16.1 would choose after placement and biome checks.

Cubiomes `MC_1_16_1` is the best existing local calculation base for this layer.

### Level 3: exact assembled natural structure

Input: world seed + structure start + surrounding generation context.

Exact means:

- same pieces;
- same orientation/mirror;
- same jigsaw/legacy structure recursion;
- same processors;
- same random variants/loot-table references;
- same terrain-dependent effects.

This requires exact 1.16.1 generator logic and RNG call ordering or an actual Minecraft-generated chunk oracle.

### Level 4: exact terrain + structure world volume

Exact means the actual final chunk blocks.

The safest source is the real 1.16.1 generator output, read from Anvil region files.

## 2. Recommended components

```
                           +-------------------------+
                           |  Minecraft 1.16.1 jar   |
                           +-------------------------+
                             | data + assets
                             v
+----------------+      +-------------------------+       +------------------+
|  Cubiomes      |----->| Local world model       |------>| 3D renderer      |
| MC_1_16_1      |      | structures/templates    |       | block models     |
+----------------+      | seed positions          |       | textures         |
  |                     | oracle chunks           |       | greedy mesh      |
  |                     +-------------------------+       +------------------+
  | seed positions               ^
  v                              |
+----------------+               |
| 2D map worker  |               |
+----------------+               |
                                 |
                       +-------------------------+
                       | 1.16.1 generation oracle|
                       | real world / Fabric tool|
                       +-------------------------+
```

## 3. Data sources

### From the client jar

At minimum inspect/extract:

- `data/minecraft/structures/**` or version-equivalent singular path;
- `data/minecraft/worldgen/**` if present;
- `assets/minecraft/blockstates/**`;
- `assets/minecraft/models/block/**`;
- `assets/minecraft/textures/block/**`;
- colormaps;
- any extra block/entity textures required by special renderers.

Do not hard-code the path shape. Enumerate the jar and inventory what 1.16.1 actually contains.

### From your mapped/decompiled 1.16.1 source

Index exact classes/functions for:

- structure placement configuration/salts;
- `ChunkGenerator` structure start paths;
- `StructurePiece` families;
- `TemplateStructurePiece` and template transforms;
- `JigsawManager`/pool-element placement;
- `StrongholdPieces`;
- `NetherBridgePieces`;
- `MineshaftPieces`;
- `WoodlandMansionPieces`;
- ocean monument building pieces;
- end-city piece generators;
- structure processors;
- Java RNG / chunk-seed helpers.

Your existing Minecraft technical library should be treated as the source-of-truth index for the version-specific port.

## 4. Renderer

Use a block-model renderer, not a cube renderer.

Required pipeline:

1. Parse palette entry to block ID + properties.
2. Resolve `assets/<ns>/blockstates/<id>.json`.
3. Select variants/multipart cases from properties.
4. Resolve model parent chain.
5. Resolve texture variables.
6. Generate model quads with UVs/cullface/tint/rotation.
7. Apply biome tint where required.
8. Cull internal faces.
9. Batch/greedy mesh where safe.
10. Handle fluids and important block entities separately.

For a structure viewer, using Ewan's `block-model-renderer` is much less work than implementing this from scratch.

## 5. Structure template normalization

Internal neutral representation:

```ts
interface StructureTemplate {
  size: [number, number, number];
  palette: BlockState[];
  blocks: {
    pos: [number, number, number];
    state: number;
    nbt?: unknown;
  }[];
  entities?: {
    pos: [number, number, number];
    nbt: unknown;
  }[];
}

interface BlockState {
  Name: string;
  Properties?: Record<string, string>;
}
```

Keep this representation lossless relative to NBT. Rendering caches should be derived, never canonical.

## 6. Seed-position service

Recommended request API:

```ts
interface SeedQuery {
  version: "1.16.1";
  seed: bigint;
  dimension: "overworld" | "nether" | "end";
  minX: number;
  minZ: number;
  maxX: number;
  maxZ: number;
  layers: string[];
}
```

Return each point with provenance:

```ts
interface StructurePoint {
  type: string;
  x: number;
  z: number;
  status: "attempt" | "viable" | "oracle_confirmed";
  source: string;
}
```

## 7. RNG policy

This deserves its own module and tests.

Never use:

- `Math.random()`;
- Mulberry32;
- a generic PCG/xorshift;
- a modern Minecraft RNG implementation without verifying 1.16.1 behavior.

For exact natural 1.16.1 generation, preserve:

- Java signed integer/long overflow semantics;
- `java.util.Random` 48-bit LCG behavior where used;
- exact seed scramblers/salts;
- exact call order;
- any separate chunk/feature/decoration seed derivations.

One extra random call changes the entire downstream structure layout.

## 8. Natural structure assembly: preferred strategy

### Strategy A: Oracle-first

Use actual Minecraft 1.16.1 to generate target chunks, then read the chunk NBT.

Advantages:

- immediately exact;
- no uncertainty about RNG order;
- includes terrain interactions;
- includes final processors and block variants.

Disadvantages:

- slower than a pure port;
- generating many far-away chunks costs CPU/disk;
- needs a local generation harness.

This is ideal for validating or rendering selected seed-map locations.

### Strategy B: Exact port

Port the relevant generator family from the mapped 1.16.1 code.

Advantages:

- fast once implemented;
- can render without creating world files;
- excellent for agents and research.

Disadvantages:

- easy to be subtly wrong;
- version drift is dangerous;
- processor/terrain context can make the port much larger than expected.

Use Strategy A as the differential oracle for Strategy B.

## 9. Suggested local cache layout

```
mc1161-viewer/
  canonical/
    client-1.16.1.jar
    server-1.16.1.jar
  extracted/
    data/
    assets/
    manifests/
  reference/
    ewan-viewer/        # cloned, not copied into your corpus
    cubiomes/
  generated/
    oracle-worlds/
    chunk-cache/
    mesh-cache/
  indexes/
    structures.jsonl
    template-pools.jsonl
    processors.jsonl
    blockstates.jsonl
    source-map.jsonl
  src/
    seed/
    nbt/
    structures/
    renderer/
    oracle/
    ui/
  tests/
    golden/
    differential/
```

## 10. Differential test suite

### A. Template NBT

For every shipped structure NBT:

- SHA-256 source bytes;
- parse;
- serialize canonical summary;
- verify size/palette/block positions.

### B. Cubiomes structure positions

For a corpus of seeds:

- generate all supported structure positions in a radius;
- generate/load the same worlds in 1.16.1;
- inspect chunk `Structures/Starts` metadata;
- compare start coordinates and type.

Use at least:

- random seeds;
- negative seeds;
- seeds near signed 64-bit boundaries;
- known speedrun seeds;
- coordinates across negative region boundaries.

### C. Strongholds

Compare first several biome-adjusted stronghold locations per seed.

### D. Procedural piece graphs

For stronghold/fortress/mineshaft/mansion/monument/end city:

- extract actual piece list/bounding boxes from generated chunks where available;
- compare piece type, bounding box, orientation, depth and parent relationships;
- only after graph identity passes compare blocks.

### E. Final chunks

Hash a neutral chunk representation:

```
(x,y,z, block-id, sorted-properties, block-entity-NBT)
```

Compare oracle vs local implementation for selected volumes.

## 11. Acceptance criteria

Do not call the project "exact 1.16.1" until:

- template parser passes the entire jar corpus;
- blockstate/model resolver renders against 1.16.1 assets rather than approximated colors;
- Cubiomes queries use `MC_1_16_1` explicitly;
- negative-coordinate region arithmetic is tested;
- each claimed exact structure family is differential-tested against actual 1.16.1 generation;
- the UI labels candidate/viable/oracle-confirmed states distinctly.
