# Agent Handoff: Build the Exact Java 1.16.1 Local Seed + Structure Viewer

## Mission

Add a version-locked local viewer capability for Minecraft Java 1.16.1 **inside the existing repository without replacing or reorganizing its unrelated functionality**. The first target is faithful local reproduction of the reference sites' behavior and rendering/data semantics, not a redesigned approximation. Project-specific polish can come after parity.

Do not use visual/worldgen approximations when exact source data, public reference code, or an oracle is available.

### Repository-integration constraint

Before changing code, inspect the current repo layout, existing 1.16.1 corpus/index/query interfaces, and agent conventions. Add the viewer/seed-map functionality behind the narrowest sensible new module or service boundary. Reuse existing canonical files and indexes where possible. Do **not** migrate the repo into the reference viewer, replace the existing library, rename unrelated directories, or make this tool the new top-level architecture.

### Parity-first constraint

When a reference implementation is available, match its observable behavior before simplifying or redesigning:

- same underlying Minecraft data and version semantics;
- same meaningful structure discovery and assembly behavior;
- same blockstate/model/texture interpretation for the 3D viewer;
- same coordinate/seed semantics for the seed map;
- equivalent camera/orbit/filter/load interactions where useful;
- deterministic outputs and URL/request state where relevant.

A lean native/Python path may be added for agent queries, tests, and automation, but it must not silently replace the high-fidelity renderer with a cube approximation.

## Non-negotiable rules

1. Version is **Java 1.16.1**, not generic 1.16 and not 1.16.5.
2. Never substitute modern worldgen constants/logic without proving they are unchanged.
3. Never use `Math.random()` or Ewan's Mulberry32 session RNG for world-seed-exact natural structure generation.
4. Every generated datum must record provenance: jar data, cubiomes, ported 1.16.1 source, or actual game oracle.
5. Preserve raw NBT and source hashes.
6. Do not infer template/worldgen JSON paths. Enumerate the actual 1.16.1 jar.
7. Use real blockstate/model/texture assets for 3D rendering.
8. Differential-test ports against actual 1.16.1 generated chunks before labeling them exact.
9. Treat this as an additive repo feature. Do not take over unrelated repo functionality or restructure the corpus to fit the viewer.
10. Prefer literal reference parity over a bespoke rewrite when public reference code/data flow exists.
11. Python/`nbtlib` spatial grids are acceptable helper representations, not the canonical visual renderer. Preserve block properties, block NBT, entities, and source template identity.
12. Do not claim `100% accuracy` from Cubiomes alone for final world blocks; distinguish structure attempts/viability from terrain-conditioned/final generation.

## Reference projects

### Structure renderer / architecture

`https://github.com/ewanhowell5195/minecraft-structure-viewer`

Important files to inspect:

- `src/nbt.js`: NBT and structure parser
- `src/world.js`: world ZIP, Anvil region and chunk reader
- `src/composables/usePacks.js`: layered jar/pack assets
- `src/composables/useStructures.js`: structure discovery, legacy plural path support
- `src/composables/useStructure.js`: loading/normalization
- `src/composables/useSession.js`: assembly session dispatch
- `src/jigsaw.js`: generic jigsaw assembly
- `src/transforms.js`: rotations, transforms, viewer PRNG
- `src/proc.js`: procedural structure registry
- `src/generators/*.js`: structure-family ports
- `tools/builtin/BuiltinExtract.java`: capture hardcoded structures from real Minecraft code
- `tools/features/FeatureExtract.java`: serialize feature registry via Minecraft codecs

Important warning: repository metadata currently has no explicit license. Clone/reference locally. Do not vendor/republish wholesale without resolving licensing.

### Seed calculation

`https://github.com/Cubitect/cubiomes`

Use:

- `MC_1_16_1`
- `setupGenerator`
- `applySeed`
- `genBiomes` / `getBiomeAt`
- `getStructureConfig`
- `getStructurePos`
- `isViableStructurePos`
- `StrongholdIter` + `initFirstStronghold` / `nextStronghold`
- `getMineshafts`
- `isSlimeChunk`

## Work plan

### Task 0: integrate without taking over the repo

1. Inspect the repository's current top-level layout and existing Minecraft 1.16.1 generation-library interfaces.
2. Identify the smallest integration boundary for: `(a)` seed query engine, `(b)` reference-parity 3D viewer, `(c)` oracle/differential tests.
3. Reuse existing canonical jar/extracted/template/index paths rather than creating duplicate canonical sources.
4. Keep reference clones/build caches isolated from the canonical library and gitignored unless intentionally tracked.
5. Record every file/directory added by this project.

Acceptance:

- existing repo commands/tests still work;
- no unrelated directories are renamed/moved;
- viewer can be removed without damaging the canonical generation library;
- agents can query the existing library without booting the UI.

### Task 1: inventory the user's canonical 1.16.1 jar

Produce:

- full SHA-256 of jar;
- list/count/hash of structure NBT files;
- all `data/minecraft/worldgen/**` files actually present;
- blockstate/model/texture inventories;
- path convention report (`structure` vs `structures`).

Use `scripts/extract-mc1161-assets.ps1` as a starting point.

Acceptance:

- no hard-coded assumption about current-version jar paths;
- manifest is deterministic and sorted.

### Task 2: bring up Ewan viewer locally with user's jar

Run `scripts/bootstrap-ewan-local.ps1`.

The script should:

- clone the reference repo;
- install from its lockfile;
- replace runtime jsDelivr loading with the installed `block-model-renderer` package;
- add `public/local-loader.html`;
- build and/or serve locally.

Then open the loader, select the user's `1.16.1.jar`, and confirm the viewer enumerates structures from that jar.

Acceptance:

- after dependencies are installed/bundled, structure viewing does not require Mojang's jar-download path or Ewan's CORS proxy;
- selecting a local jar changes the source set;
- raw structure templates render from the selected jar.

### Task 3: compare structure parser to local corpus

For every 1.16.1 NBT template:

- parse size;
- parse palette(s);
- parse blocks;
- preserve block NBT;
- count non-air blocks under an explicitly documented filter.

Record failures by file and NBT tag offset.

Acceptance: 100% parse success for canonical vanilla templates.

### Task 4: build Cubiomes 1.16.1 query worker

Compile cubiomes and the included `cubiomes/mc1161_seed_probe.c`.

Then replace the demo probe with a stable local service API that returns:

- biome tiles;
- supported structure attempts + viability;
- strongholds;
- mineshafts;
- slime chunks.

Acceptance:

- all requests pin `MC_1_16_1`;
- seeds are parsed as signed decimal input but preserved as 64-bit bit patterns;
- negative block/chunk/region coordinates use mathematical floor division;
- deterministic repeated calls are byte-identical after canonical sorting.

### Task 5: add a 2D seed map

Recommended frontend: OpenLayers, matching the public architecture credited by ezseed.

Do not put worldgen logic in UI components. Query workers/native service only.

Acceptance:

- pan/zoom does not recompute unchanged tiles;
- seed/version/dimension are URL-state;
- markers carry provenance and exactness status.

### Task 6: build an actual-1.16.1 oracle

Preferred goal: given `(seed, dimension, chunk set)`, produce actual Minecraft-generated region/chunk data.

Possible implementations:

A. automated dedicated 1.16.1 server/world generation harness;
B. minimal Fabric 1.16.1 server-side utility that forces generation of requested chunks;
C. use existing user worlds for initial golden data.

Do not replace this with a modern generator.

Acceptance:

- oracle world has recorded seed/version/jar SHA;
- requested chunks are known generated before capture;
- output region/chunk hashes are stable.

### Task 7: exact natural structure layouts

Implement one family at a time, in this order for speedrun value:

1. Bastion
2. Nether Fortress
3. Stronghold
4. Ruined Portal
5. Village
6. Shipwreck / buried treasure as useful

For each family:

1. find exact 1.16.1 source slice in the user's generation library;
2. enumerate RNG entry point and every random call that affects the piece graph;
3. port using exact Java overflow/LCG semantics;
4. compare piece graph against oracle worlds;
5. compare final blocks if needed;
6. only then mark family `exact`.

### Task 8: connect seed points to structure rendering

Clicking a seed-map marker should offer:

- `Template`: show relevant raw templates;
- `Predicted assembly`: only if a differential-tested exact port exists;
- `Oracle blocks`: generate/read actual chunks and show final world volume.

Never silently show a generic rerolled assembly as though it were the world's actual instance.

## Data provenance schema

Use a small sidecar on all calculated artifacts:

```json
{
  "minecraft_version": "1.16.1",
  "world_seed": "6090144754301628691",
  "source": "cubiomes|jar|ported-source|oracle",
  "source_revision": "...",
  "jar_sha256": "...",
  "exactness": "template_exact|position_viable|oracle_confirmed|block_exact",
  "notes": []
}
```

## Failure conditions to report loudly

- viewer source uses a modern fallback because 1.16.1 data is absent;
- a structure algorithm needs terrain context that the port does not model;
- RNG call ordering is inferred instead of traced;
- modern registry JSON is being substituted for 1.16.1 code-defined settings;
- cubiomes returns only a candidate/attempt;
- a block model/textures cannot be resolved from the selected jar;
- world/chunk data version does not match 1.16.1.

## Second-pass research guidance

Read `docs/06_SECOND_PASS_CAVEATS.md` before implementing. The short version: the suggested native Cubiomes path is good, and `nbtlib` is useful for tests/agent queries, but the claim that the structure viewer is just GZIP NBT + textured cubes is not accurate for the preferred Ewan reference. Do not downgrade the renderer or assembly system to that model.

## Definition of done

The system is useful before every structure family is ported. A strong first milestone is:

- exact local 1.16.1 NBT/model viewer;
- Cubiomes 1.16.1 map with exact/candidate labels;
- one-click actual-world chunk rendering for selected coordinates;
- differential harness ready for future generator ports.

That architecture is already materially more reliable than trying to reimplement all of Minecraft worldgen in one pass.
