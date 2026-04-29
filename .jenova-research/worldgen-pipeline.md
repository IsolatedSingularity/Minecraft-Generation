# Minecraft 1.16 World Generation Pipeline

Source: minecraft.wiki, Java Edition 1.16 source analysis. Target version: **Java Edition 1.16** (Nether Update, released June 23, 2020).

## Critical Version Distinction

1.16 uses the **OLD world generation system**. The following features are **NOT present** in 1.16:
- Multi-noise biome system (6-parameter: temperature, humidity, continentalness, erosion, weirdness, depth) -- added in 1.18
- Noise caves (cheese, spaghetti, noodle) -- added in 1.17/1.18 snapshots
- Aquifers -- added in 1.18
- Density functions -- added in 1.18
- Extended world height (Y=-64 to Y=320) -- added in 1.18
- Deepslate, tuff, dripstone -- added in 1.17+

## Seed System

- 64-bit world seed (Java `long`)
- Input: string hashed to long, or raw integer
- PRNG: Java LCG (`multiplier = 0x5DEECE66DL`, `addend = 0xBL`, `mask = (1L << 48) - 1`)
- Structure placement uses xoroshiro128++ in later versions, but 1.16 still uses the LCG-based region seed formula
- Region seed: `seed + regionX * K1 + regionZ * K2 + salt` where K1 = 341873128712, K2 = 132897987541

## Perlin Noise Mathematics (1.16)

- Gradient vectors assigned at grid vertices; dot product with distance vector to sample point
- Quintic fade function for $C^2$ continuity: $f(t) = 6t^5 - 15t^4 + 10t^3$
- First and second derivatives zero at endpoints ($t=0$, $t=1$), preventing terrain creasing
- Multiple octaves layered for fractal Brownian motion (fBm): low-frequency = continent shapes, high-frequency = fine detail
- 1.16 uses standard Java Perlin implementation (not Simplex)
- Voronoi-like jittered point distributions used for biome boundary smoothing

## Overworld Generation (1.16)

### Terrain Shape
- 3D Perlin noise density function
- Height range: Y=0 to Y=255 (old system, NOT the extended -64 to 320)
- Bedrock at Y=0-4 (random pattern)
- Sea level at Y=63
- Noise sampled at 4-block horizontal resolution, 8-block vertical resolution, then interpolated
- Terrain uses multi-octave Perlin noise with lacunarity and persistence

### Biome System (Legacy Layers)
1.16 Overworld uses the **layer-based biome system** (NOT multi-noise):
- Temperature layer
- Humidity layer
- Continent/ocean decision
- Biome selection based on temperature + humidity category
- Shore/river/ocean variant layers
- Zoom layers for detail
- Voronoi zoom for final per-block resolution (4x4 biome grid per chunk)

### Biomes (1.16 Overworld)
54 Overworld biomes including:
- Ocean variants: Warm, Lukewarm, Normal, Cold, Frozen (+ Deep variants)
- Plains, Sunflower Plains
- Desert, Desert Hills, Desert Lakes
- Forest, Flower Forest, Birch Forest, Dark Forest (+ Hills variants)
- Taiga, Giant Tree Taiga, Snowy Taiga (+ Hills/Mountains variants)
- Swamp, Swamp Hills
- Jungle, Jungle Hills, Jungle Edge, Modified Jungle, Modified Jungle Edge, Bamboo Jungle
- Mountains, Wooded Mountains, Gravelly Mountains, Shattered Savanna
- Savanna, Savanna Plateau, Shattered Savanna Plateau
- Badlands, Eroded Badlands, Wooded Badlands Plateau, Modified Badlands
- Mushroom Fields, Mushroom Field Shore
- Ice Spikes, Snowy Beach, Snowy Mountains, Snowy Tundra
- River, Frozen River, Beach, Stone Shore
- The Void (superflat)

### Cave Generation (1.16: Carvers Only)
1.16 uses **carver caves** exclusively:
- Cave carvers: random walks through terrain, creating winding tunnels
- Ravines/canyons: vertical slashes through terrain
- NO cheese caves (large open caverns from noise)
- NO spaghetti caves (thin winding tunnels from noise)
- NO noodle caves (thinner variant)
- NO aquifers (water-filled cave sections)
- Underwater caves and ravines can generate below sea level

### Ore Distribution (1.16 Overworld)
Blob-based system with uniform Y-range distribution:

| Ore | Size | Count/Chunk | Y Range | Notes |
|-----|------|-------------|---------|-------|
| Coal | 17 | 20 | 0-127 | Most common |
| Iron | 9 | 20 | 0-63 | |
| Gold | 9 | 2 | 0-31 | |
| Redstone | 8 | 8 | 0-15 | |
| Diamond | 8 | 1 | 0-15 | |
| Lapis Lazuli | 7 | 1 | Center 16, spread 16 | Triangle distribution |
| Emerald | 1 | 3-8 | 4-31 | Mountains only, single blocks |
| Dirt | 33 | 10 | 0-255 | |
| Gravel | 33 | 8 | 0-255 | |
| Granite | 33 | 10 | 0-79 | |
| Diorite | 33 | 10 | 0-79 | |
| Andesite | 33 | 10 | 0-79 | |
| Infested Stone | 9 | 7 | 0-63 | Mountains only |

### Structure Placement (1.16)
Region-based algorithm:
1. Divide world into regions (spacing x spacing chunks)
2. Use region seed to pick random position within region
3. Apply separation minimum (structure can't be in first N chunks of region)
4. Check biome at center for validity

| Structure | Spacing | Separation | Salt |
|-----------|---------|------------|------|
| Village | 32 | 8 | 10387312 |
| Desert Temple | 32 | 8 | 14357617 |
| Jungle Temple | 32 | 8 | 14357619 |
| Ocean Monument | 32 | 5 | 10387313 |
| Woodland Mansion | 80 | 20 | 10387319 |
| Pillager Outpost | 32 | 8 | 165745296 |
| Stronghold | Special ring-based | N/A | N/A |
| Nether Fortress | 27 | 4 | 30084232 |
| Bastion Remnant | 27 | 4 | 30084232 |
| Ruined Portal | 40 | 15 | 34222645 |
| Shipwreck | 24 | 4 | 165745295 |
| Ocean Ruin | 20 | 8 | 14357621 |
| Buried Treasure | 1 | 0 | 10387320 |

### Stronghold Distribution
128 strongholds across 8 concentric rings:
| Ring | Count | Min Dist (chunks) | Max Dist (chunks) |
|------|-------|--------------------|-------------------|
| 1 | 3 | 1280 | 2816 |
| 2 | 6 | 4352 | 5888 |
| 3 | 10 | 7424 | 8960 |
| 4 | 15 | 10496 | 12032 |
| 5 | 21 | 13568 | 15104 |
| 6 | 28 | 16640 | 18176 |
| 7 | 36 | 19712 | 21248 |
| 8 | 9 | 22784 | 24320 |

Strongholds are placed at random angles within each ring, with minimum angular separation enforced.

### Decoration Steps (1.16)
11 ordered steps per chunk:
1. Raw Generation
2. Lakes
3. Local Modifications
4. Underground Structures
5. Surface Structures
6. Strongholds
7. Underground Ores
8. Underground Decoration
9. Vegetal Decoration
10. Top Layer Decoration
11. Fluid Springs

## Surface Rules (All Versions)

Surface rules convert base stone into biome-specific surface blocks. Applied after terrain shaping + cave carvers, before feature decorators.

Four rule types:
1. **Block**: Replaces target coordinate with a specific block state
2. **Sequence**: Evaluates nested rules in order, stops at first match (IF-ELSE chain)
3. **Condition**: Executes nested rule only if condition passes (e.g., `stone_depth` proximity to surface)
4. **Bandlands**: Hardcoded terracotta banding for Badlands biomes

In 1.16: surface rules are procedural (not JSON-driven). Grass/dirt on top 3-4 blocks, sand in deserts, mycelium in mushroom fields, podzol in giant tree taiga, etc.

## Jigsaw Structure Assembly (1.14+)

Jigsaw-based structures (villages, bastions, End cities) use dynamic piece assembly:
1. Start with a single template piece at a known position
2. Add all jigsaw connectors to a pending list
3. While pending is not empty and depth < max_depth:
   - Pop a pending jigsaw connector
   - Choose compatible piece from template pool (matching connector labels)
   - Place piece (with rotation/mirroring), add its connectors to pending
4. Stop at max_depth or when no compatible connectors remain

Village max_depth: 6 (vanilla). Bastion max_depth: 6. End city max_depth: varies by piece type.

## Feature Placement Modifiers (All Versions)

Placed features use modifiers to control spawn conditions:
- `count`: Number of attempts per chunk
- `height_range`: Y-level bounds (uniform, triangle, or trapezoid distribution)
- `in_square`: Random XZ offset within chunk
- `surface_relative_threshold_filter`: Only place above/below surface
- `biome`: Restrict to specific biomes
- `rarity_filter`: Probability check per chunk

---

## Post-1.18 Architecture Reference

> These systems replaced 1.16 mechanics in Java Edition 1.18+. Documented for version comparison. **Do not implement for 1.16 target.**

### PRNG: xoroshiro128++ (Replaced Java LCG)
- 128-bit internal state, xorshift family
- Non-linear scrambler maps state to output (no 1:1 relationship)
- Parallel streams via jump functions for multithreaded chunk generation
- Passes PractRand statistical randomness tests

### Density Functions (Replaced Simple Heightmap)
- Scalar function $D(x,y,z)$: $D > 0$ = solid, $D \leq 0$ = air/fluid
- Primitives: `constant`, `noise`, `y_clamped_gradient`, `add`, `mul`, `min`, `max`, `range_choice`, `spline`
- Caching wrappers: `cache_2d`, `cache_once`, `cache_all_in_cell`, `flat_cache`
- Evaluated at 4x8x4 cell resolution, linearly interpolated for individual blocks
- World height extended to Y=-64 to Y=320 (384 blocks)

### Noise Router
Master data structure wiring density functions to terrain:
- `barrierNoise`: boundaries between conflicting noise functions
- `fluidLevelFloodednessNoise` + `fluidLevelSpreadNoise`: aquifer bounds
- `initialDensityWithoutJaggedness`: preliminary surface height estimate for decorators/spawning
- `finalDensity`: the master output deciding block vs air
- `lavaNoise`: if > 0.3, aquifer logic replaces water with lava

### Terrain Shaping Cubic Splines
Three spline fields, inputs are continentalness + erosion noise:
- **Offset**: baseline elevation (high continentalness = land, low = ocean basin)
- **Factor**: transition steepness (high = rolling hills, low = compressed transitions)
- **Jaggedness**: high-frequency noise amplitude (low erosion = shattered mountains)

### Noise Caves (Replaced Carver-Only Caves)

| Type | Description |
|------|-------------|
| Cheese | Large porous caverns from scaled 3D noise |
| Spaghetti | Elongated tubular tunnels connecting cheese caves |
| Noodle | Thin compressed tunnels, can pierce surface at Y>130 |

### Aquifer System
- Local water levels based on noise, decides fluid vs air per cave cell
- Y=-55 to -63: always lava (deep lava layer)
- Barrier blocks generated between conflicting fluid bodies

### Ore Distribution Changes
- Triangular/trapezoidal height distributions (replaced 1.16 uniform)
- **Reduced Air Exposure (RAE)**: ores adjacent to air face placement probability check; diamonds heavily penalized near cave walls
- **Mega-veins** (Iron/Copper): governed by core noise, not feature decorators
  - `vein_toggle > 0.0` = copper, `<= 0.0` = iron
  - `vein_ridged >= 0.0` = excluded from vein
  - `vein_gap > -0.3` + random check = place ore, else stone

---

## External Tools and References

| Tool | Purpose | URL |
|------|---------|-----|
| Misode Generator Suite | Web JSON worldgen editor (density functions, surface rules, features) | https://misode.github.io/worldgen/ |
| Misode Density Function Viewer | 2D density function evaluation visualization | https://misode.github.io/worldgen/density-function/ |
| Multi-Noise Biome Visualizer | 5D climate space Voronoi visualization (jacobsjo) | https://github.com/jacobsjo/MinecraftMultiNoiseVisualization |
| Snowcapped | Worldgen datapack authoring (noise routers, terrain splines) | https://snowcapped.jacobsjo.eu/ |
| Lithostitched API | MergedDensityFunction, worldgen modifier docs | https://www.mintlify.com/Apollounknowndev/lithostitched/ |
| DragonFightPathVisualizer | Client mod rendering dragon spline paths/nodes (mjtb49) | https://github.com/mjtb49/DragonFightPathVisualizer |
| Technical MC Docs | Aggregated reverse engineering repos + timeline (JoakimThorsen) | https://gist.github.com/JoakimThorsen/e90bd7a588af25ae529530987d9acc8a |
| Vanilla Worldgen Defaults | Exact vanilla data values by version (slicedlime) | https://github.com/slicedlime/examples/ |
| Bedrock Worldgen Docs | Official schema for features, ores, surface rules (Microsoft) | https://learn.microsoft.com/en-us/minecraft/creator/ |
| McJtyMods | Open-source custom dimension + block state registries | https://github.com/orgs/McJtyMods/repositories |

### Chunk Generation States
World generation pipeline processes chunks through ordered states:
1. `empty` -- allocated but no data
2. `structure_starts` -- structure bounding boxes placed
3. `structure_references` -- cross-chunk structure tracking
4. `biomes` -- biome data computed
5. `noise` -- terrain shape generated
6. `surface` -- surface blocks placed (grass, sand, etc.)
7. `carvers` -- cave carvers run
8. `liquid_carvers` -- underwater cave carvers
9. `features` -- ores, trees, structures placed
10. `light` -- lighting calculated
11. `spawn` -- mob spawn potential calculated
12. `heightmaps` -- heightmap data finalized
13. `full` -- ready for gameplay

### Pipeline Stage Colors (Debug Screen)
Chunk loading screen renders each stage with a specific color:

| Stage | Hex |
|-------|-----|
| empty | #000000 |
| structure_starts | #999999 |
| structure_references | #5F6191 |
| biomes | #80B252 |
| noise | #D1D1D1 |
| surface | #268097 |
| carvers | #6D665C |
| liquid_carvers | #303572 |
| features | #1C6002 |
| light | #CCCCCC |
| spawn | #F26060 |
| heightmaps | #EEEEEE |
| full | #FFFFFF |
