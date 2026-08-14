# Java 1.16.1 terrain and visualization audit

Date: 2026-08-13

## Scope

This pass replaced the active illustrative biome rasters with source-faithful
Java 1.16.1 sampling. The implementation now covers:

- Java Random construction and skip order used by the noise stacks
- Perlin, octave Perlin, octave simplex, and Double Perlin sampling
- the complete Overworld `BiomeLayers.build` selection graph
- the five-biome Nether `MultiNoiseBiomeSource` classifier
- Overworld, Nether, and End density stacks used by base-height queries
- the four generated `WORLD_SURFACE_WG` samples used by the End City gate

The active fixed-seed overview rasters are compact caches of generated biome,
height, or density samples. They are not painted terrain and do not contain a
world save.

## Independent original-JAR oracle

`VanillaBiomeOracle.java` and `VanillaTerrainCache.java` compile without
Minecraft classes. At runtime they use reflection against the private original
`minecraft-1.16.1-server.jar`, which remains outside Git.

The original JAR and the Python port agreed at all audited points. A compact
sample of the recorded parity checks follows.

| Coordinate | Overworld raw biome ID | Nether raw biome ID | Overworld height |
|---:|---:|---:|---:|
| `(0, 0)` | 12 | 171 | 69 |
| `(100, 100)` | 34 | 171 | 69 |
| `(-100, 40)` | 12 | 8 | 63 |
| `(1000, -700)` | 3 | 170 | audited separately |

An additional 25-point Overworld grid from biome-noise coordinates -128 to
128 matched raw biome IDs exactly. The automated suite retains representative
oracle points so later changes cannot silently replace the source graph with a
lookalike.

## Generated sample registry

| File | Seed | Block extent | Resolution | Stored values |
|---|---:|---:|---:|---|
| `overworld_seed_42_52480.png` | 42 | -26,240 to 26,240 | 161 square | raw biome ID, base height |
| `overworld_seed_42_center_3072.png` | 42 | -1,536 to 1,536 | 193 square | raw biome ID, base height |
| `overworld_spawn_seed_neg4172144997902289642.png` | -4172144997902289642 | -168 to 168 | 169 square | raw biome ID, base height |
| `nether_seed_42_52480.png` | 42 | -26,240 to 26,240 | 161 square | raw biome ID, Y=31 lava-density mask |

Overworld caches were exported by the original-JAR helper. The Nether cache
was exported from the separately parity-checked Python Double Perlin and
density port. Blue channel value 1 identifies the cache format.

## Figure checks

- Overworld and Nether structure overviews are square and span 52,480 by
  52,480 blocks.
- Spawn generation uses a 21 by 21 chunk extent and reveals biome, height, and
  textured surface data only after the corresponding chunk status is reached.
- End City markers describe starts, not End ships. Qualification uses four
  generated heights and the source threshold of 60.
- The dragon ensemble contains 480 exact seeded approaches. Its density adds
  the representative route progressively, and its edge bars use the fixed
  final-ensemble denominator, so neither display can move backward.
- Every graph-bound dragon segment retains its legal-edge highlight. Player
  targeted phases do not invent a graph edge.

## Deliberate remaining projections

The following items remain explicitly explanatory and are not presented as
block-for-block world saves:

- worker-thread timing in the chunk-status animation
- the dragon's reduced top-down steering projection and compressed effect time
- safe outer-gateway endpoint search
- the visible top texture chosen for each raw biome family; biome IDs and base
  heights are source-generated, while full surface builders, carvers, feature
  decoration, and block-volume chunk serialization are outside these figures

## Verification

Commands completed successfully from the repository root unless noted:

```text
python -m py_compile <all changed Python generators and core modules>
python -m unittest discover -s tests -v
cd Code
python render_all.py
```

Final result: 32 tests passed. The complete render regenerated every retained
README plot and animation. GIF decoding, dimensions, timing, final dwell,
file bounds, static flow assets, README references, supported math macros, and
the no-em-dash rule all passed automated checks.
