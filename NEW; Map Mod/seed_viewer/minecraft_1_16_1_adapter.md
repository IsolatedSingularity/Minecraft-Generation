# Minecraft 1.16.1 Adapter Notes

The renderer should remain modern. This adapter supplies 1.16.1 terrain semantics.

## World bounds

For vanilla Java 1.16.1, use the legacy 0..255 build-height model.

## Nether

The adapter should know that vanilla Nether generation has a ceiling/roof. For parity with a modern Xaero Full cave view:

- do not treat the top bedrock roof as the map surface;
- enter the generated cave volume below the roof;
- preserve lava as a strong visible surface/overlay;
- preserve Nether biome color distinctions where your style uses them.

## Heightmaps

1.16.1 chunk data exposes heightmap information, but do not make the renderer depend on it. A heightmap is a starting hint for a column scan.

This is useful because historical Xaero 1.9.0 explicitly added a setting to ignore server heightmaps, demonstrating that robust map rendering needs a fallback to actual column inspection.

## Biomes

Wrap version-specific biome storage behind:

```text
getBiome(x,y,z)
```

Even when the underlying version stores/derives biomes differently than modern releases.

## Block visual registry

Build a generated 1.16.1 registry containing:

```text
block state key
map color
texture average color
texture alpha statistics
fluid flag
tint type
light opacity
```

Generate this once from the official 1.16.1 client resources / data available in your existing Minecraft reference library.

## Important

The historical Xaero 1.10.1 build is useful for:

- old class/method evolution;
- heightmap behavior;
- early map pixel/storage logic;
- lighting comparison.

It should **not** define your cave-mode UI/semantics because modern cave-mode behavior came later.
