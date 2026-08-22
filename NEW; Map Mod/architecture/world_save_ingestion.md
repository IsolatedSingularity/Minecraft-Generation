# World-Save Ingestion

Xaero's official FAQ states that in singleplayer the map can be loaded directly from the world save. It can also convert generated world-save regions into a multiplayer-compatible map.

This is highly relevant to a seed viewer because your backend occupies a similar position: it has authoritative chunk data rather than only the player's currently loaded chunks.

## Likely subsystem split

Public references expose concepts such as:

```text
MapDimension.isUsingWorldSave()
RegionDetection
WorldDataReader
world-save region detection
```

A reasonable subsystem model is:

```text
Anvil region discovery
   |
   +--> .mca / .mcr coordinates
   |
   v
chunk NBT read
   |
   v
block states + biomes + heightmaps
   |
   v
same MapWriter / semantic-pixel pipeline
```

## Seed viewer difference

You do not need Anvil input for generated terrain. Build an adapter interface:

```text
WorldSource
  getChunk(cx, cz)
  getBlockState(x, y, z)
  getBiome(x, y, z)
  getHeightmap(type, x, z)
  minY()
  maxY()
  dimensionPolicy()
```

Implementations can be:

```text
SeedGeneratorSource
AnvilWorldSource
XaeroRegionImportSource
```

All three feed the same renderer.
