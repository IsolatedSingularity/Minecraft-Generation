# JAR Inspection Priority List

After running `Inspect-XaeroJar.ps1`, inspect in this order.

## Tier 0: map geometry

```text
MapWriter.loadPixel
MapWriter.loadPixelHelp
MapWriter.writeChunk
MapWriter.shouldOverlayCached
```

Search bytecode/decompiled output for:

```text
getHeight
Heightmap
getBlockState
getFluidState
getLightBlock
isAir
opacity
transparent
overlay
cave
fullCave
topY
```

## Tier 0: final color

```text
MapPixel.getPixelColours
```

Trace every input that contributes to the result array.

## Tier 1: cave layer identity

```text
MapProcessor.getLeafMapRegion
MapRegion constructor
LayeredRegionManager
MapLayer
```

Search for:

```text
caveLayer
caveStart
caveDepth
Integer.MIN_VALUE
-2147483648
```

The official migration FAQ uses `caves/-2147483648` for old Nether maps placed into modern Full cave mode, which is a strong clue about a special Full-layer sentinel.

## Tier 1: texture and biome data

Search:

```text
ColorResolver
Biome
MapColor
TextureAtlas
TextureAtlasSprite
alpha
resource pack
reload
```

## Tier 2: save/cache

```text
MapSaveLoad
WorldDataReader
RegionDetection
LeafRegionTexture
LeveledRegion.saveCacheTextures
```

## High-value call graph to reconstruct

```text
MapWriter.onRender
  -> writeChunk ?
  -> loadPixel
      -> loadPixelHelp
      -> OverlayBuilder
      -> MapPixel construction/update
  -> MapTile / MapTileChunk update
  -> LeafRegionTexture/updateBuffers
```

The names after `onRender` are partly inferred. Use bytecode calls to replace `?` with exact edges.
