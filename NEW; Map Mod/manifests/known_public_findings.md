# Known Public Findings

These findings are distilled from official documentation and public interoperability projects.

## Official behavior

- The mod is Xaero's World Map.
- Modern releases support cave dimensions.
- Cave mode has a `Cave Mode Top Y` setting.
- Full cave mode is recommended for Nether-like dimensions when mapping below the roof.
- `Legible Cave Maps` changes cave readability using depth-oriented behavior.
- Singleplayer can map directly from the world save.
- Generated singleplayer regions can be converted into a multiplayer-compatible map.
- Modern map data lives under `xaero/world-map`.
- Default dimension compatibility directory names include `null`, `DIM-1`, and `DIM1`.
- Map instances are stored as subdirectories under each dimension.
- Cave data is stored beneath cave-layer directories.

## Public class-surface findings

Public add-ons reference:

```text
MapProcessor
MapWriter
MapPixel
MapRegion
MapTile
MapTileChunk
OverlayBuilder
LayeredRegionManager
LeveledRegion
LeafRegionTexture
BlockStateShortShapeCache
MapSaveLoad
MapWorld
MapDimension
RegionDetection
GuiMap
```

Particularly important observed relationships:

- `MapWriter.loadPixel` sees chunk/block/fluid/opacity/cave inputs.
- `MapWriter` uses `OverlayBuilder`.
- `MapProcessor.getLeafMapRegion` takes cave layer and region coordinates.
- `MapPixel.getPixelColours` is a distinct final color stage.
- region save uses ZIP + binary data streams.
- render caches use a region-texture hierarchy.

## Public file-format findings

- A saved multiplayer region covers 512 x 512 columns.
- Public parsers model 8 x 8 subcontainers, each containing 4 x 4 tiles, each tile containing 16 x 16 pixels.
- Modern formats preserve semantic information such as block state, biome, height, light, overlays.
- Modern formats use palettes for block states and biomes.
- `.xwmc` is treated by public tools as rebuildable cache data.
