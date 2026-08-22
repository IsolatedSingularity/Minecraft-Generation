# Publicly Exposed Class Surface

This is a map of important Xaero World Map classes revealed by public interoperability projects such as XaeroPlus and xaero-world-map-bridge. It is not a dump of Xaero's source.

## Core orchestration

### `xaero.map.MapProcessor`

Observed responsibilities:

- tracks current Minecraft world, map world, dimension and multiworld IDs;
- transitions map state when worlds/dimensions change;
- runs background map processing;
- returns leaf regions with `getLeafMapRegion(caveLayer, regX, regZ, create)`;
- coordinates render pauses and map-world synchronization.

Useful conclusion: region identity is at least a function of:

```text
world / dimension / multiworld / cave-layer / region-x / region-z
```

### `xaero.map.MapWriter`

This is the highest-value rendering target.

Public mixin anchors expose methods including:

- `loadPixel`
- `loadPixelHelp`
- `writeChunk`
- `onRender`
- `shouldOverlayCached`

Inputs/interactions observed around those methods include:

- `LevelChunk`
- `BlockState`
- `FluidState`
- block opacity / `getLightBlock`
- `OverlayBuilder`
- cave and full-cave boolean state
- `MapRegion`
- current dimension

This strongly supports a column-sampling pipeline: scan a world column, choose the visible base state, construct transparent/fluid overlays, retain height/light/biome data, then turn that into map pixels.

## Pixel representation

### `xaero.map.region.MapPixel`

Observed:

- stores or references a `BlockState`;
- has a `getPixelColours(...)` stage;
- output includes a channel used as opacity by an addon.

Interpretation: Xaero separates **map data construction** from **final pixel color computation**, which is a good architecture for a web implementation too.

## Region hierarchy

Observed classes:

- `MapRegion`
- `MapTileChunk`
- `MapTile`
- `LeafRegionTexture`
- `RegionTexture`
- `LeveledRegion`
- `LayeredRegionManager`
- `MapLayer`

Public file-format tools independently establish the following practical hierarchy for saved multiplayer map data:

```text
MapRegion: 512 x 512 block-pixels
  8 x 8 TileChunks
    4 x 4 MapTiles
      16 x 16 Pixels
```

Since 8 * 4 * 16 = 512, a `MapTile` corresponds naturally to a Minecraft chunk footprint and a region corresponds to a vanilla Anvil region footprint.

## World and dimensions

### `xaero.map.world.MapWorld`
### `xaero.map.world.MapDimension`

Observed capabilities:

- current dimension selection;
- dimension lookup;
- multiplayer/singleplayer distinction;
- whether map data comes from a world save;
- detection of world-save regions.

## Storage and caches

### `xaero.map.file.MapSaveLoad`

Public injection points confirm region saves use `ZipOutputStream` plus `DataOutputStream`.

### `LeveledRegion.saveCacheTextures`

Public injection points confirm rendered cache textures are also written through ZIP/binary streams.

### `BlockStateShortShapeCache`

This cache is used around map tile updates and has IO-thread interaction. It is likely an optimization for converting BlockStates to compact render/shape information.

## Other classes worth locating in a licensed JAR

Search for:

```text
xaero.map.WorldMap
xaero.map.MapWriter
xaero.map.MapProcessor
xaero.map.region.MapPixel
xaero.map.region.MapRegion
xaero.map.region.MapTile
xaero.map.region.MapTileChunk
xaero.map.region.OverlayBuilder
xaero.map.region.LeveledRegion
xaero.map.region.LayeredRegionManager
xaero.map.region.texture.LeafRegionTexture
xaero.map.region.texture.RegionTexture
xaero.map.cache.BlockStateShortShapeCache
xaero.map.file.MapSaveLoad
xaero.map.file.RegionDetection
xaero.map.world.MapWorld
xaero.map.world.MapDimension
xaero.map.gui.GuiMap
```

Also search names containing:

```text
WorldDataReader
WorldData
Biome
Color
Colour
Light
Height
Cave
Overlay
Texture
Cache
Region
Pixel
Writer
Export
PNG
```
