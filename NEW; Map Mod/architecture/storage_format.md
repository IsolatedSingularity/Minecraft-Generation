# Xaero Map Storage Model

This section is primarily useful for interoperability and reverse-engineering. Your web viewer does not need to adopt the format.

## Modern directory layout

Official documentation describes a root like:

```text
<game-dir>/xaero/world-map/
```

with server/world folders, then dimensions such as:

```text
null      # Overworld compatibility name
DIM-1     # Nether
DIM1      # End
```

and custom dimension names derived from dimension IDs.

Each dimension can contain:

```text
dimension_config.txt
mw$.../
```

with one map instance per multiworld.

Cave data lives below a `caves/<layer>/` hierarchy in current versions.

## Region footprint

Public format tools converge on this model:

```text
X_Z.zip
  region.xaero
```

One region covers **512 x 512 block columns**, matching a vanilla `.mca` region footprint.

Logical hierarchy:

```text
Region
  8 x 8 TileChunk containers
    4 x 4 MapTiles
      16 x 16 MapPixels
```

`8 * 4 * 16 = 512`.

## Modern 6.8 format example

A public format merger targeting Xaero World Map 1.44.2 on Minecraft 1.20.1 reports format **6.8**.

Important properties:

- a version marker appears at the beginning;
- tile chunks carry compact local coordinates;
- map tiles can be absent;
- each present tile contains 16 x 16 semantic pixels;
- pixels retain height, light, biome and block-state information;
- transparent overlays can be encoded;
- block states and biome values use region-wide incremental palettes;
- palette entries can contain NBT/string values inline when first encountered and integer indexes later.

The crucial lesson is architectural: the region file is **semantic map data**, not merely an image.

## Caches

`.xwmc` files are rendered/derived cache data in modern layouts. Public interoperability tools commonly skip or invalidate them and allow Xaero to rebuild them.

Treat equivalent browser PNG/WebP tiles the same way:

```text
semantic map cache = authoritative/reusable
render image cache = disposable
```

## Import/export recommendation

If your viewer eventually imports Xaero maps:

1. parse region ZIP;
2. normalize each pixel into your `ColumnPixel`;
3. preserve unknown block-state NBT as opaque data when possible;
4. resolve biome/block identities with a version adapter;
5. discard/rebuild image caches.

If exporting to Xaero:

1. target one documented/verified format version only;
2. implement round-trip tests;
3. rebuild state and biome palettes in deterministic write order;
4. let Xaero regenerate `.xwmc`.

Do not make Xaero binary compatibility a dependency of the core seed viewer.
