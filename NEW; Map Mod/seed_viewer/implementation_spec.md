# Version-Independent Seed Viewer Implementation Spec

## Design objective

Reproduce the **useful visual logic** of a modern Xaero-style map without coupling the renderer to a particular Minecraft release or to Xaero's internal classes.

The strongest architecture is three layers:

```text
A. Versioned world model
B. Version-independent semantic map
C. Version-independent renderer + web tiler
```

## A. Versioned world model

Interface:

```ts
interface WorldAdapter {
  minY(): number;
  maxY(): number;

  getBlockState(x: number, y: number, z: number): BlockStateKey;
  getBiome(x: number, y: number, z: number): BiomeKey;
  getHeight(x: number, z: number, type: HeightmapType): number;

  blockVisual(state: BlockStateKey): BlockVisualInfo;
  biomeVisual(biome: BiomeKey): BiomeVisualInfo;
  dimensionPolicy(): DimensionPolicy;
}
```

`BlockVisualInfo`:

```ts
interface BlockVisualInfo {
  baseColor: number;
  mapColor?: number;
  alphaClass: "opaque" | "cutout" | "translucent";
  averageAlpha: number;
  isFluid: boolean;
  blocksLight: number;
  tintType: "none" | "grass" | "foliage" | "water" | "custom";
}
```

`DimensionPolicy`:

```ts
interface DimensionPolicy {
  hasSkyLight: boolean;
  hasCeiling: boolean;
  logicalHeight: number;
  roofPolicy: "none" | "nether-like" | "detect";
}
```

## B. Semantic map

Do not cache only RGBA.

```ts
interface ColumnPixel {
  y: number;
  topHeight: number;
  baseState: BlockStateKey | null;
  biome: BiomeKey | null;
  overlays: OverlayPixel[];
  blockLight: number;
  skyLight: number;
  caveDepth: number;
  flags: number;
}

interface OverlayPixel {
  state: BlockStateKey;
  alpha: number;
  light: number;
}
```

For arbitrary cave `Top Y`, a richer cache is even better:

```ts
interface VerticalVisibilityProfile {
  x: number;
  z: number;
  candidates: SurfaceCandidate[]; // sorted high -> low
}
```

A candidate records a visually meaningful solid/overlay transition. Then changing Top Y is a binary search rather than a chunk regeneration.

## C. Renderer

```ts
interface MapStyle {
  caveMode: "off" | "layered" | "full";
  topY: number;
  legibleCaves: boolean;

  biomeTint: boolean;
  terrainShade: boolean;
  lighting: "normal" | "flat";
  transparency: boolean;

  showAboveRoof: boolean;
}
```

### Surface selection

```text
profile = getVerticalProfile(x,z)

if caveMode == OFF:
    candidate = profile.surfaceCandidate()

if caveMode == LAYERED:
    candidate = profile.firstCandidateAtOrBelow(topY, cavePolicy)

if caveMode == FULL:
    candidate = profile.firstInteriorCandidate(fullCavePolicy)
```

### Pixel color

Conceptual pipeline:

```text
rgb = resolveBaseColor(candidate.baseState)

if biomeTint:
    rgb = applyBiomeTint(rgb, candidate.biome, state.tintType)

if transparency:
    rgb = compositeOverlays(rgb, candidate.overlays)

if terrainShade:
    rgb *= terrainShadeFactor(neighborHeights)

if legibleCaves and caveMode != OFF:
    rgb *= caveDepthFactor(candidate.caveDepth)
else if lighting == NORMAL:
    rgb *= lightFactor(candidate.skyLight, candidate.blockLight)

return rgba(rgb, 255)
```

Calibrate exact order after licensed-JAR inspection.

## Backend tiler

Suggested layout:

```text
world/
  metadata.json
  semantic/
    <dimension>/<z>/<x>.bin
  render/
    <style-hash>/<zoom>/<x>/<z>.webp
```

`style-hash` should include only values that change final pixel colors.

## Cache invalidation

Separate hashes:

```text
worldgenHash:
  seed
  MC version
  dimension
  generator settings

visualHash:
  resource pack
  biome/color tables
  map style
  Xaero-parity profile version
```

A style change should not invalidate worldgen.

## Browser behavior

Frontend should be dumb and fast:

1. calculate visible tile coordinates;
2. request image tiles;
3. composite markers/structures in a separate vector layer;
4. never run Minecraft worldgen in the UI thread;
5. use a worker only for cheap recoloring if semantic tiles are sent client-side.

## Structure overlay

Keep generated structures independent of terrain rasterization:

```text
terrain raster
+ structure footprint/vector layer
+ POI markers
+ route/measurement layer
```

This makes your seed viewer more powerful than Xaero without contaminating map parity.

## Suggested implementation order

### Phase 1
Overworld, opaque top surface, biome tint, height shade.

### Phase 2
water/transparency overlays and exact texture-derived colors.

### Phase 3
Nether Full cave mode and roof hiding.

### Phase 4
arbitrary Top Y with vertical visibility profiles.

### Phase 5
legible cave depth shading.

### Phase 6
resource-pack parity and golden-image calibration.

### Phase 7
optional Xaero region import/export.
