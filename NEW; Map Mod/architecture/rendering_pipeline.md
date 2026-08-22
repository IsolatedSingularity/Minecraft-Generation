# Reconstructed Rendering Pipeline

## Executive model

The most useful abstraction is:

```text
worldgen / chunk data
        |
        v
column sampler
        |
        +--> visible base block
        +--> transparent/fluid overlays
        +--> biome
        +--> height/topHeight
        +--> light
        +--> cave metadata
        |
        v
MapPixel-like semantic record
        |
        v
pixel color resolver
        |
        +--> base block/map/texture color
        +--> biome tint
        +--> transparency compositing
        +--> terrain/height shading
        +--> light or cave-depth shading
        |
        v
RGBA tile
        |
        v
region / zoom cache / GPU or browser texture
```

## 1. Column selection

**Observed + inferred.**

Public integration hooks show `MapWriter.loadPixel` reading a `LevelChunk`, `BlockState`, fluid state and block opacity. This makes a vertical block-column scan the central primitive.

For a surface map:

1. choose an initial top Y, often from a heightmap or chunk/world metadata;
2. inspect the block at `(x, y, z)`;
3. peel blocks that should behave as transparent overlays;
4. stop on the primary visible map surface;
5. retain both the base surface and one or more overlays.

For a cave map:

1. derive a search ceiling from cave mode and `Top Y`;
2. intentionally ignore roof/ceiling surfaces when necessary;
3. find the first interior surface satisfying cave visibility rules;
4. record cave depth / layer metadata.

The exact skip/stop tests should be confirmed in `MapWriter.loadPixel` and `loadPixelHelp`.

## 2. Overlay construction

**Observed.**

`OverlayBuilder` and `shouldOverlayCached(...)` are visible in public mixin targets. Fluids are explicitly queried. Therefore transparent surfaces are not merely discarded. They can be represented as overlays composited over a lower base state.

Examples likely to matter:

- water;
- ice/glass-like blocks;
- leaves;
- snow layers;
- flowers / non-full blocks;
- modded blocks with alpha;
- resource-pack-dependent texture transparency.

Modern changelogs indicate actual block texture transparency became important for deciding transparency, so a faithful web renderer should not classify transparency solely by hard-coded block categories.

## 3. Semantic pixel

A practical semantic pixel for your viewer:

```text
ColumnPixel
  surfaceY
  topHeight
  baseStateId
  biomeId
  overlays[]
  blockLight
  skyLight
  caveDepth
  caveLayer
  flags
```

Keep this representation independent of Minecraft version and independent of the browser.

## 4. Color resolution

**Observed + inferred.**

`MapPixel.getPixelColours(...)` is an explicit late-stage color function.

Recommended web equivalent:

```text
base color
  -> biome tint
  -> overlay composition
  -> terrain/slope shade
  -> lighting or cave-depth shade
  -> final RGBA
```

Do not bake biome tint or lighting into your worldgen cache if you want resource-pack changes, style changes or multiple map modes without regenerating the terrain.

## 5. Biome tint

Modern Xaero versions retain biome information per pixel. Public format readers show biome information in region data, and public renderers use it for biome tint.

Your resolver should support at least:

- grass color;
- foliage color;
- water color;
- dimension/biome overrides;
- version-specific biome climate coloring;
- optional resource-pack/custom resolver inputs.

For exact Minecraft parity, use version-specific biome and color tables.

## 6. Height and terrain shading

Xaero visually encodes terrain relief. Old and modern file formats retain height information, including a distinction between a surface/top height in newer formats.

A good approximation is a neighbor-gradient shade:

```text
dx = h(x+1,z) - h(x-1,z)
dz = h(x,z+1) - h(x,z-1)

slopeShade = clamp(kx*dx + kz*dz, minShade, maxShade)
```

However, do not assume this is Xaero's exact formula. Confirm the relevant color/shading code from your licensed JAR and calibrate against exported PNGs.

## 7. Lighting

Xaero exposes actual light information in map records and has changed light behavior across versions.

Maintain separate fields for:

```text
skyLight
blockLight
caveDepth
```

Then implement render modes:

```text
NORMAL_LIGHTING = f(skyLight, blockLight, time/profile)
LEGIBLE_CAVE    = g(caveDepth or distance below cave ceiling)
FLAT            = 1.0
```

This prevents a seed viewer from needing to rerun worldgen just to change visual style.

## 8. Tiling

Use a two-level tiling scheme:

```text
semantic terrain tile: 16 x 16 or 64 x 64 columns
render tile:           256 x 256 or 512 x 512 pixels
zoom pyramid:          downsampled render products
```

The semantic tile should preserve map data; browser image tiles are disposable caches.

## 9. What is known vs uncertain

High confidence:

- column-based map sampling;
- base state + overlay model;
- biome per pixel in modern format;
- height + light retention;
- cave/full-cave modes;
- 512 x 512 saved region footprint;
- layered region/cache system.

Needs JAR confirmation for pixel-perfect parity:

- exact surface skip predicate;
- exact alpha threshold and texture sampling;
- exact biome tint order;
- exact height/slope shade formula;
- exact light transfer curve;
- exact cave interior selection;
- exact Full cave visibility rule around Nether roof;
- exact downsampling/filtering used for zoom caches.
