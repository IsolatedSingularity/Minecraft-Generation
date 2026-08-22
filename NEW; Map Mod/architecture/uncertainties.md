# Open Questions to Resolve from the Licensed JAR

These are the high-value unknowns.

## P0: exact cave surface rule

Find in:

```text
xaero.map.MapWriter.loadPixel
xaero.map.MapWriter.loadPixelHelp
```

Questions:

- How is the starting Y selected?
- How are heightmaps trusted/ignored?
- What qualifies as transparent?
- How does Full cave mode enter the interior below a roof?
- How does Layered mode quantize or label cave layers?
- How are caveStart and caveDepth generated?

## P0: exact pixel shading

Find in:

```text
xaero.map.region.MapPixel.getPixelColours
```

Questions:

- order of biome tint, alpha, height shade and lighting;
- slope/neighbor inputs;
- light transfer function;
- cave-depth transfer function;
- color-space behavior.

## P1: texture transparency

Find code referencing:

```text
sprite
atlas
texture
alpha
opacity
transparent
resource pack
reload
```

Need to determine whether a texture is classified by:

- any alpha < 255;
- average alpha;
- sampled texels;
- baked model/face information;
- material/block properties plus texture alpha.

## P1: zoom/cache generation

Search:

```text
LeafRegionTexture
RegionTexture
LeveledRegion
updateBuffers
saveCacheTextures
```

Determine:

- texture resolution per region level;
- mip/downsample policy;
- interpolation mode;
- cache invalidation.

## P2: world-save fast path

Search:

```text
WorldDataReader
RegionDetection
detectRegions
mca
xwmc
```

Determine how much semantic processing is shared with live chunk rendering.

## Recommended evidence capture

For every candidate method, save:

```text
javap bytecode
decompiled method
method descriptor
called-method list
string literals
version hash
```

Then compare current and 1.16.1 builds.
