# Cave Mode Reconstruction

The screenshot is especially useful because the map is in the Nether and displays `Top Y: 97`.

## Official behavior

Xaero documentation exposes three conceptual states:

### OFF

Render the normal above-ground/surface map.

### Layered

Use a cave-map layer associated with a chosen **Cave Mode Top Y**. The top Y matters and cave maps can be separated into layers.

### Full

Ignore the ordinary top-Y cutoff and treat the dimension as a full cave volume from the world's top toward the bottom. Xaero's FAQ recommends this for Nether-like dimensions and describes toggling below/above the bedrock roof.

### Legible Cave Maps

Use depth-oriented cave shading instead of ordinary block lighting so distinct vertical cave structures are easier to read.

## What the screenshot likely represents

The map appears to be:

```text
dimension = Nether
cave rendering = active
Top Y = 97
roof/upper solids = not treated as the visible map surface
lava = bright overlay/base feature
interior terrain = dark red/brown/teal biome-colored surfaces
```

The `Top Y` display does not by itself prove whether the selected cave type is Layered or Full because the UI can display the setting even when Full ignores it.

## Seed-viewer model

Represent cave parameters explicitly:

```json
{
  "mode": "off | layered | full",
  "topY": 97,
  "legible": true,
  "showAboveRoof": false,
  "layerStep": 16
}
```

`layerStep` is your own implementation parameter until the exact Xaero layer quantization is confirmed.

## Layered pseudocode

```text
renderLayeredColumn(x, z, topY):
    y = clamp(topY, minY, maxY)

    while y >= minY:
        state = block(x,y,z)

        if shouldSkipAsCeilingOrTransparent(state, x,y,z):
            y -= 1
            continue

        if isInteriorSurfaceCandidate(x,y,z):
            return buildPixel(x,y,z)

        y -= 1

    return unknownPixel
```

The exact meaning of "interior surface candidate" is the key unknown. Inspect `MapWriter.loadPixel` / `loadPixelHelp`.

## Full cave pseudocode

Do not simply render the Nether roof.

A robust behavioral approximation:

```text
renderFullCaveColumn(x,z):
    roofBottom = detectPersistentCeilingOrConfiguredRoof(x,z)

    y = roofBottom - 1
    while y >= minY:
        if isInteriorSurfaceCandidate(x,y,z):
            return buildPixel(x,y,z)
        y -= 1

    return unknownPixel
```

For a version-independent implementation, make roof detection a dimension adapter policy rather than hard-coding Nether Y=127.

## Better roof detection

Possible policies, in order of preference:

1. **Dimension semantics**: Nether adapter knows the generator has a bedrock ceiling.
2. **Generated density topology**: find an upper connected solid ceiling and enter the first interior air cavity.
3. **Configurable cutoff**: user defines roof cut Y.
4. **Hard-coded legacy fallback**: acceptable only for a strict 1.16.x Nether adapter.

## Legible cave shading

A useful reconstruction:

```text
depth = caveCeilingY - surfaceY
normalized = clamp(depth / maxReadableDepth, 0, 1)
light = mix(brightFactor, darkFactor, normalized)
```

Xaero's exact curve may use other cave metadata. Store depth separately so the curve can be calibrated later.

## Why this matters for an online seed viewer

A seed viewer already owns complete generated terrain, unlike a normal multiplayer client. That means it can render Full cave mode much more cleanly:

- no exploration dependency;
- no cache holes;
- arbitrary top Y;
- instant layer switching if semantic column profiles are cached;
- server-side high-resolution exports.

A particularly strong design is to cache not a single pixel per column, but a compact **vertical visibility profile** containing every transition that could become a map surface. Then any top Y can be resolved without regenerating the chunk.
