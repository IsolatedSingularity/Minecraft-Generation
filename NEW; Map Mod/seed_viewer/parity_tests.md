# Golden-Image Parity Test Plan

This is the fastest route from "Xaero-like" to "very close to Xaero".

## 1. Build controlled worlds

Use the same seed and generated coordinates in:

- a local Minecraft profile containing Xaero;
- your seed viewer.

Test regions should include:

- flat plains;
- ocean + shore;
- forest;
- mountains/cliffs;
- snow;
- transparent blocks;
- Nether wastes;
- warped forest;
- crimson forest;
- basalt delta;
- lava sea;
- bedrock roof;
- cave intersections at several Y values.

## 2. Export/reference

Xaero supports PNG map export. For each test:

```text
dimension
coordinate rectangle
cave mode
Top Y
legible cave setting
resource pack
time/lighting profile
```

record an exported PNG and configuration.

## 3. Align raster coordinates

Crop/warp only once to establish the exact block-to-pixel transform. Do not visually eyeball screenshots.

## 4. Metrics

Compute:

```text
pixel exact match %
mean absolute RGB error
95th percentile RGB error
SSIM-like structural comparison
edge mismatch %
```

Also make a diff heatmap.

## 5. Isolate systems

Run toggles one at a time:

```text
flat/no tint/no light
+ biome tint
+ transparency
+ height shading
+ lighting
+ cave selection
+ legible cave shading
```

This reveals which formula is wrong.

## 6. Cave-specific tests

At one Nether `(x,z)` rectangle export:

```text
Full
Layered TopY 120
Layered TopY 97
Layered TopY 64
Layered TopY 32
```

Compare selected `surfaceY` first, before comparing RGB. If geometry selection is wrong, color tuning is wasted work.

## 7. Regression corpus

Keep a small permanent corpus:

```text
golden/
  mc-1.16.1/
  mc-modern/
  metadata.jsonl
```

Every renderer change should rerun the corpus.
