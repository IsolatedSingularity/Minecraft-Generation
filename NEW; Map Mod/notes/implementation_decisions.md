# Recommended Decisions for the Seed Viewer

## Reuse the concepts, not the binary architecture

Copying Xaero's exact region/cache system would add complexity without helping a web seed viewer.

Keep:

- semantic pixels;
- overlays;
- per-pixel biome;
- height/top height;
- independent lighting;
- cave layers/top Y;
- disposable render caches.

Replace:

- Java/Minecraft `BlockState` references with stable namespaced IDs;
- Xaero region ZIPs with your own indexed binary tile format;
- GPU client cache hierarchy with web tile pyramids;
- live-world exploration updates with deterministic seed generation.

## Best optimization: vertical visibility profiles

For arbitrary cave Y, the expensive operation is repeatedly scanning the same vertical columns.

During chunk generation, emit a compact list of all potentially visible surfaces.

Then:

```text
surface mode -> candidate 0
layered y=97 -> first candidate <= 97 matching interior policy
layered y=64 -> first candidate <= 64 matching interior policy
full -> first candidate after roof transition
```

This converts cave-Y scrubbing into cheap selection.

## Keep colors late-bound

Do not write final colors into the semantic cache.

Reasons:

- resource packs;
- biome-style changes;
- Xaero parity tuning;
- day/night;
- legible caves;
- custom map themes.

## Make parity a named renderer profile

Example:

```text
rendererProfile = "xaero-modern-1"
```

Later you can add:

```text
vanilla-map
satellite
terrain
structure-debug
```

without changing worldgen.
