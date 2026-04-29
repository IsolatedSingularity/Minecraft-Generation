# Minecraft-Generation Audit Checklist
# Used by @shinra-minecraft during audit sessions.
# Read this file when running a full audit.

## Version Compliance (1.16)
- [ ] No post-1.16 features: noise caves, multi-noise biomes, aquifers, extended height?
- [ ] Height range Y=0 to Y=255 (not Y=-64 to Y=320)?
- [ ] Biome system: layer-based (temperature/humidity), NOT multi-noise 6-parameter?
- [ ] Cave system: carver caves (random walk + ravines), NOT noise caves?
- [ ] No post-1.16 blocks: deepslate, tuff, dripstone, copper, amethyst?
- [ ] Structure salts: correct values from worldgen-pipeline.md?

## Dimension Coverage
- [ ] Overworld: terrain heightmap, ore distribution, biome placement, caves?
- [ ] Nether: 5 biomes (1.16 headline), terrain cross-section, lava sea Y=31?
- [ ] Nether structures: fortress + bastion remnant with correct salts?
- [ ] Nether ores: ancient debris (peak Y=8-22, always buried)?
- [ ] End: outer islands, End City placement, obsidian pillars?
- [ ] Ender Dragon: state model (7 or 11 phases)? Pathfinding nodes?

## Algorithm Accuracy
- [ ] LCG seed formula: correct implementation for structure placement?
- [ ] Perlin noise: proper multi-octave implementation (not sine-wave approximation)?
- [ ] Ore vein generation: correct per-ore Y-range and vein count statistics?
- [ ] Biome layer system: temperature/humidity layer stack correct?
- [ ] Structure bounding boxes: non-overlapping placement logic?

## Code Quality
- [ ] camelCase: all variables, functions, class methods?
- [ ] Type annotations: on all public functions?
- [ ] numpy vectorized ops: no nested Python loops for tile-level operations?
- [ ] Code consolidation: shared code in Code/core/ (not duplicated across scripts)?
- [ ] Dark theme visualization: plt.style.use('dark_background') with #0D1117?
- [ ] Output paths: all visualization to Plots/ directory?
- [ ] No backup files (e.g., _backup.py) in tracked code?

## Implementation Coverage
- [ ] How many scripts exist? What does each implement?
- [ ] Which dimensions are covered?
- [ ] Gap analysis: what is missing vs existing-repo-analysis.md?
- [ ] Core library (Code/core/): what utilities exist vs what is still inline?

## Output Format
```
## Minecraft-Generation Audit
**Date**: [date]
**Agent**: @shinra-minecraft
**Version Target**: Java Edition 1.16

### Version Compliance
| # | Check | Status | Evidence | Notes |
|---|-------|--------|----------|-------|

### Dimension Coverage
| # | Dimension | Feature | Status | Evidence |
|---|-----------|---------|--------|----------|

### Algorithm Accuracy
| # | Check | Status | Evidence | Notes |
|---|-------|--------|----------|-------|

### Code Quality
| # | Check | Status | Evidence | Notes |
|---|-------|--------|----------|-------|

### Implementation Gaps (severity-ranked)
| # | Severity | Missing Feature | Priority Phase | Effort |
|---|----------|----------------|----------------|--------|

### Summary
[2-3 sentences on overall state and recommended next steps]
```
