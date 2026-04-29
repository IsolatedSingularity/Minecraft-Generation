# Minecraft 1.16 Biome Reference

Source: minecraft.wiki. Target version: **Java Edition 1.16**.

## Biome System Architecture (1.16)

### Overworld: Layer-Based System
1.16 uses the **legacy layer system**, NOT the multi-noise system introduced in 1.18.

The biome is determined through a chain of "layer" transformations:
1. **Island layer**: Random continent/ocean decision
2. **Climate layers**: Temperature and humidity assignment
3. **Biome layer**: Maps (temperature, humidity) pair to a biome ID
4. **Shore layer**: Adds beaches, stone shores at biome/ocean boundaries
5. **River layer**: Carves river biomes through terrain
6. **Hills layer**: Replaces some biomes with "Hills" variants
7. **Zoom layers**: Progressively doubles resolution (bilinear interpolation with randomness)
8. **Voronoi zoom**: Final per-block biome resolution (each chunk stores 4x4 biome grid at section level)

### Overworld Biome Categories

**Snowy** (temperature < 0.2):
- Snowy Tundra, Snowy Taiga, Snowy Mountains, Ice Spikes, Frozen River, Snowy Beach, Snowy Taiga Hills, Snowy Taiga Mountains

**Cold** (temperature 0.2-0.5):
- Mountains, Wooded Mountains, Gravelly Mountains, Modified Gravelly Mountains, Taiga, Taiga Hills, Taiga Mountains, Giant Tree Taiga, Giant Tree Taiga Hills, Giant Spruce Taiga, Giant Spruce Taiga Hills, Stone Shore

**Temperate** (temperature 0.5-0.95):
- Plains, Sunflower Plains, Forest, Flower Forest, Birch Forest, Tall Birch Forest, Birch Forest Hills, Tall Birch Hills, Dark Forest, Dark Forest Hills, Swamp, Swamp Hills, Jungle, Jungle Hills, Jungle Edge, Modified Jungle, Modified Jungle Edge, Bamboo Jungle, Bamboo Jungle Hills, River, Beach, Mushroom Fields, Mushroom Field Shore

**Warm/Dry** (temperature > 0.95):
- Desert, Desert Hills, Desert Lakes, Savanna, Savanna Plateau, Shattered Savanna, Shattered Savanna Plateau, Badlands, Eroded Badlands, Wooded Badlands Plateau, Modified Wooded Badlands Plateau, Modified Badlands Plateau

**Ocean** (special):
- Ocean, Deep Ocean, Warm Ocean, Lukewarm Ocean, Deep Lukewarm Ocean, Cold Ocean, Deep Cold Ocean, Frozen Ocean, Deep Frozen Ocean

### Nether Biome Distribution (1.16)
The Nether uses a simplified 3D Voronoi-based biome picker (NOT the layer system):

| Biome | Weight | Distribution | Surface Block | Fog Color |
|-------|--------|-------------|---------------|-----------|
| Nether Wastes | ~36.3% | Most common | Netherrack | Red-orange |
| Crimson Forest | ~22.2% | Second most | Crimson Nylium | Dark red |
| Soul Sand Valley | ~17.1% | Third | Soul Sand/Soul Soil | Blue |
| Basalt Deltas | ~15.9% | Fourth | Basalt/Blackstone | Light gray |
| Warped Forest | ~8.5% | Rarest | Warped Nylium | Purple |

### End Biomes (1.16 Java Edition)
End biome placement is radial:

| Biome | Location | Condition |
|-------|----------|-----------|
| The End | Central island | Within ~1000 blocks of origin |
| Small End Islands | Transition zone | Between central and outer islands |
| End Midlands | Outer islands | Default outer island biome |
| End Highlands | Outer islands | Highest elevation outer terrain. End cities generate here |
| End Barrens | Outer island edges | Thin perimeter of outer islands |

End biome selection uses moisture noise: higher moisture = Highlands, lower = Barrens/Midlands.

## Mob Spawning by Biome (1.16 Nether)

| Biome | Hostile | Passive/Neutral |
|-------|---------|-----------------|
| Nether Wastes | Ghasts, Magma Cubes, Zombified Piglins | Piglins, Striders, Endermen (rare) |
| Crimson Forest | Hoglins | Piglins, Zombified Piglins, Striders |
| Soul Sand Valley | Skeletons, Ghasts | Endermen (rare), Striders |
| Basalt Deltas | Magma Cubes (high), Ghasts | -- |
| Warped Forest | -- | Endermen, Striders |

**Nether Fortress** (structure, any biome): Blazes, Wither Skeletons (+ 20% Skeleton replacement in soul sand valley/wastes)
**Bastion Remnant** (structure, not basalt deltas): Piglin Brutes (generation only, don't respawn)

## Biome-Specific Generation Details

### Crimson Forest
- Floor: Crimson nylium with patches of bare netherrack or nether wart blocks
- Ceiling: Small patches of nether wart blocks and weeping vines
- Vegetation: Huge crimson fungi (tree equivalent), weeping vines hanging from fungi/ceiling, shroomlights, crimson roots, crimson/warped fungus
- Unique: Fog similar to nether wastes (dark red tint)

### Warped Forest
- Floor: Warped nylium with patches of bare netherrack or warped wart blocks
- Vegetation: Huge warped fungi, twisting vines (grow upward), nether sprouts, shroomlights, warped roots, warped/crimson fungus
- Unique: Magenta-purple fog, NO music plays in this biome
- Safest Nether biome (only endermen + striders, no hostile mobs)

### Soul Sand Valley
- Floor: Soul sand and soul soil
- Features: Exposed nether fossils (bone blocks, 14 designs), basalt pillars (floor to ceiling), soul fire (blue), blue fog
- Terrain: Vast open grottos

### Basalt Deltas
- Floor: Basalt and blackstone, chaotic/uneven surface
- Features: Magma block deltas (constrained lava + magma blocks), basalt columns near lava ocean borders
- Unique: Light gray fog, falling ash particles, NO bastion remnants generate here
- Highest magma cube spawn rate of any biome

### Nether Wastes
- Floor: Netherrack
- Features: Glowstone ceiling clusters, lava springs from ceiling, gravel/soul sand shores at lava sea
- Original Nether biome, renamed from "Nether" in 1.16

---

## Post-1.18 Biome Architecture

> Replaced the layer-based system in Java Edition 1.18+. Documented for version comparison. **1.16 uses the layer system described above.**

### Multi-Noise Biome Source
Biomes selected via `MultiNoiseBiomeSource`: 5D climate parameters mapped to biomes using nearest-neighbor Euclidean distance search.

Each 4x4x4 block section samples 6 values:

| Parameter | Controls | Example |
|-----------|----------|---------|
| Temperature | Heat index (frozen vs tropical) | Frozen biomes < 0.0, deserts > 0.5 |
| Humidity | Precipitation/flora density | Jungles = wet, deserts = dry |
| Continentalness | Ocean vs inland (macro landmass) | Low = ocean basin, high = deep inland |
| Erosion | Flatness | High = swamp/meadow, low = jagged mountains |
| Weirdness | High-contrast biome variations | Peaks/valleys formula, micro-biomes |
| Depth | Vertical biome transitions | Underground biomes independent of surface |

Each biome has an ideal 5D coordinate. Generator assigns the biome whose climate point is closest.

### Peaks and Valleys Formula
Terrain ridge calculation from weirdness noise:
$$P\&V = 1 - |3 \times |\text{weirdness}| - 2|$$

Creates cyclical terrain oscillating from deep valleys to prominent peaks.

### Biome Border Smoothing
- Normalized sparse convolution + randomized jittering on sample points
- Prevents biome borders from aligning with chunk grid
- Standard Voronoi tessellation with displacement for natural transitions

### Underground Biomes (1.18+)
- **Lush Caves**: azalea, glow berries, moss, dripleaf, clay pools
- **Dripstone Caves**: dripstone, pointed dripstone, copper ore
- **Deep Dark** (1.19): sculk, sculk sensors, ancient cities
