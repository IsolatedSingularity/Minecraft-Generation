# Minecraft Procedural Generation and Mob Pathing Analysis
###### Under the direction of various open community sources such as the [Minecraft Wiki](https://minecraft.wiki/), referencing technical insights from [Alan Zucconi’s world generation articles](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/) and discussions on [Reddit](https://www.reddit.com/r/Minecraft/) and [Gaming Stack Exchange](https://gaming.stackexchange.com/).

![alt text](Plots/VillageDistribution.png "Sample Village Distribution in the Overworld")

## Objective

The procedural world generation in Minecraft involves intricate algorithms that produce diverse biomes, structures, and entity behaviors. This repository focuses on modeling several key aspects:

1. **Village Generation:** Simulating village placement across a large coordinate range (e.g., −10,000 to 10,000). Villages appear only in certain biomes (Plains, Savanna, Taiga, Snowy Plains, Desert [1]) and are influenced by a “biome stability” metric. This metric affects how likely a village is to spawn and adapt to local conditions.

2. **Stronghold Distribution:** Representing strongholds as generated in concentric rings around the origin [2]. By using deterministic polar-to-Cartesian conversions and slight randomness, we mimic the known radii and counts per ring, aligning with in-game patterns.

3. **Ender Dragon Pathing:** Modeling the Ender Dragon’s movement in the End dimension as a graph traversal problem [3]. Nodes represent structures like the fountain center, inner and outer rings, and obsidian towers. Edges are assigned probabilities, and by sampling these probabilities, we can determine the dragon’s likely path. This simulates the dragon’s circling, perching, and strafing behaviors, incorporating logic on choosing edges with certain probabilities.

4. **Noisy Biome-Based Suitability Maps:** Employing multiple frequencies of sine/cosine noise and random perturbations to generate complex, fragmented biome patterns [4]. Each biome’s suitability score for village spawning arises from these noisy temperature/humidity fields, avoiding overly uniform shapes.

## Code Functionality

### Village Generation with Reduced Density

Naive random distributions produce unrealistically dense village placements. Instead, we divide the entire area into 32x32 chunk regions (512x512 blocks). Each region is assigned at most one potential village, with a certain probability. This aligns better with Minecraft’s actual village rarity and distribution [1].

We define trust or suitability scores:
- Plains: High suitability (∼1.0)
- Savanna: Moderate to high (∼0.9)
- Taiga: Moderate (∼0.8)
- Snowy Plains: Lower (∼0.7)
- Desert: Even lower but non-zero (∼0.6)
- Other biomes: Near zero (0 to 0.1)

These scores derive from noise-based temperature/humidity maps. This approach provides a more realistic representation of biome-dependent village placement.

### Stronghold Distribution

Strongholds are arranged in concentric rings with known radii and counts [2]. By mapping rings to polar coordinates, we place strongholds at even angular intervals, adding slight random variation for authenticity. This captures the known pattern of stronghold distribution around the world spawn point.

### Ender Dragon Pathing

We represent the Ender Dragon’s movement as a graph traversal [3]:
- Nodes: Fountain center, inner and outer rings, towers.
- Edges: Each edge (u → v) has probability p_(u→v). The dragon samples these probabilities when moving.
- Behavior: Different edges correspond to typical dragon actions (circling pillars, perching, strafing attacks). By simulating multiple steps, we derive probable flight paths and states.

### Noisy Biome-Based Suitability Maps

To avoid large uniform regions, we combine multiple sine/cosine terms and inject random perturbations. This ensures each biome emerges from a complex noise field, breaking clean boundaries and producing more natural transitions [4].

![alt text](Plots/NoisyBiomeSuitability.png "Noisy Biome Suitability Map")

## Caveats

- High complexity noise layering can be computationally expensive.
- The probabilities for villages, strongholds, and dragon movements are heuristic. Perfect authenticity would require analyzing Minecraft’s actual code or reverse-engineered parameters.
- The dragon pathing model is simplified and does not capture all in-game conditions (like crystal healing phases or player proximity triggers).

## Next Steps

- Integrate temporal dynamics: Seasonal changes or evolving conditions affecting biome distributions.
- Refine noise: Replace manual sine/cosine functions with Perlin or Simplex noise for more realism.
- Combine systems: Study correlations between stronghold placement and village distributions.
- Enhance dragon logic: Introduce state machines or triggers for more complex dragon behavior.

## References

1. Minecraft Wiki: [Village](https://minecraft.wiki/w/Village)  
2. Sportskeeda Wiki: [Stronghold](https://wiki.sportskeeda.com/minecraft/stronghold)  
3. Minecraft Wiki: [Ender Dragon](https://minecraft.wiki/w/Ender_Dragon)  
4. Reddit Discussion on Village Algorithm: [Reddit Thread](https://www.reddit.com/r/Minecraft/comments/ahab1g/what_is_minecrafts_algorithm_for_deciding_the/)  
   Alan Zucconi’s World Generation Articles: [Zucconi’s Blog](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/)


