# Minecraft Procedural Generation and Mob Pathing Analysis
###### Under the supervision of Community Wiki Knowledge ([Minecraft Wiki](https://minecraft.wiki/)), drawing from multiple technical sources such as [Alan Zucconi’s world generation articles](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/) and community discussions on [Reddit](https://www.reddit.com/r/Minecraft/) and [Gaming Stack Exchange](https://gaming.stackexchange.com/).

![alt text](Plots/villageDistributionExample.png "Example: Village Distribution in the Overworld")

## Objective

Minecraft’s procedural world generation involves complex algorithms to produce diverse biomes, structures, and entity behaviors. This repository models several core aspects of that generation and analysis, focusing on:

- **Village Generation:** Simulating village placement across a large 20,000 x 20,000 block region (–10,000 to 10,000). Villages only spawn in certain biomes (e.g., Plains, Savanna, Taiga, Snowy Plains, Desert [1]) and each biome has a “biome stability” score. This stability influences how likely a village is to appear and adapt to the local environment.

- **Stronghold Distribution:** Representing strongholds as placed in concentric rings around the origin [2]. By assigning deterministic polar coordinates to stronghold locations and then mapping to Cartesian space, we visualize their distribution. We base this on known radii and stronghold counts per ring, adding slight randomness to emulate the in-game procedural noise.

- **Ender Dragon Pathing and Towers:** Modeling the Ender Dragon’s movement in the End as a graph traversal problem [3]. The End dimension features nodes representing the fountain, inner rings, outer rings, and towers. Each edge has a probability weight. By sampling these probabilities, we can determine the dragon’s likely flight paths. Higher probabilities reflect common behaviors (circling pillars), while lower probabilities might represent rarer strafing attacks.

- **Noisy Biome-Based Suitability Maps:** Using layered sine/cosine functions (and potentially Perlin noise) to create temperature-humidity maps. From these, each biome gets a suitability score for village spawning. Introducing small random perturbations avoids overly well-defined biome shapes, yielding fragmented, realistic biome transitions [4].

## Code Functionality

### Village Generation with Reduced Density

Originally, naive random distributions produced unrealistically dense village placements. We refined this:

1. Divide the world into 32x32 chunk regions (512x512 blocks).
2. Place at most one village per region, controlled by a certain probability.
3. Assign biome suitability based on noise fields for temperature/humidity and threshold conditions. Valid biome sets (Plains, Savanna, Taiga, Snowy Plains, Desert) yield higher suitability.

**Equations (Temperature/Humidity Mapping):**
  
$$
\text{temperature}(x,z)=\sin(x/3000) + 0.5\cos(z/2000) + \cdots
$$
$$
\text{humidity}(x,z)=\cos(x/3500) + 0.4\sin(z/2500) + \cdots
$$

**Suitability Assignment:**
- Plains: [0.9,1.0]
- Savanna: [0.8,0.9]
- Taiga: [0.7,0.8]
- Snowy Plains: [0.6,0.7]
- Desert: [0.5,0.6]
- Others: [0,0.1]

![alt text](Plots/biomeSuitabilityMap.png "Noisy Biome Suitability Map")

### Stronghold Distribution

Strongholds appear in concentric rings at known distances [2]. We implement:

- Deterministic polar-to-Cartesian conversion for each stronghold.
- Even angular spacing.
- Slight noise to mimic the procedural generation.

This produces a pattern similar to actual stronghold placement in Minecraft.

### Ender Dragon Pathing

Representing the Ender Dragon’s movement as a graph problem [3]:

- Nodes: fountain center `f`, inner ring nodes `i1,…,i8`, outer ring nodes `o1,…,o12`, and possibly central towers.
- Edges: Each `(u,v)` has a probability `p_uv`.
  
**Condition:**
$$
\sum_{v:(u,v)\in E} p_{uv} = 1
$$

By sampling edges from this distribution, we get plausible flight paths. High-probability edges reflect commonly observed behaviors (e.g., circling towers), while low-probability edges correspond to rarer actions.

**Code Snippet:**
```python
currentNode = 'fountain'
neighbors = ['pillar_1', 'pillar_2', 'center_node_1']
probabilities = [0.5, 0.3, 0.2]  # must sum to 1.0

nextNode = np.random.choice(neighbors, p=probabilities)
