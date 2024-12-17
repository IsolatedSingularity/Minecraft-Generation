# Minecraft Procedural Generation and Entity Pathing Analysis
###### Collaborations and References include [Minecraft Wiki](https://minecraft.wiki/), [Sportskeeda Wiki](https://wiki.sportskeeda.com/minecraft), community discussions on [Reddit](https://www.reddit.com/r/Minecraft/), and procedural generation works from [Alan Zucconi](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/).

![Ender Dragon Path](https://github.com/IsolatedSingularity/Minecraft-Generation/blob/main/Plots/ender_dragon_pathing_graph_adjusted.png?raw=true)

## Objective
This repository provides a deep exploration of Minecraft's procedural world generation and mob behavior algorithms. Key contributions include:

1. **Village Distribution with Biome Suitability:** Simulate villages over a large 20,000 x 20,000 region divided into 32x32 chunk-sized regions. Each region probabilistically spawns villages based on biome suitability calculated using layered sine/cosine noise fields.

2. **Stronghold Distribution in Rings:** Model strongholds placed in concentric rings around the origin. Randomize angular and radial positions to emulate in-game noise, while respecting the known ring parameters.

3. **Ender Dragon Pathing Visualization:** Model the Ender Dragon's movement as a graph traversal problem. Nodes represent key positions (fountain, pillars, and center nodes), while edges encode flight path probabilities. Adjust node colors to visualize distances from the fountain.

4. **Noise-Based Biome Mapping:** Implement temperature and humidity fields using Perlin-like sine/cosine functions. Threshold values determine biome suitability for villages, producing realistic transitions and fragmentation.

Each feature is visualized and mathematically formalized, providing detailed insights into Minecraft’s world generation mechanics.

---

## Equations and Concepts

### Temperature and Humidity Fields
We generate temperature $$T$$ and humidity $$H$$ fields using a combination of sine, cosine, and noise functions:

$$
T(x,z) = \sin\left(\frac{x}{3000}\right) + 0.5 \cos\left(\frac{z}{2000}\right) + 0.3 \sin\left(\frac{x+z}{1000}\right) + 0.2 \cos\left(\frac{x-z}{1500}\right) + \mathcal{N}(0,0.1)
$$

$$
H(x,z) = \cos\left(\frac{x}{3500}\right) + 0.4 \sin\left(\frac{z}{2500}\right) + 0.3 \cos\left(\frac{x+2z}{1800}\right) + 0.2 \sin\left(\frac{2x-z}{1200}\right) + \mathcal{N}(0,0.1)
$$

These fields are thresholded to assign biome suitability for villages:

$$
\text{Suitability} = \begin{cases} 
[0.9, 1.0] & \text{Plains} \\
[0.8, 0.9] & \text{Savanna} \\
[0.7, 0.8] & \text{Taiga} \\
[0.6, 0.7] & \text{Snowy Plains} \\
[0.5, 0.6] & \text{Desert} \\
[0, 0.1] & \text{Otherwise}
\end{cases}
$$

---

## Code Functionality
The following sections describe the implemented functions in great detail, with corresponding code snippets and explanations.

### 1. **Village Distribution with Biome Suitability**

**Description:**
This function simulates village placement across a massive world region, ensuring villages appear only in suitable biomes. The world is divided into 32x32 chunk regions (512x512 blocks each). For each region:

- The center is calculated as the average position of the region's boundaries.
- A probability threshold (e.g., 40%) determines if a village spawns.
- Biome suitability is computed using pre-defined noise-based temperature/humidity fields.
- Random perturbations are applied to village positions within each region to avoid overly regular patterns.

Mathematically, suitability depends on thresholded $$T(x,z)$$ and $$H(x,z)$$, and village positions are generated based on the equations:

$$
\text{Village Position} = (x_r + \mathcal{U}(-\delta, \delta), z_r + \mathcal{U}(-\delta, \delta))
$$

where $$x_r, z_r$$ are the region centers and $$\delta$$ represents random perturbations.

<details>
  <summary><i>Village Distribution Python Function</i></summary>

```python
# Define region parameters
worldCoordinateRange = 10000  # (-10,000 to 10,000)
regionBlockSize = 512  # Each region is 512x512 blocks
numberOfRegionsPerAxis = worldCoordinateRange * 2 // regionBlockSize

# Initialize village positions
villageXPositions = []
villageZPositions = []

# Generate villages
for xRegion in range(-numberOfRegionsPerAxis // 2, numberOfRegionsPerAxis // 2):
    for zRegion in range(-numberOfRegionsPerAxis // 2, numberOfRegionsPerAxis // 2):
        regionCenterX = xRegion * regionBlockSize + regionBlockSize // 2
        regionCenterZ = zRegion * regionBlockSize + regionBlockSize // 2

        if np.random.rand() < 0.4:  # 40% spawn chance
            villageXPositions.append(regionCenterX + np.random.uniform(-regionBlockSize // 4, regionBlockSize // 4))
            villageZPositions.append(regionCenterZ + np.random.uniform(-regionBlockSize // 4, regionBlockSize // 4))

# Compute biome suitability
biomeSuitability = np.random.rand(len(villageXPositions))

# Plot village positions
plt.scatter(villageXPositions, villageZPositions, c=biomeSuitability, cmap='BuPu', alpha=0.6)
plt.title("Village Distribution in the Overworld")
plt.show()
```
</details>

**Output:**
![Village Distribution](https://github.com/IsolatedSingularity/Minecraft-Generation/blob/main/Plots/village_distribution.png?raw=true)

---

### 2. **Stronghold Distribution in Rings**

**Description:**
Strongholds generate in concentric rings centered at the world origin, with fixed counts and radii per ring. The positions are defined using polar coordinates:

$$
(r, \theta) \to (x, z) = (r \cos(\theta), r \sin(\theta))
$$

Radial positions are uniformly sampled within ring bounds, while angular positions are perturbed to add randomness:

$$
\theta_i = \theta_i^{(0)} + \mathcal{N}(0, \sigma)
$$

<details>
  <summary><i>Stronghold Ring Python Function</i></summary>

```python
# Ring definitions
ringDefinitions = [
    {'radius': (1280, 2816), 'count': 3, 'color': '#440154'},
    {'radius': (4352, 5888), 'count': 6, 'color': '#443983'},
]

# Generate strongholds
for ring in ringDefinitions:
    r_min, r_max = ring['radius']
    count = ring['count']
    angles = np.linspace(0, 2*np.pi, count, endpoint=False) + np.random.normal(0, np.pi/(count*2), count)
    radii = np.random.uniform(r_min, r_max, count)
    x_positions = radii * np.cos(angles)
    z_positions = radii * np.sin(angles)

    plt.scatter(x_positions, z_positions, color=ring['color'], s=100)

plt.title("Stronghold Distribution in Rings")
plt.show()
```
</details>

**Output:**
![Stronghold Distribution](https://github.com/IsolatedSingularity/Minecraft-Generation/blob/main/Plots/stronghold_distribution.png?raw=true)

---

### 3. **Ender Dragon Pathing Graph**

**Description:**
The Ender Dragon’s movement is modeled as a graph with nodes representing critical positions and edges defining flight paths. Node distances are visualized using a colormap based on their Euclidean distance:

$$
\text{Distance} = \sqrt{x^2 + z^2}
$$

<details>
  <summary><i>Ender Dragon Path Python Function</i></summary>

```python
# Define node positions
outerNodeAngles = np.linspace(0, 2*np.pi, 12, endpoint=False)
outerNodes = [(100*np.cos(a), 100*np.sin(a)) for a in outerNodeAngles]
centerNode = (0, 0)

# Graph
G = nx.Graph()
G.add_nodes_from(outerNodes + [centerNode])
G.add_edges_from([(centerNode, node) for node in outerNodes])

# Color nodes based on distance
colors = [np.linalg.norm(node) for node in outerNodes]
nx.draw(G, pos={n: n for n in G.nodes}, node_color=colors, cmap='winter', with_labels=False)
plt.title("Ender Dragon Path in the End")
plt.show()
```
</details>

**Output:**
![Ender Dragon Path](https://github.com/IsolatedSingularity/Minecraft-Generation/blob/main/Plots/ender_dragon_pathing_graph_adjusted.png?raw=true)

---

## Caveats
- The biome suitability map uses heuristic noise functions, which are simplified compared to Minecraft’s actual Perlin noise.
- The Ender Dragon’s path probabilities are static; real behavior depends on its state (e.g., circling, perching).

## Next Steps
- Incorporate Perlin noise for biome transitions.
- Extend the dragon path model with Markov chains to simulate dynamic behavior.

## References
1. [Minecraft Wiki: Village](https://minecraft.wiki/)
2. [Minecraft Wiki: Stronghold](https://minecraft.wiki/)
3. [Minecraft Wiki: Ender Dragon](https://minecraft.wiki/)
4. [Alan Zucconi’s Procedural Generation Articles](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/)
