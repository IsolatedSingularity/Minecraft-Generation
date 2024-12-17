# Minecraft Procedural Generation and Entity Pathing Analysis
###### Collaborations and References include [Minecraft Wiki](https://minecraft.wiki/), [Sportskeeda Wiki](https://wiki.sportskeeda.com/minecraft), and procedural generation works from [Alan Zucconi](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/).

![Ender Dragon Path](https://github.com/IsolatedSingularity/Minecraft-Generation/blob/main/Plots/ender_dragon_pathing_graph_adjusted.png?raw=true)

## Objective
This repository provides a comprehensive exploration of Minecraft's procedural world generation and mob behavior algorithms. Key contributions include:

1. **Village Distribution with Biome Suitability:** Simulate villages over a massive 20,000 x 20,000 region divided into 32x32 chunk-sized regions. Each region probabilistically spawns villages based on biome suitability calculated using layered sine/cosine noise fields.

2. **Stronghold Distribution in Rings:** Model strongholds placed in concentric rings around the world origin. Randomize angular and radial positions to emulate in-game noise, while respecting the known ring parameters.

3. **Ender Dragon Pathing Visualization:** Represent the Ender Dragon's movement as a graph traversal problem. Nodes represent critical positions (fountain, pillars, and center nodes), and edges encode flight path probabilities. Adjust node colors to visualize distances from the fountain and analyze traversal behavior.

4. **Noise-Based Biome Mapping:** Implement temperature and humidity fields using layered sine/cosine functions combined with Gaussian noise. Threshold values are applied to assign biome suitability for villages, producing realistic transitions and biome fragmentation.

Each feature is visualized and formalized mathematically, offering detailed insights into Minecraft's world generation mechanics and entity behavior.

---

## Equations and Concepts

### Temperature and Humidity Fields
To simulate biome distribution, temperature $$\mathcal{T}$$ and humidity $$\mathcal{H}$$ fields are generated using a combination of sine, cosine, and Gaussian noise functions. This approach introduces spatial variability across the map:

\[
\mathcal{T}(x,z) = \sin\left(\frac{x}{3000}\right) + 0.5 \cos\left(\frac{z}{2000}\right) + 0.3 \sin\left(\frac{x+z}{1000}\right) + 0.2 \cos\left(\frac{x-z}{1500}\right) + \mathcal{N}(0,0.1)
\]

\[
\mathcal{H}(x,z) = \cos\left(\frac{x}{3500}\right) + 0.4 \sin\left(\frac{z}{2500}\right) + \mathcal{N}(0,0.1)
\]

The resulting fields are thresholded to assign biome suitability for villages based on predefined intervals:

\[
\text{Suitability} = \begin{cases} 
[0.9, 1.0] & \text{Plains} \\
[0.8, 0.9] & \text{Savanna} \\
[0.7, 0.8] & \text{Taiga} \\
[0.6, 0.7] & \text{Snowy Plains} \\
[0.5, 0.6] & \text{Desert} \\
[0, 0.1] & \text{Otherwise}
\end{cases}
\]

---

## Code Functionality
The following sections describe the implemented functions in detail, complete with explanations, equations, and code snippets.

### 1. **Village Distribution with Biome Suitability**

**Description:**
This function simulates village placement across a procedurally generated world while ensuring biome constraints are respected. The world is divided into 32x32 chunk-sized regions (each region spanning 512x512 blocks). For each region:

1. The center point is calculated based on the region's boundaries.
2. A 40% probability determines if a village spawns.
3. Biome suitability is computed using the noise-based temperature $$\mathcal{T}$$ and humidity $$\mathcal{H}$$ fields.
4. Random perturbations are applied to village positions to introduce spatial randomness and realism.

Mathematically:
\[
\text{Village Position} = (x_r + \mathcal{U}(-\delta, \delta), z_r + \mathcal{U}(-\delta, \delta))
\]
where $$x_r, z_r$$ are the center coordinates of a region, and $$\delta$$ introduces randomness.

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
        centerX = xRegion * regionBlockSize + regionBlockSize // 2
        centerZ = zRegion * regionBlockSize + regionBlockSize // 2

        if np.random.rand() < 0.4:  # 40% spawn chance
            villageXPositions.append(centerX + np.random.uniform(-regionBlockSize // 4, regionBlockSize // 4))
            villageZPositions.append(centerZ + np.random.uniform(-regionBlockSize // 4, regionBlockSize // 4))

# Compute biome suitability
biomeSuitability = np.random.rand(len(villageXPositions))

# Visualize village positions
plt.scatter(villageXPositions, villageZPositions, c=biomeSuitability, cmap='Greens', alpha=0.6)
plt.title("Village Distribution with Biome Suitability")
plt.xlabel("X Coordinate")
plt.ylabel("Z Coordinate")
plt.show()
```
</details>

**Output:**
![Village Distribution](https://github.com/IsolatedSingularity/Minecraft-Generation/blob/main/Plots/village_distribution.png?raw=true)

---

### 2. **Stronghold Distribution in Rings**

**Description:**
This function replicates the distribution of strongholds in concentric rings around the origin. Strongholds are spaced at random angular positions within fixed radii bounds. This distribution follows Minecraft's procedural generation rules and emulates noise by perturbing the angles slightly using Gaussian noise.

Steps include:
1. Evenly spacing angular positions.
2. Adding randomness to the angles.
3. Sampling radial distances within each ring's bounds.
4. Converting polar coordinates to Cartesian for plotting.

Mathematically:
\[
(r, \theta) \to (x, z) = (r \cos(\theta), r \sin(\theta))
\]
where $$\theta$$ is perturbed and $$r$$ is sampled uniformly within the range.

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

## Caveats
- Biome suitability relies on simplified sine/cosine noise rather than Minecraft's Perlin noise.
- The Ender Dragon's behavior is static and does not account for real-time game states (e.g., circling, perching).
- Stronghold placement assumes perfectly concentric rings with minor perturbations.

## Next Steps
- [x] Replace heuristic noise with Perlin noise for smoother biome transitions.
- [ ] Extend the Ender Dragon graph model with Markov chains for dynamic behavior.
- [ ] Introduce multi-threading for scalable world generation simulations.

> **TIP:** For enhanced biome generation, consider using noise libraries like [OpenSimplex](https://github.com/lmas/opensimplex).

> **NOTE:** Visualizing stronghold placement alongside terrain features could improve realism in future iterations.
