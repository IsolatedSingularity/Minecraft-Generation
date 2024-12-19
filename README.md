# Minecraft Procedural Generation and Entity Pathing Analysis
###### Collaborations and References include [Minecraft Wiki](https://minecraft.wiki/), [Sportskeeda Wiki](https://wiki.sportskeeda.com/minecraft), and procedural generation works from [Alan Zucconi](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/).

![Ender Dragon Path](https://github.com/IsolatedSingularity/Minecraft-Generation/blob/main/Plots/ender_dragon_pathing_graph_adjusted.png?raw=true)

## Objective
This repository provides a deep exploration of Minecraft's procedural world generation and mob behavior algorithms. Key contributions include:

1. **Village Distribution with Biome Suitability:** Simulate villages over a large 20,000 x 20,000 region divided into 32x32 chunk-sized regions. Each region probabilistically spawns villages based on biome suitability calculated using layered sine/cosine noise fields.

2. **Stronghold Distribution in Rings:** Model strongholds placed in concentric rings around the origin. Randomize angular and radial positions to emulate in-game noise, while respecting the known ring parameters.

3. **Ender Dragon Pathing Visualization:** Model the Ender Dragon's movement as a graph traversal problem. Nodes represent key positions (fountain, pillars, and center nodes), while edges encode flight path probabilities. Adjust node colors to visualize distances from the fountain.

Each feature is visualized and mathematically formalized, providing detailed insights into Minecraft’s world generation mechanics.

---

## Equations and Concepts

### Temperature and Humidity Fields
We generate temperature $$\mathcal{T}$$ and humidity $$\mathcal{H}$$ fields using a combination of sine, cosine, and Gaussian noise functions:

$$
\mathcal{T}(x,z) = \sin\left(\frac{x}{3000}\right) + 0.5 \cos\left(\frac{z}{2000}\right) + 0.3 \sin\left(\frac{x+z}{1000}\right) + 0.2 \cos\left(\frac{x-z}{1500}\right) + \mathcal{N}(0,0.1)
$$

$$
\mathcal{H}(x,z) = \cos\left(\frac{x}{3500}\right) + 0.4 \sin\left(\frac{z}{2500}\right) + \mathcal{N}(0,0.1)
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

Expanding upon this, temperature and humidity are used as inputs into procedural biome classification systems. Noise layers emulate realistic transitions between biomes. By combining sine and cosine waves with Gaussian noise, the approach ensures smooth but varied gradients, mimicking natural environmental variability. This methodology parallels applications in procedural graphics and environmental simulations. Future work could integrate Perlin noise or OpenSimplex noise for more organic transitions.

---

## Code Functionality

### Village Distribution with Biome Suitability

**Description:**
This function generates a spatial distribution of villages across a procedurally generated Minecraft-like world. The world is divided into 32x32 chunk regions (each chunk spanning 512x512 blocks). A central point for each region is computed, and villages are spawned probabilistically, ensuring that only biomes with high suitability values allow village generation. 

Biome suitability values are derived from noise-based temperature $$\mathcal{T}(x,z)$$ and humidity $$\mathcal{H}(x,z)$$ fields, ensuring that villages appear naturally in Plains, Savanna, and similar biomes. Random perturbations are added to village positions to break rigid patterns and create a realistic appearance. This simulation provides insights into procedural generation algorithms while controlling spatial randomness and biome constraints.

Mathematically:
$$\text{Village Position} = (x_r + \mathcal{U}(-\delta, \delta), z_r + \mathcal{U}(-\delta, \delta))$$

where $$x_r, z_r$$ are the center of a region and $$\delta$$ represents perturbations.

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

# Visualize village positions
plt.scatter(villageXPositions, villageZPositions, alpha=0.6)
plt.title("Village Distribution with Biome Suitability")
plt.xlabel("X Coordinate")
plt.ylabel("Z Coordinate")
plt.show()
```

![Village Distribution](https://github.com/IsolatedSingularity/Minecraft-Generation/blob/main/Plots/village_distribution.png?raw=true)

---

### Stronghold Distribution in Rings

**Description:**
Strongholds in Minecraft are placed in concentric rings centered around the origin. Each ring contains a specific number of strongholds, and their positions are determined based on polar coordinates. The algorithm generates strongholds by sampling their angular positions (with slight randomness) and radial distances within the bounds of each ring.

This approach models Minecraft's behavior closely by respecting the ring radii while introducing randomness to emulate procedural generation. The use of polar coordinates simplifies the calculation of positions and ensures that the strongholds are distributed naturally within their respective rings.

Mathematically:
$$(r, \theta) \to (x, z) = (r \cos(\theta), r \sin(\theta))$$

where $$\theta$$ is sampled with a slight perturbation, and $$r$$ is uniformly sampled within each ring's radius bounds.

```python
# Ring definitions
ringDefinitions = [
    {'radius': (1280, 2816), 'count': 3},
    {'radius': (4352, 5888), 'count': 6},
]

# Generate strongholds
for ring in ringDefinitions:
    r_min, r_max = ring['radius']
    count = ring['count']
    angles = np.linspace(0, 2*np.pi, count, endpoint=False) + np.random.normal(0, np.pi/(count*2), count)
    radii = np.random.uniform(r_min, r_max, count)
    x_positions = radii * np.cos(angles)
    z_positions = radii * np.sin(angles)

    plt.scatter(x_positions, z_positions, s=100)

plt.title("Stronghold Distribution in Rings")
plt.show()
```

![Stronghold Distribution](https://github.com/IsolatedSingularity/Minecraft-Generation/blob/main/Plots/stronghold_distribution.png?raw=true)

---

### Ender Dragon Pathing Graph

**Description:**
The Ender Dragon's movement in the End is modeled as a graph traversal problem. Nodes in the graph represent key positions, including the central fountain and the tops of the obsidian pillars that surround it. Each node is connected to the fountain by edges, which correspond to possible flight paths the dragon can take.

Edges in the graph have probabilities proportional to the inverse degree of the connected vertices:
$$P(i \to j) = \frac{1}{\text{deg}(j)}$$
where $$\text{deg}(j)$$ is the number of edges connected to vertex $$j$$. This reflects the Ender Dragon's tendency to choose paths toward less-connected nodes, balancing traversal efficiency and randomness.

Mathematically, the graph is represented as an adjacency matrix $$A$$, where $$A[i, j] = 1$$ if there is an edge between nodes $$i$$ and $$j$$. The traversal algorithm utilizes a weighted Markov chain to model the dragon's flight probabilities, ensuring smooth yet unpredictable movements.

In a higher-dimensional perspective, the dragon's pathing can be viewed as a traversal over a simplicial complex, where nodes represent 0-simplices, edges represent 1-simplices, and potential flight paths correspond to 2-simplices. This approach aligns with topological methods used to study dynamic systems in constrained environments.

```python
import networkx as nx

# Define nodes and edges
nodes = ['Fountain', 'Pillar1', 'Pillar2', 'Pillar3']
edges = [('Fountain', 'Pillar1'), ('Fountain', 'Pillar2'), ('Pillar1', 'Pillar3')]

# Create graph
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# Visualize graph
nx.draw(G, with_labels=True)
plt.show()
```

![Ender Dragon Path](https://github.com/IsolatedSingularity/Minecraft-Generation/blob/main/Plots/ender_dragon_pathing_graph_adjusted.png?raw=true)

---

## Caveats
- Biome suitability is based on simplified sine/cosine noise rather than Perlin noise, leading to less organic transitions.
- Dragon pathing is static and does not include real-time behavior changes.
- Stronghold placement assumes perfectly concentric rings.

## Next Steps
- [x] Replace heuristic noise with Perlin noise for more realistic biome transitions.
- [ ] Extend the Ender Dragon model using Markov chains for dynamic behavior.
- [ ] Simulate larger worlds efficiently with parallel computation for scalability.

> [!TIP]
> Explore dynamic node traversal using weighted Markov chains to simulate Ender Dragon behavior realistically.

> [!NOTE]
> For enhanced realism, integrate noise generation libraries such as [OpenSimplex](https://github.com/lmas/opensimplex) for biome mapping.
