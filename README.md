# Minecraft Procedural Generation and Mob Pathing Analysis
###### Under collaboration with references from [Minecraft Wiki](https://minecraft.wiki/), [Sportskeeda Wiki](https://wiki.sportskeeda.com/minecraft), [Reddit community discussions](https://www.reddit.com/r/Minecraft/), and theoretical inspirations from [Alan Zucconi’s world generation articles](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/).

![Village Distribution](https://github.com/IsolatedSingularity/Minecraft-Generation/blob/main/Plots/village_distribution.png?raw=true)

## Objective

Minecraft’s world is generated through a complex interplay of noise functions, biome rules, and procedural algorithms to produce vast, dynamic landscapes. In addition, certain structures (like strongholds) and entity behaviors (like the Ender Dragon’s pathing in the End) follow distinct procedural rules. This repository explores:

1. **Village Generation with Biome Suitability:** Simulate a large world region (−10,000 to 10,000) and place villages in 32x32 chunk regions (512x512 blocks). Assign biome-based suitability scores to determine the probability and distribution of villages, ensuring a realistic density and fragmentation in biome appearance.

2. **Stronghold Distribution:** Represent strongholds in concentric rings around the world origin, following known radii and counts from the Minecraft Wiki. Introducing slight randomness in angles and radii simulates procedural noise in actual gameplay.

3. **Ender Dragon Pathing:** Model the Ender Dragon’s movements in the End dimension as a graph traversal problem. The dragon’s next move at a node (representing fountain, pillars, center nodes) is chosen by sampling from edges with certain probabilities. By doing so, we capture the dragon’s circling, perching, and strafing behaviors and connect it to node-based probability distributions.

4. **Noisy Biome Maps:** Use combined sine/cosine noise layers and random perturbations to produce biome stability maps. These are used for assigning suitability values to locations and ensuring a non-uniform, organic biome distribution without overly defined shapes.

## Equations and Concepts

### Biome Suitability and Noise Functions

We define temperature and humidity fields to determine biomes:

\[
\text{temperature}(x,z) = \sin\left(\frac{x}{3000}\right) + 0.5\cos\left(\frac{z}{2000}\right) + \cdots
\]

\[
\text{humidity}(x,z) = \cos\left(\frac{x}{3500}\right) + 0.4\sin\left(\frac{z}{2500}\right) + \cdots
\]

From these, we assign suitability scores depending on the biome thresholds. For instance:

\[
\begin{aligned}
&\text{If Plains: } \text{Suitability} \in [0.9, 1.0]\\
&\text{If Savanna: } \text{Suitability} \in [0.8, 0.9]\\
&\text{If Taiga: } \text{Suitability} \in [0.7, 0.8]\\
&\text{If Snowy Plains: } \text{Suitability} \in [0.6, 0.7]\\
&\text{If Desert: } \text{Suitability} \in [0.5, 0.6]\\
&\text{Otherwise: } \text{Suitability} \in [0, 0.1]
\end{aligned}
\]

This ensures distinct ranges for each biome and adds complexity to the resulting distribution.

### Stronghold Distribution in Rings

Strongholds appear in rings around the origin [\[2\]](https://wiki.sportskeeda.com/minecraft/stronghold). For ring $k$, with radius range $(R_{\min}, R_{\max})$ and $n_k$ strongholds, we generate angles evenly spaced:

\[
\theta_j = \frac{2\pi j}{n_k}, \quad j=0,\dots,n_k-1
\]

Add slight randomness:

\[
\theta_j' = \theta_j + \mathcal{N}(0,\sigma_\theta), \quad r_j = U(R_{\min}+100, R_{\max}-100)
\]

Then convert to Cartesian:

\[
x_j = r_j \cos(\theta_j'), \quad y_j = r_j \sin(\theta_j')
\]

### Ender Dragon Pathing Probability

Consider a node in the End dimension graph (fountain, center, inner, outer nodes). Let the set of edges from this node be $E(u)$ with probabilities $p_{uv}$ for each edge $(u,v)$:

\[
\sum_{v:(u,v) \in E} p_{uv} = 1
\]

The dragon’s next move from node $u$ is chosen by sampling from the distribution $\{p_{uv}\}$. This simulates observed behaviors: higher $p_{uv}$ edges represent common dragon flights, while lower $p_{uv}$ edges represent rare events.

## Code Functionality

Below is a snippet of the Python code. We rely on `numpy`, `matplotlib`, `networkx` for visualization, and standard Python libraries.

<details>
  <summary><b>Show Code</b></summary>

```python
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge
from matplotlib.cm import get_cmap

# Global style & colors
plt.style.use('dark_background')
backgroundColor = '#1C1C1C'
gridLineColor = '#383838'
textFontColor = 'white'
downloadPath = os.path.join(os.path.expanduser("~"), "Downloads")

# Village Distribution (Reduced Density)
figureVillage, axesVillage = plt.subplots(figsize=(12, 12))
axesVillage.set_facecolor(backgroundColor)
figureVillage.patch.set_facecolor(backgroundColor)

worldCoordinateRange = 10000
regionBlockSize = 512
numberOfRegionsPerAxis = worldCoordinateRange * 2 // regionBlockSize

villageXPositions = []
villageZPositions = []
for xRegion in range(-numberOfRegionsPerAxis // 2, numberOfRegionsPerAxis // 2):
    for zRegion in range(-numberOfRegionsPerAxis // 2, numberOfRegionsPerAxis // 2):
        regionCenterX = xRegion * regionBlockSize + regionBlockSize // 2
        regionCenterZ = zRegion * regionBlockSize + regionBlockSize // 2
        # 40% chance village spawns
        if np.random.rand() < 0.4:
            villageXPositions.append(regionCenterX + np.random.uniform(-regionBlockSize // 4, regionBlockSize // 4))
            villageZPositions.append(regionCenterZ + np.random.uniform(-regionBlockSize // 4, regionBlockSize // 4))

biomeSuitabilityArray = np.random.rand(len(villageXPositions))
scatterPlot = axesVillage.scatter(villageXPositions, villageZPositions,
                                  c=biomeSuitabilityArray,
                                  cmap='BuPu',
                                  alpha=0.6)

axesVillage.set_xlim(-worldCoordinateRange, worldCoordinateRange)
axesVillage.set_ylim(-worldCoordinateRange, worldCoordinateRange)
axesVillage.set_title('Village Distribution in the Overworld', color=textFontColor, pad=20)
axesVillage.set_xlabel('X Coordinate', color=textFontColor)
axesVillage.set_ylabel('Z Coordinate', color=textFontColor)
axesVillage.grid(True, color=gridLineColor, linestyle='--', alpha=0.5)

colorBar = plt.colorbar(scatterPlot)
colorBar.set_label('Biome Suitability', color=textFontColor)
colorBar.ax.yaxis.set_tick_params(color=textFontColor)
plt.setp(plt.getp(colorBar.ax.axes, 'yticklabels'), color=textFontColor)

plt.tight_layout()
plt.savefig(os.path.join(downloadPath, "village_distribution.png"), dpi=1000,
            bbox_inches="tight", facecolor=backgroundColor)
plt.show()

# Ender Dragon Pathing Graph
figureDragon, axesDragon = plt.subplots(figsize=(12, 12))
axesDragon.set_facecolor(backgroundColor)
figureDragon.patch.set_facecolor(backgroundColor)

outerNodeCount = 12
innerNodeCount = 8
centerNodeCount = 4
outerRingRadius = 100
innerRingRadius = 60
centerRingRadius = 30

outerAngles = np.linspace(0, 2*np.pi, outerNodeCount, endpoint=False)
innerAngles = np.linspace(0, 2*np.pi, innerNodeCount, endpoint=False)
centerAngles = np.linspace(0, 2*np.pi, centerNodeCount, endpoint=False)

outerPositions = [(outerRingRadius*np.cos(a), outerRingRadius*np.sin(a)) for a in outerAngles]
innerPositions = [(innerRingRadius*np.cos(a), innerRingRadius*np.sin(a)) for a in innerAngles]
centerPositions = [(centerRingRadius*np.cos(a), centerRingRadius*np.sin(a)) for a in centerAngles]
centralPosition = (0,0)

graphDragon = nx.Graph()
graphDragon.add_nodes_from(outerPositions)
graphDragon.add_nodes_from(innerPositions)
graphDragon.add_nodes_from(centerPositions)
graphDragon.add_node(centralPosition)

# Add edges for rings, center connections, etc. (omitted here for brevity)
# ...

# Assign colors and draw
nodeColorArray = (
    ['#440154'] * len(outerPositions) +
    ['#31688E'] * len(innerPositions) +
    ['#35B779'] * len(centerPositions) +
    ['#FDE725']
)
posDict = {n: n for n in graphDragon.nodes()}
nx.draw_networkx_edges(graphDragon, posDict, edge_color='#E0E0E0', width=1.5, ax=axesDragon)
nx.draw_networkx_nodes(graphDragon, posDict, node_color=nodeColorArray, node_size=60, ax=axesDragon)

axesDragon.set_title('Ender Dragon Path in the End', color=textFontColor, pad=20, fontsize=14)
axesDragon.set_xlabel('X Coordinate', color=textFontColor)
axesDragon.set_ylabel('Z Coordinate', color=textFontColor)
axesDragon.tick_params(colors=textFontColor)
axesDragon.grid(True, color=gridLineColor, linestyle='--', alpha=0.5)
axesDragon.set_xlim(-120,120)
axesDragon.set_ylim(-120,120)

legendElements = [
    Line2D([0], [0], marker='o', color='w', label='Outer Ring', markerfacecolor='#440154', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Inner Ring', markerfacecolor='#31688E', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Center Nodes', markerfacecolor='#35B779', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Central Node', markerfacecolor='#FDE725', markersize=10)
]
axesDragon.legend(handles=legendElements,
                  loc='upper right',
                  facecolor=backgroundColor,
                  edgecolor='#383838',
                  labelcolor='white')

plt.tight_layout()
plt.savefig(os.path.join(downloadPath, "ender_dragon_pathing_graph.png"), dpi=1000,
            bbox_inches="tight", facecolor=backgroundColor)
plt.show()

# Adjusted Ender Dragon Graph with Distances
# ...
# (Similar code as above; see final code snippet)
# ...

# Stronghold Distribution
figureStrongholds, axesStrongholds = plt.subplots(figsize=(12, 12))
axesStrongholds.set_facecolor(backgroundColor)
figureStrongholds.patch.set_facecolor(backgroundColor)

ringDefinitionList = [
    {'radius': (1280, 2816), 'count': 3, 'color': '#440154'},
    {'radius': (4352, 5888), 'count': 6, 'color': '#443983'},
    {'radius': (7424, 8960), 'count': 10, 'color': '#31688E'},
    {'radius': (10496, 12032), 'count': 15, 'color': '#21918C'},
    {'radius': (13568, 15104), 'count': 21, 'color': '#35B779'},
    {'radius': (16640, 18176), 'count': 28, 'color': '#90D743'},
    {'radius': (19712, 21248), 'count': 36, 'color': '#FDE725'},
    {'radius': (22784, 24320), 'count': 9, 'color': '#440154'}
]

for ringDef in ringDefinitionList:
    # Draw annular rings and scatter points (omitted for brevity)
    # ...

axesStrongholds.set_title('Stronghold Distribution in the Overworld', color=textFontColor, pad=20)
# ...
# Save and show plot
