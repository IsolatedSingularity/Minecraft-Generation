#%% Initializing prerequisites - 차단하다


# Import modules
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.patches import Wedge
from matplotlib.cm import get_cmap

# Configure global style and colors
plt.style.use('dark_background')
backgroundColor = '#1C1C1C'
gridLineColor = '#383838'
textFontColor = 'white'
downloadPath = os.path.join(os.path.expanduser("~"), "Downloads")


#%% Village distribution and biome suitability [Overworld] - 차단하다


# Create figure and axes
figureVillage, axesVillage = plt.subplots(figsize=(12, 12))
axesVillage.set_facecolor(backgroundColor)
figureVillage.patch.set_facecolor(backgroundColor)

# Define world and region parameters
worldCoordinateRange = 10000  # Total coordinate range for the world (-10,000 to 10,000)
regionBlockSize = 512  # Size of each region in blocks (32x32 chunks)
numberOfRegionsPerAxis = worldCoordinateRange * 2 // regionBlockSize  # Number of regions along one axis

# Generate village positions with reduced density (one per region)
villageXPositions = []
villageZPositions = []

for xRegion in range(-numberOfRegionsPerAxis // 2, numberOfRegionsPerAxis // 2):
    for zRegion in range(-numberOfRegionsPerAxis // 2, numberOfRegionsPerAxis // 2):
        # Calculate the center of the current region
        regionCenterX = xRegion * regionBlockSize + regionBlockSize // 2
        regionCenterZ = zRegion * regionBlockSize + regionBlockSize // 2

        # Randomly decide if a village will spawn in this region (e.g., 40% chance)
        if np.random.rand() < 0.4:  # Adjust probability as needed
            villageXPositions.append(regionCenterX + np.random.uniform(-regionBlockSize // 4, regionBlockSize // 4))
            villageZPositions.append(regionCenterZ + np.random.uniform(-regionBlockSize // 4, regionBlockSize // 4))

# Generate biome suitability values (random for demonstration)
biomeSuitabilityArray = np.random.rand(len(villageXPositions))

# Plot villages with BuPu colormap
scatterPlot = axesVillage.scatter(villageXPositions, villageZPositions,
                                  c=biomeSuitabilityArray,
                                  cmap='BuPu',
                                  alpha=0.6)

# Customize axes appearance
axesVillage.set_xlim(-worldCoordinateRange, worldCoordinateRange)
axesVillage.set_ylim(-worldCoordinateRange, worldCoordinateRange)
axesVillage.set_title('Village Distribution in the Overworld', color=textFontColor, pad=20)
axesVillage.set_xlabel('X Coordinate', color=textFontColor)
axesVillage.set_ylabel('Z Coordinate', color=textFontColor)
axesVillage.grid(True, color=gridLineColor, linestyle='--', alpha=0.5)

# Add colorbar
colorBar = plt.colorbar(scatterPlot)
colorBar.set_label('Biome Suitability', color=textFontColor)
colorBar.ax.yaxis.set_tick_params(color=textFontColor)
plt.setp(plt.getp(colorBar.ax.axes, 'yticklabels'), color=textFontColor)

# Save plot to file with a generic download path
plt.tight_layout()
plt.savefig(os.path.join(downloadPath, "village_distribution.png"), dpi=1000, bbox_inches="tight", facecolor=backgroundColor)
plt.show()




#%% Ender dragon pathing graph [End] - 차단하다


# Create figure and axes
figureDragon, axesDragon = plt.subplots(figsize=(12, 12))
axesDragon.set_facecolor(backgroundColor)
figureDragon.patch.set_facecolor(backgroundColor)

# Define node counts and radii
outerNodeCount = 12
innerNodeCount = 8
centerNodeCount = 4
outerRingRadius = 100
innerRingRadius = 60
centerRingRadius = 30

# Define node angle arrays
outerNodeAngles = np.linspace(0, 2*np.pi, outerNodeCount, endpoint=False)
innerNodeAngles = np.linspace(0, 2*np.pi, innerNodeCount, endpoint=False)
centerNodeAngles = np.linspace(0, 2*np.pi, centerNodeCount, endpoint=False)

# Compute node positions
outerNodePositions = [(outerRingRadius*np.cos(a), outerRingRadius*np.sin(a)) for a in outerNodeAngles]
innerNodePositions = [(innerRingRadius*np.cos(a), innerRingRadius*np.sin(a)) for a in innerNodeAngles]
centerNodePositions = [(centerRingRadius*np.cos(a), centerRingRadius*np.sin(a)) for a in centerNodeAngles]
centralNodePosition = (0, 0)

# Create graph and add nodes
graphDragon = nx.Graph()
graphDragon.add_nodes_from(outerNodePositions)
graphDragon.add_nodes_from(innerNodePositions)
graphDragon.add_nodes_from(centerNodePositions)
graphDragon.add_node(centralNodePosition)

# Add edges for outer ring
for i in range(outerNodeCount):
    graphDragon.add_edge(outerNodePositions[i], outerNodePositions[(i+1) % outerNodeCount])

# Add edges for inner ring
for i in range(innerNodeCount):
    graphDragon.add_edge(innerNodePositions[i], innerNodePositions[(i+1) % innerNodeCount])

# Add edges for center ring
for i in range(centerNodeCount):
    graphDragon.add_edge(centerNodePositions[i], centerNodePositions[(i+1) % centerNodeCount])

# Add edges from central node to center ring nodes
for centerNode in centerNodePositions:
    graphDragon.add_edge(centralNodePosition, centerNode)

# Define outer-to-inner connections
outerToInnerConnections = [
    (0,0), (1,1), (2,1), (3,2), (4,3), (5,3),
    (6,4), (7,5), (8,5), (9,6), (10,7), (11,7)
]
for outerIndex, innerIndex in outerToInnerConnections:
    graphDragon.add_edge(outerNodePositions[outerIndex], innerNodePositions[innerIndex])

# Define inner-to-center connections
innerToCenterConnections = [
    (0,0), (1,0), (2,1), (3,1), (3,2), (4,2),
    (5,2), (5,3), (6,3), (7,0), (7,3)
]
for innerNodeIndex, centerNodeIndex in innerToCenterConnections:
    graphDragon.add_edge(innerNodePositions[innerNodeIndex], centerNodePositions[centerNodeIndex])

# Add special outer-to-center connection
graphDragon.add_edge(outerNodePositions[0], centerNodePositions[0])

# Add missed edge
graphDragon.add_edge(innerNodePositions[1], centerNodePositions[1])

# Assign colors to different node groups
nodeColorArray = (
    ['#440154'] * len(outerNodePositions) +
    ['#31688E'] * len(innerNodePositions) +
    ['#35B779'] * len(centerNodePositions) +
    ['#FDE725']
)
nodePositionsDictionary = {n: n for n in graphDragon.nodes()}
nx.draw_networkx_edges(graphDragon, nodePositionsDictionary, edge_color='#E0E0E0', width=1.5, ax=axesDragon)
nx.draw_networkx_nodes(graphDragon, nodePositionsDictionary, node_color=nodeColorArray, node_size=60, ax=axesDragon)

# Customize plot appearance
axesDragon.set_title('Ender Dragon Path in the End', color=textFontColor, pad=20, fontsize=14)
axesDragon.set_xlabel('X Coordinate', color=textFontColor)
axesDragon.set_ylabel('Z Coordinate', color=textFontColor)
axesDragon.tick_params(colors=textFontColor)
axesDragon.grid(True, color=gridLineColor, linestyle='--', alpha=0.5)
axesDragon.set_xlim(-120, 120)
axesDragon.set_ylim(-120, 120)

# Create and add legend
legendElementList = [
    Line2D([0], [0], marker='o', color='w', label='Outer Ring', markerfacecolor='#440154', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Inner Ring', markerfacecolor='#31688E', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Center Nodes', markerfacecolor='#35B779', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Central Node', markerfacecolor='#FDE725', markersize=10)
]
axesDragon.legend(handles=legendElementList,
                  loc='upper right',
                  facecolor=backgroundColor,
                  edgecolor=gridLineColor,
                  labelcolor='white')

plt.tight_layout()
plt.savefig(os.path.join(downloadPath, "ender_dragon_pathing_graph.png"), dpi=1000, bbox_inches="tight", facecolor=backgroundColor)
plt.show()



#%% Embedded ender drgaon pathing graph [End] - 차단하다


# Definining plot preamble
figureDragonAdjusted, axesDragonAdjusted = plt.subplots(figsize=(12, 12))
axesDragonAdjusted.set_facecolor(backgroundColor)
figureDragonAdjusted.patch.set_facecolor(backgroundColor)

# Define graph vertices angles in polar coordinates
outerNodeAngles = np.linspace(0, 2 * np.pi, outerNodeCount, endpoint=False)
innerNodeAngles = np.linspace(0, 2 * np.pi, innerNodeCount, endpoint=False)
centerNodeAngles = np.linspace(0, 2 * np.pi, centerNodeCount, endpoint=False)

# Defining graph vertices
outerNodePositions = [(outerRingRadius * np.cos(a), outerRingRadius * np.sin(a)) for a in outerNodeAngles]
innerNodePositions = [(innerRingRadius * np.cos(a), innerRingRadius * np.sin(a)) for a in innerNodeAngles]
centerNodePositions = [(centerRingRadius * np.cos(a), centerRingRadius * np.sin(a)) for a in centerNodeAngles]
centralNodePosition = (0, 0)
graphDragonAdjusted = nx.Graph()
graphDragonAdjusted.add_nodes_from(outerNodePositions + innerNodePositions + centerNodePositions + [centralNodePosition])

# Adding graph edges
for i in range(outerNodeCount):
    graphDragonAdjusted.add_edge(outerNodePositions[i], outerNodePositions[(i + 1) % outerNodeCount])
for i in range(innerNodeCount):
    graphDragonAdjusted.add_edge(innerNodePositions[i], innerNodePositions[(i + 1) % innerNodeCount])
for i in range(centerNodeCount):
    graphDragonAdjusted.add_edge(centerNodePositions[i], centerNodePositions[(i + 1) % centerNodeCount])
for centerNode in centerNodePositions:
    graphDragonAdjusted.add_edge(centralNodePosition, centerNode)
for outerIndex, innerIndex in outerToInnerConnections:
    graphDragonAdjusted.add_edge(outerNodePositions[outerIndex], innerNodePositions[innerIndex])
for innerNodeIndex, centerNodeIndex in innerToCenterConnections:
    graphDragonAdjusted.add_edge(innerNodePositions[innerNodeIndex], centerNodePositions[centerNodeIndex])
graphDragonAdjusted.add_edge(outerNodePositions[0], centerNodePositions[0])
graphDragonAdjusted.add_edge(innerNodePositions[1], centerNodePositions[1])
nodePositionsDictionary = {n: n for n in graphDragonAdjusted.nodes()}
nx.draw_networkx_edges(graphDragonAdjusted, nodePositionsDictionary, edge_color='#FFFFFF', width=1.5, ax=axesDragonAdjusted)
nx.draw_networkx_nodes(graphDragonAdjusted, nodePositionsDictionary, node_color=nodeColorArray, node_size=100, ax=axesDragonAdjusted)

# Defining (normalized dimensionless) distance function and colour map on vertex rings
colorMapObject = get_cmap('winter')
nodePositionsList = outerNodePositions + innerNodePositions + centerNodePositions + [centralNodePosition]
distanceFromCenterArray = [np.linalg.norm(pos) for pos in nodePositionsList]
normalizedDistanceArray = np.array(distanceFromCenterArray) / max(distanceFromCenterArray)
nodeColorArray = [colorMapObject(norm) for norm in normalizedDistanceArray]

#% Add end towers and end fountain as separate graphs
towerRingRadius = 76
towerAnglesArray = np.linspace(0, 2 * np.pi, 10, endpoint=False)
towerPositionList = [(towerRingRadius * np.cos(a), towerRingRadius * np.sin(a)) for a in towerAnglesArray]

# Plotting the towers and fountain
for towerPosition in towerPositionList:
    towerCirclePatch = plt.Circle(towerPosition,
                                  radius=6,
                                  color='#4B0082',
                                  alpha=0.7)
    axesDragonAdjusted.add_patch(towerCirclePatch)

fountainCirclePatch = plt.Circle((0, 0),
                                 radius=8,
                                 color='#808080',
                                 alpha=0.7)
axesDragonAdjusted.add_patch(fountainCirclePatch)


# Adding different figure elements
axesDragonAdjusted.set_title('Embedded Ender Dragon Path in the End', color=textFontColor, pad=20)
axesDragonAdjusted.set_xlabel('X Coordinate', color=textFontColor)
axesDragonAdjusted.set_ylabel('Z Coordinate', color=textFontColor)
axesDragonAdjusted.tick_params(colors=textFontColor)
axesDragonAdjusted.grid(True, color=gridLineColor, linestyle='--', alpha=0.5)
axesDragonAdjusted.set_xlim(-120, 120)
axesDragonAdjusted.set_ylim(-120, 120)
axesDragonAdjusted.set_aspect('equal')

legendElementList = [
    Line2D([0], [0], marker='o', color='w', label='Outer Ring', markerfacecolor=colorMapObject(0.25), markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Inner Ring', markerfacecolor=colorMapObject(0.5), markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Center Nodes', markerfacecolor=colorMapObject(0.75), markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Central Node', markerfacecolor=colorMapObject(1.0), markersize=10),
    Line2D([0], [0], marker='o', color='w', label='End Towers', markerfacecolor='#4B0082', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Central Fountain', markerfacecolor='#808080', markersize=10)
]
axesDragonAdjusted.legend(handles=legendElementList,
                          loc='upper right',
                          facecolor=backgroundColor,
                          edgecolor=gridLineColor,
                          labelcolor='white')

scalarMappableObject = plt.cm.ScalarMappable(cmap=colorMapObject)
scalarMappableObject.set_array(normalizedDistanceArray)
colorBar = plt.colorbar(scalarMappableObject, shrink=0.8)
colorBar.set_label('Distance from Fountain', color=textFontColor)
colorBar.ax.yaxis.set_tick_params(color=textFontColor)
plt.setp(colorBar.ax.yaxis.get_ticklabels(), color=textFontColor)

# Compactifing layout and saving the figure
plt.tight_layout()
plt.savefig(os.path.join(downloadPath, "ender_dragon_pathing_graph_adjusted.png"), dpi=1000,
            bbox_inches="tight", facecolor=backgroundColor)
plt.show()



#%% Stronghold distribution [Overworld] - 차단하다


# Initializing figure
figureStrongholds, axesStrongholds = plt.subplots(figsize=(12, 12))
axesStrongholds.set_facecolor(backgroundColor)
figureStrongholds.patch.set_facecolor(backgroundColor)

# Initializing ring generation ranges
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

# Looping over stronghold generation rings
for ringDefinition in ringDefinitionList:
    ringMinimumRadius, ringMaximumRadius = ringDefinition['radius']
    ringCount = ringDefinition['count']
    ringColor = ringDefinition['color']
    
    annularWedgePatch = Wedge((0,0), ringMaximumRadius, 0, 360, width=(ringMaximumRadius - ringMinimumRadius),
                              color=ringColor, alpha=0.2)
    axesStrongholds.add_patch(annularWedgePatch)
    
    nodeAngleArray = np.linspace(0, 2*np.pi, ringCount, endpoint=False)
    nodeAngleArray += np.random.normal(0, np.pi/(ringCount*2), ringCount)
    nodeRadiusArray = np.random.uniform(ringMinimumRadius+100, ringMaximumRadius-100, ringCount)
    nodeXCoordinateArray = nodeRadiusArray * np.cos(nodeAngleArray)
    nodeYCoordinateArray = nodeRadiusArray * np.sin(nodeAngleArray)
    axesStrongholds.scatter(nodeXCoordinateArray, nodeYCoordinateArray, c=ringColor, s=100)

# Configuring axis and plot settings
axesStrongholds.set_aspect('equal', 'box')
axesStrongholds.grid(True, color=gridLineColor, linestyle='--', alpha=0.5)
axesStrongholds.set_title('Stronghold Distribution in the Overworld', color=textFontColor, pad=20)
axesStrongholds.set_xlabel('X Coordinate', color=textFontColor)
axesStrongholds.set_ylabel('Z Coordinate', color=textFontColor)

legendHandleList = [plt.Line2D([0],[0], color=r['color'], lw=4) for r in ringDefinitionList]
legendLabelList = [f'Ring {i+1}' for i in range(len(ringDefinitionList))]
axesStrongholds.legend(legendHandleList, legendLabelList, loc='upper right', facecolor=backgroundColor, edgecolor=gridLineColor, labelcolor='white')

plotDisplayLimitRadius = max(r['radius'][1] for r in ringDefinitionList) + 1000
axesStrongholds.set_xlim(-plotDisplayLimitRadius, plotDisplayLimitRadius)
axesStrongholds.set_ylim(-plotDisplayLimitRadius, plotDisplayLimitRadius)

# Compactifing layout and saving the figure
plt.tight_layout()
plt.savefig(os.path.join(downloadPath, "stronghold_distribution.png"), dpi=1000,
            bbox_inches="tight", facecolor=backgroundColor)
plt.show()



# %%
