export const BIOMES = new Map([
  [0, "Ocean"], [1, "Plains"], [2, "Desert"], [3, "Mountains"],
  [4, "Forest"], [5, "Taiga"], [6, "Swamp"], [7, "River"],
  [8, "Nether Wastes"], [9, "The End"], [10, "Frozen Ocean"],
  [11, "Frozen River"], [12, "Snowy Tundra"], [13, "Snowy Mountains"],
  [14, "Mushroom Fields"], [15, "Mushroom Field Shore"], [16, "Beach"],
  [17, "Desert Hills"], [18, "Wooded Hills"], [19, "Taiga Hills"],
  [20, "Mountain Edge"], [21, "Jungle"], [22, "Jungle Hills"],
  [23, "Jungle Edge"], [24, "Deep Ocean"], [25, "Stone Shore"],
  [26, "Snowy Beach"], [27, "Birch Forest"], [28, "Birch Forest Hills"],
  [29, "Dark Forest"], [30, "Snowy Taiga"], [31, "Snowy Taiga Hills"],
  [32, "Giant Tree Taiga"], [33, "Giant Tree Taiga Hills"],
  [34, "Wooded Mountains"], [35, "Savanna"], [36, "Savanna Plateau"],
  [37, "Badlands"], [38, "Wooded Badlands Plateau"], [39, "Badlands Plateau"],
  [40, "Small End Islands"], [41, "End Midlands"], [42, "End Highlands"],
  [43, "End Barrens"], [44, "Warm Ocean"], [45, "Lukewarm Ocean"],
  [46, "Cold Ocean"], [47, "Deep Warm Ocean"], [48, "Deep Lukewarm Ocean"],
  [49, "Deep Cold Ocean"], [50, "Deep Frozen Ocean"], [127, "The Void"],
  [129, "Sunflower Plains"], [130, "Desert Lakes"], [131, "Gravelly Mountains"],
  [132, "Flower Forest"], [133, "Taiga Mountains"], [134, "Swamp Hills"],
  [140, "Ice Spikes"], [149, "Modified Jungle"], [151, "Modified Jungle Edge"],
  [155, "Tall Birch Forest"], [156, "Tall Birch Hills"],
  [157, "Dark Forest Hills"], [158, "Snowy Taiga Mountains"],
  [160, "Giant Spruce Taiga"], [161, "Giant Spruce Taiga Hills"],
  [162, "Modified Gravelly Mountains"], [163, "Shattered Savanna"],
  [164, "Shattered Savanna Plateau"], [165, "Eroded Badlands"],
  [166, "Modified Wooded Badlands Plateau"], [167, "Modified Badlands Plateau"],
  [168, "Bamboo Jungle"], [169, "Bamboo Jungle Hills"],
  [170, "Soul Sand Valley"], [171, "Crimson Forest"],
  [172, "Warped Forest"], [173, "Basalt Deltas"]
])

const NETHER_BIOMES = new Set([8, 170, 171, 172, 173])
const END_BIOMES = new Set([9, 40, 41, 42, 43])

export function biomeName(id) {
  return BIOMES.get(id) ?? `Unknown biome (${id})`
}

export function biomesForDimension(dimension) {
  const included = dimension === -1
    ? NETHER_BIOMES
    : dimension === 1
      ? END_BIOMES
      : null
  return [...BIOMES.entries()].filter(([id]) => {
    if (included) return included.has(id)
    return id !== 127 && !NETHER_BIOMES.has(id) && !END_BIOMES.has(id)
  })
}

export const STRUCTURES = [
  { id: 0, key: "village", label: "Village", colour: "#d8b45b", dims: [0] },
  { id: 1, key: "desert_pyramid", label: "Desert Pyramid", colour: "#e0c36d", dims: [0] },
  { id: 2, key: "jungle_temple", label: "Jungle Temple", colour: "#54a55b", dims: [0] },
  { id: 3, key: "swamp_hut", label: "Swamp Hut", colour: "#6f9a73", dims: [0] },
  { id: 4, key: "igloo", label: "Igloo", colour: "#dceef2", dims: [0] },
  { id: 5, key: "ocean_ruin", label: "Ocean Ruin", colour: "#4b9cbf", dims: [0] },
  { id: 6, key: "shipwreck", label: "Shipwreck", colour: "#b47846", dims: [0] },
  { id: 7, key: "monument", label: "Ocean Monument", colour: "#52b4a7", dims: [0] },
  { id: 8, key: "mansion", label: "Woodland Mansion", colour: "#7f6052", dims: [0] },
  { id: 9, key: "outpost", label: "Pillager Outpost", colour: "#b46353", dims: [0] },
  { id: 10, key: "ruined_portal", label: "Ruined Portal", colour: "#a26bd2", dims: [0, -1] },
  { id: 11, key: "fortress", label: "Nether Fortress", colour: "#8f3d3f", dims: [-1] },
  { id: 12, key: "bastion", label: "Bastion Remnant", colour: "#ce7450", dims: [-1] },
  { id: 13, key: "end_city", label: "End City", colour: "#b78bc5", dims: [1] },
  { id: 14, key: "stronghold", label: "Stronghold", colour: "#77d4d0", dims: [0] }
]

export const DIMENSIONS = { overworld: 0, nether: -1, end: 1 }
