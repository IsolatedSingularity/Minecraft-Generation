const COLORS = new Map([
  [0, [40, 83, 151]], [1, [123, 187, 74]], [2, [219, 201, 119]],
  [3, [96, 96, 96]], [4, [52, 126, 57]], [5, [44, 112, 77]],
  [6, [71, 107, 80]], [7, [49, 101, 159]], [8, [108, 54, 32]],
  [9, [125, 116, 179]], [10, [80, 113, 175]], [11, [91, 140, 192]],
  [12, [222, 236, 240]], [13, [228, 239, 242]], [14, [118, 94, 137]],
  [15, [113, 87, 133]], [16, [231, 221, 179]], [17, [210, 183, 101]],
  [18, [44, 107, 49]], [19, [50, 101, 68]], [20, [105, 105, 105]],
  [21, [45, 132, 54]], [22, [38, 111, 49]], [23, [68, 138, 68]],
  [24, [27, 60, 113]], [25, [123, 123, 123]], [26, [222, 233, 235]],
  [27, [78, 137, 72]], [28, [65, 117, 62]], [29, [44, 79, 47]],
  [30, [61, 103, 88]], [31, [53, 91, 80]], [32, [76, 108, 83]],
  [33, [67, 95, 77]], [34, [83, 92, 83]], [35, [177, 183, 74]],
  [36, [160, 166, 64]], [37, [184, 93, 58]], [38, [160, 83, 54]],
  [39, [174, 98, 65]], [40, [86, 79, 120]], [41, [118, 96, 150]],
  [42, [139, 111, 166]], [43, [99, 85, 133]], [44, [55, 132, 186]],
  [45, [47, 118, 175]], [46, [51, 98, 157]], [47, [28, 83, 137]],
  [48, [25, 72, 126]], [49, [27, 67, 116]], [50, [45, 86, 143]],
  [127, [26, 24, 31]], [129, [146, 198, 78]], [130, [226, 193, 112]],
  [131, [115, 108, 107]], [132, [77, 147, 74]], [133, [53, 119, 83]],
  [134, [85, 121, 89]], [140, [232, 243, 246]], [149, [58, 151, 70]],
  [151, [87, 157, 82]], [155, [98, 157, 86]], [156, [82, 139, 76]],
  [157, [58, 95, 57]], [158, [74, 116, 98]], [160, [88, 124, 92]],
  [161, [74, 108, 86]], [162, [111, 111, 106]], [163, [191, 194, 81]],
  [164, [170, 174, 70]], [165, [195, 107, 69]], [166, [171, 94, 62]],
  [167, [186, 111, 76]], [168, [62, 158, 73]], [169, [50, 133, 60]],
  [170, [80, 63, 56]], [171, [136, 37, 57]], [172, [35, 120, 111]],
  [173, [73, 71, 68]]
])

export function biomeColor(id) {
  const direct = COLORS.get(id)
  if (direct) return direct
  const base = id >= 128 ? COLORS.get(id - 128) : null
  return base ?? [92, 104, 95]
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
