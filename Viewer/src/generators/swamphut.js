import { statePicker } from "../transforms.js"

// Direct block-for-block port of SwampHutGenerator.generate in Java 1.16.1.
// Terrain-dependent post extension is represented by one visible log below
// each corner; the surrounding terrain itself is intentionally omitted.
export async function runSwampHut() {
  const { palette, stateFor } = statePicker()
  const cells = new Map()
  const key = (x, y, z) => `${x},${y},${z}`
  const put = (name, x, y, z, properties) => cells.set(key(x, y + 1, z), { name, properties })
  const remove = (x, y, z) => cells.delete(key(x, y + 1, z))
  const fill = (name, x0, y0, z0, x1, y1, z1, properties) => {
    for (let y = y0; y <= y1; y++) for (let z = z0; z <= z1; z++) for (let x = x0; x <= x1; x++)
      put(name, x, y, z, properties)
  }

  fill("minecraft:spruce_planks", 1, 1, 1, 5, 1, 7)
  fill("minecraft:spruce_planks", 1, 4, 2, 5, 4, 7)
  fill("minecraft:spruce_planks", 2, 1, 0, 4, 1, 0)
  fill("minecraft:spruce_planks", 2, 2, 2, 3, 3, 2)
  fill("minecraft:spruce_planks", 1, 2, 3, 1, 3, 6)
  fill("minecraft:spruce_planks", 5, 2, 3, 5, 3, 6)
  fill("minecraft:spruce_planks", 2, 2, 7, 4, 3, 7)
  for (const [x, z] of [[1, 2], [5, 2], [1, 7], [5, 7]]) fill("minecraft:oak_log", x, -1, z, x, 3, z, { axis: "y" })
  put("minecraft:oak_fence", 2, 3, 2)
  put("minecraft:oak_fence", 3, 3, 7)
  remove(1, 3, 4); remove(5, 3, 4); remove(5, 3, 5)
  put("minecraft:potted_red_mushroom", 1, 3, 5)
  put("minecraft:crafting_table", 3, 2, 6)
  put("minecraft:cauldron", 4, 2, 6, { level: "0" })
  put("minecraft:oak_fence", 1, 2, 1)
  put("minecraft:oak_fence", 5, 2, 1)

  const stair = (facing, shape = "straight") => ({ facing, half: "bottom", shape, waterlogged: "false" })
  fill("minecraft:spruce_stairs", 0, 4, 1, 6, 4, 1, stair("north"))
  fill("minecraft:spruce_stairs", 0, 4, 2, 0, 4, 7, stair("east"))
  fill("minecraft:spruce_stairs", 6, 4, 2, 6, 4, 7, stair("west"))
  fill("minecraft:spruce_stairs", 0, 4, 8, 6, 4, 8, stair("south"))
  put("minecraft:spruce_stairs", 0, 4, 1, stair("north", "outer_right"))
  put("minecraft:spruce_stairs", 6, 4, 1, stair("north", "outer_left"))
  put("minecraft:spruce_stairs", 0, 4, 8, stair("south", "outer_left"))
  put("minecraft:spruce_stairs", 6, 4, 8, stair("south", "outer_right"))

  const blocks = [...cells.entries()].map(([position, cell]) => ({
    state: stateFor(cell.name, cell.properties),
    pos: position.split(",").map(Number)
  }))
  return {
    structure: {
      size: [7, 6, 9], palette, blocks,
      entities: [
        { pos: [2.5, 3, 5.5], nbt: { id: "minecraft:witch" } },
        { pos: [2.5, 3, 5.5], nbt: { id: "minecraft:cat" } }
      ]
    },
    maxDepth: 1
  }
}
