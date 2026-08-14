import { rand32, rnd, shuffle, statePicker } from "../transforms.js"

// end spikes (EndSpikeFeature): parametric, so built in code

// the podium's portal ring sits at the End's surface height
const PORTAL_Y = 62

// Exact block rules from 1.16.1 EndPortalFeature.generate. The feature is
// code-generated in vanilla (there is no structure NBT in the client jar).
function buildExitPortal(stateFor, blocks, active) {
  const within = (x, y, z, radius) => x * x + y * y + z * z <= radius * radius
  for (let x = -4; x <= 4; x++) for (let y = -1; y <= 32; y++) for (let z = -4; z <= 4; z++) {
    const inner = within(x, y, z, 2.5)
    if (!inner && !within(x, y, z, 3.5)) continue
    let name
    if (y < 0) name = inner ? "minecraft:bedrock" : "minecraft:end_stone"
    else if (y > 0) continue // air above the fountain is not emitted
    else if (!inner) name = "minecraft:bedrock"
    else if (active) name = "minecraft:end_portal"
    else continue
    blocks.push({ state: stateFor(name), pos: [x, PORTAL_Y + y, z] })
  }
  for (let y = 0; y < 4; y++) blocks.push({ state: stateFor("minecraft:bedrock"), pos: [0, PORTAL_Y + y, 0] })
  for (const [x, z, facing] of [[0, -1, "north"], [0, 1, "south"], [-1, 0, "west"], [1, 0, "east"]]) {
    blocks.push({ state: stateFor("minecraft:wall_torch", { facing, lit: "true" }), pos: [x, PORTAL_Y + 2, z] })
  }
}

function buildSpike(stateFor, blocks, entities, cx, cz, size) {
  const radius = 2 + Math.floor(size / 3)
  const height = 76 + size * 3
  const guarded = size === 1 || size === 2
  const obsidian = stateFor("minecraft:obsidian")

  for (let x = cx - radius; x <= cx + radius; x++) {
    for (let z = cz - radius; z <= cz + radius; z++) {
      if ((cx - x) * (cx - x) + (cz - z) * (cz - z) > radius * radius + 1) continue
      for (let y = 0; y < height; y++) blocks.push({ state: obsidian, pos: [x, y, z] })
    }
  }

  if (guarded) {
    for (let dx = -2; dx <= 2; dx++) {
      for (let dz = -2; dz <= 2; dz++) {
        for (let dy = 0; dy <= 3; dy++) {
          const xSide = Math.abs(dx) === 2, zSide = Math.abs(dz) === 2, top = dy === 3
          if (!xSide && !zSide && !top) continue
          const xEdge = dx === -2 || dx === 2 || top
          const zEdge = dz === -2 || dz === 2 || top
          blocks.push({
            state: stateFor("minecraft:iron_bars", {
              north: String(xEdge && dz !== -2),
              south: String(xEdge && dz !== 2),
              west: String(zEdge && dx !== -2),
              east: String(zEdge && dx !== 2)
            }),
            pos: [cx + dx, height + dy, cz + dz]
          })
        }
      }
    }
  }

  blocks.push({ state: stateFor("minecraft:bedrock"), pos: [cx, height, cz] })
  blocks.push({ state: stateFor("minecraft:fire"), pos: [cx, height + 1, cz] })
  entities.push({ pos: [cx + 0.5, height + 1, cz + 0.5], nbt: { id: "minecraft:end_crystal" } })
}

function normalise(palette, blocks, entities) {
  const lo = [Infinity, 0, Infinity], hi = [-Infinity, 0, -Infinity]
  for (const b of blocks) {
    lo[0] = Math.min(lo[0], b.pos[0]); lo[2] = Math.min(lo[2], b.pos[2])
    hi[0] = Math.max(hi[0], b.pos[0]); hi[1] = Math.max(hi[1], b.pos[1]); hi[2] = Math.max(hi[2], b.pos[2])
  }
  for (const b of blocks) { b.pos = [b.pos[0] - lo[0], b.pos[1], b.pos[2] - lo[2]] }
  for (const e of entities) { e.pos = [e.pos[0] - lo[0], e.pos[1], e.pos[2] - lo[2]] }
  return {
    size: [hi[0] - lo[0] + 1, hi[1] + 1, hi[2] - lo[2] + 1],
    palette, blocks, entities,
    anchor: [-lo[0], 0, -lo[2]]
  }
}

export const makeEndSpikes = active => async (_loadStruct, { seed } = {}) => {
  const rand = rnd(seed ?? rand32())
  const sizes = shuffle([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], rand)
  const { palette, stateFor } = statePicker()
  const blocks = [], entities = []

  for (let i = 0; i < 10; i++) {
    const cx = Math.floor(42 * Math.cos(2 * (-Math.PI + Math.PI / 10 * i)))
    const cz = Math.floor(42 * Math.sin(2 * (-Math.PI + Math.PI / 10 * i)))
    buildSpike(stateFor, blocks, entities, cx, cz, sizes[i])
  }

  buildExitPortal(stateFor, blocks, active)

  // anchor on the portal (the base nbt's origin) so the camera stays with it
  const structure = normalise(palette, blocks, entities)
  structure.anchor = [structure.anchor[0], PORTAL_Y, structure.anchor[2]]
  return { structure, maxDepth: 1 }
}

export const runEndSpikes = makeEndSpikes(false)
export const runEndSpikesActive = makeEndSpikes(true)

// a single spike is deterministic per size, so each size is its own entry
const OPEN_SIZES = [0, 3, 4, 5, 6, 7, 8, 9]

export const makeEndSpikeSize = size => async () => {
  const { palette, stateFor } = statePicker()
  const blocks = [], entities = []
  buildSpike(stateFor, blocks, entities, 0, 0, size)
  return { structure: normalise(palette, blocks, entities), maxDepth: 1 }
}

export const runEndSpike = async (loadStruct, { seed } = {}) => {
  const rand = rnd(seed ?? rand32())
  return makeEndSpikeSize(OPEN_SIZES[Math.floor(rand() * OPEN_SIZES.length)])()
}
