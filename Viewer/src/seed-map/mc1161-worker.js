import createCubiomes from "./generated/mc1161.js"
import wasmUrl from "./generated/mc1161.wasm?url"

const modulePromise = createCubiomes({
  locateFile: path => path.endsWith(".wasm") ? wasmUrl : path
})

let apiPromise
const contexts = new Map()

function splitSeed(seedText) {
  const seed = BigInt.asUintN(64, BigInt(seedText))
  return [Number((seed >> 32n) & 0xffffffffn), Number(seed & 0xffffffffn)]
}

async function api() {
  if (!apiPromise) {
    apiPromise = modulePromise.then(module => {
      const bindings = {
        module,
        create: module.cwrap("mc_create", "number", ["number", "number", "number"]),
        destroy: module.cwrap("mc_destroy", null, ["number"]),
        biomeColors: module.cwrap("mc_biome_colors", "number", ["number"]),
        biomeTile: module.cwrap("mc_biome_tile", "number", ["number", "number", "number", "number", "number", "number", "number", "number"]),
        heightTile: module.cwrap("mc_height_tile", "number", ["number", "number", "number", "number", "number", "number", "number"]),
        structures: module.cwrap("mc_structures", "number", ["number", "number", "number", "number", "number", "number", "number", "number"]),
        stride: module.cwrap("mc_structure_stride", "number", [])()
      }
      const colorsPointer = module._malloc(256 * 3)
      try {
        const result = bindings.biomeColors(colorsPointer)
        if (result) throw new Error(`Cubiomes biome palette failed (${result})`)
        bindings.colors = new Uint8Array(module.HEAPU8.buffer, colorsPointer, 256 * 3).slice()
      } finally {
        module._free(colorsPointer)
      }
      return bindings
    })
  }
  return apiPromise
}

async function contextFor(seed, dimension) {
  const bindings = await api()
  const key = `${seed}:${dimension}`
  if (contexts.has(key)) return contexts.get(key)
  const [high, low] = splitSeed(seed)
  const pointer = bindings.create(high, low, dimension)
  if (!pointer) throw new Error("Cubiomes context allocation failed")
  contexts.set(key, pointer)
  while (contexts.size > 4) {
    const [oldKey, oldPointer] = contexts.entries().next().value
    bindings.destroy(oldPointer)
    contexts.delete(oldKey)
  }
  return pointer
}

function shadePixel(heights, width, x, y) {
  const left = heights[y * width + Math.max(0, x - 1)]
  const right = heights[y * width + Math.min(width - 1, x + 1)]
  const top = heights[Math.max(0, y - 1) * width + x]
  const bottom = heights[Math.min(heights.length / width - 1, y + 1) * width + x]
  return Math.max(0.58, Math.min(1.2, 0.89 + (left - right + top - bottom) * 0.025))
}

async function renderTile(message) {
  const bindings = await api()
  const context = await contextFor(message.seed, message.dimension)
  const count = message.width * message.height
  const idsPointer = bindings.module._malloc(count * 4)
  try {
    const result = bindings.biomeTile(
      context,
      message.scale,
      message.sampleX,
      message.sampleZ,
      message.width,
      message.height,
      message.sampleY,
      idsPointer
    )
    if (result) throw new Error(`Cubiomes biome generation failed (${result})`)
    const ids = new Int32Array(bindings.module.HEAP32.buffer, idsPointer, count).slice()

    let heights = null
    let heightWidth = 0
    let directStride = 1
    if (message.terrain && message.scale <= 16) {
      const directSurface = message.dimension !== 0
      // Preserve one density sample per displayed pixel through the ordinary
      // overview zoom. The bounded worker pool keeps dimension changes quick.
      directStride = 1
      const blocksWide = message.width * message.scale
      const blocksHigh = message.height * message.scale
      heightWidth = directSurface ? Math.ceil(message.width / directStride) + 2 : Math.ceil(blocksWide / 4) + 2
      const heightHeight = directSurface ? Math.ceil(message.height / directStride) + 2 : Math.ceil(blocksHigh / 4) + 2
      const heightsPointer = bindings.module._malloc(heightWidth * heightHeight * 4)
      try {
        const heightResult = bindings.heightTile(
          context,
          directSurface ? message.scale * directStride : message.scale,
          directSurface ? Math.floor(message.sampleX / directStride) - 1 : Math.floor(message.sampleX * message.scale / 4) - 1,
          directSurface ? Math.floor(message.sampleZ / directStride) - 1 : Math.floor(message.sampleZ * message.scale / 4) - 1,
          heightWidth,
          heightHeight,
          heightsPointer
        )
        if (!heightResult)
          heights = new Float32Array(bindings.module.HEAPF32.buffer, heightsPointer, heightWidth * heightHeight).slice()
      } finally {
        bindings.module._free(heightsPointer)
      }
    }

    const rgba = new Uint8Array(count * 4)
    for (let y = 0; y < message.height; y++) {
      for (let x = 0; x < message.width; x++) {
        const index = y * message.width + x
        const biomeId = ids[index]
        const colour = message.biomes
          ? bindings.colors.subarray(biomeId * 3, biomeId * 3 + 3)
          : [36, 39, 43]
        let shade = 1
        let surfaceHeight = null
        if (heights) {
          const directSurface = message.dimension !== 0
          const hx = directSurface ? Math.floor(x / directStride) + 1 : Math.min(heightWidth - 2, Math.floor(x * message.scale / 4) + 1)
          const hy = directSurface ? Math.floor(y / directStride) + 1 : Math.min(heights.length / heightWidth - 2, Math.floor(y * message.scale / 4) + 1)
          shade = shadePixel(heights, heightWidth, hx, hy)
          surfaceHeight = heights[hy * heightWidth + hx]
        }
        const isEndVoid = message.dimension === 1 && surfaceHeight <= 0
        const base = isEndVoid ? [12, 8, 22] : colour
        rgba[index * 4] = Math.round(base[0] * shade)
        rgba[index * 4 + 1] = Math.round(base[1] * shade)
        rgba[index * 4 + 2] = Math.round(base[2] * shade)
        rgba[index * 4 + 3] = 255
      }
    }
    return rgba
  } finally {
    bindings.module._free(idsPointer)
  }
}

async function biomePoint(message) {
  const bindings = await api()
  const context = await contextFor(message.seed, message.dimension)
  const pointer = bindings.module._malloc(4)
  try {
    const result = bindings.biomeTile(
      context,
      1,
      message.x,
      message.z,
      1,
      1,
      message.y,
      pointer
    )
    if (result) throw new Error(`Cubiomes biome lookup failed (${result})`)
    return bindings.module.HEAP32[pointer >> 2]
  } finally {
    bindings.module._free(pointer)
  }
}

async function queryStructures(message) {
  const bindings = await api()
  const context = await contextFor(message.seed, message.dimension)
  const capacity = 32768
  const pointer = bindings.module._malloc(capacity * bindings.stride)
  try {
    const count = bindings.structures(
      context,
      message.minX,
      message.minZ,
      message.maxX,
      message.maxZ,
      message.mask,
      pointer,
      capacity
    )
    if (count < 0) throw new Error(`Structure result capacity exceeded (${Math.abs(count)})`)
    const view = new DataView(bindings.module.HEAPU8.buffer, pointer, count * bindings.stride)
    const hits = []
    for (let index = 0; index < count; index++) {
      const offset = index * bindings.stride
      hits.push({
        type: view.getInt32(offset, true),
        x: view.getInt32(offset + 4, true),
        z: view.getInt32(offset + 8, true),
        viable: Boolean(view.getInt32(offset + 12, true)),
        terrainSensitive: Boolean(view.getInt32(offset + 16, true))
      })
    }
    return hits
  } finally {
    bindings.module._free(pointer)
  }
}

self.onmessage = async event => {
  const { id, type } = event.data
  try {
    if (type === "ready") {
      const bindings = await api()
      self.postMessage({ id, ok: true, colors: Array.from(bindings.colors) })
      return
    }
    if (type === "tile") {
      const rgba = await renderTile(event.data)
      self.postMessage({ id, ok: true, rgba: rgba.buffer }, [rgba.buffer])
      return
    }
    if (type === "structures") {
      const hits = await queryStructures(event.data)
      self.postMessage({ id, ok: true, hits })
      return
    }
    if (type === "biomePoint") {
      const biome = await biomePoint(event.data)
      self.postMessage({ id, ok: true, biome })
      return
    }
    throw new Error(`Unknown worker request: ${type}`)
  } catch (error) {
    self.postMessage({ id, ok: false, error: String(error?.message ?? error) })
  }
}
