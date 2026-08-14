import "ol/ol.css"
import "./style.css"

import OLMap from "ol/Map.js"
import View from "ol/View.js"
import Projection from "ol/proj/Projection.js"
import TileGrid from "ol/tilegrid/TileGrid.js"
import DataTileSource from "ol/source/DataTile.js"
import VectorSource from "ol/source/Vector.js"
import WebGLTileLayer from "ol/layer/WebGLTile.js"
import VectorLayer from "ol/layer/Vector.js"
import Feature from "ol/Feature.js"
import Point from "ol/geom/Point.js"
import LineString from "ol/geom/LineString.js"
import Overlay from "ol/Overlay.js"
import { Circle as CircleStyle, Fill, Stroke, Style, Text } from "ol/style.js"
import { FullScreen, ScaleLine, defaults as defaultControls } from "ol/control.js"

import { biomeName, biomesForDimension, DIMENSIONS, STRUCTURES } from "./biomes.js"

const WORLD_LIMIT = 33_554_432
// Smaller tiles preserve per-pixel density accuracy while letting the map show
// completed center tiles progressively instead of waiting on large 256px jobs.
const TILE_SIZE = 128
const RESOLUTIONS = [256, 64, 16, 8, 4, 2, 1]
const DEFAULT_TYPES = new Set()

const elements = Object.fromEntries([
  "seed-form", "seed", "dimension", "biomes-toggle", "terrain-toggle", "grid-toggle",
  "viable-only", "structure-list", "select-all", "clear-all", "goto-form", "goto-x",
  "goto-z", "cursor", "biome-readout", "biome-legend", "status", "popup",
  "collapse-layers", "open-layers"
].map(id => [id, document.getElementById(id)]))

let requestId = 0
const pending = new Map()
const WORKER_COUNT = Math.min(4, Math.max(2, navigator.hardwareConcurrency || 2))
let workers = []
let nextWorker = 0
function resetWorker() {
  for (const worker of workers) worker.terminate()
  for (const request of pending.values()) request.reject(new Error("dimension changed"))
  pending.clear()
  workers = Array.from({ length: WORKER_COUNT }, () => {
    const worker = new Worker(new URL("./mc1161-worker.js", import.meta.url), { type: "module" })
    worker.onmessage = event => {
      const request = pending.get(event.data.id)
      if (!request) return
      pending.delete(event.data.id)
      event.data.ok ? request.resolve(event.data) : request.reject(new Error(event.data.error))
    }
    return worker
  })
  nextWorker = 0
}
resetWorker()

function callWorker(body) {
  const id = ++requestId
  const promise = new Promise((resolve, reject) => pending.set(id, { resolve, reject }))
  workers[nextWorker++ % workers.length].postMessage({ id, ...body })
  return promise
}

function parseSeed(value) {
  const text = String(value).trim()
  if (!/^[+-]?\d+$/.test(text)) throw new Error("Enter a signed decimal Java world seed")
  const seed = BigInt(text)
  if (seed < -(1n << 63n) || seed > (1n << 63n) - 1n)
    throw new Error("Seed must fit a signed Java 64-bit long")
  return seed.toString()
}

function parseHash() {
  const values = new URLSearchParams(location.hash.replace(/^#/, ""))
  const seed = values.get("seed")
  if (seed) {
    try { elements.seed.value = parseSeed(seed) } catch {}
  }
  const dimension = values.get("dim")
  if (dimension in DIMENSIONS) elements.dimension.value = dimension
  if (values.has("terrain")) elements["terrain-toggle"].checked = values.get("terrain") !== "0"
  const selected = values.get("layers")
  return {
    x: Number(values.get("x")) || 0,
    z: Number(values.get("z")) || 0,
    resolution: RESOLUTIONS.includes(Number(values.get("r"))) ? Number(values.get("r")) : 16,
    selected: values.has("layers")
      ? new Set(selected.split(",").filter(Boolean))
      : new Set(DEFAULT_TYPES)
  }
}

const initial = parseHash()
let seed = parseSeed(elements.seed.value)
let selectedTypes = initial.selected
let structureRequestToken = 0
const structureCache = new Map()
let biomeColors = null
let biomeHoverTimer = null
let biomeHoverToken = 0
const dimensionButtons = [...document.querySelectorAll("[data-dimension]")]

function dimensionLabel() {
  return elements.dimension.options[elements.dimension.selectedIndex]?.text ?? elements.dimension.value
}

function renderDimensionButtons() {
  for (const button of dimensionButtons)
    button.setAttribute("aria-pressed", String(button.dataset.dimension === elements.dimension.value))
}

const projection = new Projection({
  code: "MINECRAFT:BLOCKS",
  units: "m",
  extent: [-WORLD_LIMIT, -WORLD_LIMIT, WORLD_LIMIT, WORLD_LIMIT]
})
const tileGrid = new TileGrid({
  extent: projection.getExtent(),
  origin: [-WORLD_LIMIT, WORLD_LIMIT],
  resolutions: RESOLUTIONS,
  tileSize: TILE_SIZE
})

function sampleY(scale) {
  if (elements.dimension.value === "overworld") return scale === 1 ? 63 : 15
  if (elements.dimension.value === "nether") return scale === 1 ? 64 : 16
  return scale === 1 ? 64 : 16
}

const terrainSource = new DataTileSource({
  projection,
  tileGrid,
  tileSize: [TILE_SIZE, TILE_SIZE],
  bandCount: 4,
  wrapX: false,
  transition: 100,
  loader: async (z, x, y) => {
    const extent = tileGrid.getTileCoordExtent([z, x, y])
    const scale = RESOLUTIONS[z]
    const response = await callWorker({
      type: "tile",
      seed,
      dimension: DIMENSIONS[elements.dimension.value],
      scale,
      sampleX: Math.floor(extent[0] / scale),
      sampleZ: Math.floor(-extent[3] / scale),
      sampleY: sampleY(scale),
      width: TILE_SIZE,
      height: TILE_SIZE,
      biomes: elements["biomes-toggle"].checked,
      terrain: elements["terrain-toggle"].checked
    })
    return new Uint8Array(response.rgba)
  }
})
const terrainLayer = new WebGLTileLayer({ source: terrainSource })

const structureSource = new VectorSource()
const styleCache = new Map()
function structureStyle(feature) {
  const type = STRUCTURES[feature.get("type")]
  const viable = feature.get("viable")
  const cacheKey = `${type.id}:${viable}`
  if (!styleCache.has(cacheKey)) {
    styleCache.set(cacheKey, new Style({
      image: new CircleStyle({
        radius: viable ? 6 : 5,
        fill: new Fill({ color: viable ? type.colour : "rgba(39,43,48,.75)" }),
        stroke: new Stroke({ color: viable ? "#f5f0dc" : type.colour, width: viable ? 1.5 : 2 })
      }),
      text: new Text({
        text: type.label.slice(0, 1),
        font: "600 9px system-ui",
        fill: new Fill({ color: viable ? "#16191d" : "#f5f0dc" })
      })
    }))
  }
  return styleCache.get(cacheKey)
}
const structureLayer = new VectorLayer({ source: structureSource, style: structureStyle, zIndex: 5 })

const endLandmarkSource = new VectorSource()
const endLandmarkLayer = new VectorLayer({
  source: endLandmarkSource,
  zIndex: 4,
  style: new Style({
    image: new CircleStyle({
      radius: 5,
      fill: new Fill({ color: "#f0cf67" }),
      stroke: new Stroke({ color: "#2b1e39", width: 2 })
    }),
    text: new Text({
      text: "Exit portal",
      offsetY: 14,
      font: "600 10px system-ui",
      fill: new Fill({ color: "#f3df9a" }),
      stroke: new Stroke({ color: "rgba(20,12,28,.92)", width: 3 })
    })
  })
})

const gridSource = new VectorSource()
const gridLayer = new VectorLayer({
  source: gridSource,
  visible: false,
  zIndex: 3,
  style: new Style({ stroke: new Stroke({ color: "rgba(229,238,231,.25)", width: 1 }) })
})

const map = new OLMap({
  target: "map",
  layers: [terrainLayer, gridLayer, endLandmarkLayer, structureLayer],
  controls: defaultControls({ zoom: true, rotate: false, attribution: false }).extend([
    new FullScreen(),
    new ScaleLine({ units: "metric", bar: true, minWidth: 100 })
  ]),
  view: new View({
    projection,
    center: [initial.x, -initial.z],
    resolutions: RESOLUTIONS,
    resolution: initial.resolution,
    constrainOnlyCenter: true,
    extent: projection.getExtent()
  })
})

const popupOverlay = new Overlay({
  element: elements.popup,
  offset: [0, -12],
  positioning: "bottom-center",
  stopEvent: true
})
map.addOverlay(popupOverlay)

function selectedMask() {
  return STRUCTURES.reduce((mask, type) => selectedTypes.has(type.key) ? (mask | (1 << type.id)) : mask, 0)
}

function structureKey(type, x, z) {
  return `${type}:${x}:${z}`
}

function dimensionTypes() {
  const dimension = DIMENSIONS[elements.dimension.value]
  return STRUCTURES.filter(type => type.dims.includes(dimension))
}

function renderStructureControls() {
  const dimension = DIMENSIONS[elements.dimension.value]
  elements["structure-list"].replaceChildren(...STRUCTURES.map(type => {
    const label = document.createElement("label")
    label.className = "structure-toggle"
    label.dataset.dimensionVisible = String(type.dims.includes(dimension))
    const input = document.createElement("input")
    input.type = "checkbox"
    input.checked = selectedTypes.has(type.key)
    input.addEventListener("change", () => {
      input.checked ? selectedTypes.add(type.key) : selectedTypes.delete(type.key)
      updateStructures()
      syncHash()
    })
    const dot = document.createElement("span")
    dot.className = "marker-dot"
    dot.style.background = type.colour
    const text = document.createElement("span")
    text.textContent = type.label
    label.append(input, dot, text)
    return label
  }))
}

function renderBiomeLegend() {
  if (!biomeColors) return
  const dimension = DIMENSIONS[elements.dimension.value]
  elements["biome-legend"].replaceChildren(...biomesForDimension(dimension).map(([id, name]) => {
    const row = document.createElement("div")
    row.className = "biome-legend-row"
    const swatch = document.createElement("span")
    swatch.className = "biome-swatch"
    swatch.style.background = `rgb(${biomeColors[id * 3]}, ${biomeColors[id * 3 + 1]}, ${biomeColors[id * 3 + 2]})`
    const label = document.createElement("span")
    label.textContent = name
    const code = document.createElement("span")
    code.className = "biome-id"
    code.textContent = String(id)
    row.append(swatch, label, code)
    return row
  }))
}

function updateGrid() {
  gridSource.clear()
  if (!elements["grid-toggle"].checked) return
  const resolution = map.getView().getResolution()
  if (resolution > 16) return
  const extent = map.getView().calculateExtent(map.getSize())
  const minX = Math.floor(extent[0] / 16) * 16
  const maxX = Math.ceil(extent[2] / 16) * 16
  const minZ = Math.floor(-extent[3] / 16) * 16
  const maxZ = Math.ceil(-extent[1] / 16) * 16
  if ((maxX - minX) / 16 + (maxZ - minZ) / 16 > 700) return
  const lines = []
  for (let x = minX; x <= maxX; x += 16)
    lines.push(new Feature(new LineString([[x, -minZ], [x, -maxZ]])))
  for (let z = minZ; z <= maxZ; z += 16)
    lines.push(new Feature(new LineString([[minX, -z], [maxX, -z]])))
  gridSource.addFeatures(lines)
}

function updateEndLandmark() {
  endLandmarkSource.clear()
  if (elements.dimension.value === "end")
    endLandmarkSource.addFeature(new Feature(new Point([0, 0])))
}

async function updateStructures() {
  const token = ++structureRequestToken
  const size = map.getSize()
  if (!size) return
  const raw = map.getView().calculateExtent(size)
  const resolution = map.getView().getResolution()
  const quant = Math.max(512, Math.round(resolution * 128))
  const minX = Math.floor(raw[0] / quant) * quant
  const maxX = Math.ceil(raw[2] / quant) * quant
  const minZ = Math.floor(-raw[3] / quant) * quant
  const maxZ = Math.ceil(-raw[1] / quant) * quant
  const mask = selectedMask()

  if (!mask) {
    structureSource.clear()
    elements.status.textContent = `${dimensionLabel()} · Java 1.16.1 · Cubiomes WASM · seed ${seed}`
    return
  }
  if (maxX - minX > 140_000 || maxZ - minZ > 140_000) {
    structureSource.clear()
    elements.status.textContent = `${dimensionLabel()} biome map ready · zoom in to calculate structure markers`
    return
  }

  const key = [seed, elements.dimension.value, minX, minZ, maxX, maxZ, mask].join(":")
  elements.status.textContent = `Calculating ${dimensionLabel()} Java 1.16.1 structures…`
  try {
    let hits = structureCache.get(key)
    if (!hits) {
      const response = await callWorker({
        type: "structures",
        seed,
        dimension: DIMENSIONS[elements.dimension.value],
        minX, minZ, maxX, maxZ, mask
      })
      hits = response.hits
      structureCache.set(key, hits)
      if (structureCache.size > 24) structureCache.delete(structureCache.keys().next().value)
    }
    if (token !== structureRequestToken) return
    const viableOnly = elements["viable-only"].checked
    const unique = new Map()
    for (const hit of hits) {
      if (viableOnly && !hit.viable) continue
      unique.set(structureKey(hit.type, hit.x, hit.z), hit)
    }
    const features = [...unique.values()].map(hit => new Feature({
      geometry: new Point([hit.x, -hit.z]),
      ...hit
    }))
    structureSource.clear()
    structureSource.addFeatures(features)
    elements.status.textContent = `Ready · ${dimensionLabel()} · ${features.length} markers · seed ${seed}`
  } catch (error) {
    if (token !== structureRequestToken) return
    elements.status.textContent = `Structure query failed: ${error.message}`
  }
}

function refreshWorld() {
  structureCache.clear()
  terrainSource.setKey([
    seed,
    elements.dimension.value,
    elements["biomes-toggle"].checked ? 1 : 0,
    elements["terrain-toggle"].checked ? 1 : 0
  ].join(":"))
  terrainSource.refresh()
  updateStructures()
  updateGrid()
  updateEndLandmark()
  syncHash()
}

function syncHash() {
  const center = map.getView().getCenter() ?? [0, 0]
  const params = new URLSearchParams({
    seed,
    x: String(Math.round(center[0])),
    z: String(Math.round(-center[1])),
    r: String(map.getView().getResolution()),
    dim: elements.dimension.value,
    terrain: elements["terrain-toggle"].checked ? "1" : "0",
    layers: [...selectedTypes].join(",")
  })
  history.replaceState(null, "", `#${params}`)
}

elements["seed-form"].addEventListener("submit", event => {
  event.preventDefault()
  try {
    seed = parseSeed(elements.seed.value)
    elements.seed.value = seed
    resetWorker()
    refreshWorld()
  } catch (error) {
    elements.status.textContent = error.message
  }
})

elements.dimension.addEventListener("change", () => {
  resetWorker()
  renderDimensionButtons()
  renderStructureControls()
  renderBiomeLegend()
  refreshWorld()
})

for (const button of dimensionButtons) {
  button.addEventListener("click", () => {
    const next = button.dataset.dimension
    if (!(next in DIMENSIONS) || next === elements.dimension.value) return
    elements.dimension.value = next
    elements.dimension.dispatchEvent(new Event("change"))
  })
}

for (const id of ["biomes-toggle", "terrain-toggle"]) {
  elements[id].addEventListener("change", refreshWorld)
}
elements["grid-toggle"].addEventListener("change", () => {
  gridLayer.setVisible(elements["grid-toggle"].checked)
  updateGrid()
})
elements["viable-only"].addEventListener("change", updateStructures)

elements["select-all"].addEventListener("click", () => {
  for (const type of dimensionTypes()) selectedTypes.add(type.key)
  renderStructureControls()
  updateStructures()
  syncHash()
})
elements["clear-all"].addEventListener("click", () => {
  for (const type of dimensionTypes()) selectedTypes.delete(type.key)
  renderStructureControls()
  updateStructures()
  syncHash()
})

elements["goto-form"].addEventListener("submit", event => {
  event.preventDefault()
  const x = Number(elements["goto-x"].value)
  const z = Number(elements["goto-z"].value)
  if (!Number.isFinite(x) || !Number.isFinite(z)) return
  map.getView().animate({ center: [x, -z], duration: 300 })
})

elements["collapse-layers"].addEventListener("click", () => document.body.classList.add("layers-collapsed"))
elements["open-layers"].addEventListener("click", () => document.body.classList.remove("layers-collapsed"))

map.on("pointermove", event => {
  const x = Math.round(event.coordinate[0])
  const z = Math.round(-event.coordinate[1])
  elements.cursor.textContent = `X ${x.toLocaleString()} / Z ${z.toLocaleString()}`
  clearTimeout(biomeHoverTimer)
  const token = ++biomeHoverToken
  biomeHoverTimer = setTimeout(async () => {
    try {
      const response = await callWorker({
        type: "biomePoint",
        seed,
        dimension: DIMENSIONS[elements.dimension.value],
        x,
        z,
        y: sampleY(1)
      })
      if (token === biomeHoverToken)
        elements["biome-readout"].textContent = `${biomeName(response.biome)} / ID ${response.biome}`
    } catch (error) {
      if (token === biomeHoverToken)
        elements["biome-readout"].textContent = `Biome lookup failed: ${error.message}`
    }
  }, 70)
})
map.on("moveend", () => {
  updateGrid()
  updateStructures()
  syncHash()
})
map.on("singleclick", event => {
  const feature = map.forEachFeatureAtPixel(event.pixel, candidate => candidate, {
    layerFilter: layer => layer === structureLayer,
    hitTolerance: 6
  })
  if (!feature) {
    popupOverlay.setPosition(undefined)
    elements.popup.hidden = true
    return
  }
  const type = STRUCTURES[feature.get("type")]
  const viable = feature.get("viable")
  const terrainSensitive = feature.get("terrainSensitive")
  elements.popup.innerHTML = `
    <strong>${type.label}</strong>
    <span>X ${feature.get("x").toLocaleString()} · Z ${feature.get("z").toLocaleString()}</span>
    <em>${viable ? "biome viable" : "generation attempt"}${terrainSensitive ? " · terrain gate remains" : ""}</em>
  `
  elements.popup.hidden = false
  popupOverlay.setPosition(feature.getGeometry().getCoordinates())
})

renderStructureControls()
renderDimensionButtons()
gridLayer.setVisible(elements["grid-toggle"].checked)
updateEndLandmark()
callWorker({ type: "ready" }).then(response => {
  biomeColors = response.colors
  renderBiomeLegend()
  elements.status.textContent = `Ready · ${dimensionLabel()} · Java 1.16.1 · seed ${seed}`
  updateStructures()
}).catch(error => {
  elements.status.textContent = `Cubiomes failed to load: ${error.message}`
})
