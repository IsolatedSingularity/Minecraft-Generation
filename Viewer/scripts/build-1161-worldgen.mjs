import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { writeZip } from "../tools/builtin/zip.js"

const here = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(here, "../..")
const sourceRoot = path.join(root, "Game Reference/08_mc_1_16_1_agent_reference/source/java/net/minecraft/structure")
const output = path.join(root, "Assets/minecraft_1_16_1/viewer/worldgen_registry.zip")
const sourceFiles = [
  "PlainsVillageData.java", "DesertVillageData.java", "SavannaVillageData.java",
  "SnowyVillageData.java", "TaigaVillageData.java", "PillagerOutpostGenerator.java",
  "BastionData.java", "BastionUnitsData.java", "HoglinStableData.java",
  "BastionTreasureData.java", "BastionBridgeData.java"
]

function matchingParen(text, open) {
  let depth = 0, quote = false, escape = false
  for (let i = open; i < text.length; i++) {
    const ch = text[i]
    if (quote) {
      if (escape) escape = false
      else if (ch === "\\") escape = true
      else if (ch === '"') quote = false
      continue
    }
    if (ch === '"') { quote = true; continue }
    if (ch === "(") depth++
    else if (ch === ")" && --depth === 0) return i
  }
  throw new Error(`unbalanced Java expression at ${open}`)
}

function calls(text, marker) {
  const out = []
  let at = 0
  while ((at = text.indexOf(marker, at)) >= 0) {
    const open = at + marker.length - 1
    const close = matchingParen(text, open)
    out.push(text.slice(open + 1, close))
    at = close + 1
  }
  return out
}

function splitArgs(text) {
  const out = []
  let start = 0, round = 0, square = 0, curly = 0, quote = false, escape = false
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (quote) {
      if (escape) escape = false
      else if (ch === "\\") escape = true
      else if (ch === '"') quote = false
      continue
    }
    if (ch === '"') { quote = true; continue }
    if (ch === "(") round++
    else if (ch === ")") round--
    else if (ch === "[") square++
    else if (ch === "]") square--
    else if (ch === "{") curly++
    else if (ch === "}") curly--
    else if (ch === "," && round === 0 && square === 0 && curly === 0) {
      out.push(text.slice(start, i).trim())
      start = i + 1
    }
  }
  out.push(text.slice(start).trim())
  return out
}

const firstString = text => text.match(/"([^"\\]*(?:\\.[^"\\]*)*)"/)?.[1]
const id = value => value.includes(":") ? value : `minecraft:${value}`

function singleElement(expression, projection) {
  if (/EmptyPoolElement\.INSTANCE/.test(expression) || /FeaturePoolElement\s*\(/.test(expression))
    return { element_type: "minecraft:empty_pool_element" }

  if (/ListPoolElement\s*\(/.test(expression)) {
    const elements = []
    for (const marker of ["new LegacySinglePoolElement(", "new SinglePoolElement("]) {
      for (const body of calls(expression, marker)) {
        const location = firstString(body)
        if (location) elements.push({
          element_type: marker.includes("Legacy") ? "minecraft:legacy_single_pool_element" : "minecraft:single_pool_element",
          location: id(location), processors: [], projection
        })
      }
    }
    return elements.length ? { element_type: "minecraft:list_pool_element", elements, projection } : null
  }

  const legacy = /LegacySinglePoolElement\s*\(/.test(expression)
  if (!legacy && !/SinglePoolElement\s*\(/.test(expression)) return null
  const location = firstString(expression)
  return location ? {
    element_type: legacy ? "minecraft:legacy_single_pool_element" : "minecraft:single_pool_element",
    location: id(location), processors: [], projection
  } : null
}

function pairBodies(expression) {
  const found = []
  for (const marker of ["Pair.of(", "new Pair<>("]) {
    let at = 0
    while ((at = expression.indexOf(marker, at)) >= 0) {
      const open = at + marker.length - 1
      const close = matchingParen(expression, open)
      found.push({ at, body: expression.slice(open + 1, close) })
      at = close + 1
    }
  }
  return found.sort((a, b) => a.at - b.at).map(entry => entry.body)
}

function parsePool(body, source) {
  const args = splitArgs(body)
  if (args.length < 4) return null
  const name = firstString(args[0])
  const fallback = firstString(args[1])
  if (!name || !fallback) return null
  const projection = /TERRAIN_MATCHING/.test(args.at(-1)) ? "terrain_matching" : "rigid"
  const elements = []
  for (const pair of pairBodies(args[2])) {
    const pairArgs = splitArgs(pair)
    const weight = Number.parseInt(pairArgs.at(-1), 10)
    const element = singleElement(pairArgs[0], projection)
    if (element && Number.isInteger(weight)) elements.push({ weight, element })
  }
  if (!elements.length) throw new Error(`no elements parsed for ${name} in ${source}`)
  return { name, json: { fallback: id(fallback), elements } }
}

const files = new Map()
const manifest = { version: "Java 1.16.1", generatedFrom: [], pools: [] }
for (const file of sourceFiles) {
  const full = path.join(sourceRoot, file)
  const source = fs.readFileSync(full, "utf8")
  manifest.generatedFrom.push(`net/minecraft/structure/${file}`)
  for (const body of calls(source, "new StructurePool(")) {
    const pool = parsePool(body, file)
    if (!pool) continue
    const rel = `data/minecraft/worldgen/template_pool/${pool.name}.json`
    files.set(rel, Buffer.from(JSON.stringify(pool.json)))
    manifest.pools.push(pool.name)
  }
}

const starts = [
  ["village/plains/town_centers", 6], ["village/desert/town_centers", 6],
  ["village/savanna/town_centers", 6], ["village/snowy/town_centers", 6],
  ["village/taiga/town_centers", 6], ["pillager_outpost/base_plates", 7],
  ["bastion/units/base", 60], ["bastion/hoglin_stable/origin", 60],
  ["bastion/treasure/starters", 60], ["bastion/bridge/start", 60]
]
for (const [start_pool, size] of starts) {
  const slug = start_pool.replaceAll("/", "_")
  const json = { type: "minecraft:jigsaw", start_pool: id(start_pool), size, max_distance_from_center: 80 }
  files.set(`data/minecraft/worldgen/structure/showcase/${slug}.json`, Buffer.from(JSON.stringify(json)))
}

manifest.pools.sort()
files.set("viewer/worldgen_registry_manifest.json", Buffer.from(JSON.stringify(manifest, null, 2) + "\n"))
fs.mkdirSync(path.dirname(output), { recursive: true })
fs.writeFileSync(output, writeZip(new Map([...files].sort(([a], [b]) => a.localeCompare(b)))))
console.log(`Built ${output}`)
console.log(`${manifest.pools.length} exact 1.16.1 template pools, ${starts.length} start definitions`)
