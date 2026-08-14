import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { readZip, unzipEntry } from "../tools/builtin/zip.js"

const here = path.dirname(fileURLToPath(import.meta.url))
const zipPath = path.resolve(here, "../../Assets/minecraft_1_16_1/viewer/worldgen_registry.zip")
const files = readZip(fs.readFileSync(zipPath))
const readJson = name => JSON.parse(unzipEntry(files.get(name)).toString("utf8"))

for (const required of [
  "village/plains/town_centers", "village/desert/houses", "pillager_outpost/base_plates",
  "bastion/bridge/start", "bastion/units/base", "bastion/hoglin_stable/origin", "bastion/treasure/starters"
]) {
  const name = `data/minecraft/worldgen/template_pool/${required}.json`
  if (!files.has(name)) throw new Error(`missing pool: ${required}`)
  const pool = readJson(name)
  if (!pool.elements?.length) throw new Error(`empty pool: ${required}`)
}

const bridge = readJson("data/minecraft/worldgen/template_pool/bastion/bridge/start.json")
if (bridge.elements[0]?.element?.location !== "minecraft:bastion/bridge/starting_pieces/entrance_base")
  throw new Error("bridge starter does not match 1.16.1 source")
const plains = readJson("data/minecraft/worldgen/template_pool/village/plains/town_centers.json")
if (plains.elements.reduce((sum, entry) => sum + entry.weight, 0) !== 204)
  throw new Error("plains town-center weights do not match 1.16.1 source")

console.log(`PASS: ${files.size} Java 1.16.1 worldgen registry entries`)
