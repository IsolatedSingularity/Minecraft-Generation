import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { readZip } from "../tools/builtin/zip.js"

const path = resolve("../Assets/minecraft_1_16_1/viewer/client_structure_assets.zip")
const archive = readZip(readFileSync(path))
const names = [...archive.keys()]
const count = prefix => names.filter(name => name.startsWith(prefix)).length

const inventory = {
  blockstates: count("assets/minecraft/blockstates/"),
  models: count("assets/minecraft/models/"),
  textures: count("assets/minecraft/textures/"),
  structures: count("data/minecraft/structures/")
}

const expected = { blockstates: 764, models: 2485, textures: 2084, structures: 866 }
for (const [kind, total] of Object.entries(expected)) {
  if (inventory[kind] !== total)
    throw new Error(`Unexpected ${kind} inventory: expected ${total}, got ${inventory[kind]}`)
}

for (const required of [
  "assets/minecraft/blockstates/stone_bricks.json",
  "assets/minecraft/models/block/stone_bricks.json",
  "assets/minecraft/textures/block/stone_bricks.png",
  "data/minecraft/structures/village/plains/houses/plains_small_house_1.nbt",
  "data/minecraft/structures/end_city/ship.nbt"
]) {
  if (!archive.has(required)) throw new Error(`Missing required client asset: ${required}`)
}

console.log(`PASS: ${archive.size} version-locked Java 1.16.1 client viewer assets`)
