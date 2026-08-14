import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { readZip } from "../tools/builtin/zip.js"

const archive = readZip(readFileSync(resolve("src/assets/minecraft-1.16.1-builtins.zip")))
const names = [...archive.keys()]
const fortress = names.filter(name => name.includes("/nether_fortress/") && name.endsWith(".nbt"))
const stronghold = names.filter(name => name.includes("/stronghold/") && name.endsWith(".nbt"))
const masks = names.filter(name => name.includes("/stronghold/") && name.endsWith(".rand.json"))

if (archive.size !== 43 || fortress.length !== 13 || stronghold.length !== 15 || masks.length !== 15)
  throw new Error(`Unexpected built-in inventory: ${archive.size} total, ${fortress.length}/${stronghold.length}/${masks.length}`)

for (const required of [
  "data/minecraft/structure/builtin/nether_fortress/bridge_crossing.nbt",
  "data/minecraft/structure/builtin/stronghold/stairs_down.nbt",
  "data/minecraft/structure/builtin/stronghold/portal_room.rand.json"
]) {
  if (!archive.has(required)) throw new Error(`Missing required built-in: ${required}`)
}

console.log("PASS: 43 source-checked Java 1.16.1 fortress/stronghold files")
