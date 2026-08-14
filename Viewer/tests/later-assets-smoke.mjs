import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { readZip, unzipEntry } from "../tools/builtin/zip.js"

const archive = readZip(readFileSync(resolve("../Assets/minecraft_later_versions/viewer/structure_assets.zip")))
const names = [...archive.keys()]
const templateCount = names.filter(name => /^data\/minecraft\/structure\/(ancient_city|trial_chambers)\/.+\.nbt$/.test(name)).length

if (templateCount !== 228) throw new Error(`Unexpected later-version template count: ${templateCount}`)
for (const required of [
  "data/minecraft/worldgen/structure/ancient_city.json",
  "data/minecraft/worldgen/structure/trial_chambers.json",
  "data/minecraft/worldgen/template_pool/ancient_city/city_center.json",
  "data/minecraft/worldgen/template_pool/trial_chambers/chamber/end.json",
  "assets/minecraft/blockstates/sculk.json",
  "assets/minecraft/blockstates/trial_spawner.json"
]) {
  if (!archive.has(required)) throw new Error(`Missing later-version viewer asset: ${required}`)
}

const manifest = JSON.parse(unzipEntry(archive.get("viewer/later_structure_manifest.json")).toString("utf8"))
if (manifest.sourceVersion !== "Minecraft Java 1.21" || manifest.structureTemplates !== 228)
  throw new Error("Later-version viewer manifest is inconsistent")

console.log(`PASS: ${templateCount} selected Minecraft Java 1.21 structure templates`)
