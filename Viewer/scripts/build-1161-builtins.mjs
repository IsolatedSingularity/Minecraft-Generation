import { existsSync, readFileSync, readdirSync, writeFileSync, mkdirSync } from "node:fs"
import { dirname, join, relative, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { writeZip } from "../tools/builtin/zip.js"

const viewerRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const sourceRoot = resolve(
  viewerRoot,
  "../.oracle-bin/ewan-structure-viewer/bundled/builtin"
)
const output = join(viewerRoot, "src/assets/minecraft-1.16.1-builtins.zip")
const selected = [
  "data/minecraft/structure/builtin/nether_fortress",
  "data/minecraft/structure/builtin/stronghold"
]

if (!existsSync(sourceRoot)) {
  throw new Error(`Reference built-ins not found: ${sourceRoot}`)
}

const files = new Map()
for (const directory of selected) {
  const absolute = join(sourceRoot, directory)
  for (const entry of readdirSync(absolute, { withFileTypes: true })) {
    if (!entry.isFile() || !/\.(nbt|rand\.json)$/.test(entry.name)) continue
    const path = join(absolute, entry.name)
    files.set(relative(sourceRoot, path).replaceAll("\\", "/"), readFileSync(path))
  }
}

mkdirSync(dirname(output), { recursive: true })
writeFileSync(output, writeZip(files))
console.log(`Wrote ${relative(viewerRoot, output)} with ${files.size} source-checked files`)
