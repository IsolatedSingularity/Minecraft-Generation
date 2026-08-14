import { runDesertPyramid1161, runJungleTemple1161 } from "../src/generators/legacy1161.js"
import { runSwampHut } from "../src/generators/swamphut.js"
import { runEndSpikesActive } from "../src/generators/endspikes.js"

const nameCount = (structure, name) => structure.blocks.filter(block => structure.palette[block.state]?.Name === name).length

const desert = (await runDesertPyramid1161()).structure
if (desert.size[0] !== 21 || desert.size[2] !== 21 || nameCount(desert, "minecraft:chest") !== 4 || nameCount(desert, "minecraft:tnt") !== 9)
  throw new Error("Desert pyramid source-port invariant failed")

const jungle = (await runJungleTemple1161(null, { seed: 0x11610001 })).structure
if (jungle.size[0] !== 12 || jungle.size[2] !== 15 || nameCount(jungle, "minecraft:chest") !== 2 || nameCount(jungle, "minecraft:dispenser") !== 2)
  throw new Error("Jungle temple source-port invariant failed")

const hut = (await runSwampHut()).structure
if (hut.size[0] !== 7 || hut.size[2] !== 9 || hut.entities.length !== 2)
  throw new Error("Swamp hut source-port invariant failed")

const arena = (await runEndSpikesActive(null, { seed: 0x11610001 })).structure
if (arena.entities.length !== 10 || nameCount(arena, "minecraft:end_portal") === 0)
  throw new Error("End arena source-port invariant failed")

console.log(`PASS: Java 1.16.1 code structures (${desert.blocks.length}/${jungle.blocks.length}/${hut.blocks.length}/${arena.blocks.length} blocks)`)
