Deeply scrape all info of how these sites work and the code behind it so i can reproduce it exactly locally (for testing, personal project nothing weird):

https://ezseed.net/seed-map/#seed=6090144754301628691&x=2513&z=419&s=20.38739745134612&ed=java-26.2

https://www.minecraftmaps.com/tools/structure-viewer

(even better)

https://structure-viewer.ewanhowell.com/

like locally i have all the minecraft files so i should be able to just extract the info and render the buildings like that. or i want a seed viewer thats accurate like the websites. what i have locally are shitty approximations and not the real deal. you may ask me clarifying questions if you have. we need to get this info locally to my agents so maybe a detailed md or multiple ones witht ehf ull code idk if you can scrape it and i can download

Goal: give your agents enough source-backed material and runnable scaffolding to build a genuinely local, version-locked Minecraft Java 1.16.1 seed + structure viewer, without confusing templates, predicted structure positions, generated assemblies, and actual final world blocks.

I did the deep pass and packaged the useful results here:




The biggest discoveries

Ewan Howell's viewer is by far the most valuable one. Its complete current implementation is public on GitHub. It is Vue/Vite/Three.js, loads Minecraft jars and packs, parses NBT and actual world saves, renders Minecraft block models, handles jigsaw assembly, implements several procedural structure families, streams .mca chunks, and even contains Java extraction tooling that executes Minecraft's own generation classes against a capturing fake world to turn hardcoded structures into NBT.

I mapped the important implementation paths for your agents, especially src/nbt.js, world.js, jigsaw.js, transforms.js, usePacks.js, useStructures.js, useSession.js, generators/*, and the extraction tools. Its current generator registry includes strongholds, fortresses, mineshafts, monuments, End cities, mansions, igloos, pyramids, dungeons and more.

There is one very important trap: Ewan's viewer ?seed= / Re-roll seed is not the Minecraft world seed. Its viewer assembly uses a custom 32-bit Mulberry32 PRNG, and its URL session seed is only eight hexadecimal digits. Therefore, feeding your 64-bit Java seed into that assembly system does not reconstruct the naturally generated 1.16.1 structure instance.

Ewan's project also normally has online pieces that I removed from the local design. It downloads Mojang jars through the author's CORS proxy and imports block-model-renderer from jsDelivr at runtime. The bootstrap I made instead lets Vite bundle the installed renderer and lets you provide your own 1.16.1.jar directly through the documented loadPacks({base: bytes}) API.

I found another 1.16.1 contamination issue: Ewan's current app automatically adds its modern builtin.zip and features.zip after the selected base jar. My bootstrap disables those in pure 1.16.1 mode until you generate 1.16.1-specific equivalents, rather than letting modern data silently enter your corpus.

ezseed is fundamentally the other half of the system. Its own documentation says Java biome/structure calculations use Cubiomes compiled to WebAssembly, calculations happen locally, the map uses OpenLayers, and concurrent web workers process map data. It also admits that terrain-dependent structure results can be candidates rather than guaranteed final placements.

That is particularly useful because Cubiomes itself explicitly distinguishes:

MC_1_16_1

from

MC_1_16_5 / the generic MC_1_16 alias.

So your seed engine should pin MC_1_16_1, then use Cubiomes APIs such as getStructurePos(), isViableStructurePos(), StrongholdIter, getMineshafts(), biome generation, and slime chunk logic. Cubiomes documents the structure-attempt versus biome-viability distinction explicitly.

There's also a useful shortcut I found: Cubiomes Viewer already exists as a local desktop seed/map application, and its source explicitly contains MC_1_16_1 handling. I would keep that around as an independent oracle while your agents implement your custom UI.

MinecraftMaps is much less interesting technically. Their own explanation says they parse the real template's size, palette, and blocks, skip air-like blocks, then render each block as a cube. Their texture mode uses real textures, but stairs, fences, etc. remain full cubes. Their assembled End Cities are generated separately by running the assembly algorithm. So it is a good NBT/template correctness reference, but it is exactly the kind of simplified geometry you are trying to get away from.
What I think you should actually build

The robust architecture is:

                    WORLD SEED
                        |
                        v
             Cubiomes MC_1_16_1
          biome + structure positions
                        |
                  [2D seed map]
                        |
                  click marker
                        |
          +-------------+-------------+
          |                           |
          v                           v
   1.16.1 jar data            exact 1.16.1 generator
 templates/models/textures       OR actual game
          |                           |
          v                           v
   Ewan-style renderer         generated .mca chunks
          |                           |
          +-------------+-------------+
                        |
                        v
                exact 3D inspection

The critical distinction is that the system should label results as template_exact, position_viable, assembly_exact, or block_exact. Cubiomes finding a bastion at (x,z) is not the same thing as knowing precisely which bastion pieces Minecraft 1.16.1 chose. And a reconstructed piece graph is not necessarily the same as the final blocks after processors and terrain interaction.

For true final correctness, actual chunks generated by Minecraft 1.16.1 are the gold standard. Conveniently, Ewan's world.js already contains a serious Anvil/world reader, including region location tables, chunk decompression, dimensions, NBT parsing and lazy region caching.
What is in the download

The bundle contains the detailed reverse engineering notes plus runnable scaffolding:

    AGENT_HANDOFF.md, a fairly detailed implementation sequence and acceptance criteria

    docs/00_FINDINGS_MATRIX.md, what each site actually does

    docs/01_EWAN_STRUCTURE_VIEWER.md, deep source map and 1.16.1 adaptation notes

    docs/02_EZSEED_CUBIOMES.md, Cubiomes/WASM seed-map architecture

    docs/03_MINECRAFTMAPS_VIEWER.md, NBT/template methodology

    docs/04_MC_1_16_1_EXACT_ARCHITECTURE.md, the full exactness architecture and differential-testing strategy

    docs/05_CUBIOMES_VIEWER_READYMADE.md, ready-made local validation reference

    scripts/bootstrap-ewan-local.ps1, clones Ewan's viewer, removes the runtime CDN dependency, disables modern supplemental data in pure 1.16.1 mode, adds the local loader, builds, and runs it

    scripts/local-loader.html, selects your own client jar/pack files and sends the bytes directly into Ewan's documented embed API

    scripts/extract-mc1161-assets.ps1, deterministic jar extraction plus SHA-256 manifest

    cubiomes/mc1161_seed_probe.c, an actual starter program using MC_1_16_1 that emits JSONL structure coordinates and viability

    CHECKSUMS.sha256 and a source inventory

One legal/source note matters: the Ewan repository is public, but GitHub currently reports no explicit repository license. I therefore did not stuff his complete source into your download. The bootstrap clones the public repository locally and applies our adapter instead. I likewise did not try to republish MinecraftMaps or ezseed's proprietary frontend bundles. For ezseed, the valuable calculation layer is already available cleanly through upstream Cubiomes.

For your existing 1.16.1 generation library, I would hand AGENT_HANDOFF.md + the zip to an agent and have it start with Task 1 through Task 4. That gets you an exact local template renderer and exact 1.16.1 seed-position engine first. Then I would attack Bastion → Fortress → Stronghold natural assembly with differential tests against real generated worlds, since those are the highest-value pieces for the speedrunning use case.

Nice! Can you modify that bundle minimally and create a new one based on:

i) it shouuld be clear to the agents that this will be an addition to the repo and shouldnt take over the full functionality and in essence we are trying to literally recearaet or rip exactly what is being done there (dont worry ill put my own spin on it) but now locally

ii) i had gemini do the same research but way elss detailed however might be useful:

To reproduce these tools perfectly locally for your Jenova agent framework, you need to bypass the web layer entirely. These websites don't run Minecraft instances; they rely on heavily optimized, reverse-engineered C libraries compiled to WebAssembly (for seed mapping) and client-side WebGL (for structure rendering).

By running the core libraries directly via Python or C, you achieve 100% accuracy, zero telemetry, and eliminate the web-stack bloat. This allows your agents to parse seed generation math and 3D structure topologies natively and offline.

Here is the exact architectural breakdown and the lean code to implement both capabilities locally.
Part 1: Seed Mapping (Replicating ezseed.net)

How the site works:

ezseed.net and ChunkBase do not use Mojang's code. They use an open-source C library called Cubiomes (written by Cubitect). Cubiomes perfectly replicates Minecraft's java.util.Random PRNG and biome/structure generation algorithms. The website just compiles this C code into WebAssembly (WASM) to run in the browser using OpenLayers for the 2D UI.

Your Local / Agent Solution:

You will use the native cubiomes C library. Since you focus on lean, performant systems, you can compile it natively and wrap it for Python.

    Get the Core Engine:

    Clone the repository:

Bash

git clone https://github.com/Cubitect/cubiomes.git
cd cubiomes
make # Compiles into libcubiomes.a / libcubiomes.so

    Hooking it into Jenova (Python):

    Instead of writing complex ctypes wrappers manually, you can use the community Python wrapper cubiomes-python or just write a tiny C-script that takes a seed and coordinates, and outputs JSON for your Python agents to ingest via standard I/O.

Note: If you want a visual tool for yourself without writing the UI, use Cubiomes Viewer (a native C++/Qt offline desktop app). It is exactly what ezseed is, but running natively on your OS with zero web overhead.
Part 2: Structure Rendering (Replicating Ewan Howell)

How the site works:

Minecraft saves structures (like villages or strongholds) as .nbt (Named Binary Tag) files. These are just GZIP-compressed binary dictionaries containing a 3D grid of blocks. Ewan's site uses Javascript (pako for gzip extraction) to parse the NBT tree, and then uses Three.js (WebGL) to draw the blocks by mapping the NBT block names (e.g., minecraft:oak_log) to flat texture images.

Your Local / Agent Solution:

Since you have the local Minecraft files, you have everything you need. You will extract the NBT files from the game .jar and parse them directly into memory using Python.

    Extracting the Raw Data (Local OS Level):

    Minecraft hides its structures and textures inside the version .jar. You can treat it as a zip file.

Bash

# Extract all vanilla structures (NBT files) to a local folder
7z x %appdata%\.minecraft\versions\1.21\1.21.jar data/minecraft/structures/

# Extract all block textures (if you want to render them later)
7z x %appdata%\.minecraft\versions\1.21\1.21.jar assets/minecraft/textures/block/

    Parsing the NBT in Python (Agent Logic):

    To let Jenova "see" the structure natively, you need to parse the NBT into a spatial tensor or 3D dictionary. Use the lean nbtlib library (pip install nbtlib).

Here is the exact Python script to parse an .nbt file into a lean, queryable 3D grid:

Python

import nbtlib

def load_structure_to_grid(nbt_filepath: str):
    """
    Reads a Minecraft .nbt structure file and returns its dimensions
    and a spatial dictionary of block states.
    """
    # nbtlib automatically handles the GZIP decompression
    nbt_data = nbtlib.load(nbt_filepath)
    root = nbt_data.root

    # 1. Get the bounding box dimensions [X, Y, Z]
    dims = [int(c) for c in root['size']]

    # 2. Extract the palette (the unique blocks used in this structure)
    # We strip "minecraft:" to keep the data lean
    palette = [block['Name'].replace('minecraft:', '') for block in root['palette']]

    # 3. Map the blocks to a 3D coordinate dictionary
    # Format: {(x, y, z): "block_name"}
    world_grid = {}

    for block in root['blocks']:
        # NBT stores positions as arrays of 3 ints
        x, y, z = (int(c) for c in block['pos'])

        # 'state' is the index pointing to the palette list
        state_idx = int(block['state'])
        block_name = palette[state_idx]

        # Skip air blocks to save memory in the agent's context
        if block_name != "air" and block_name != "cave_air":
            world_grid[(x, y, z)] = block_name

    return dims, world_grid

# --- Example Usage ---
if __name__ == "__main__":
    # Path to an extracted NBT file (e.g., an igloo or ship)
    target_file = "./data/minecraft/structures/igloo/top.nbt"

    dimensions, grid = load_structure_to_grid(target_file)
    print(f"Structure Size: X:{dimensions[0]} Y:{dimensions[1]} Z:{dimensions[2]}")

    # Your agent can now query exact coordinates
    # e.g., "What is at 2, 1, 3?"
    print(f"Block at (2, 1, 3): {grid.get((2, 1, 3), 'empty')}")

3. Rendering the NBT Locally (Optional)

If you want to visually render the structures the way Ewan Howell's site does, but locally:

    Lean CLI approach: You can write a Python script that iterates through your world_grid dictionary and outputs an .obj or .ply 3D model file. You can then open it in a lean local viewer like Windows 3D Viewer or Blender.

    Native UI approach: Use pyvista or ursina in Python. You would map the strings (e.g., oak_log) in your parsed world_grid to the texture .png files you extracted from the .jar earlier, rendering them as voxel cubes.

By utilizing cubiomes for seed mapping and nbtlib for structure parsing, your Jenova framework gains the exact same capabilities as those websites, completely untethered from external APIs or browsers.

are you confident the agetnts can 1 for 1 recreate whats done on the side and i can load it like that? would it be useful to put you on deep research mode to deeply serach and scrape the sites? any blind spots or caveats that can be fixed?

Goal: keep the original plan intact, but make the agents treat this as a bounded addition to your existing repo and aim for reference-site parity rather than a simplified rewrite.

Done. I left the original bundle untouched and made a minimally revised v2:




The ZIP passed an integrity test. SHA-256 is d2f1b483de2145ff3f101cc6163b4cad83e7a4391c69bbbe25c6902076965597.
What I changed

I made three surgical changes. First, the handoff now says explicitly that this is an additive subsystem, not permission to refactor your generation library into a new viewer project. Agents have a new Task 0 requiring them to inspect the repo, reuse the existing canonical corpus/indexes, keep the viewer removable, and leave unrelated commands/directories/functionality alone.

Second, I made the goal much more explicit:

    reference parity first

Meaning: before adding their own abstractions, they should reproduce the reference tools' meaningful behavior, asset interpretation, seed semantics, rendering behavior and interactions as literally as practical locally.

So an agent should not decide:

    "I'll make a simpler Python voxel viewer, close enough."

when the point is to reproduce what Ewan is doing.

Third, I incorporated Gemini's useful ideas while explicitly correcting the misleading bits.
Am I confident they can reproduce this 1:1?

For Ewan's structure viewer itself: yes, quite strongly.

This is the fortunate case because its current implementation is public source rather than a black box. Its own documentation describes loading the vanilla client jar plus packs, real structure files, greedy meshing and texture atlases, face culling, animated water/lava/fire, interactive doors/gates, jigsaw assembly, procedural structure generators, world-save loading and streaming, .glb/.obj exports, and even walk mode.

So your agents aren't trying to infer the renderer from screenshots. They can inspect the implementation.

There are, however, two different meanings of "1:1":

1. Reproduce Ewan's application locally: high confidence.

2. Given Minecraft world seed 6090144754301628691, reproduce the exact naturally generated 1.16.1 Bastion/Stronghold/etc. down to blocks: achievable, but that is an additional worldgen problem and is not automatically solved by cloning Ewan.

That distinction is the main thing I want your agents never to lose track of.

Ewan's viewer session generator currently uses its own deterministic 32-bit PRNG machinery for viewer assemblies rather than treating that value as Minecraft's 64-bit world seed. Its current session machinery therefore cannot just be passed your Java seed and assumed to represent the naturally generated structure.

That is why the bundle separates:

template_exact
position_exact / position_viable
assembly_exact
final block_exact

Gemini found something useful, but oversimplified the hard half

Its Cubiomes recommendation is good.

Ezseed itself currently says its Java biome/structure calculations use Cubiomes compiled to WebAssembly, calculations run client-side, and its map rendering uses OpenLayers.

Running Cubiomes natively instead of compiling it to WASM is probably better for your internal agent-facing API. You can have:

agent
  ↓ JSON
mc1161-worldgen.exe
  ↓
native Cubiomes

rather than forcing agents through a browser.

The existing bundle already starts that path with:

cubiomes/mc1161_seed_probe.c

And yes, wrapping it with Python eventually is fine.

But Gemini's description of Ewan as basically:

    GZIP NBT → Three.js → textured cubes

is substantially too simple.

For example, its statement about pako doesn't match the current public implementation. Ewan's current nbt.js detects GZIP and uses the browser's DecompressionStream("gzip").

More importantly, the rendering system isn't just:

oak_stairs.nbt
→ oak_stairs.png
→ cube

It is resolving Minecraft asset semantics.

That includes blockstates, model definitions, textures and much more. This is precisely why recreating Ewan is so much better than writing our own Ursina/PyVista block cubes.
nbtlib is still useful

I absolutely would keep Gemini's Python representation, but as an agent/query layer, not the visual implementation.

For example:

grid[(x, y, z)] = {
    "Name": "minecraft:oak_stairs",
    "Properties": {
        "facing": "west",
        "half": "bottom",
        "shape": "outer_left",
        "waterlogged": "false"
    },
    "nbt": ...
}

not merely:

grid[(x, y, z)] = "oak_stairs"

The latter throws away information you need to reproduce the actual geometry.

That Python representation will be excellent for questions like:

    Which blocks are inside this Bastion template?

    Find every chest/jigsaw/gold block.

    What occupies (x,y,z)?

    Compare these two structure variants.

But Ewan remains the renderer target.
Biggest remaining blind spots

There are five things I'd consider the actual danger zones now:

    1.16.1 RNG call ordering for natural structures. Bastion, Fortress, Stronghold, ruined portals, villages, etc. need the correct generator entry point and Java RNG consumption order. A mathematically similar modern implementation can still pick different pieces.

    Terrain-conditioned generation. Cubiomes is extremely useful, but even ezseed's own current documentation says some structures are represented as candidates when terrain is not completely simulated. So "Cubiomes says structure here" and "these exact final blocks exist here" must remain separate claims.

    Hardcoded 1.16.1 generation. Having all NBT/templates from the jar does not mean you possess all the generation rules as data. Some behavior lives in code. This is where your mapped/decompiled 1.16.1 generation library becomes enormously useful.

    Rendering fidelity. Stairs, fences, panes, waterlogging, multipart states, model parent inheritance, UV rotation, tinting, transparency, biome colors, block entities, animated materials, etc. are where simplistic local viewers turn into the "shitty approximation" problem again.

    Modern contamination. Running Ewan against a 1.16.1 jar without controlling its modern supplemental assets can accidentally create a Frankenstein 1.16.1/modern viewer. The existing handoff already tells agents to prevent this.

MinecraftMaps is useful primarily as a validation target for the simple template layer. Its own current description says it parses size, palette, and blocks, skips air-like blocks, and turns the rest into colored cubes. That's much easier to reproduce, but it's also explicitly less sophisticated than where we want to end up.
Would putting me in Deep Research mode help?

Yes, but I would change what we spend the research budget on.

I don't think another giant generic search of "how does Ewan work?" is the best use anymore. We already found the jackpot: his source code.

The highest-value deep investigation now would be:

Minecraft Java 1.16.1 exact natural structure generation
        ↓
Bastion
Fortress
Stronghold
Village
Ruined Portal
        ↓
exact RNG initialization
exact RNG consumption
piece graph construction
rotation/mirroring
processors
terrain adjustment
chunk generation stage
        ↓
compare to actual generated 1.16.1 worlds

That is the piece that gets you from:

    "We have a beautiful structure viewer."

to:

    "Click this Bastion marker on this 1.16.1 seed and I can show you the Bastion that is actually there."

For literal ezseed UI parity, a separate deeper pass over its public frontend assets, workers, caching/tile protocol, URL-state behavior and network behavior would also be useful. The live site confirms that it has configurable concurrent web workers, URL hash synchronization, overlays, structure filters, nearest searches, stronghold rings, slime chunks and multiple other layers.

But I'd prioritize Minecraft correctness over pixel-perfect ezseed UI cloning.

So my confidence now is roughly:

Ewan viewer locally: ~95%+ that your agents can get extremely close because we have the implementation.

MinecraftMaps viewer locally: ~99%, because it's conceptually much simpler.

Ezseed-like 1.16.1 seed map: ~90–95% for the functionality that matters because Cubiomes gives us the core calculation engine.

Exact world-seed → exact naturally generated 3D structure without using Minecraft as an oracle: this is the part that still needs serious version-specific engineering.

The revised bundle now makes that hierarchy much harder for an agent to misunderstand.


