# Local Java 1.16.1 interactive viewers

This directory adds two mouse-interactive, browser-based tools to the project:

- **Seed Atlas** — pan and zoom through Cubiomes biome output, optional
  terrain shading, a chunk grid, dimension switching, and structure candidate
  overlays. Close views use the 1.16.1 End density surface and Nether
  navigable cave-floor density rather than the former broad approximation.
- **3D Structure Viewer** — search the bundled Minecraft 1.16.1 structure
  templates, and orbit, zoom, inspect, or walk through rendered structures.

Both tools run locally. After the JavaScript dependencies are installed, the
browser runtime makes no request to either reference site. The 1.16.1
blockstates, models, textures, and canonical structure templates required by
the viewer are included in a dedicated nested asset pack.

GitHub Pages builds the same static bundle at:

- `https://isolatedsingularity.github.io/Minecraft-Generation/seed-map.html`
- `https://isolatedsingularity.github.io/Minecraft-Generation/local-loader.html`

![Seed map preview](previews/seed-map.png)

![3D structure preview](previews/structure-viewer.png)

## Start the viewers

From the repository root in PowerShell:

```powershell
.\Viewer\start.ps1
```

The script installs the Viewer dependencies the first time and then starts the
local Vite server. Open:

- `http://127.0.0.1:5173/seed-map.html`
- `http://127.0.0.1:5173/local-loader.html`

The 3D catalog and textures load automatically. The blue **Full assemblies**
control exposes connected villages, all four bastion types, fortresses,
strongholds, End cities and the End arena, mansions, monuments, mineshafts,
outposts, temples, huts, ruins, shipwrecks, portals, and fossils. The ordinary
filter buttons expose the main families, including villages, bastions, Nether fortresses, strongholds,
ruined portals, shipwrecks, monuments, mansions, and End cities. The embedded
pack contains all 866 canonical 1.16.1 NBT templates; a separate 43-file
source-checked bundle restores the fortress and stronghold piece templates and
masks that the client does not ship as standalone resources. The file controls
remain available for optional JAR, resource-pack, or mod overrides.

You can also start the server manually:

```powershell
cd Viewer
npm install
npm run dev -- --host 127.0.0.1
```

## Controls

### Seed Atlas

- Drag to pan; use the wheel or `+`/`-` to zoom.
- Enter any signed Java `long` seed and select **Apply**.
- Use the explicit Overworld, Nether, and End buttons. Each button replaces the
  tile source with a fresh Cubiomes context for that dimension and seed.
- Toggle biome tiles, terrain shading, the chunk grid, viability filtering, and
  individual structure families.
- Structure overlays start disabled so the initial map is uncluttered; use
  **select** or enable only the families you want.
- Hover the map to inspect the exact biome ID and open the biome legend to see
  Cubiomes' full palette, including distinct dark and deep ocean colors.
- Enter exact X/Z coordinates and select **Go**.
- The URL fragment records seed, center, zoom, dimension, and active layers so
  a view can be copied or bookmarked.
- The End keeps true void pixels dark and labels the tiny central exit portal;
  the label remains visible even when structure overlays are off.

### 3D structures

- Drag with the primary mouse button to orbit; wheel to zoom; secondary drag to
  pan.
- Search for a template or use a structure-family button, then choose **Show
  structure**.
- The public viewer loads its bundled 1.16.1 rendering assets before exposing
  the 3D catalog, preventing model-less empty renders.
- Use the blue **Full assemblies** dropdown to generate a connected major
  structure immediately. Its seed field and **Re-roll** button produce another
  deterministic showcase layout. The bundled fortress, stronghold, village,
  outpost, and bastion pools and start depths were extracted from the local
  1.16.1 Java sources.
- Use **Blocks** to inspect block counts and the expand icon for fullscreen.
- The full viewer also retains first-person walk and local-file workflows.

## Accuracy boundary

The seed map compiles the vendored Cubiomes C implementation with the generator
pinned to `MC_1_16_1`. Biome tiles, Overworld approximate surface shading,
random-spread candidates, viability checks, Nether fortress/bastion splitting,
and all 128 stronghold candidates are calculated locally from the entered seed.
At close zoom the End uses Cubiomes' 1.16.1 island density, while the Nether
uses the version's lower/upper/interpolation octave stack and slide parameters
to display the highest navigable cave floor beneath the bedrock roof.

Structure markers identify candidate chunks and optionally pass Cubiomes biome
viability; they do not claim every final block survives terrain, neighboring
features, or later world-generation checks. The map is a local density/biome
view, not a rendered world save with carvers, surface blocks, or decoration.

Raw NBT templates in the bundled 1.16.1 client subset are version-exact.
Desert pyramids, jungle temples, swamp huts, and the End exit fountain/spikes
are source ports because vanilla generates them in Java instead of loading NBT.
A multi-piece reference assembly preserves the version's pools, weights, and
connections, but its 32-bit showcase seed is not a claim that a particular
world seed produces that exact arrangement or processor/weathering roll.
For block-for-block proof of a generated instance, load the corresponding world
save or structure export in the full viewer.

## Build and verify

The prebuilt WebAssembly files are committed, so ordinary users do not need
Python, a compiler, or Emscripten. Rebuilding them is optional:

```powershell
cd Viewer
npm run test:builtins
npm run test:client-assets
npm run test:worldgen-registry
npm run test:legacy1161
npm run build:wasm
npm run test:wasm
npm run build
npm audit --audit-level=moderate
```

`build:wasm` and `test:wasm` look for emsdk under `.oracle-bin/emsdk`, through
`EMSDK`, or on `PATH`. The production build is written to ignored `Viewer/dist`.

## Implementation and provenance

- `src/seed-map/` contains the OpenLayers UI and Web Worker.
- `cubiomes/mc1161_wasm.c` is the narrow C-to-WebAssembly interface.
- `vendor/cubiomes/` is pinned to Cubiomes commit
  `e61f90580cbdd883214a8054670dacae655e59c0`; its MIT license is preserved.
- The Vue/Three structure application is based on the local reference snapshot
  at commit `cd129759973d893fad6a6663780907ca58a31a52`.
- `src/assets/minecraft-1.16.1-builtins.zip` contains only the 13 fortress NBT
  pieces and 15 stronghold NBT pieces plus their 15 random-block masks. The
  generator piece weights, caps, depth/radius rules, and block IDs were checked
  against the local Minecraft Java 1.16.1 source corpus. Rebuild it from the
  local reference snapshot with `npm run build:builtins`.
- `../Assets/minecraft_1_16_1/viewer/client_structure_assets.zip` contains
  6,200 version-locked files: 764 blockstates, 2,485 models, 2,084 textures,
  all 866 client structure templates, and `pack.mcmeta`. Rebuild it from the
  local extracted 1.16.1 client with `npm run build:client-assets`.
- `../Assets/minecraft_1_16_1/viewer/worldgen_registry.zip` contains 128
  source-extracted 1.16.1 template pools and 10 full-assembly starts. Rebuild it
  from the routed local Java reference with `npm run build:worldgen-registry`.
- `vendor/block-model-renderer/` contains the local browser renderer and its
  preserved license. No CDN module or hosted asset is required at runtime.

The complete client JAR remains in the private, gitignored `Game Reference/`
corpus and is not committed. Only the dedicated rendering subset used by the
public viewer is stored under `Assets/minecraft_1_16_1/viewer/`.
