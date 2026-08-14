# Findings matrix

| Target | Public code status | What it actually gives you | Use for exact 1.16.1 |
|---|---|---|---|
| Ewan Structure Viewer | Full current GitHub repo is public; no explicit repository license visible at research snapshot | Real jar/pack asset loading, NBT/world parsing, block-model rendering, jigsaw/procedural assembly, hardcoded-piece extraction | **Primary 3D/reference architecture**, but replace viewer RNG/version assumptions and disable modern supplemental bundles |
| ezseed | Site behavior/credits are public; no matching public source repo was found in this research pass | Cubiomes-in-WASM seed calculations, worker-based map UI, OpenLayers rendering, site-specific finders | **Behavioral reference**. Rebuild on upstream Cubiomes rather than copying its site JS |
| MinecraftMaps Structure Viewer | Site explains its method; no source repo identified; site says all rights reserved | NBT size/palette/blocks parser, cube/template viewer, texture/map-color display, preassembled End City examples | **Parser/UI reference only**, not natural seed-exact assembly |
| Cubiomes | MIT source | Java biome/structure seed calculations, explicit `MC_1_16_1`, strongholds/mineshafts/etc. | **Primary seed-position engine** |
| Cubiomes Viewer | GPLv3 source | Ready-made local GUI around Cubiomes | **Validation baseline** for your custom seed map |

## Important distinction

A site can be visually impressive while still solving only one of these layers:

1. `template_exact`: bytes saved in a vanilla `.nbt` template;
2. `position_viable`: seed/region/biome math predicts a structure start;
3. `assembly_exact`: exact piece graph and rotations for that world seed/version;
4. `block_exact`: final blocks after processors, terrain interaction and chunk generation.

Your existing "shitty approximations" are most likely crossing those boundaries without labeling them. The architecture in this bundle keeps them separate and promotes a result only after differential validation.
