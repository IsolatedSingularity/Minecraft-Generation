# Minecraft Java 1.16.1 surface texture subset

This directory contains only the block textures actively used by the generated
terrain figures. They were copied from the version-locked local 1.16.1 client
JAR at `assets/minecraft/textures/block/`; the original nested path is retained
below this directory.

The private source corpus, JARs, unused assets, and extraction tooling remain
outside Git. See the local gitignored `ASSET_PERMISSION.md` and the repository
guidance in `AGENTS.md` before changing this subset.

The renderer currently uses sixteen textures: grass, sand, stone, water,
snow, gravel, mycelium, podzol, red sand, netherrack, crimson and warped
nylium, soul sand, basalt, lava, and End stone.

`terrain_samples/` contains four small generated PNG datasets for the fixed
README seeds. Red stores the raw biome ID and blue stores the cache-format
version. In Overworld samples, green stores the `WORLD_SURFACE_WG` height. In
the Nether sample, green marks negative-density lava cavities at Y=31. The
Overworld files were exported from the original 1.16.1 server JAR with
`Audits/VanillaTerrainCache.java`; the Nether file was exported from the
source-checked Python port with `Audits/export_nether_terrain_cache.py`.
These are generated numerical samples, not copied source code, chunk saves,
or a hidden hand-painted terrain layer.
