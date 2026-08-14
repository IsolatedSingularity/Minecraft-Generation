# Interactive viewer accuracy audit (Minecraft Java 1.16.1)

## Scope

This audit covers `Viewer/seed-map.html` and `Viewer/local-loader.html`. It does
not change or certify the older static structure-generation plots in `Code/`.

## Seed Atlas

- The C wrapper calls `setupGenerator(..., MC_1_16_1, ...)` and applies the
  entered signed Java `long` as its 64-bit world seed.
- Biome pixels, point inspection, random-spread candidates, viability checks,
  Nether fortress/bastion splitting, and the 128 stronghold positions all come
  from the same vendored Cubiomes build.
- Colors now come directly from Cubiomes `initBiomeColors`; there is no separate
  hand-tuned JavaScript palette. In particular, ocean is `#000070` and deep
  ocean is `#000030`.
- At close zoom the End calls Cubiomes `mapEndSurfaceHeight`; zero-height pixels
  remain void. The Nether uses the exact 1.16.1 cave-generator octave stack and
  slide parameters, then reports the highest navigable floor below Y=122.
- The WASM smoke fixture checks independent known seed-42 biome and structure
  coordinates, repeat determinism, non-flat bounded Nether relief, central End
  land plus the void gap, all 128 strongholds, and the dark-water palette.

### Boundary

The biome/density map is version-locked and deterministic. Overworld relief
remains Cubiomes `mapApproxHeight`. Nether relief represents a density-derived
navigable cave floor, not exposed terrain with carvers and surface replacement.
Candidate markers do not prove every final block survives later generation.

## 3D structures

- Raw templates and rendering assets come from a path-preserving subset of the
  exact 1.16.1 client extraction. It contains all 866 canonical client
  templates and the blockstates, models, and textures needed to render them.
  The complete client JAR is not committed.
- The clean, neutral dark-grid orbit presentation follows the Ewan Howell
  viewer snapshot used as the interaction baseline; the bright green grid from
  the discarded reference is not used.
- Mojang's client JAR lacks standalone NBT resources for the hard-coded Nether
  fortress and stronghold generators. The repository therefore includes only
  13 fortress pieces, 15 stronghold pieces, and 15 stronghold random-block
  masks from the local reference snapshot.
- Fortress piece weights/caps and stronghold weights/caps, conditional pieces,
  depth 50, and radius 112 were compared with the mapped Java 1.16.1
  `NetherFortressGenerator` and `StrongholdGenerator` sources. The selected NBT
  palettes use block IDs available in 1.16.1.
- The assembly registry is generated from the mapped 1.16.1 Java pool classes:
  128 template pools and 10 declared starts cover five village types, the
  pillager outpost, and all four bastion types. List-pool elements remain
  colocated instead of silently discarding companion templates.
- Desert pyramid, jungle temple, swamp hut, and End portal/spike structures are
  direct source ports because those structures have no canonical client NBT.
  Smoke tests check dimensions and characteristic chests, TNT, dispensers,
  entities, portal blocks, and ten End crystals.

### Boundary

The generated assemblies reproduce checked piece selection, pool weight, and
connection rules for a deterministic interactive showcase seed. Processor
weathering and the exact natural world-seed RNG stream are not yet reproduced.
They are not coupled to a Seed Atlas coordinate or terrain realization. That
coupling belongs in a later, separately validated structure-generation phase;
the existing plots remain untouched here.
