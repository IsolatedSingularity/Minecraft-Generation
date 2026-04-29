# Minecraft-Generation Repository Analysis

Repository: `IsolatedSingularity/Minecraft-Generation`
Branch: `main`
Language: Python 100%
Analysis date: 2026-04-17

## File Structure

```
Minecraft-Generation/
  .gitignore
  README.md
  requirements.txt
  Code/
    README.md
    core/
      __init__.py
      constants.py
      lcg.py
      noise.py
    dragon_pathfinding.py
    end_dimension_overview.py
    minecraftAnimations.py
    minecraftExtendedAnimations.py
    minecraftGeneration.py
    minecraftMathematicalAnalysis.py
    minecraftStructureAnalysis.py
    minecraftStructureAnalysis_backup.py
    multi_structure_generation.py
    oneshot_dragon.py
    seed_loading.py
    stronghold_distribution.py
    structure_placement.py
  Plots/
    (17 output images/GIFs)
```

## What Is Implemented

### Core Library (`Code/core/`)
1. **MinecraftLCG**: Exact Java `Random` LCG (`0x5DEECE66DL`, 48-bit modulus) with `next_bits`, `next_int` (rejection sampling), `next_float`, `next_double`
2. **generate_region_seed()**: Minecraft's quadratic hash formula for structure placement
3. **noise.py**: Simplified Perlin-like noise via multi-frequency sine waves (NOT true Perlin)
4. **constants.py**: Visual palette, village/fortress/monument spacings + salts, 8 stronghold rings (128 total), 7 dragon states, dragon arena dimensions, biome classification colors

### Visualization Scripts (14 files)
| File | What It Does |
|------|-------------|
| `dragon_pathfinding.py` | EnderDragonAI with 7-state machine, 25-node graph pathfinding, crystal-dependent perch probability `P = 1/(3+crystals)`, fireball mechanics. Animated GIF |
| `end_dimension_overview.py` | 20 gateway positions (radius 96), outer island rings with overflow void gaps (370,720 / 524,288). Multi-panel figure |
| `stronghold_distribution.py` | 128 strongholds across 8 rings, polar coordinate placement with jitter. Publication PNG |
| `structure_placement.py` | Village placement with spiral-order region scanning, exact seed formula display, animated GIF |
| `seed_loading.py` | World gen pipeline animation with proper Perlin (permutation table + fade + lerp + grad), BiomeGenerator (6 parameters), chunk spiral loading |
| `multi_structure_generation.py` | 8 structure types with proper salts, K1/K2 constants. Animated GIF |
| `minecraftMathematicalAnalysis.py` | LCG analysis, stronghold triangulation, nearest-neighbor routes, distance probability, seed viability |
| `minecraftStructureAnalysis.py` | Full JavaLCG, proper PerlinNoise (gradient vectors, fade, fBm), 8 structures with authentic salts, triangular distribution |
| `oneshot_dragon.py` | 3D one-shot exploit: normal arrow physics (gravity + drag) vs integer overflow exploit (2^31-1). 3D animation |
| `minecraftGeneration.py` | Original monolithic script: village scatter, dragon pathfinding graph, strongholds. Saves to ~/Downloads |
| `minecraftAnimations.py` | Legacy animator: structure placement + dragon pathfinding animations |
| `minecraftExtendedAnimations.py` | 6-panel comprehensive analysis + 4-panel speedrunning analysis |
| `minecraftStructureAnalysis_backup.py` | Backup of original structure analysis (should be removed/gitignored) |

## What Is Missing

### High Priority (Core Features)
1. **Nether dimension visualization**: Fortress generation defined in constants but never visualized standalone. No Nether biome distribution, no bastion remnant placement, no lava sea terrain
2. **Overworld terrain heightmap**: Noise fields used for biomes but actual 3D terrain cross-section not generated
3. **Cave generation**: No carver cave simulation (random walks, ravines). Especially important for 1.16 which uses ONLY carver caves
4. **Ore distribution modeling**: No vein placement visualization or statistical analysis

### Medium Priority (Completeness)
5. **Nether biome distribution**: 5 biomes with known percentages (36.3% Wastes, 22.2% Crimson, etc.) but no visualization
6. **End island noise generation**: Outer islands use specific noise formula but it's not visualized beyond the overview
7. **Structure exclusion zones**: Structures have spacing/separation but mutual exclusion not modeled
8. **Biome-specific structure filtering**: Village variants by biome, temples restricted to specific biomes

### Low Priority (Polish)
9. **Unified CLI entry point**: No `__main__.py` or argument parser
10. **Test suite**: pytest in requirements but no test files
11. **Code deduplication**: LCG implemented 5+ times independently across files; `core/` module exists but most scripts don't import from it
12. **Backup file**: `minecraftStructureAnalysis_backup.py` should be removed
13. **Output path inconsistency**: `minecraftGeneration.py` saves to `~/Downloads`, others save to `Plots/`

## Tech Stack

- **numpy**: Core numerical computation
- **matplotlib** (+animation, patches, mplot3d): All visualization
- **networkx**: Dragon pathfinding graph
- **scipy** (spatial.distance, stats, optimize): Distance analysis, probability
- **seaborn**: Extended animations
- **pillow**: GIF writing via PillowWriter

## Coding Conventions

- **camelCase** for variables (matches user preference): `villageXPositions`, `outerNodeAngles`, `backgroundColor`
- **PascalCase** for classes: `MinecraftLCG`, `EnderDragonAI`, `StructurePlacementSimulator`
- **Dark theme**: `plt.style.use('dark_background')` with `#0D1117` GitHub-dark palette
- **#%%** cell markers in original script (Spyder/Jupyter style)
- Docstrings present in newer files, sparse in legacy code
- Korean comments in `minecraftGeneration.py`
- French prose in `README.md`

## Config State

| Item | Status |
|------|--------|
| `.gitignore` | Present. Ignores `__pycache__`, `.venv`, `.vscode/`, `Jenova/`, `audit-*.md` |
| `requirements.txt` | numpy, scipy, matplotlib, seaborn, pillow, networkx. pytest/black commented out |
| `.github/` | Not present |
| `Jenova/` junction | Not present (gitignore rule exists) |
| Git LFS | Configured |

## Implementation Gap Priority for Agent

Based on what exists vs what's missing, the agent should focus on:

1. **Nether dimension** (biome distribution, fortress/bastion placement, lava sea terrain, ancient debris distribution) -- this is the 1.16 headline feature
2. **Overworld terrain cross-section** (heightmap from noise, cave carver simulation)
3. **Ore distribution analysis** (per-ore Y-range visualization, vein statistics)
4. **Code consolidation** (migrate standalone LCG/noise implementations to `core/`)
5. **Ender Dragon AI refinement** (add missing phases 2-4, 6-7, expand to full 11-phase model matching wiki data)
