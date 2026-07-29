# Code directory

The active Python modules reproduce selected Java Edition 1.16.1 random streams and generate the figures used in the root README. Shared numerical logic lives in `core/`; rendering scripts consume it without duplicating constants.

## Generate and test

From the repository root:

```powershell
py -3 -m pip install -r requirements.txt
py -3 Code/render_all.py
py -3 -m unittest discover -s tests -v
```

`render_all.py` writes static PNGs first and GIFs second. Every active visual uses a fixed presentation seed. Individual functions accept alternate seeds for experiments.

## Active generators

| Script | Output | Scope |
|---|---|---|
| `dragon_pathfinding.py` | `dragon_pathfinding.gif`, three phase clips, `dragon_trajectory_ensemble.gif` | Exact 24-node topology and phase rolls, reduced-order 2D motion |
| `end_dimension_overview.py` | `end_dimension_overview.png` | Seeded End samples, exact central geometry, schematic End City assembly |
| `seed_loading.py` | `seed_loading.gif` | Full 1.16.1 status order, exact population-seed texture, illustrative dependency timing |
| `structure_placement.py` | `structure_placement.gif` | Exact village candidate stage |
| `multi_structure_generation.py` | `multi_structure_generation.gif` | Exact Nether candidate grids and fortress or bastion split |
| `minecraftStructureAnalysis.py` | `structure_analysis.png` | Candidate statistics and triangulation experiment |
| `stronghold_distribution.py` | `stronghold_rings.png` | Seeded ring candidates and noisy bearing intersections |
| `oneshot_dragon.py` | `dragon_trajectory_ensemble.gif` | Compatibility entry point for the ensemble generator |

## Shared core

| Module | Responsibility |
|---|---|
| `core/lcg.py` | Java `Random`, signed 64-bit wrapping, region seeds, population seeds |
| `core/structures.py` | Village, fortress, bastion, and ruined-portal candidate rules |
| `core/strongholds.py` | The 128-candidate, eight-ring iterator |
| `core/dragon.py` | Dragon path nodes, adjacency, graph search, phase rolls, simulations |
| `core/end_generation.py` | End simplex sampling, pillar-seed shuffle, spikes, gateways |
| `core/style.py` | Shared scientific palette and axis styling |
| `core/rendering.py` | Adaptive GIF palette optimization and metadata inspection |
| `core/constants.py` | Version-specific constants shared by all modules |

## Exactness conventions

An `exact` label applies only to the named stage:

- Structure points are exact candidate chunks before biome and start validation.
- Stronghold points are exact seeded ring candidates before the biome relocation search.
- Dragon nodes and decision rolls are source-faithful; continuous top-down curves are visual interpolation.
- End sample qualification follows the seeded simplex branch; point size is not a terrain height or island boundary.
- Chunk status order and seed mixing are exact; animation timing is not wall-clock scheduling.
- The End City figure is a piece-family schematic, not a seeded structure reconstruction.

This separation is intentional. It keeps a clean figure from quietly claiming more than the implemented model can prove.

## Numerical validation

`tests/test_algorithms.py` covers:

- published Java `Random` vectors for `nextInt`, `nextDouble`, and signed `nextLong`;
- structure candidate windows and the 2-in-5 fortress split;
- dragon node radii, graph connectivity, and perch probability;
- stronghold count and ring populations;
- gateway radius, pillar-seed derivation, spike heights, and cage count.

Rendering validation checks dimensions, frame counts, file size, and readable assets separately from numerical invariants.

## Legacy scripts

Older exploratory scripts remain for project history, but they are not called by `render_all.py` and should not be treated as Java 1.16.1 reference implementations. New work should import the shared core instead of copying their formulas.
