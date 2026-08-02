# Open Problems: Minecraft-Generation

This document catalogs open problems, unresolved challenges, and improvement areas for the **Minecraft-Generation** procedural world generation and mathematical analysis codebase.

---

## 1. Algorithmic & Implementation Problems

- **Multi-Octave 3D Noise Performance Optimization**
  - **Problem**: Accelerating 3D multi-octave Perlin noise and density function evaluations (`core/noise.py`) for terrain and cave generation using Numba JIT compilation or C-extension bindings.
  - **Context**: Pure Python loops for spline-based Continentalness, Erosion, and Weirdness (1.18+ world generation) become computationally prohibitive when rendering high-resolution volumetric slices.
- **Multi-Structure Spline Biome Integration**
  - **Problem**: Extending structure placement algorithms (`structure_placement.py`, `multi_structure_generation.py`) beyond Java 1.16.1 region-salt candidate windows to include full multi-noise biome parameter space validation.

---

## 2. Bugs & Unresolved Issues

- **Matplotlib Headless Animation Memory Leaks**
  - **Problem**: Sequential execution of animation renderers in `render_all.py` (`minecraftAnimations.py`, `minecraftExtendedAnimations.py`) can leak Matplotlib figure memory when generating large GIF files (`Plots/`), requiring explicit figure disposal and garbage collection.

---

## 3. Theoretical & Scientific Problems

- **Stronghold Ring Angular Anisotropy Under 48-Bit LCG**
  - **Problem**: Quantifying statistical deviations from uniform angular distribution across the 128 strongholds in 8 concentric polar rings caused by the low-order bit correlations of Java's 48-bit Linear Congruential Generator.
- **Ender Dragon Markov Chain Stationary Distribution**
  - **Problem**: Deriving the exact stationary distribution of the 25-node Ender Dragon pathfinding graph as a function of remaining obsidian end crystals, where $P(\text{perch}) = 1 / (3 + \text{crystals\_alive})$.

---

## 4. Code Maintenance & Refactoring Opportunities

- **Duplicate Script Cleanup (`minecraftStructureAnalysis_backup.py`)**
  - **Opportunity**: `Code/minecraftStructureAnalysis_backup.py` (17 KB) is tracked alongside `Code/minecraftStructureAnalysis.py` (9 KB). Comparing, merging unique methods, and removing the backup file will prevent codebase drift.
