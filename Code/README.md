# Code Directory - Minecraft Procedural Generation Analysis

This directory contains the complete Python implementation for analyzing and visualizing Minecraft's procedural generation algorithms. The codebase provides mathematical precision in simulating authentic Minecraft world generation mechanics, pathfinding systems, and structure placement algorithms.

## Module Architecture

### Core Library (`core/`)

The `core/` module provides centralized utilities for all generation algorithms:

| Module | Purpose |
|--------|---------|
| `constants.py` | Minecraft generation constants, colors, ring definitions |
| `lcg.py` | Linear Congruential Generator (Java Random implementation) |
| `noise.py` | Perlin noise generation for terrain simulation |
| `structures.py` | Exact Java 1.16.1 candidate-stage structure placement |
| `end_generation.py` | End geometry, simplex fields, and 32-bit overflow rings |
| `minecraft_visuals.py` | Illustrative pixel-art terrain backdrops for exact overlays |

```python
from core import MinecraftLCG, generate_region_seed, simple_perlin_noise
from core.constants import STRONGHOLD_RINGS, BIOME_COLORS
```

### Primary Visualizations

#### `dragon_pathfinding.py`
**Enhanced Ender Dragon AI Visualization**

Real-time simulation of dragon behavior with:

- 24 pathfinding nodes across 60, 40, and 20-block rings
- 7 behavioral states with probability-weighted transitions
- Crystal destruction mechanics affecting perch probability
- Source-shaped central-island projection with exact spike footprints
- Graph-based navigation with weighted edges

```python
class EnderDragonAI:
    def get_perch_probability(self):
        """P(perch) = 1 / (3 + crystals_alive)"""
        return 1.0 / (3.0 + self.crystals_alive)
```

**Outputs**: `Plots/dragon_pathfinding_hero.gif`, detail clips, and `Plots/dragon_trajectory_ensemble.gif`

#### `end_dimension_overview.py`
**Three-Panel End Dimension Overview**

The left panel maps genuine signed 32-bit overflow rings across a 2.2-million-block view, the upper-right panel shows central fight geometry, and the lower-right panel magnifies the first distant terrain and void band.

**Output**: `Plots/end_dimension_overview.png`

#### structure_placement.py
**Java 1.16.1 Village Candidate Animation**

The generator shows one exact candidate attempt per 32 x 32 chunk region:

- Java-compatible region seed with village salt 10387312
- Two Java Random nextInt(24) offsets
- Explicit 24 x 24 chunk candidate window
- Explicit eight-chunk excluded margins and exact candidate trace

**Output**: Plots/structure_placement.gif

#### `seed_loading.py`
**Java 1.16.1 Chunk Status Dependency Wave**

Shows the exact thirteen-status order from `EMPTY` through `FULL` as an explicitly modeled dependency wave over an illustrative terrain backdrop.

**Output**: `Plots/seed_loading.gif`

#### `multi_structure_generation.py`
**Java 1.16.1 Nether Candidate Layers**

Shows the shared 27 x 27 fortress and bastion grid, its exact 2/5 versus 3/5 type roll, and the independent 25 x 25 ruined-portal grid.

**Output**: `Plots/multi_structure_generation.gif`

#### stronghold_distribution.py
**Java 1.16.1 Stronghold Candidate Rings**

The generator follows the pre-1.19.3 Java ring iterator shared with the static structure dashboard:

- 128 seeded candidates across eight rings
- Ring counts 3, 6, 10, 15, 21, 28, 36, 9
- First-ring candidate range 1,408 to 2,688 blocks
- Approximate candidates shown before the 112-block biome search

**Output**: Plots/stronghold_rings.png

### Legacy Animation Systems

#### `minecraftAnimations.py`
**Primary Functions**: Original animation systems for core Minecraft algorithms
- **Structure Placement Animation**: Grid-based village placement
- **Dragon Pathfinding Animation**: Basic AI state transitions

**Key Features**:
- Authentic Linear Congruential Generator implementation
- Salt-based region seed calculation following Java's Random specification
- Real-time pathfinding graph visualization with weighted edges
- Publication-quality animation rendering (150+ DPI)

#### `minecraftExtendedAnimations.py`
**Primary Functions**: Advanced evolutionary animations for comprehensive analysis
- **Comprehensive Analysis Animation**: Six-panel dynamic visualization showing noise field evolution, biome classification, and structure distribution
- **Speedrunning Analysis Animation**: Four-panel optimization strategy visualization including triangulation and route planning

```python
class MinecraftExtendedAnimator:
    def animate_comprehensive_analysis(self, frames=200, interval=100):
        """6-panel analysis: temperature/humidity fields, biomes, structures"""

    def animate_speedrunning_analysis(self, frames=150, interval=120):
        """4-panel speedrunning optimization with triangulation strategy"""
```

**Advanced Capabilities**:
- Multi-layered noise field generation with temporal evolution
- Dynamic biome classification with threshold-based decision trees
- Progressive stronghold ring revelation with mathematical precision
- Speedrunning route optimization with probability analysis

### Analysis Frameworks

#### minecraftStructureAnalysis.py
**Primary Functions**: Static Java 1.16.1 candidate and structure analysis

The four-panel dashboard separates exact seeded candidate placement from illustrative terrain context. It includes:

- Village candidates across exact 32 x 32 chunk regions
- One expanded 24 x 24 candidate window with Java Random offsets
- Shared fortress and bastion candidates beside independent ruined portals
- Complete 576-pair offset distribution and exact 40/60 Nether type split

#### `minecraftMathematicalAnalysis.py`
**Primary Functions**: Mathematical foundation analysis and speedrunning optimization
- **LCG Pattern Analysis**: Linear Congruential Generator behavior visualization
- **Speedrunning Optimization**: Stronghold triangulation, route planning, and seed viability assessment
- **Probabilistic Analysis**: Structure placement probability distributions

```python
class MinecraftMathAnalyzer:
    def visualize_speedrunning_optimization(self):
        """4-panel speedrunning strategy analysis"""

    def calculate_stronghold_triangulation(self):
        """Eye of ender throw triangulation mathematics"""
```

**Mathematical Precision**:
- Bit-perfect Java Random implementation matching Minecraft's RNG
- Polar coordinate stronghold placement with ring constraints
- Probability-weighted pathfinding algorithms
- Distance optimization for competitive speedrunning

### Legacy Foundation

#### `minecraftGeneration.py`
**Primary Functions**: Original foundational analysis implementation
- Core algorithm development and validation
- Reference implementations for comparison
- Historical development documentation

## Technical Specifications

### Dependencies
```python
# Core Libraries
import numpy as np              # Mathematical operations and array handling
import matplotlib.pyplot as plt # Publication-quality visualization
import matplotlib.animation     # Dynamic animation generation
import networkx as nx          # Graph-based pathfinding analysis
import scipy                   # Statistical analysis and optimization

# Specialized Modules
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import cdist
```

### Mathematical Constants
```python
# Java Random LCG Implementation
LCG_MULTIPLIER = 0x5DEECE66D
LCG_ADDEND = 0xB
LCG_MODULUS = 2**48

# Structure Salt Values (Authentic Minecraft)
VILLAGE_SALT = 10387312
FORTRESS_SALT = 30084232

# Stronghold Ring Parameters for Java 1.16.1
STRONGHOLD_RINGS = [
    {'count': 3, 'min_radius': 1408, 'max_radius': 2688},
    {'count': 6, 'min_radius': 4352, 'max_radius': 5888},
    {'count': 10, 'min_radius': 7424, 'max_radius': 8960},
    {'count': 15, 'min_radius': 10496, 'max_radius': 12032},
    {'count': 21, 'min_radius': 13568, 'max_radius': 15104},
    {'count': 28, 'min_radius': 16640, 'max_radius': 18176},
    {'count': 36, 'min_radius': 19712, 'max_radius': 21248},
    {'count': 9, 'min_radius': 22784, 'max_radius': 24320},
]

```

### Performance Characteristics

| Module | Complexity | Memory Usage | Render Time |
|--------|------------|--------------|-------------|
| `minecraftAnimations.py` | O(n²) regions | 2-4 GB | 30-60 seconds |
| `minecraftExtendedAnimations.py` | O(n³) temporal | 4-8 GB | 90-180 seconds |
| `minecraftStructureAnalysis.py` | O(n²) spatial | 1-2 GB | 15-30 seconds |
| `minecraftMathematicalAnalysis.py` | O(n log n) | 512 MB - 1 GB | 10-20 seconds |

## Algorithm Implementations

### Linear Congruential Generator
Authentic Java Random implementation ensuring bit-perfect Minecraft compatibility:

```python
def lcg_next(seed):
    """Generate next LCG value using Java's exact algorithm"""
    return (0x5DEECE66D * seed + 0xB) % (2**48)
```

### Structure Placement Algorithm

For Java 1.16.1 village candidates, use the 48-bit Java seed setup and the fixed structure-set window:

worldSeed + regionX * 341873128712 + regionZ * 132897987541 + 10387312
candidateChunkX = regionX * 32 + nextInt(24)
candidateChunkZ = regionZ * 32 + nextInt(24)

This is one candidate attempt per region. Biome viability is a separate check.

### Stronghold Ring Mathematics

Use the shared generator in Code/core/strongholds.py:

- Seed a Java 1.16.1 LCG from the world seed
- Start at 128 chunks with +/-40 chunk radius jitter
- Increase each ring center by 192 chunks
- Advance evenly within each ring
- Rotate each new ring with the next LCG double
- Keep the final eight-ring population at 128 candidates

The final stronghold location still comes from the vanilla biome search.

### Basic Animation Generation
```python
from minecraftAnimations import MinecraftAnimator

# Initialize with seed
animator = MinecraftAnimator(world_seed=42)

# Generate structure placement animation
animator.animate_structure_placement("structure_placement.gif")

# Generate dragon pathfinding animation
animator.animate_dragon_pathfinding("dragon_pathfinding.gif")
```

### Comprehensive Analysis
```python
from minecraftExtendedAnimations import MinecraftExtendedAnimator

# Advanced analysis with extended parameters
extended_animator = MinecraftExtendedAnimator(world_seed=42, world_size=20000)

# Create comprehensive 6-panel analysis
extended_animator.animate_comprehensive_analysis(frames=200)

# Create speedrunning optimization analysis
extended_animator.animate_speedrunning_analysis(frames=150)
```

### Static Visualization
```python
from minecraftStructureAnalysis import MinecraftStructureAnalyzer

# High-resolution static analysis
analyzer = MinecraftStructureAnalyzer(world_seed=42, world_size=20000)

# Generate comprehensive structure visualization
analyzer.visualize_comprehensive_structure_analysis()
```

## Examples Directory

The `Examples/` subdirectory contains reference implementations demonstrating:
- **Code Quality Standards**: Professional documentation and structure patterns
- **Visualization Excellence**: Publication-quality figure generation
- **Mathematical Rigor**: Precise algorithm implementation with validation
- **Educational Value**: Clear explanations and comprehensive analysis

These examples serve as templates for maintaining consistency across the codebase and establishing quality benchmarks for future development.

## Output Specifications

### Animation Formats
- **Primary**: High-quality GIF (150+ DPI)
- **Alternative**: MP4 with H.264 encoding (requires FFmpeg)
- **Frame Rate**: 10-20 FPS optimized for educational viewing
- **Resolution**: 1200x1200+ pixels for publication quality

### Static Image Formats
- **Primary**: PNG with 300+ DPI
- **Color Space**: RGB with scientific color palettes
- **Transparency**: Alpha channel support for overlay applications
- **Compression**: Lossless for mathematical precision

## Validation and Testing

All implementations undergo rigorous validation against:
- **Known Seed Databases**: Cross-reference with speedrunning community findings
- **Minecraft Source Analysis**: Verification against decompiled Java implementations
- **Mathematical Consistency**: Ensuring deterministic reproducibility across runs
- **Performance Benchmarks**: Optimized for large-scale analysis requirements

> [!TIP]
> For optimal performance, run analyses on systems with 8+ GB RAM and multi-core processors. Large-scale animations may require 16+ GB for complex temporal evolution sequences.

> [!NOTE]
> All random number generation follows Java's LCG specification exactly, ensuring compatibility with Minecraft's authentic generation algorithms for research and speedrunning applications.
