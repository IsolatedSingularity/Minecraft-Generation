# Minecraft Procedural Generation Analysis
###### Mathematical exploration of Minecraft's world generation algorithms. References include [Minecraft Wiki](https://minecraft.wiki/), [Sportskeeda Wiki](https://wiki.sportskeeda.com/minecraft), and procedural generation works from [Alan Zucconi](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/).

![Ender Dragon Pathfinding](Plots/dragon_pathfinding.gif)

---

## Objective

This repository implements a comprehensive exploration of **Minecraft's procedural generation algorithms** through mathematical analysis and dynamic visualization. The project combines rigorous computational modeling with publication-quality animations to reveal the sophisticated deterministic systems underlying Minecraft's infinite world generation.

*Every seed tells a story. Every block has a purpose. Every death to a baby zombie is statistically inevitable.*

The implementation spans the complete generation pipeline—from 48-bit LCG seeds to the dragon's final breath:

| Domain | Description | Status |
|--------|-------------|--------|
| **Structure Placement** | Salt-based deterministic algorithms for villages, temples, and fortresses | ✅ Animated |
| **Dragon Pathfinding** | Graph-based navigation with probability-weighted state transitions | ✅ 3D Simulated |
| **Stronghold Distribution** | Polar coordinate mathematics across 8 concentric rings | ✅ Visualized |
| **End Dimension Layout** | Gateway positioning and outer island overflow rings | ✅ Mapped |
| **One-Shot Mechanics** | Statistics overflow arrow velocity exploitation | ✅ Simulated |
| **Seed Loading** | Chunk generation with biome noise layer visualization | ✅ Animated |
| **Multi-Structure Generation** | Parallel structure type spawning across regions | ✅ Animated |

**Goal:** Simulate and visualize Minecraft's procedural generation systems with mathematical precision—for speedrunning optimization, educational exploration, and the eternal question: *"Why did my portal spawn over lava?"*

---

## Mathematical Foundations

Minecraft's world generation employs sophisticated **deterministic algorithms** that create seemingly random but entirely predictable patterns. At its core lies Java's Linear Congruential Generator:

$$X_{n+1} = (aX_n + c) \bmod m$$

where $a = 25214903917$ (`0x5DEECE66D`), $c = 11$ (`0xB`), and $m = 2^{48}$.

This single equation is responsible for every creeper spawn position, every diamond vein, and every stronghold placement. *The universe runs on 48 bits.*

### Structure Placement Mathematics

Structure placement utilizes **region-based salt randomization** ensuring deterministic distribution. For standard structures (villages, temples, outposts):

$$S_{\text{region}} = S_{\text{world}} + R_x \cdot 341873128712 + R_z \cdot 132897987541 + \text{salt}$$

Within each region, the structure position uses triangular distribution for natural clustering:

```python
def get_structure_chunk(region_x, region_z, spacing, separation, salt):
    """
    Calculate structure chunk position within a region.
    
    The formula is deterministic—same seed, same world, same village.
    Your speedrun luck was decided 14 years ago.
    """
    region_seed = world_seed + region_x * K1 + region_z * K2 + salt
    rng = JavaLCG(region_seed)
    
    pos_x = rng.next_int(spacing - separation)
    pos_z = rng.next_int(spacing - separation)
    
    return region_x * spacing + pos_x, region_z * spacing + pos_z
```

### Stronghold Ring Distribution

128 strongholds generate in **8 concentric rings** around world spawn, following polar coordinate mathematics with angular jitter:

$$r = r_{\min} + \mathcal{U}(0, r_{\max} - r_{\min})$$

$$\theta = \frac{2\pi \cdot i}{n} + \text{noise}$$

| Ring | Count | Inner Radius | Outer Radius | Notes |
|------|-------|--------------|--------------|-------|
| 1 | 3 | 1,408 | 2,688 | *The speedrunner's prayer* |
| 2 | 6 | 4,480 | 5,760 | |
| 3 | 10 | 7,552 | 8,832 | |
| 4 | 15 | 10,624 | 11,904 | |
| 5 | 21 | 13,696 | 14,976 | |
| 6 | 28 | 16,768 | 18,048 | |
| 7 | 36 | 19,840 | 21,120 | |
| 8 | 9 | 22,912 | 24,192 | *Here be dragons (and crashes)* |

### Dragon State Machine

The Ender Dragon operates on a **probability-weighted finite state machine**. The critical perch probability formula:

$$P(\text{perch}) = \frac{1}{3 + n_{\text{crystals}}}$$

| Crystals Alive | Perch Probability | Implication |
|----------------|-------------------|-------------|
| 10 | 7.7% | *"Why won't it land?!"* |
| 5 | 12.5% | Halfway there |
| 0 | 33.3% | *Time to one-cycle* |

---

## Visualizations

### 1. Ender Dragon Pathfinding

![Dragon Pathfinding Animation](Plots/dragon_pathfinding.gif)

Real-time visualization of the Ender Dragon's behavioral navigation system:

- **Graph-based pathfinding** across 25+ nodes in outer, inner, and center rings
- **7 behavioral states**: HOLDING → STRAFING → APPROACH → LANDING → PERCHING → TAKEOFF → CHARGING
- **Dynamic crystal destruction** affecting perch probability calculation
- **Fireball spawning** during strafing attack phases
- **State machine diagram** with real-time active state highlighting

The dragon doesn't chase you—it follows a deterministic graph. Your death was mathematically predetermined.

### 2. Structure Placement Algorithm

![Structure Placement Animation](Plots/structure_placement.gif)

Step-by-step visualization of Minecraft's village generation algorithm:

- **Spiral region scan** expanding outward from world spawn
- **Real-time seed calculation** displaying 64-bit hex values
- **Biome suitability evaluation** across Plains, Desert, Savanna, and Taiga
- **Spawn probability visualization** with formula breakdown
- **Live statistics**: regions scanned, structures found, spawn rate percentage

Watch the algorithm decide, in real-time, why your spawn has no village within 2,000 blocks.

### 3. Stronghold Ring Distribution

![Stronghold Distribution](Plots/stronghold_rings.png)

Publication-quality polar coordinate visualization of all 128 strongholds:

- **8 concentric rings** with distinct color coding per ring
- **Angular distribution analysis** showing near-even spacing with noise
- **Distance scale overlay** for speedrunning route optimization
- **Ring statistics panel** with counts and radius bounds

*Ring 1 is always between 1,408-2,688 blocks. This is not a suggestion—it's mathematics.*

### 4. End Dimension Overview

![End Dimension Layout](Plots/end_dimension_overview.png)

Comprehensive map of the End dimension's structure:

- **Central island** with exit portal, 10 obsidian pillars, and crystal positions
- **20 End Gateway positions** calculated via: $x = \lfloor 96 \cdot \cos(\pi k / 10) \rfloor$, $z = \lfloor 96 \cdot \sin(\pi k / 10) \rfloor$
- **Outer island rings** including the overflow void gaps at 370,720 and 524,288 blocks
- **Mathematical formulas** for gateway and island generation

The End isn't infinite—it's bounded by arithmetic overflow. The void gaps are a *feature*.

### 5. One-Shot Dragon Kill Simulation

![One-Shot Technique](Plots/oneshot_dragon.gif)

3D animated simulation of the legendary MCSR one-shot technique:

- **Statistics overflow exploitation** building arrow velocity beyond normal limits
- **Dragon perching mechanics** in full 3D with wing animation
- **Arrow trajectory visualization** from player position to dragon head
- **Velocity graph** showing the precise moment of integer overflow
- **Damage calculation breakdown** explaining instant kill mechanics

*When the statistics counter wraps around, physics becomes optional.*

### 6. Seed Loading Animation

![Seed Loading](Plots/seed_loading.gif)

Visualization of world generation as chunks load from seed:

- **Chunk-by-chunk generation** expanding from spawn in spiral pattern
- **Biome noise layers** showing temperature, humidity, and continentalness
- **Structure spawning** as regions complete evaluation
- **LCG state visualization** tracking random number generation
- **Progress statistics** with timing and chunk counts

Every Minecraft world begins with a single 64-bit number. Watch that number become a universe.

### 7. Multi-Structure Generation

![Multi-Structure Generation](Plots/multi_structure_generation.gif)

Animated comparison of parallel structure type generation:

- **Simultaneous village, outpost, and temple spawning** across shared regions
- **Salt differentiation visualization** showing how the same region produces different results
- **Biome overlay** demonstrating structure-biome requirements
- **Collision detection** between mutually exclusive structures
- **Regional grid system** with structure density analysis

Different salt, different fate. Same seed, different structures.

### 8. Structure Analysis

![Structure Analysis](Plots/structure_analysis.png)

Comprehensive multi-panel analysis dashboard:

- **Biome parameter maps**: Temperature, Humidity, Continentalness noise fields
- **Structure distribution plots** for villages, strongholds, and multi-type overlays  
- **Distance histograms** from spawn
- **Generation formulas** with exact Minecraft mathematics
- **Seed entropy analysis** and statistics summary

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/IsolatedSingularity/Minecraft-Generation.git
cd Minecraft-Generation

# Create virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install numpy matplotlib networkx scipy pillow seaborn

# Generate visualizations
python Code/dragon_pathfinding.py      # Dragon behavioral animation
python Code/structure_placement.py     # Village placement animation
python Code/stronghold_distribution.py # Stronghold rings plot
python Code/end_dimension_overview.py  # End dimension map
python Code/oneshot_dragon.py          # One-shot 3D simulation
python Code/seed_loading.py            # Chunk generation animation
python Code/multi_structure_generation.py  # Multi-structure animation
python Code/minecraftStructureAnalysis.py  # Comprehensive analysis
```

### Project Structure

```
Minecraft-Generation/
├── README.md                           # You are here
├── requirements.txt
├── Code/
│   ├── core/                           # Shared utilities
│   │   ├── __init__.py
│   │   ├── constants.py                # Minecraft generation constants
│   │   ├── lcg.py                      # Java LCG implementation
│   │   └── noise.py                    # Perlin noise
│   ├── dragon_pathfinding.py           # Dragon behavioral visualization
│   ├── structure_placement.py          # Village algorithm animation
│   ├── stronghold_distribution.py      # Ring distribution plot
│   ├── end_dimension_overview.py       # End dimension map
│   ├── oneshot_dragon.py               # One-shot 3D simulation
│   ├── seed_loading.py                 # Chunk loading animation
│   ├── multi_structure_generation.py   # Multi-structure animation
│   ├── minecraftStructureAnalysis.py   # Comprehensive analyzer
│   └── [legacy modules]                # Original implementations
└── Plots/                              # Generated visualizations
    ├── dragon_pathfinding.gif
    ├── structure_placement.gif
    ├── stronghold_rings.png
    ├── end_dimension_overview.png
    ├── oneshot_dragon.gif
    ├── seed_loading.gif
    ├── multi_structure_generation.gif
    └── structure_analysis.png
```

---

## Technical Specifications

| Component | Specification |
|-----------|---------------|
| **Python** | 3.10+ |
| **Animation FPS** | 15-20 |
| **Static DPI** | 300 |
| **Animation DPI** | 100-200 |
| **LCG Precision** | 48-bit (Java-accurate) |

### Dependencies

```
numpy>=1.21.0
matplotlib>=3.5.0
networkx>=2.6
scipy>=1.7.0
pillow>=8.0.0
seaborn>=0.11.0
```

---

## References

1. **[Minecraft Wiki](https://minecraft.wiki/)** — Definitive game mechanics documentation
2. **[Alan Zucconi](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/)** — Procedural generation deep dives
3. **Java Random Implementation** — OpenJDK LCG source code
4. **MCSR Community** — Speedrunning optimization research

---

*Author: Jeffrey Morais*

---

> [!TIP]
> For speedrunning: First ring strongholds are at 1,408-2,688 blocks. Triangulate with 2 eye throws minimum. The math doesn't lie—your throws do.

> [!NOTE]  
> All visualizations use authentic Minecraft algorithms verified against game decompilation. Seeds produce results identical to Java Edition.

> [!CAUTION]
> Side effects of understanding these algorithms include: inability to enjoy "random" generation, compulsive seed analysis, and explaining to non-players why 48-bit integers matter.

---

<details>
<summary>🥚 The Scroll of Forbidden Knowledge</summary>

```
The ancient texts speak of seeds most cursed:

Seed 164311266871034 - Where villages fear to spawn
Seed 1785852800490 - The stronghold that wasn't  
Seed 27594263 - Portal room behind bedrock

Some seeds are best left unplanted.

Also, did you know Herobrine's removal was never actually implemented?
The changelog lies. He watches through the perlin noise.
Always 3 chunks behind. Always listening for footsteps.

The generation is deterministic.
Your survival is not.

- Translated from the Ender Tongue, circa 2011
```

</details>
