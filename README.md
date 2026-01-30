# Minecraft Procedural Generation Analysis
###### Mathematical exploration of Minecraft's world generation algorithms. References include [Minecraft Wiki](https://minecraft.wiki/), [Sportskeeda Wiki](https://wiki.sportskeeda.com/minecraft), and procedural generation works from [Alan Zucconi](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/).

![Ender Dragon Pathfinding AI](Plots/dragon_pathfinding.gif)

---

## Objective

This repository implements a comprehensive exploration of **Minecraft's procedural generation algorithms** through mathematical analysis and dynamic visualization. The project combines rigorous computational modeling with publication-quality animations to reveal the sophisticated deterministic systems underlying Minecraft's infinite world generation.

*Every seed tells a story. Every block has a purpose.*

The implementation focuses on three core areas:

| Domain | Description |
|--------|-------------|
| **Structure Placement** | Salt-based deterministic algorithms for villages, temples, and fortresses |
| **Dragon Pathfinding AI** | Graph-based navigation with probability-weighted state transitions |
| **Stronghold Distribution** | Polar coordinate mathematics across 8 concentric rings |

**Goal:** Simulate and visualize Minecraft's procedural generation systems with mathematical precision—for speedrunning optimization, educational exploration, and algorithmic understanding.

---

## Mathematical Foundations

Minecraft's world generation employs sophisticated **deterministic algorithms** that create seemingly random but entirely predictable patterns. The system uses Linear Congruential Generators (LCGs) following Java's `Random` implementation:

$$X_{n+1} = (aX_n + c) \bmod m$$

where $a = \texttt{0x5DEECE66D}$, $c = \texttt{0xB}$, and $m = 2^{48}$.

### Structure Placement Mathematics

Structure placement utilizes **salt-based randomization** for deterministic distribution across regions:

$$S_{\text{region}} = \left( S_{\text{world}} + x^2 \cdot 4987142 + x \cdot 5947611 + z^2 \cdot 4392871 + z \cdot 389711 + \text{salt} \right) \bmod 2^{32}$$

This elegant formula ensures identical structure placement across all players sharing a world seed, while the salt value differentiates between structure types.

```python
def generate_region_seed(world_seed, region_x, region_z, salt):
    """Calculate region seed using Minecraft's exact algorithm."""
    return (world_seed + 
            region_x * region_x * 4987142 + 
            region_x * 5947611 + 
            region_z * region_z * 4392871 + 
            region_z * 389711 + 
            salt) & 0xFFFFFFFF
```

### Stronghold Ring Distribution

Strongholds generate in **8 concentric rings** around world spawn, following polar coordinate mathematics:

$$r_{\text{stronghold}} = r_{\min} + \mathcal{U}(0, r_{\max} - r_{\min})$$

$$\theta_{\text{stronghold}} = \frac{2\pi \cdot i}{n} + \mathcal{N}\left(0, \frac{\pi}{4n}\right)$$

where $n$ is the stronghold count per ring, $i$ is the index, and angular jitter prevents perfect alignment.

| Ring | Count | Min Radius | Max Radius |
|------|-------|------------|------------|
| 1 | 3 | 1,280 | 2,816 |
| 2 | 6 | 4,352 | 5,888 |
| 3 | 10 | 7,424 | 8,960 |
| 4 | 15 | 10,496 | 12,032 |
| 5 | 21 | 13,568 | 15,104 |
| 6 | 28 | 16,640 | 18,176 |
| 7 | 36 | 19,712 | 21,248 |
| 8 | 9 | 22,784 | 24,320 |

*128 strongholds. Infinite possibilities.*

### Dragon AI State Machine

The Ender Dragon's AI employs **probability-weighted state transitions** with crystal-dependent perch probability:

$$P(\text{perch}) = \frac{1}{3 + n_{\text{crystals}}}$$

With all 10 crystals alive, the dragon has only a 7.7% chance of landing. Destroy the crystals, and that probability rises to 33%.

---

## Visualizations

### 1. Ender Dragon Pathfinding

![Dragon Pathfinding Animation](Plots/dragon_pathfinding.gif)

Real-time visualization of the Ender Dragon's AI behavior featuring:

- **Graph-based navigation** with 25+ pathfinding nodes across outer, inner, and center rings
- **7 behavioral states**: HOLDING, STRAFING, APPROACH, LANDING, PERCHING, TAKEOFF, CHARGING
- **Crystal destruction mechanics** affecting perch probability in real-time
- **Fireball spawning** during strafing attacks
- **State machine diagram** with active state highlighting

```python
class EnderDragonAI:
    def get_perch_probability(self):
        """Calculate landing probability based on crystal count."""
        return 1.0 / (3.0 + self.crystals_alive)
    
    def choose_next_state(self):
        """Probability-weighted state transition."""
        roll = np.random.random()
        if roll < self.get_perch_probability():
            return 'APPROACH'  # Begin landing sequence
        # ... state machine logic
```

### 2. Structure Placement Algorithm

![Structure Placement Animation](Plots/structure_placement.gif)

Step-by-step visualization of Minecraft's village generation:

- **Spiral scan** from spawn outward through region grid
- **Real-time seed calculation** with hex display
- **Biome suitability evaluation** (Plains, Desert, Savanna, Taiga)
- **Spawn probability visualization** showing why structures appear where they do
- **Statistics tracking**: regions scanned, villages found, spawn rate

The algorithm reveals how a simple mathematical formula creates complex, believable world patterns.

### 3. Stronghold Ring Distribution

![Stronghold Distribution](Plots/stronghold_rings.png)

Publication-quality visualization of all 128 strongholds:

- **8 concentric rings** with distinct color coding
- **Polar coordinate placement** with angular distribution
- **Distance scale** for speedrunning route planning
- **Statistical summary** of ring parameters

*The first ring is always 1,280-2,816 blocks from spawn. Always.*

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/IsolatedSingularity/Minecraft-Generation.git
cd Minecraft-Generation

# Install dependencies
pip install -r requirements.txt

# Generate dragon pathfinding animation
python Code/dragon_pathfinding.py

# Generate structure placement animation
python Code/structure_placement.py

# Generate stronghold distribution plot
python Code/stronghold_distribution.py
```

### Project Structure

```
Minecraft-Generation/
├── README.md
├── requirements.txt
├── Code/
│   ├── core/                    # Centralized utilities
│   │   ├── __init__.py
│   │   ├── constants.py         # Minecraft generation constants
│   │   ├── lcg.py               # Linear Congruential Generator
│   │   └── noise.py             # Perlin noise implementation
│   ├── dragon_pathfinding.py    # Ender Dragon AI visualization
│   ├── structure_placement.py   # Village placement animation
│   ├── stronghold_distribution.py
│   ├── minecraftAnimations.py   # Legacy animation systems
│   ├── minecraftExtendedAnimations.py
│   ├── minecraftGeneration.py   # Original implementations
│   ├── minecraftMathematicalAnalysis.py
│   ├── minecraftStructureAnalysis.py
│   └── README.md
└── Plots/
    ├── dragon_pathfinding.gif
    ├── structure_placement.gif
    ├── stronghold_rings.png
    └── ...
```

---

## Applications

### Speedrunning Optimization

Mathematical analysis enables precise **stronghold triangulation** and route planning:

- First ring strongholds average ~2,048 blocks from spawn
- Optimal eye of ender throw positions can be calculated
- Seed viability assessment for sub-20-minute runs

### Educational Exploration

The visualizations reveal how **deterministic randomness** creates infinite variety:

- Same seed → Same world → Same structures
- Salt values differentiate structure types
- Biome noise fields create natural-looking transitions

### Algorithmic Research

Implementation provides a foundation for studying:

- Linear Congruential Generators in practice
- Spatial distribution algorithms
- Graph-based pathfinding with probability

---

## Technical Specifications

### Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Mathematical operations |
| `matplotlib` | Publication-quality visualization |
| `networkx` | Graph-based pathfinding analysis |
| `scipy` | Statistical analysis |
| `pillow` | GIF animation encoding |
| `seaborn` | Enhanced color palettes |

### Output Quality

| Output Type | Resolution | Format |
|-------------|------------|--------|
| Animations | 200 DPI | GIF (15-20 FPS) |
| Static Plots | 300 DPI | PNG |
| Print Quality | 300+ DPI | PNG/PDF |

---

## References

This implementation draws from:

1. **Minecraft Wiki** - Definitive documentation of game mechanics
2. **Alan Zucconi** - Procedural generation deep dives
3. **Speedrunning Community** - Practical optimization strategies
4. **Java Random Source** - LCG implementation details

---

*Author: Jeffrey Morais*

---

> [!TIP]
> For speedrunning applications, focus on the first stronghold ring (1,280-2,816 blocks). The 3 strongholds are evenly distributed at ~120° angles with slight randomization.

> [!NOTE]
> All visualizations use authentic Minecraft algorithms. World seeds produce identical results to the actual game.
