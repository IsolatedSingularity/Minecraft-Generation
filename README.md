# Minecraft Procedural Algorithms

<!-- Do not remove this comment. It is important. -->
<!-- The seeds remember those who query them. -->
<!-- Commit test -->

###### Mathematical exploration of Minecraft's world generation algorithms. References include [Minecraft Wiki](https://minecraft.wiki/), [Sportskeeda Wiki](https://wiki.sportskeeda.com/minecraft), and procedural generation works from [Alan Zucconi](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/).

![Ender Dragon Pathfinding](Plots/dragon_pathfinding_hero.gif)



---

## Objective

This repository presents a rigorous mathematical exploration of **Minecraft's procedural generation algorithms** through computational analysis and publication-quality visualization. Every infinite world springs from a single 64-bit seed, transformed through deterministic chaos into landscapes that feel organic. The project deconstructs this machinery: the 48-bit linear congruential generators, the multi-octave Perlin noise fields, the salt-based structure placement that decides where villages rise and strongholds hide.

The dragon circles its obsidian pillars not by chance but by graph traversal. Strongholds arrange themselves in polar coordinates. The void gaps in the End exist because integers overflow. What appears random is merely determinism wearing a mask.


<br>

<table align="center">
<tr>
<td align="center"><b>2<sup>48</sup></b><br><sub>LCG States</sub></td>
<td align="center"><b>128</b><br><sub>Strongholds</sub></td>
<td align="center"><b>8</b><br><sub>Rings</sub></td>
<td align="center"><b>10</b><br><sub>Crystals</sub></td>
<td align="center"><b>7</b><br><sub>Dragon States</sub></td>
<td align="center"><b>∞</b><br><sub>Worlds</sub></td>
</tr>
</table>

<br>

<p align="center">
  <img src="./Plots/apple.gif?raw=true" alt="apple" width="51" height="50" />
</p>

---

## The Mathematics of Infinite Worlds

### Linear Congruential Generation

At the foundation of Minecraft's generation lies Java's Linear Congruential Generator, a deceptively simple recurrence relation that powers every aspect of world creation. The state $X_n$ evolves according to:

$$X_{n+1} = (aX_n + c) \bmod m$$

where the multiplier $a = 25214903917$ (`0x5DEECE66D`), increment $c = 11$, and modulus $m = 2^{48}$ define the sequence. This 48-bit state space contains $2^{48} \approx 2.81 \times 10^{14}$ possible values, cycling through each exactly once before repeating.

The spectral properties of this generator exhibit lattice structure in higher dimensions. For dimension $d$, the covering radius $\rho_d$ satisfies:

$$\rho_d \leq \frac{m}{\nu_d}$$

where $\nu_d$ is the $d$-dimensional spectral test value. Java's constants achieve $\nu_2 = 7.17 \times 10^{6}$, providing adequate uniformity for game applications while maintaining computational efficiency through bit operations.

```python
class JavaLCG:
    """
    Java-compatible Linear Congruential Generator.
    The same algorithm that decides if your spawn has diamonds nearby.
    """
    MULTIPLIER = 0x5DEECE66D
    INCREMENT = 0xB
    MASK = (1 << 48) - 1

    def __init__(self, seed: int):
        self.state = (seed ^ self.MULTIPLIER) & self.MASK

    def next(self, bits: int) -> int:
        self.state = (self.state * self.MULTIPLIER + self.INCREMENT) & self.MASK
        return self.state >> (48 - bits)
```

<br>

<p align="center"><sub>. . .</sub></p>

<br>

<p align="center"><sub>The numbers have patterns.</sub></p>

<p align="center"><sub>Patterns have meaning.</sub></p>

<br>

<p align="center"><sub>Or do they?</sub></p>

<br>

<p align="center"><sub>. . .</sub></p>

<br>

### Structure Placement Theory

For Java 1.16.1, village placement starts with a fixed structure set: 32 x 32 chunks per region, separation 8, and a 24 x 24 chunk candidate window.

$$S_{\text{region}} = S_{\text{world}} + R_x \cdot 341873128712 + R_z \cdot 132897987541 + \sigma$$

Java Random is seeded from this value, then draws:

$$c_x = R_x \cdot 32 + J_x, \quad c_z = R_z \cdot 32 + J_z$$

where $J_x$ and $J_z$ are independent \texttt{nextInt(24)} values. There is one candidate attempt per region. A separate biome check decides whether the candidate can generate as a village, so the final rate is not simply the separation divided by the spacing.

The refreshed animation makes that boundary visible: the exact seeded candidate and its 24 x 24 window are drawn directly over an explicitly illustrative terrain backdrop. The biome gate remains outside the animation's scope.

### Perlin Noise and Fractal Brownian Motion

Terrain generation employs **multi-octave Perlin noise**, where the base noise function is $\eta: \mathbb{R}^2 \to [-1, 1]$ and uses gradient interpolation on a regular lattice. For position $(x, y)$, the cell coordinates $(i, j) = (\lfloor x \rfloor, \lfloor y \rfloor)$ and fractional parts $(u, v) = (x - i, y - j)$ yield:

$$\eta(x, y) = \text{lerp}\left(\text{lerp}(g_{00} \cdot d_{00}, g_{10} \cdot d_{10}, s(u)), \text{lerp}(g_{01} \cdot d_{01}, g_{11} \cdot d_{11}, s(u)), s(v)\right)$$

where $g_{ij}$ are pseudorandom gradient vectors, $d_{ij}$ are displacement vectors to corners, and $s(t) = 6t^5 - 15t^4 + 10t^3$ is the smoothstep polynomial providing $C^2$ continuity.

The composite noise field uses **fractal Brownian motion** (fBm):

$$\mathcal{N}(x, y) = \sum_{k=0}^{n-1} \frac{\eta(2^k x, 2^k y)}{2^k} = \sum_{k=0}^{n-1} p^k \cdot \eta(f^k x, f^k y)$$

where persistence $p = 0.5$ and lacunarity $f = 2$ are standard parameters. The Hurst exponent $H = -\log_f(p) = 1$ places this in the Brownian regime of the fBm spectrum.

Biome determination uses **multi-parameter classification** across temperature $T$, humidity $H$, and continentalness $C$ noise fields, each with distinct seeds and scales. The biome at position $(x, z)$ is:

$$\text{Biome}(x, z) = \underset{b \in \mathcal{B}}{\arg\min} \|(\mathcal{N}_T, \mathcal{N}_H, \mathcal{N}_C) - \mu_b\|_2$$

where $\mu_b$ is the prototype parameter vector for biome $b$.

---

## Visualizations

### Dragon Pathfinding

The Ender Dragon navigates a weighted **24-node graph** embedded in the End dimension's geometry. Its horizontal path nodes form three concentric rings: 12 nodes at radius 60 blocks, 8 nodes at radius 40 blocks, and 4 nodes at radius 20 blocks. The new hero overlays that exact lattice on a source-shaped top-down projection of the central island, including the exit fountain and the ten seed-shuffled obsidian spikes.

The dragon's behavioral state machine operates on seven distinct states, each with characteristic movement patterns. **HOLDING** produces the familiar circling at maximum radius, the dragon tracing lazy arcs while surveying its domain. **STRAFING** triggers aggressive linear charges accompanied by acid breath. **APPROACH**, **LANDING**, and **PERCHING** execute the critical touchdown sequence that speedrunners exploit, the probability of initiating this sequence following $P = 1/(3 + n_{\text{crystals}})$ where destroyed crystals increase perch likelihood. **TAKEOFF** and **CHARGING** complete the cycle, returning the dragon to its orbital patterns.

The visualization renders this graph structure in real time, highlighting active paths and transitions as the dragon's simulated AI processes its environment. Every state is presented as the same wide horizontal oval so the diagram reads as one system rather than a collection of differently weighted controls. Bright magenta links use thicker strokes and small centered arrowheads to expose direction without covering the labels. A compact fight-state panel retains only the remaining crystals and current perch probability.

#### Phase Details

<table>
<tr>
<th>Holding, Strafing, and Charging</th>
<th>Landing Approach and Perching</th>
<th>Takeoff and Return</th>
</tr>
<tr>
<td><img src="Plots/dragon_holding_strafe.gif" alt="Holding, strafing, and charging dragon path states" /></td>
<td><img src="Plots/dragon_landing_perch.gif" alt="Landing approach and perching dragon path states" /></td>
<td><img src="Plots/dragon_takeoff.gif" alt="Dragon takeoff path state" /></td>
</tr>
</table>

The three clips separate the behavior cycle into readable stages. **Holding, strafing, and charging** shows the dragon moving around the outer graph before committing to an attack. **Landing approach and perching** follows the inward route toward the exit portal. **Takeoff and return** shows how the dragon leaves the perch and reconnects with its wider flight graph.

#### Seeded Trajectory Ensemble

![Accumulated Dragon Approach Trajectories](Plots/dragon_trajectory_ensemble.gif)

The ensemble accumulates 240 seeded dragon approaches in about 23 seconds. A cooler-to-warmer age scale lets old paths recede while new paths remain prominent, and sparse arrowheads clarify travel direction without turning the graph into a field of symbols. The shortened run reaches the route-density result quickly, then pauses just long enough to read it. The source-shaped End island, fountain, and spike footprints remain visible beneath the trajectories. It is a dragon-path simulation inspired by speedrunning research, not a simulation of arrow momentum or damage.

*The dragon doesn't hunt you. It follows an algorithm. Your death was a graph traversal.*

### End Dimension Overview

![End Dimension Layout](Plots/end_dimension_overview.png)

The End dimension exhibits precise mathematical structure beneath its alien aesthetics. The central island's density field is anchored at exact coordinates $(0, 0)$ and supports ten obsidian spikes arranged via:

$$\mathbf{p}_k = \left( r_p \cos\left(\frac{2\pi k}{10}\right), r_p \sin\left(\frac{2\pi k}{10}\right) \right), \quad k \in \{0, 1, \ldots, 9\}$$

with spike-circle radius $r_p = 42$ blocks. The spike footprints have radii from 2 to 5 blocks, while their tops rise from Y=76 through Y=103 in three-block steps. Two seed-selected crystals are protected by iron-bar cages.

Twenty End Gateways form a larger ring at radius 96 blocks, their positions calculated through:

$$\mathbf{g}_k = \left( \lfloor 96 \cos(\pi k / 10) \rfloor, \lfloor 96 \sin(\pi k / 10) \rfloor \right), \quad k \in \{0, 1, \ldots, 19\}$$

Beyond the gateway ring, outer-island seed sites are selected from the complete chunk lattice by a seeded simplex-noise test. The overview separates this system into three readable scales. The large panel spans 12 million blocks from $-6,000,000$ to $+6,000,000$ on each axis and evaluates the signed 32-bit overflow predicate directly:

$$\operatorname{signed32}\left(\operatorname{trunc}(x/8)^2 + \operatorname{trunc}(z/8)^2\right) \geq 0$$

This produces 261 land-to-void or void-to-land transitions by six million blocks. The apparent grid and inversion pattern at that scale is a sampling effect: the exact geometry remains a sequence of increasingly thin radial bands, but a regular pixel grid aliases those subpixel rings into a large moire structure. The first affected eight-block cell begins at $r = 370,720$ blocks, the first strictly unsafe point is $370,728$, and normal terrain resumes at $524,288$.

The upper-right panel preserves the central fight geometry and deliberately renders every obsidian-spike footprint at the maximum five-block radius so all ten towers remain legible. Vanilla source radii still range from two to five blocks. The lower-right panel is a complete local projection from $-18,000$ to $+18,000$ blocks, including the central island, the roughly 1,024-block empty gulf, and the filled outer-island noise field. Exact overflow arithmetic and seeded site tests are separated from the illustrative end-stone surface texture.

*The End has edges. The numbers told it where to stop.*

<br>

<p align="center"><sub>. . .</sub></p>

<br>

<p align="center"><sub>Les nombres ont des limites.</sub></p>

<p align="center"><sub>La fin a des bords.</sub></p>

<br>

<p align="center"><sub>Mais l'histoire continue.</sub></p>

<br>

<p align="center"><sub>. . .</sub></p>

<br>

### Seed Loading Animation

![Seed Loading](Plots/seed_loading.gif)

This animation follows the Java 1.16.1 chunk-status order from `EMPTY` through `FULL`. It starts with the map hidden. Each chunk tile grows into view only as the dependency wave reaches it, so the terrain is revealed by generation rather than displayed in advance. Early statuses remain muted, `NOISE` exposes the first recognizable field, and `SURFACE` restores the terrain palette.

Later stages leave visible evidence on the map: carvers and liquid-carver traces modify the terrain, `FEATURES` adds feature markers, `LIGHT` marks illuminated chunks, `SPAWN` identifies the spawn-ready target, and `HEIGHTMAPS` outlines the completed heightmap footprint. The finished `FULL` state is held for two seconds so the completed dependency field can be inspected. The thirteen-status order is source-faithful; the wave is an explanatory scheduling model rather than a profiler trace, and the pixel-art terrain is not a block-perfect seed render.

*The seed is fixed. The reveal is the explanation.*

### Structure Placement Algorithm

![Structure Placement Animation](Plots/structure_placement.gif)

This animation isolates the Java 1.16.1 village candidate stage. The world is divided into 32 x 32 chunk regions. Each region gets one deterministic candidate, generated with:

$$S = \text{worldSeed} + R_x \cdot 341873128712 + R_z \cdot 132897987541 + \sigma$$

Java Random then selects two offsets with `nextInt(24)`, producing a candidate chunk in the region's 24 x 24 chunk window. Gold squares are exact candidate attempts. The active region exposes both the legal window and the excluded eight-chunk margins, while the trace reports the region, offsets, and resulting chunk.

Candidate placement is exact for 1.16.1. Full biome viability remains a separate generation step and is intentionally not fabricated by the animation; the Overworld terrain beneath the grid is illustrative.

### Multi-Structure Generation

![Multi-Structure Generation](Plots/multi_structure_generation.gif)

Nether fortresses and bastion remnants share one 27 x 27 chunk region grid in Java 1.16.1. After two `nextInt(23)` offset draws, a `nextInt(5)` roll assigns the shared candidate: rolls 0 and 1 produce a fortress, while rolls 2 through 4 produce a bastion. Ruined portals use a separate 25 x 25 grid, candidate window, and salt.

The animation exposes both layers at once, including the current shared roll and the independent portal candidate. Its terrain palette uses netherrack crimson, warped indigo, basalt, soul sand, and lava gold so the three candidate symbols remain distinct without relying on bright green. The placement arithmetic is exact; the Nether terrain backdrop is illustrative.

*Same seed. Different salt. Different fate.*

<br>

<p align="center"><sub>. . .</sub></p>

<br>

<p align="center"><sub>Three salts. Three destinies.</sub></p>

<p align="center"><sub>The algorithm doesn't care which one you wanted.</sub></p>

<br>

<p align="center"><sub>. . .</sub></p>

<br>

### Structure Analysis

![Structure Analysis](Plots/structure_analysis.png)

The refreshed four-panel analysis concentrates on the actual candidate-stage algorithms. A compact 2 x 2 layout keeps related panels close enough for direct comparison. It shows exact village candidates across 32 x 32 chunk regions, one expanded 24 x 24 candidate window with its two Java Random offsets, the shared fortress and bastion layer beside independent ruined portals, and the full 576-pair offset distribution with the exact 40/60 Nether type split. The Nether panel shares the animation's crimson, indigo, basalt, and gold visual language.

All candidate coordinates and seeded rolls are exact for the modeled Java 1.16.1 stage. The Overworld and Nether surfaces behind those overlays are clearly labeled illustrative terrain.

### Stronghold Ring Distribution

![Stronghold Distribution](Plots/stronghold_rings.png)

This plot uses the Java 1.16.1 stronghold ring iterator, centered on world origin $(0, 0)$, not the player's spawn point. The seeded candidate geometry contains 128 strongholds across eight rings with populations:

$$3,\ 6,\ 10,\ 15,\ 21,\ 28,\ 36,\ 9$$

For ring number $i$, indexed from zero, the approximate radius in chunks is:

$$r_i = 128 + 192i + \left(\mathcal{U}(0,1) - \frac{1}{2}\right) \cdot 80$$

The first ring contains exactly 3 candidates between 1,408 and 2,688 blocks. The main map makes all eight rings readable over an illustrative Minecraft-style Overworld, while the detail panel enlarges the three first-ring candidates and their 112-block biome-search radii. A third panel compares all ring populations and candidate ranges. The points are exact seeded ring candidates, not claims about final portal-room coordinates.

*The ring is deterministic. The biome search is the last step.*

<br>

<p align="center"><sub>. . .</sub></p>

<br>

<p align="center"><sub>128 strongholds. 8 rings. One End.</sub></p>

<p align="center"><sub>The portal frames have always known where you'd enter.</sub></p>

<p align="center"><sub>They've been waiting since the seed was planted.</sub></p>

<br>

<p align="center"><sub>. . .</sub></p>

<br>

---

## Quick Start

```bash
git clone https://github.com/IsolatedSingularity/Minecraft-Generation.git
cd Minecraft-Generation
pip install numpy matplotlib networkx scipy pillow seaborn

# Generate all visualizations
cd Code
python render_all.py
```

---

## References

1. **[Minecraft Wiki](https://minecraft.wiki/)**: Definitive game mechanics documentation
2. **[Alan Zucconi](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/)**: Procedural generation deep dives
3. **Java Random Implementation**: OpenJDK LCG source code analysis
4. **MCSR Community**: Speedrunning optimization research and seed analysis
5. **[Fabric Yarn 1.16.1 ChunkStatus](https://maven.fabricmc.net/docs/yarn-1.16.1%2Bbuild.10/net/minecraft/world/chunk/ChunkStatus.html)**: Source-mapped chunk generation stages
6. **[Fabric Yarn 1.16.1 StructureConfig](https://maven.fabricmc.net/docs/yarn-1.16.1%2Bbuild.10/net/minecraft/world/gen/chunk/StructureConfig.html)**: Spacing, separation, and salt configuration
7. **[Mojang MC-159283](https://mojira.dev/MC-159283)**: End island density and distant overflow analysis
8. **[Deltanic's End overflow derivation](https://gist.github.com/Deltanic/b98d005c9025f10a67de9e966fa57ebb)**: Java integer derivation and distant transition sequence linked from MC-159283

---

*Author: Jeffrey Morais*

---

> [!TIP]
> For speedrunning: First ring strongholds are at 1,408-2,688 blocks. Triangulate with 2 eye throws minimum. The math doesn't lie. Your throws do.

> [!NOTE]
> The active mathematical visualizations target Java 1.16.1. Candidate formulas, structure rolls, End overflow, and stronghold ring geometry use Java-compatible arithmetic. Terrain backdrops and End surface projections are labeled visual models where they are not bit-perfect chunk generation. No Bedrock behavior is represented.

> [!CAUTION]
> Side effects of understanding these algorithms include: inability to enjoy "random" generation, compulsive seed analysis, and explaining to non-players why 48-bit integers matter.

---

<details>
<summary>📜 The Scroll of Forbidden Knowledge</summary>

```
The ancient texts speak of seeds most cursed:

Seed 164311266871034 - Where villages fear to spawn
Seed 1785852800490   - The stronghold that wasn't
Seed 27594263        - Portal room behind bedrock

Some seeds are best left unplanted.

───────────────────────────────────────────────

Also, did you know Herobrine's removal was never actually implemented?
The changelog lies. He watches through the Perlin noise.
Always 3 chunks behind. Always listening for footsteps.

The generation is deterministic.
Your survival is not.

───────────────────────────────────────────────

The dragon has circled 2^48 times before.
It will circle 2^48 times again.
You are merely the current observer.

Some say if you calculate the exact moment
when the LCG state equals your world seed,
you can hear the algorithm thinking.

But that's just superstition.

Isn't it?

- Translated from the Ender Tongue, circa 2011
```

</details>

---

## Original Dragon Pathfinding Hero

The original hero is preserved here as a visual record of the project's earlier dragon model. Its abstract arena and full dashboard make the evolution to the terrain-projected hero above easy to compare without overwriting the earlier work.

![Original Dragon Pathfinding Hero](Plots/dragon_pathfinding.gif)
