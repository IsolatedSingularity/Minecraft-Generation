# Minecraft Procedural Algorithms

<!-- Do not remove this comment. It is important. -->
<!-- The seeds remember those who query them. -->
<!-- Commit test -->

###### Mathematical exploration of Minecraft's world generation algorithms. References include [Minecraft Wiki](https://minecraft.wiki/), [Sportskeeda Wiki](https://wiki.sportskeeda.com/minecraft), and procedural generation works from [Alan Zucconi](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/).

![Ender Dragon Pathfinding](Plots/dragon_pathfinding.gif)



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

The refreshed animation makes that boundary visible: exact candidate placement first, readable biome-pass preview second.

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

![Dragon Pathfinding Animation](Plots/dragon_pathfinding.gif)

The Ender Dragon navigates a **directed acyclic graph** embedded in the End dimension's geometry. The pathfinding system maintains 25+ nodes distributed across three concentric rings: outer nodes at radius 100 blocks for circling behavior, inner nodes at 60 blocks for strafing approaches, and center nodes at 30 blocks for landing preparation.

The dragon's behavioral state machine operates on seven distinct states, each with characteristic movement patterns. **HOLDING** produces the familiar circling at maximum radius, the dragon tracing lazy arcs while surveying its domain. **STRAFING** triggers aggressive linear charges accompanied by acid breath. **APPROACH**, **LANDING**, and **PERCHING** execute the critical touchdown sequence that speedrunners exploit, the probability of initiating this sequence following $P = 1/(3 + n_{\text{crystals}})$ where destroyed crystals increase perch likelihood. **TAKEOFF** and **CHARGING** complete the cycle, returning the dragon to its orbital patterns.

The visualization renders the path graph in real time while the right-hand RPG-style state wheel tracks behavior. With all ten crystals alive, every state node is linked like a fully connected stat build. Each destroyed crystal removes part of that network, while the central ring updates the exact perch probability and the surviving crystal count.

#### Phase Details

These three clips isolate the movement families compressed into the main loop. The first follows wide holding arcs and attack transitions, the second traces the inward landing sequence and perch, and the third shows how the dragon reconnects with the outer graph after takeoff.
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

#### Seeded Trajectory Ensemble

![Accumulated Dragon Approach Trajectories](Plots/dragon_trajectory_ensemble.gif)

The ensemble now begins empty and gradually accumulates 420 seeded dragon approaches over roughly sixteen seconds. Pale blue density marks repeatedly occupied routes, while the brighter green strokes identify the newest approaches so the field can be read as it forms. It is a dragon-path simulation inspired by speedrunning research, not a simulation of arrow momentum or damage.

*The dragon doesn't hunt you. It follows an algorithm. Your death was a graph traversal.*

### End Dimension Overview

![End Dimension Layout](Plots/end_dimension_overview.png)

The End dimension exhibits precise mathematical structure beneath its alien aesthetics. The central island, 200 blocks in diameter, hosts the exit portal at exact coordinates $(0, 0)$ surrounded by ten obsidian pillars arranged via:

$$\mathbf{p}_k = \left( r_p \cos\left(\frac{2\pi k}{10}\right), r_p \sin\left(\frac{2\pi k}{10}\right) \right), \quad k \in \{0, 1, \ldots, 9\}$$

with pillar radius $r_p = 76$ blocks. Crystals atop these pillars follow a height sequence encoding their cage protection status.

Twenty End Gateways form a larger ring at radius 96 blocks, their positions calculated through:

$$\mathbf{g}_k = \left( \lfloor 96 \cos(\pi k / 10) \rfloor, \lfloor 96 \sin(\pi k / 10) \rfloor \right), \quad k \in \{0, 1, \ldots, 19\}$$

Beyond the gateway ring, outer islands generate in a pseudo-infinite expanse, but not truly infinite. At $r = 370,720$ blocks, integer arithmetic overflow creates a void gap where no islands spawn. A second gap appears at $r = 524,288$ blocks. These are not bugs but *consequences of binary representation*.

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

This animation presents the Java 1.16.1 loading story as one focused chunk field. Generation statuses travel outward as a dependency wave, and the compact control strip below shows which stage is currently advancing.

The population-seed mixing and status order are exact. The apparent timing is a deterministic educational schedule rather than a wall-clock profiler trace.

*The seed is fixed. The reveal is the explanation.*

### Structure Placement Algorithm

![Structure Placement Animation](Plots/structure_placement.gif)

This animation isolates the Java 1.16.1 village candidate stage. The world is divided into 32 x 32 chunk regions. Each region gets one deterministic candidate, generated with:

$$S = \text{worldSeed} + R_x \cdot 341873128712 + R_z \cdot 132897987541 + \sigma$$

Java Random then selects two offsets with `nextInt(24)`, producing a candidate chunk in the region's 24 x 24 chunk window. House-shaped markers show the deterministic candidates, while the blue outline follows the active region and its legal placement window.

The terrain is a faint deterministic context layer for orientation, not a bit-exact biome claim. Candidate placement remains exact for Java 1.16.1, while full terrain and biome viability remain separate generation steps.

### Multi-Structure Generation

![Multi-Structure Generation](Plots/multi_structure_generation.gif)

This animation compares Java 1.16.1 Nether fortress, bastion-remnant, and ruined-portal candidates. Fortresses and bastions share 27-chunk regions with separation 4 and salt $30084232$; one shared candidate is classified with the source-faithful 2:3 fortress-to-bastion split. Ruined portals use independent 25-chunk regions with separation 10 and salt $34222645$.

The colored Nether terrain is a deterministic context layer, not bit-exact Nether biome generation. Castle, shield, and portal symbols distinguish the exact candidate-stage outputs, while the right-side legend keeps the plot area unobstructed.

*Same seed. Different salt. Different fate.*

<br>

<p align="center"><sub>. . .</sub></p>

<br>

<p align="center"><sub>Two salts. Three outcomes.</sub></p>

<p align="center"><sub>The algorithm doesn't care which one you wanted.</sub></p>

<br>

<p align="center"><sub>. . .</sub></p>

<br>

### Stronghold Ring Distribution

![Stronghold Distribution](Plots/stronghold_rings.png)

This plot now uses the Java 1.16.1 stronghold ring iterator, centered on world origin '(0, 0)', not the player's spawn point. The seeded candidate geometry contains 128 strongholds across eight rings with populations:

$$3,\ 6,\ 10,\ 15,\ 21,\ 28,\ 36,\ 9$$

For ring number $i$, indexed from zero, the approximate radius in chunks is:

$$r_i = 128 + 192i + \left(\mathcal{U}(0,1) - \frac{1}{2}\right) \cdot 80$$

The first ring contains exactly 3 candidates between 1,408 and 2,688 blocks. The remaining ring ranges are shown directly in the figure. Java 1.16.1 then searches around each candidate for a valid biome, so the plotted points are the exact seeded ring candidates, not claims about final portal-room coordinates.
The right subplot repeats a two-throw triangulation experiment 1,800 times with independent Gaussian bearing noise of $\sigma_\theta = 1.2^\circ$. The point cloud shows how angular error spreads the estimate; the star is the true candidate, the X is the median estimate, and the dashed circle is the 112-block biome-search radius. This noise model is an uncertainty demonstration, not a vanilla generation rule.

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
python Code/render_all.py
```

---

## References

1. **[Minecraft Wiki](https://minecraft.wiki/)**: Definitive game mechanics documentation
2. **[Alan Zucconi](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/)**: Procedural generation deep dives
3. **Java Random Implementation**: OpenJDK LCG source code analysis
4. **MCSR Community**: Speedrunning optimization research and seed analysis

---

*Author: Jeffrey Morais*

---

> [!TIP]
> For speedrunning: First ring strongholds are at 1,408-2,688 blocks. Triangulate with 2 eye throws minimum. The math doesn't lie. Your throws do.

> [!NOTE]
> These four refreshed assets target Java 1.16.1. Candidate formulas and stronghold ring geometry use Java-compatible RNG. Biome panels are labeled compact previews, not bit-perfect full-world biome generation. No Bedrock behavior is represented.

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
