# Minecraft Generation

<!-- Do not remove this comment. It is important. -->
<!-- The seeds remember those who query them. -->

###### A mathematical study of Minecraft's procedural-generation algorithms, with supporting analysis of pathing systems such as Ender Dragon flight behaviour.

![Ender Dragon Pathfinding](Plots/dragon_pathfinding_hero.gif)

## Objective

Minecraft Generation explains how a compact seed becomes a world. Its primary subject is procedural generation: Java's pseudorandom number generator, noise fields, structure-region arithmetic, stronghold rings, chunk-status progression, and the unusual integer behaviour of the distant End. Pathing algorithms are included where they reveal the same central idea, namely that apparently organic behaviour can arise from deterministic rules.

The repository combines source-shaped Python models with publication-quality figures and animations. Exact arithmetic is identified as exact. Terrain textures and other reduced-order surfaces are identified as illustrative when they are designed to explain a mechanism rather than reproduce every block from a vanilla save.

<table align="center">
<tr>
<td align="center"><b>2<sup>48</sup></b><br><sub>Java Random states</sub></td>
<td align="center"><b>128</b><br><sub>Strongholds</sub></td>
<td align="center"><b>8</b><br><sub>Stronghold rings</sub></td>
<td align="center"><b>24</b><br><sub>Dragon path nodes</sub></td>
<td align="center"><b>10</b><br><sub>End spikes</sub></td>
<td align="center"><b>20</b><br><sub>End gateways</sub></td>
</tr>
</table>

<p align="center">
  <img src="./Plots/apple.gif?raw=true" alt="apple" width="51" height="50" />
</p>

## Mathematical Foundation

### Java's 48-bit generator

A Minecraft Java world accepts a 64-bit seed, but many Java 1.16.1 placement decisions pass through the 48-bit internal state of `java.util.Random`. If the current state is $X_n$, the next state is

$$
X_{n+1} = (aX_n + c) \bmod 2^{48},
$$

with $a = 25214903917$ (`0x5DEECE66D`) and $c = 11$. Multiplication stretches and folds the current state around a finite ring of integers. The addition prevents zero from becoming a fixed point. The high bits of the new state are then returned to the caller.

The implementation in [`Code/core/lcg.py`](Code/core/lcg.py) matters because a mathematically plausible random generator is not sufficient. Reproducing Minecraft's candidate chunks requires the same bit width, overflow, seeding transform, and bounded-integer rejection rule as Java.

```python
class MinecraftLCG:
    MULTIPLIER = 0x5DEECE66D
    ADDEND = 0xB
    MASK = (1 << 48) - 1

    def next_bits(self, bits):
        self.seed = (self.MULTIPLIER * self.seed + self.ADDEND) & self.MASK
        return self.seed >> (48 - bits)
```

### Noise, scale, and Brownian composition

Random numbers alone do not make terrain. A noise function assigns nearby coordinates similar values, so moving a short distance usually changes the field only a little. Minecraft combines fields evaluated at several scales. Broad layers provide continents and climate regions, while finer layers provide local variation.

A useful explanatory model is fractal Brownian motion:

$$
\mathcal{N}(x,z) = \sum_{k=0}^{n-1} p^k\,\eta(f^k x, f^k z),
$$

where $\eta$ is a smooth base-noise field, $f$ is lacunarity, and $p$ is persistence. Increasing $f$ makes each octave finer. Multiplying by $p$ makes fine octaves less influential. With $f=2$ and $p=1/2$, each layer doubles in frequency and halves in amplitude.

The repository uses source-shaped simplex and gradient-noise components, then explicitly labels reduced biome and terrain fields as explanatory models. This separation is important: an illustration can accurately teach scale composition without claiming to be a block-for-block world export.

## Visualizations

### Dragon Pathfinding

The Ender Dragon moves through a graph rather than choosing an arbitrary point in continuous space. Java 1.16.1 defines 24 horizontal path nodes arranged in three rings: 12 near radius 60 blocks, 8 near radius 40, and 4 near radius 20. Edges determine which transitions are legal. A shortest-path search then minimizes the sum of Euclidean edge lengths,

$$
d(v) = \min_{u \rightarrow v}\left[d(u) + \lVert \mathbf{x}_u - \mathbf{x}_v \rVert_2\right].
$$

The fight state also changes the usable graph. While crystals remain, the dragon can use the outer nodes. After the crystals are gone, its search is restricted toward the inner graph. The simplified perch decision follows

$$
P(\text{perch}) = \frac{1}{3+n_{\mathrm{crystals}}}.
$$

This explains a familiar gameplay pattern: destroying crystals does more than remove healing. It also increases the chance that a holding phase commits to a landing approach.

[`Code/core/dragon.py`](Code/core/dragon.py) uses a priority queue to traverse only source-allowed edges. The important detail is that path cost and state restrictions are kept separate.

```python
minimum_node = 0 if crystals_alive > 0 else 12
allowed = set(range(minimum_node, 24))

for neighbor in adjacency[current]:
    if neighbor not in allowed:
        continue
    weight = np.linalg.norm(DRAGON_NODES[current] - DRAGON_NODES[neighbor])
```

![Dragon path graph and fight state](Plots/dragon_pathfinding_hero.gif)

The large left panel places the graph over the central End island. Node rings, spike footprints, cages, the exit fountain, and the active route share the same block-coordinate system. The right side presents the seven-state behavioural cycle: Holding, Strafing, Approach, Landing, Perching, Takeoff, and Charging. The lower state panel reports remaining crystals and the corresponding perch probability. This is a reduced-order flight simulation grounded in the Java 1.16.1 node topology, not a complete reimplementation of every dragon controller.

#### Phase details

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

The clips isolate the major phases so the path and state transition can be read without waiting through a full fight cycle. Their geometry and state colours match the main animation.

### Trajectory Distribution and Degeneracy

A single shortest path says what one simulated approach did. An ensemble asks which parts of the graph repeatedly attract routes. For trajectories $\gamma_i(t)$, a simple occupancy field is

$$
D(x,z) = \sum_i \sum_t K_h\!\left((x,z)-\gamma_i(t)\right),
$$

where $K_h$ deposits a small amount of density around every sampled position. High values identify shared corridors and near-degenerate route choices, meaning several valid approaches occupy almost the same region.

Minecraft relevance comes from repeatability. Speedrunning strategies depend less on one attractive route than on locations that many plausible routes revisit. The current animation focuses on the spatial ensemble. A more explicit intersection-frequency analysis belongs to the next development cycle.

[`Code/core/dragon.py`](Code/core/dragon.py) constructs each approach from legal node paths and smooth, deterministic interpolation.

```python
def path_coordinates(indices, samples_per_edge=10, bend=1.2):
    points = []
    for start, end in zip(indices[:-1], indices[1:]):
        points.extend(smooth_segment(
            DRAGON_NODES[start], DRAGON_NODES[end],
            samples=samples_per_edge, bend=bend,
        ))
    return np.asarray(points)
```

![Accumulated Dragon Approach Trajectories](Plots/dragon_trajectory_ensemble.gif)

The image accumulates 240 seeded approaches. Older paths cool and recede; recent paths remain warmer and brighter. Arrowheads indicate direction, while the End island and spike footprints provide physical scale. Dense luminous corridors mark graph regions that many approaches share. The figure models flight-path distribution, not arrow momentum or dragon-damage mechanics.

### End Dimension Structure

The End contains meaningful geometry at three very different scales. Near the origin, ten obsidian spikes occupy a seed-shuffled ring with nominal angular positions

$$
\mathbf{p}_k = 42\left(\cos\frac{2\pi k}{10},\ \sin\frac{2\pi k}{10}\right),
\qquad k=0,\ldots,9.
$$

Twenty post-fight gateways occupy a larger ring of radius 96 blocks,

$$
\mathbf{g}_k = \left(\left\lfloor 96\cos\frac{\pi k}{10}\right\rfloor,
\left\lfloor 96\sin\frac{\pi k}{10}\right\rfloor\right).
$$

Far from the origin, Java's signed 32-bit arithmetic changes the End density calculation. Let

$$
q(x,z) = \mathrm{signed32}\!\left(
\mathrm{trunc}(x/8)^2 + \mathrm{trunc}(z/8)^2
\right).
$$

Terrain remains in the normal arithmetic branch when $q(x,z) \ge 0$. Repeated signed overflow creates thinner radial land and void bands. When sampled on a regular image grid, those bands alias into a folded, lattice-like field. The pattern should not be read as copies of the central island.

[`Code/core/end_generation.py`](Code/core/end_generation.py) evaluates the wrap explicitly, avoiding unsupported plotting macros and avoiding floating-point guesses about Java overflow.

```python
sample_x = np.trunc(x / 8.0).astype(np.int64)
sample_z = np.trunc(z / 8.0).astype(np.int64)
unsigned = (sample_x * sample_x + sample_z * sample_z) & 0xFFFFFFFF
signed = np.where(unsigned >= 0x80000000, unsigned - 0x100000000, unsigned)
generated = signed >= 0
```

![End Dimension Structure](Plots/end_dimension_overview.png)

Panel (a) samples the exact overflow predicate from $-6,000,000$ to $+6,000,000$ blocks on both axes. The first affected eight-block cell begins at 370,720 blocks, the first strictly unsafe point is 370,728, and normal terrain resumes at 524,288. Panel (b) shows the fight-scale island, ten spikes, two seed-selected cages, the emphasized exit fountain, and the 20-gateway ring. Tower outlines are neutral so the iron cages and crystals remain visually distinct. Panel (c) zooms into the first outer-island band. Each mark is a simplex-qualified source site, kept visually separate to communicate that the ring consists of many islands rather than continuous End stone.

The redesign was guided by the archived references in [`resources/references/end/`](resources/references/end/), while the rendered figure remains generated from repository code.

### Radial World Generation

Chunk generation is not a single operation. Java 1.16.1 advances chunks through an ordered status chain from `EMPTY` to `FULL`. If status $s_j$ depends on a neighbourhood completed at status $s_{j-1}$, generation propagates as a dependency wave rather than revealing the entire map at once.

The terrain revealed by that wave is connected to the earlier Brownian-motion discussion. Broad noise layers establish large regions before fine layers and surface rules add detail. The animation does not claim that Minecraft literally evaluates one fBm equation at each status. Instead, the equation explains why recognizable large-scale form can appear before the final surface is complete.

[`Code/seed_loading.py`](Code/seed_loading.py) keeps the source status order exact and models scheduling with a deterministic wave.

```python
phase = progress * wave_extent - distances * dependency_lag
stages = np.floor(phase).astype(int)
hidden = phase < 0.0
stages = np.clip(stages, 0, len(STATUS_NAMES) - 1)
```

![Radial World Generation](Plots/seed_loading.gif)

The axes identify chunk position around the target chunk. Tiles grow into view only when the wave reaches them. Early statuses remain muted; `NOISE` reveals the first field, `SURFACE` restores biome colour, and later stages leave visible evidence for carvers, features, lighting, spawn readiness, and heightmaps. The final `FULL` state is held so the completed dependency field can be read. The timing is explanatory, not a profiler trace.

### Overworld Structure Generation

Many Java 1.16.1 structures begin with random-spread regions. For a structure with spacing $d$, separation $s$, salt $\sigma$, world seed $W$, and region coordinate $(R_x,R_z)$, the region seed is

$$
S = W + 341873128712R_x + 132897987541R_z + \sigma.
$$

Java Random draws two offsets from the candidate window $w=d-s$,

$$
c_x = dR_x + J_x, \qquad c_z = dR_z + J_z,
\qquad J_x,J_z \in \{0,\ldots,w-1\}.
$$

Villages, desert pyramids, jungle pyramids, swamp huts, and pillager outposts all use a 32 by 32 chunk grid with an eight-chunk separation in this model, but their salts differ. Different salts produce different candidates even when the grid is shared. Candidate arithmetic answers where a structure may try to start. A biome gate then answers whether that structure family is compatible with the local environment.

[`Code/core/structures.py`](Code/core/structures.py) separates those two questions. This keeps the exact candidate calculation testable while allowing the visualization to state honestly that its biome boundaries are illustrative.

```python
window = config.spacing - config.separation
offset_x = random.next_int(window)
offset_z = random.next_int(window)
chunk_x = region_x * config.spacing + offset_x
chunk_z = region_z * config.spacing + offset_z

compatible = structure_biome_compatible(config.name, biome_name)
```

![Overworld Structure Generation](Plots/structure_placement.gif)

The main map is measured in chunks. Thin white squares show 32 by 32 placement regions, the cyan outline shows the active region, and the dashed cyan square shows its 24 by 24 random window. Marker shape and colour distinguish the five structure families. Only candidates compatible with the displayed biome category are plotted. The right legend names both structure markers and biome colours. Candidate positions and salts are Java-compatible for 1.16.1; the biome field is a deterministic explanatory surface, so the points are not claims about final structures in a vanilla seed export.

### Nether Structure Generation

Nether fortresses and bastion remnants share one Java 1.16.1 grid. Each 27 by 27 chunk region has a 23 by 23 candidate window. After drawing the two offsets, Java Random draws

$$
r = \mathrm{nextInt}(5).
$$

Rolls 0 and 1 select a fortress, while rolls 2, 3, and 4 select a bastion. The candidate-stage split is therefore

$$
P(\text{fortress}) = \frac{2}{5}, \qquad
P(\text{bastion}) = \frac{3}{5}.
$$

Ruined portals use a separate 25 by 25 grid, a 15 by 15 candidate window, and a different salt. Sharing a dimension does not imply sharing a random sequence.

[`Code/core/structures.py`](Code/core/structures.py) makes the shared roll explicit.

```python
offset_x = random.next_int(23)
offset_z = random.next_int(23)
type_roll = random.next_int(5)
structure_type = 'fortress' if type_roll < 2 else 'bastion'
```

![Nether Structure Generation](Plots/multi_structure_generation.gif)

The animation overlays the shared fortress-or-bastion layer and the independent ruined-portal layer. The trace identifies the active region, offsets, and shared type roll. Crimson, warped, basalt, soul-sand, netherrack, and lava colours provide Nether context without changing the exact placement arithmetic. A fuller biome legend and terrain redesign are intentionally reserved for the next cycle.

### Stronghold Ring Distribution

Strongholds do not use the rectangular random-spread grid. Java 1.16.1 advances around the origin in polar coordinates. For ring index $i$, a useful form of the candidate radius in chunks is

$$
r_i = 128 + 192i + \left(U-\frac{1}{2}\right)80,
\qquad U \sim \mathcal{U}(0,1).
$$

Angles advance around each ring, and a seeded angular offset rotates the next ring. The eight ring populations are

$$
3,\ 6,\ 10,\ 15,\ 21,\ 28,\ 36,\ 9,
$$

which sum to 128 candidates. The first ring lies between 1,408 and 2,688 blocks from world origin. These are pre-biome-search candidates, not guaranteed portal-room coordinates.

[`Code/core/strongholds.py`](Code/core/strongholds.py) preserves the ring iterator and Java rounding.

```python
radius_chunks = (
    4 * 32 + 6 * ring_index * 32
    + (random.next_double() - 0.5) * 32 * 2.5
)
x = java_round(math.cos(angle) * radius_chunks) * 16
z = java_round(math.sin(angle) * radius_chunks) * 16
```

![Stronghold Ring Distribution](Plots/stronghold_rings.png)

Panel (a) shows all 128 candidates over a faded, varied Overworld biome field. Coloured bands encode each ring's allowed radial interval; thicker boundaries and enlarged dots keep the exact geometry dominant over the illustrative terrain. Panel (b) enlarges the three first-ring candidates. Dashed gold circles show the 112-block biome-search neighbourhood around each preliminary point. Panel (c) gives two quantities for every ring: bar height is candidate count, while the label above the bar gives the radial search band in thousands of blocks. The background is illustrative, but the seeded candidate positions, ring counts, and ranges are exact for the modeled iterator.

## Quick Start

```bash
git clone https://github.com/IsolatedSingularity/Minecraft-Generation.git
cd Minecraft-Generation
pip install -r requirements.txt

# Run the numerical and asset checks
python -m unittest discover -s tests -v

# Generate all active visualizations
cd Code
python render_all.py
```

Run `render_all.py` from inside `Code/`. The visualization modules use sibling imports, so invoking `python Code/render_all.py` from the repository root is not supported.

## Scope and Accuracy

> [!NOTE]
> Active mathematical visualizations target Java Edition 1.16.1. Java Random, candidate-region arithmetic, stronghold ring geometry, gateway positions, dragon topology, and signed-integer overflow are implemented with Java-compatible conventions. Terrain backdrops, reduced biome fields, the chunk scheduling wave, and selected End surface projections are explanatory models unless stated otherwise. No Bedrock behaviour is represented.

> [!TIP]
> For speedrunning, first-ring stronghold candidates occur 1,408 to 2,688 blocks from origin. Eye triangulation still targets the final biome-adjusted stronghold, not merely the preliminary ring point.

## References

1. [Minecraft Wiki](https://minecraft.wiki/): mechanics and historical version context.
2. [Alan Zucconi, Minecraft World Generation](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/): accessible procedural-generation background.
3. [OpenJDK `java.util.Random`](https://github.com/openjdk/jdk/blob/master/src/java.base/share/classes/java/util/Random.java): Java LCG behaviour.
4. [Fabric Yarn 1.16.1 `ChunkStatus`](https://maven.fabricmc.net/docs/yarn-1.16.1%2Bbuild.10/net/minecraft/world/chunk/ChunkStatus.html): source-mapped chunk status order.
5. [Fabric Yarn 1.16.1 `StructureConfig`](https://maven.fabricmc.net/docs/yarn-1.16.1%2Bbuild.10/net/minecraft/world/gen/chunk/StructureConfig.html): spacing, separation, and salt configuration.
6. [Fabric Yarn 1.16.1 `StructureFeature`](https://maven.fabricmc.net/docs/yarn-1.16.1%2Bbuild.10/net/minecraft/world/gen/feature/StructureFeature.html): structure families and candidate-stage source context.
7. [Mojang MC-159283](https://mojira.dev/MC-159283): distant End density and integer-overflow behaviour.
8. [Deltanic's End overflow derivation](https://gist.github.com/Deltanic/b98d005c9025f10a67de9e966fa57ebb): transition sequence linked from MC-159283.

*Author: Jeffrey Morais*

## Legacy Simulations

The original dragon animation is retained as a record of the project's earlier reduced-order model. It uses an abstract arena and a larger dashboard rather than the newer source-shaped central-island projection. Its movement is less constrained by the current 24-node presentation, but its smooth interpolation remains useful when comparing how visual assumptions change the apparent character of the same state machine.

The legacy asset is not used as evidence for current numerical claims. It is presented after the active figures so readers encounter the tested model first.

![Original Dragon Pathfinding Hero](Plots/dragon_pathfinding.gif)

## Scroll of Forbidden Knowledge

<details>
<summary>The Scroll of Forbidden Knowledge</summary>

```
The ancient texts speak of seeds most cursed:

Seed 164311266871034 - Where villages fear to spawn
Seed 1785852800490   - The stronghold that wasn't
Seed 27594263        - Portal room behind bedrock

Some seeds are best left unplanted.

Also, did you know Herobrine's removal was never actually implemented?
The changelog lies. He watches through the Perlin noise.
Always 3 chunks behind. Always listening for footsteps.

The generation is deterministic.
Your survival is not.

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
