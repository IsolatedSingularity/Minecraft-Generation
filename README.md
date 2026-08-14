# Minecraft Generation

[![Tests](https://github.com/IsolatedSingularity/Minecraft-Generation/actions/workflows/tests.yml/badge.svg)](https://github.com/IsolatedSingularity/Minecraft-Generation/actions/workflows/tests.yml)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Minecraft Java 1.16.1](https://img.shields.io/badge/Minecraft%20Java-1.16.1-62B47A)
![Scope](https://img.shields.io/badge/scope-source--checked%20models-8B6FD6)

<!-- Do not remove this comment. It is important. -->
<!-- The seeds remember those who query them. -->
<!--
LOCAL AGENT NOTE: A private, version-locked Minecraft Java 1.16.1 corpus may
exist at `Game Reference/`. It is deliberately gitignored and must never be
committed. When source verification is relevant and the user has asked for it,
read `Game Reference/README.md`, then route through
`Game Reference/08_mc_1_16_1_agent_reference/START_HERE.md`. Use its index and
topic-specific retrieval conservatively; do not recursively read the corpus,
raw JAR extractions, assets, tooling environments, or all mapped sources.
-->

###### One seed goes in. Villages, strongholds, strange coastlines, and one very stubborn dragon come out.

![Ender Dragon Pathfinding](Plots/dragon_pathfinding_hero.gif)

## The Question

How does one integer become a world?

There is no single world-generation formula hiding behind the terrain. A seed starts a pseudorandom stream, that stream chooses offsets, noise gives nearby coordinates some memory of one another, and layers of biome and structure rules decide what survives. Even the Ender Dragon, for all its theatrical circling, is fenced in by a graph, a phase machine, and a handful of probabilities.

This repository pulls those rules apart with tested Python and generated figures. The arithmetic is allowed to be fussy. The pictures are allowed to be beautiful. Neither is allowed to bluff.

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

![World-generation data flow](Plots/world_generation_flow.svg)

### Where the map ends and Minecraft begins

Two labels keep the pictures honest:

- **Source-faithful** means the implementation follows the Java 1.16.1 arithmetic, RNG order, biome graph, density stack, graph topology, or placement constants named in that section.
- **Explanatory schedule or projection** marks the remaining reductions, chiefly worker timing and the dragon's top-down steering view.

That boundary matters. The active terrain maps now use the full 1.16.1 biome selection and base-height paths, with vanilla block textures for the surface. Structure points remain candidate-stage points unless a later gate is explicitly applied.

## Mathematical Foundation

*Before there are mountains or monuments, there is a number moving through a very small machine.*

### Java's 48-bit clockwork

Minecraft needs surprise that can be replayed exactly. Give two worlds the same seed and the same sequence of calls, and they must receive the same answers. Java's `Random` class does this with a 48-bit internal state that advances one rigid step at a time.

If the current state is $X_n$, the next state is

$$
X_{n+1} = (aX_n+c) \bmod 2^{48},
$$

where

$$
a=25214903917,\qquad c=11.
$$

This can be read as four small operations:

1. Multiply the current state by a fixed number.
2. Add 11.
3. Keep only the lowest 48 bits.
4. Return selected high bits as the next random value.

The modulus is the edge of the machine. Anything beyond 48 bits wraps back into its finite state space. NumPy's default generator could draw equally convincing dots, but they would not be Minecraft's dots.

[`Code/core/lcg.py`](Code/core/lcg.py) performs the same masked update:

```python
class MinecraftLCG:
    MULTIPLIER = 0x5DEECE66D
    ADDEND = 0xB
    MASK = (1 << 48) - 1

    def next_bits(self, bits):
        self.seed = (self.MULTIPLIER * self.seed + self.ADDEND) & self.MASK
        return self.seed >> (48 - bits)
```

The hexadecimal mask is $2^{48}-1$, so `& MASK` keeps the same low 48 bits as the modulus. The figure below follows 64 exact updates. Its bit raster makes the important sleight of hand visible: the state keeps all 48 bits, while `next_bits(16)` hands only the highest 16 back to the caller.

![Java 48-bit LCG state and bit extraction](Plots/lcg_bit_extraction.png)

### Noise, scale, and Brownian composition

*A landscape needs memory. Nearby blocks should feel related, even when the horizon does not.*

Independent random values snap from one coordinate to the next. Terrain needs slower changes underneath the fine ones, so a hillside can belong to a region and a region can belong to a continent.

A useful explanatory model is fractal Brownian motion:

$$
N(x,z)=\sum_{k=0}^{n-1}p^k\,\eta(f^k x,f^k z).
$$

Here:

- $(x,z)$ is a horizontal world coordinate.
- $\eta$ is one smooth noise field.
- $k$ identifies a scale, often called an octave.
- $f$ increases the spatial frequency from one octave to the next.
- $p$ reduces the amplitude of finer octaves.

For $f=2$ and $p=1/2$, each new layer changes twice as quickly and speaks half as loudly. The broad layer carries the silhouette; the later octaves roughen its edges.

![Brownian noise composition flow](Plots/noise_composition_flow.svg)

The Brownian figure is the gentler mathematical doorway, not a substitute for Minecraft's generator. [`Code/core/vanilla_noise.py`](Code/core/vanilla_noise.py), [`Code/core/vanilla_biomes.py`](Code/core/vanilla_biomes.py), and [`Code/core/vanilla_terrain.py`](Code/core/vanilla_terrain.py) carry the real job: Java Random initialization, octave Perlin and Double Perlin samplers, the complete Overworld biome-layer graph, Nether multi-noise selection, and the three-dimensional density stacks used for base height.

![Weighted Brownian noise octaves and their sum](Plots/brownian_noise_composition.png)

The renderer groups raw biome IDs only for a legible legend, then samples the selected vanilla 1.16.1 block textures in world coordinates. Grass, water, sand, snow, stone, mycelium, nylium, soul sand, basalt, lava, and End stone all come from the small provenance-tracked asset subset in [`Assets/minecraft_1_16_1`](Assets/minecraft_1_16_1/README.md).

## Visualizations

### Dragon Pathfinding

*Twenty-four waypoints hang above the End. The dragon still has to turn between them.*

The Ender Dragon does not choose every position in the sky independently. Java 1.16.1 defines 24 horizontal navigation nodes arranged in three rings:

- 12 outer nodes near a radius of 60 blocks
- 8 middle nodes near a radius of 40 blocks
- 4 inner nodes near a radius of 20 blocks

Edges state which node transitions are legal. Fight phases decide which goals are relevant, and continuous steering carries the dragon between those goals.

![Dragon navigation data flow](Plots/dragon_navigation_flow.svg)

For an edge from node $u$ to node $v$, the top-down travel cost is its Euclidean length:

$$
w(u,v)=\lVert\mathbf{x}_{u}-\mathbf{x}_{v}\rVert_{2}.
$$

The shortest known cost to a node $v$ is then

$$
d(v)=\min_{u\rightarrow v}\!\left[d(u)+w(u,v)\right].
$$

The equation says: to reach $v$, consider every legal predecessor $u$, add the cost already required to reach $u$, add the final edge length, and keep the cheapest result.

Crystals also affect the simplified perch decision:

$$
P(\mathrm{perch})=\frac{1}{3+n_{\mathrm{crystals}}}.
$$

With ten crystals alive, the chance is $1/13$, or about $7.7\%$. With no crystals alive, it becomes $1/3$, or about $33.3\%$. Destroying crystals therefore changes both healing pressure and the probability of a landing attempt.

[`Code/core/dragon.py`](Code/core/dragon.py) keeps graph restrictions separate from edge cost:

```python
minimum_node = 0 if crystals_alive > 0 else 12
allowed = set(range(minimum_node, 24))

for neighbor in adjacency[current]:
    if neighbor not in allowed:
        continue
    weight = np.linalg.norm(DRAGON_NODES[current] - DRAGON_NODES[neighbor])
```

The graph selects meaningful targets. Every graph-bound route is expanded through legal decoded edges, then a reduced top-down integrator applies the source movement terms: wrapped yaw error, a $\pm50^\circ$ turn clamp, retained turn momentum, alignment-sensitive acceleration, and velocity damping. Path targets advance at the source's ten-block completion boundary. Dense time samples keep the visible trail curved without inventing points between simulated positions. This is still a two-dimensional projection of a three-dimensional controller, not a block-exact replay.

![Dragon path graph and fight state](Plots/dragon_pathfinding_hero.gif)

The left panel puts the island, node graph, enlarged spikes, cages, fountain, dragon, fireball, breath clouds, explosions, and recent trail in one block-coordinate system. Grey lines show every legal edge. The graph edge currently being traversed receives a thin glow beneath the continuous steering trail, so the route decision and the dragon's curved response can be read together. Strafe and Charge target the player outside the navigation graph, so those phases correctly show no fabricated graph edge.

The sprite is split into a rigid body and two wing layers. Airborne frames cycle the wing span through a visible but restrained flap, while perched states hold the wings folded. This is sprite articulation only; it does not move the simulated position or smooth away a steering sample.

The right panel shows all 11 Java 1.16.1 phase types and highlights every one at least once. Solid arrows follow source-confirmed phase changes: Holding can enter Strafe or Landing Approach; Landing Approach enters Landing and then Sitting Scanning; scanning can attack, take off, or select Charging Player; attacking enters Flaming; and Takeoff, Strafe, and Charging Player return to Holding. Dashed arrows identify initialization and damage-triggered paths. In particular, sufficient damage while sitting or hovering forces Takeoff, while lethal airborne damage can enter Dying. The currently selected phase alone receives a thick white outline.

The dashboard labels the probability as the next Holding-path landing roll. Its value is $1/(3+n_\mathrm{crystals})$, not a continuous per-frame chance. Diamond crystal indicators mirror the surviving, non-circular destruction order on the island. Crystal destruction remains an external scripted demonstration event. The Strafe example launches a translucent purple dragon fireball whose impact cloud grows from radius 3 toward radius 7 and lingers with a transparent fade. Sitting Flaming separately displays a growing, fading radius-5 breath cloud, followed by a visible damage pulse that triggers the source-valid Takeoff path. The timings and radii follow the audited phase and entity sources; their top-down particle rendering is illustrative.

#### The fight, closer in

<table>
<tr>
<th>Holding and Strafing</th>
<th>Landing, Perched Decisions, and Charging</th>
<th>Takeoff and Return</th>
</tr>
<tr>
<td><img src="Plots/dragon_holding_strafe.gif" alt="Holding and strafing dragon path states" /></td>
<td><img src="Plots/dragon_landing_perch.gif" alt="Landing, perched decisions, and charging dragon path states" /></td>
<td><img src="Plots/dragon_takeoff.gif" alt="Dragon takeoff path state" /></td>
</tr>
</table>

The shorter clips use the same larger dragon, enlarged towers, highlighted graph edges, source-shaped steering, and phase colours. They linger a little longer than the hero's original cuts, but keep enough pace to feel like a fight rather than a slideshow.

### Trajectory Distribution and Degeneracy

*One flight is an anecdote. Four hundred and eighty flights begin to reveal the corridors.*

One route only tells us what happened once. An ensemble asks which cells keep mattering when the starting node and the player's direction change.

For trajectories $\gamma_i(t)$, a spatial occupancy field can be written as

$$
D(x,z)=\sum_{i}\sum_{t}K_{h}\!\left((x,z)-\gamma_{i}(t)\right),
$$

where $K_h$ deposits density near each sampled flight position. Bright regions are visited repeatedly.

The heatmap uses a stricter count. A trajectory contributes at most once to a spatial cell:

$$
F_{a,b}=\sum_{i}\mathbf{1}\!\left[\gamma_{i}\ \mathrm{enters\ cell}\ (a,b)\right].
$$

This prevents a slow trajectory from inflating a cell merely because it supplied many nearby samples. High $F_{a,b}$ values identify repeatable approach corridors. The right-hand chart avoids the arbitrary choice of raster bin size altogether by counting how many distinct approaches use each decoded legal graph edge.

[`Code/dragon_pathfinding.py`](Code/dragon_pathfinding.py) forms a binary grid contribution for each trajectory:

```python
histogram, _, _ = np.histogram2d(path[:, 1], path[:, 0], bins=(bins, bins))
contributions.append(histogram > 0)
cumulative = np.cumsum(np.asarray(contributions), axis=0)
```

![Accumulated Dragon Approach Trajectories](Plots/dragon_trajectory_ensemble.gif)

The figure accumulates 480 seeded landing approaches in fixed batches. Each seed chooses a current outer-ring node. Player positions are spread deterministically across a 24-to-48-block annulus; as Java 1.16.1 does, the landing phase selects the radius-40 node opposite that player and appends the exit portal as the final target. The intervening node route uses only decoded legal edges. The same steering reduction used by the hero supplies the continuous positions, and the active graph edge glows beneath the representative dragon's trail.

The representative route is added progressively while the dragon advances by arc length, so the sprite, local trail, highlighted edge, and live density agree at every frame. The density palette is capped at the 98.5th percentile for contrast, and that cap is stated in the panel. The underlying counts are untouched. On the right, every bar uses the final 480-route ensemble as its denominator. Counts only move upward, so a late batch can never make an earlier result appear to shrink.

### End Dimension Structure

*The End is tidy near the portal, scattered beyond the gulf, and numerically haunted at the world border.*

This figure jumps between three scales that rarely fit in the same conversation: the central fight, the first outer islands, and an integer-overflow scar millions of blocks away.

Ten spike positions occupy a nominal radius of 42 blocks:

$$
p_{k}=42\left(\cos\frac{2\pi k}{10},\ \sin\frac{2\pi k}{10}\right),
\qquad k=0,\ldots,9.
$$

Twenty post-fight gateways occupy a radius of 96 blocks:

$$
g_{k}=\left(
\left\lfloor96\cos\frac{\pi k}{10}\right\rfloor,
\left\lfloor96\sin\frac{\pi k}{10}\right\rfloor
\right).
$$

Far from the origin, Java's signed 32-bit arithmetic changes the End density calculation. First define the eight-block sample coordinates

$$
u_{x}=\mathrm{trunc}(x/8),\qquad u_{z}=\mathrm{trunc}(z/8).
$$

Then compute

$$
q(x,z)=\mathrm{signed32}\!\left(u_{x}^{2}+u_{z}^{2}\right).
$$

The relevant terrain branch remains valid when $q(x,z)\ge 0$. When the signed integer wraps below zero, an invalid square-root path produces a void band. The first affected eight-block cell starts at 370,720 blocks and the first strictly void sample occurs at 370,728 blocks.

[`Code/core/end_generation.py`](Code/core/end_generation.py) applies the wrap explicitly:

```python
sample_x = np.trunc(x / 8.0).astype(np.int64)
sample_z = np.trunc(z / 8.0).astype(np.int64)
unsigned = (sample_x * sample_x + sample_z * sample_z) & 0xFFFFFFFF
signed = np.where(unsigned >= 0x80000000, unsigned - 0x100000000, unsigned)
generated = signed >= 0
```

![End Dimension Structure](Plots/end_dimension_overview.png)

Panel (a) samples the exact predicate across approximately $\pm30$ million blocks, near the ordinary world-border scale. The void bands remain centered on the real origin, as described by [Mojang issue MC-159283](https://bugs-legacy.mojang.com/browse/MC-159283). At this scale the bands become thinner than the image grid, so they alias into a checkerboard-like lattice. Those apparent circles are sampling artifacts, not a secret collection of new End origins.

Panel (b) shows the fight-scale island, spike ring, emphasized cages, active exit fountain, and all 20 central gateways. Panel (c) shows the first outer-island source band as many separate seed sites rather than one continuous ring of End stone.

### End Structure Generation

*Past the thousand-block gulf, islands offer possibilities. The height check decides which ones become cities.*

End cities use a random-spread grid before biome and terrain-height checks. For a 20-chunk spacing and 11-chunk separation, the candidate window is

$$
w=20-11=9\ \mathrm{chunks}.
$$

End cities use the center-biased form of the random-spread rule. For each axis, Java Random draws twice from $\{0,\ldots,8\}$ and integer-averages the results, making central offsets more likely than edge offsets. The placement salt is 10387313.

End cities follow a center-biased candidate window and then an outer-island height gate. Gateways follow a separate path: begin on the central radius-96 ring, project the direction toward radius 1,024, then search for a safe outer-island endpoint.

[`Code/core/end_generation.py`](Code/core/end_generation.py) uses the exact End City candidate grid and derives the source rotation from `chunkX + chunkZ * 10387313`. The four rotated samples now come from the ported three-dimensional End density generator and its `WORLD_SURFACE_WG` base-height query.

For gateway $k$, the ideal outer direction is

$$
o_{k}=1024\left(\cos\frac{2\pi k}{20},\ \sin\frac{2\pi k}{20}\right).
$$

The plot then snaps that ideal vector to the nearest qualified outer-island source site, standing in for the safe-position search performed by the gateway system.

For candidate origin $(x,z)$ and source-selected offsets $(\Delta x,\Delta z)$, the displayed gate is

$$
H_{\min}=\min\{H(x,z),H(x+\Delta x,z),H(x,z+\Delta z),H(x+\Delta x,z+\Delta z)\},
\qquad H_{\min}\geq60.
$$

The offset signs come from one of the four rotations and have five-block magnitude. Candidate placement, rotation, sample geometry, generated heights, minimum operation, and threshold all follow the 1.16.1 path.

![End Structure Generation](Plots/end_structure_generation.png)

The left panel textures the generated End-stone surface and marks qualified End City starts with purpur squares. Cyan diamonds preserve the separate outer-gateway direction model; they are not End City points.

The right panel is the generated base-height field itself. The pale contour is height 60, grey crosses fail the four-sample gate, and purpur squares pass it. No End ship is plotted or implied. Both legends sit above the data and use the symbols present in their panel.

### Radial World Generation

*A chunk cannot finish alone. Its neighbours have homework too.*

A chunk does not move directly from nonexistent to finished. Java 1.16.1 advances it through an ordered sequence of statuses such as biomes, noise, surface, carvers, features, lighting, spawn preparation, heightmaps, and full completion.

The ordered pipeline runs from `EMPTY` through structure starts and references, biomes, noise, surface, carvers, features, lighting, spawn preparation, heightmaps, and finally `FULL`.

Neighbouring chunks introduce dependencies, so a center chunk can advance only while wider shells have reached the statuses it requires. The displayed status at Chebyshev distance $d$ is

$$
s_{d}(t)=\min\left(\left\lfloor22t-(10-d)\right\rfloor,T(d)\right),
$$

where $t$ is normalized animation progress and $T(d)$ is the source-required terminal status for that shell. The schedule is explanatory, but the terminal dependency profile is source-mapped.

This outside-in schedule determines *when* a chunk may expose each result. Its final targets come directly from Java 1.16.1 `ChunkStatus`: Chebyshev distance 0 reaches `FULL`, distance 1 reaches `FEATURES`, distance 2 reaches `LIQUID_CARVERS`, and distances 3 through 10 reach `STRUCTURE_STARTS`. The surface beneath it is not a painted proxy. It is the exact biome and base-height sample for seed `-4172144997902289642` across the displayed 21 by 21 chunks.

[`Code/seed_loading.py`](Code/seed_loading.py) implements that relationship directly:

```python
target[distances <= 10] = STRUCTURE_STARTS
target[distances == 2] = LIQUID_CARVERS
target[distances == 1] = FEATURES
target[distances == 0] = FULL
work = progress * 22
stages = np.clip(np.floor(work - (10 - distances)), 0, target)
```

![Radial World Generation](Plots/seed_loading.gif)

The main panel begins empty. A chunk first appears as work metadata, then exposes its selected biome, generated height, and textured surface only after reaching the corresponding status. Because most of the 21 by 21 dependency footprint stops at structure starts, most of it never pretends to have a finished surface. The larger tracker on the right uses Java 1.16.1's status colours. Worker timing remains explanatory; the status order, terminal dependency rings, biome IDs, heights, and visible textures are source-backed.

### Overworld Structure Generation

*The grid proposes. The biome and terrain get the final word.*

Many structures begin by dividing the chunk plane into placement regions. For world seed $W$, region coordinate $(R_x,R_z)$, and structure salt $\sigma$, the region seed is

$$
S=W+341873128712R_{x}+132897987541R_{z}+\sigma.
$$

For spacing $d$ and separation $s$, the usable candidate window is

$$
w=d-s.
$$

Most structures in this figure use a center-biased offset:

$$
J_{x}=\left\lfloor\frac{A_{x}+B_{x}}{2}\right\rfloor,
\qquad A_{x},B_{x}\in\{0,\ldots,w-1\}.
$$

with the same construction for $J_z$. Averaging two draws makes central offsets more common than edge offsets. Ocean monuments use a uniform draw instead. The candidate chunk is

$$
c_{x}=dR_{x}+J_{x},\qquad c_{z}=dR_{z}+J_{z}.
$$

![Structure-candidate data flow](Plots/structure_candidate_flow.svg)

[`Code/core/structures.py`](Code/core/structures.py) keeps the exact offset rule separate from later biome and terrain qualification:

```python
if config.uniform:
    offset_x = random.next_int(window)
else:
    offset_x = (random.next_int(window) + random.next_int(window)) // 2
```

The visualization includes villages, desert pyramids, jungle pyramids, swamp huts, pillager outposts, igloos, woodland mansions, ocean monuments, shipwrecks, ocean ruins, and ruined portals. Each family keeps its own spacing, separation, salt, and offset distribution. Pillager outposts additionally apply their source-level one-in-five roll and nearby-village exclusion. All other points remain candidate-stage positions, so the map does not silently apply gates that belong to later generation steps.

![Overworld Structure Generation](Plots/structure_placement.gif)

The square map spans 3,280 by 3,280 chunks, or 52,480 by 52,480 blocks, close to the full stronghold-ring view. It retains every in-bounds candidate for the displayed structure families. Faint 32-chunk lines provide a common reference, while the cyan outline shows the active structure region and the dashed fill shows its usable candidate window. A fixed central inset keeps the local placement geometry readable.

Candidates use compact, structure-specific symbols so all in-bounds points remain visible at the wider scale. The right side identifies every symbol and textured terrain family. Underneath them, seed 42 is sampled through the full Overworld biome graph and `WORLD_SURFACE_WG` height path. The background is exact coordinate context, but it is still not being used as a hidden structure filter.

### Nether Structure Generation

*Fortresses and bastions share a throw of the dice. Ruined portals bring their own.*

Nether fortresses and bastion remnants share one Java 1.16.1 candidate grid. Each 27 by 27 chunk region has a 23 by 23 candidate window. After the candidate offsets, Java Random draws

$$
r=\mathrm{nextInt}(5).
$$

Rolls 0 and 1 choose a fortress. Rolls 2, 3, and 4 choose a bastion:

$$
P(\mathrm{fortress})=\frac{2}{5},\qquad
P(\mathrm{bastion})=\frac{3}{5}.
$$

Ruined portals use an independent 25 by 25 region grid, a 15 by 15 candidate window, and a different seed path. Sharing a dimension does not mean sharing a random sequence.

[`Code/core/structures.py`](Code/core/structures.py) makes the shared type roll explicit:

```python
offset_x = random.next_int(23)
offset_z = random.next_int(23)
type_roll = random.next_int(5)
structure_type = 'fortress' if type_roll < 2 else 'bastion'
```

![Nether Structure Generation](Plots/multi_structure_generation.gif)

The square map uses the same 3,280 by 3,280 chunk span as the Overworld figure, which is 52,480 by 52,480 Nether blocks. Subtle red lines show every fifth shared fortress-or-bastion region boundary and purple dotted lines show every fifth independent ruined-portal boundary. The 256-chunk-wide detail inset is deliberately looser than the old close-up, giving both active grids room to breathe.

Compact symbols distinguish every displayed fortress, bastion, and ruined-portal candidate without hiding the wider field. The backdrop runs the four Java 1.16.1 Double Perlin fields and selects the nearest of the five multi-noise prototypes. Lava is separately sampled as negative density at Y=31, below the lava sea, and is never labeled as a sixth biome. Every exact candidate within the frame is retained: fortresses and ruined portals accept all five displayed biome classes, while bastions exclude basalt deltas at the later biome gate.

### Stronghold Ring Distribution

*The search begins as a spiral, long before an Eye of Ender enters the story.*

Strongholds do not use a rectangular random-spread grid. Java 1.16.1 advances through polar coordinates. A useful expression for the candidate radius in chunks is

$$
r_{i}=128+192i+\left(U-\frac12\right)80,
\qquad U\sim U(0,1).
$$

The ring index $i$ moves the baseline radius outward. The random term moves an individual candidate within that ring's radial band. Angles divide the ring among its candidates, and a seeded angular offset rotates the following ring.

The eight populations are

$$
3,\ 6,\ 10,\ 15,\ 21,\ 28,\ 36,\ 9,
$$

which sum to 128.

Starting from the world seed, Java Random chooses the initial angle and a radius inside the current ring band. Minecraft converts that polar position to chunk coordinates, rounds it, and searches up to 112 blocks for a valid biome before fixing the final start.

[`Code/core/strongholds.py`](Code/core/strongholds.py) preserves the ring iterator and Java rounding:

```python
radius_chunks = (
    4 * 32 + 6 * ring_index * 32
    + (random.next_double() - 0.5) * 32 * 2.5
)
x = java_round(math.cos(angle) * radius_chunks) * 16
z = java_round(math.sin(angle) * radius_chunks) * 16
```

![Stronghold Ring Distribution](Plots/stronghold_rings.png)

Panel (a) shows all 128 pre-biome-search candidates. Coloured radial bands show each ring's allowed range, while enlarged dots and stronger boundaries keep the exact geometry above the faded biome field.

Panel (b) is a coordinate-consistent view of the same seed near the first ring. Dashed gold circles show the 112-block biome-search neighbourhood around each preliminary candidate. The backdrop uses the same exact seed-42 biome and base-height sample as the structure overview without changing the stronghold arithmetic.

Panel (c) separates two quantities that were previously easy to confuse. Bar height is the number of candidates in a ring. The compact label beneath each ring gives its radial band in thousands of blocks. The first ring spans 1,408 to 2,688 blocks from the world origin.

## Quick Start

*Run the checks, step into `Code/`, and let the figures rebuild themselves.*

```bash
git clone https://github.com/IsolatedSingularity/Minecraft-Generation.git
cd Minecraft-Generation
pip install -r requirements.txt

# Run numerical and asset checks
python -m unittest discover -s tests -v

# Generate all active visualizations
cd Code
python render_all.py
```

Run `render_all.py` from inside `Code/`. The visualization modules use sibling imports, so invoking `python Code/render_all.py` from the repository root is not supported.

## Scope and Accuracy

*A useful picture should make its limits as visible as its result.*

> [!NOTE]
> Active mathematical visualizations target Java Edition 1.16.1. Java Random, candidate-region arithmetic, stronghold rings, dragon graph topology, Overworld biome layers, Nether multi-noise fields, base terrain density, End City height samples, and signed-integer overflow use source-faithful conventions. Chunk reveal timing, continuous two-dimensional dragon steering, and safe outer-gateway destinations remain explicit projections. No Bedrock behaviour is represented.

> [!IMPORTANT]
> The terrain maps never inject rare biomes for display. A biome appears only where the selected 1.16.1 seed places it. The legend may therefore contain a family that is absent from a particular crop.

> [!TIP]
> For speedrunning, first-ring stronghold candidates occur 1,408 to 2,688 blocks from origin. Eye triangulation still targets the final biome-adjusted stronghold, not merely the preliminary ring point.

## References

1. [Minecraft Wiki](https://minecraft.wiki/): mechanics, biome appearance, and historical version context.
2. [OpenJDK `java.util.Random`](https://github.com/openjdk/jdk/blob/master/src/java.base/share/classes/java/util/Random.java): Java LCG behaviour and bounded-integer rejection.
3. [Fabric Yarn 1.16.1 `ChunkStatus`](https://maven.fabricmc.net/docs/yarn-1.16.1%2Bbuild.10/net/minecraft/world/chunk/ChunkStatus.html): source-mapped chunk status order.
4. [Fabric Yarn 1.16.1 `StructureConfig`](https://maven.fabricmc.net/docs/yarn-1.16.1%2Bbuild.10/net/minecraft/world/gen/chunk/StructureConfig.html): spacing, separation, and salt configuration.
5. [Fabric Yarn 1.16.1 `StructureFeature`](https://maven.fabricmc.net/docs/yarn-1.16.1%2Bbuild.10/net/minecraft/world/gen/feature/StructureFeature.html): structure families and candidate-stage context.
6. [Fabric Yarn `EndCityFeature`](https://maven.fabricmc.net/docs/yarn-1.17.1%2Bbuild.10/net/minecraft/world/gen/feature/EndCityFeature.html): End City terrain-height qualification retained from the 1.16 generation family.
7. [Mojang MC-159283](https://bugs-legacy.mojang.com/browse/MC-159283): distant End terrain loss caused by integer overflow.
8. [Deltanic's End overflow derivation](https://gist.github.com/Deltanic/b98d005c9025f10a67de9e966fa57ebb): transition sequence linked from the Mojang issue.
9. [Alan Zucconi, Minecraft World Generation](https://www.alanzucconi.com/2022/06/05/minecraft-world-generation/): accessible procedural-generation background.
10. [Fabric Yarn 1.16.1 `PhaseType`](https://maven.fabricmc.net/docs/yarn-1.16.1%2Bbuild.10/net/minecraft/entity/boss/dragon/phase/PhaseType.html): the complete 11-type Ender Dragon phase taxonomy.

*Author: Jeffrey Morais*

## Legacy Simulations

The original dragon animation stays as a fossil from an earlier model. Its arena is abstract, its dashboard is larger, and its curves are freer than the source allows. That looseness helped motivate the current steering work, but the old graph, effects, and geometry are not evidence for any numerical claim above.

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
