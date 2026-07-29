# Minecraft Procedural Algorithms

<!-- Do not remove this comment. It is important. -->
<!-- The seeds remember those who query them. -->

###### A visual study of deterministic systems in Minecraft: Java Edition 1.16.1

![Ender Dragon pathfinding and phase state](Plots/dragon_pathfinding.gif)

*Figure 1. A scripted, single-loop tour of the dragon's 24-node path graph, phase state, and surviving End crystals. The animation is 1600 x 900 at 12 fps and is palette-optimized for GitHub.*

## Objective

Every Java world begins with a signed 64-bit seed. A 48-bit pseudorandom state, salted structure grids, noise samplers, graph search, and phase-specific rules turn that seed into a place that feels discovered rather than computed.

This repository studies those mechanisms through reproducible scientific visuals. It is not a replacement for the game engine. It isolates the part of each algorithm being discussed, states what is exact, and marks what has been simplified for legibility.

<table align="center">
<tr>
<td align="center"><b>2<sup>48</sup></b><br><sub>Java Random states</sub></td>
<td align="center"><b>24</b><br><sub>Dragon path nodes</sub></td>
<td align="center"><b>10</b><br><sub>End spikes</sub></td>
<td align="center"><b>20</b><br><sub>End gateways</sub></td>
<td align="center"><b>128</b><br><sub>Strongholds</sub></td>
<td align="center"><b>8</b><br><sub>Stronghold rings</sub></td>
</tr>
</table>

<p align="center">
  <img src="./Plots/apple.gif?raw=true" alt="apple" width="51" height="50" />
</p>

## Accuracy boundary

| System | Source-faithful component | Visual abstraction |
|---|---|---|
| Java Random | 48-bit recurrence, bounded `nextInt`, signed `nextLong`, region and population seed mixing | None |
| Dragon | 24 horizontal nodes, adjacency bitmasks, crystal-dependent node subset, weighted search, holding-phase rolls | Smooth top-down motion between targets, scripted hero sequence |
| End | Pillar-seed derivation, spike shuffle and geometry, gateway ring, seeded simplex qualification | Sample density and point size, schematic End City assembly |
| Structures | Java Random candidate chunks, salts, spacing, separation, fortress or bastion split | Biome and terrain start checks are outside the candidate plots |
| Strongholds | Seeded eight-ring candidate iterator | Final biome relocation and portal-room layout are outside the plot |
| Chunk loading | Java 1.16.1 status order and population-seed texture | The outward dependency wave is explanatory timing, not a profiler trace |

The visual seeds are fixed so screenshots and GIFs remain stable. Core functions accept other seeds for stochastic experiments.

## Java Random

Java's `Random` state follows

$$
X_{n+1} = (25214903917X_n + 11) \bmod 2^{48}.
$$

The public seed is first mixed with the multiplier, then each call advances the state and returns selected high bits. The implementation also reproduces rejection sampling for non-power-of-two `nextInt` bounds and Java's signed composition of `nextLong`.

Large-feature candidate grids use

$$
S_r = S_w + 341873128712R_x + 132897987541R_z + \sigma,
$$

where $S_w$ is the world seed, $(R_x,R_z)$ is the structure region, and $\sigma$ is the structure salt.

## Dragon pathfinding

The dragon has 24 horizontal path nodes arranged on radii 60, 40, and 20 blocks, with populations 12, 8, and 4. Their vertical coordinates are selected from the End heightmap in game. The figure preserves the source adjacency bitmasks and uses distance-weighted graph search. When no crystals remain, search is restricted to nodes 12 through 23.

During the holding phase, the landing-approach roll is

$$
P(\text{landing approach}) = \frac{1}{n_{\text{crystals}} + 3}.
$$

This is a roll at a decision point, not a constant probability per game tick. The right side of Figure 1 groups the readable phase state and crystal state without turning the animation into a dashboard.

### Phase details

<table>
<tr>
<th>Holding, strafing, charging</th>
<th>Landing approach and perch</th>
<th>Takeoff and return</th>
</tr>
<tr>
<td><img src="Plots/dragon_holding_strafe.gif" alt="Holding, strafing, and charging dragon path states" /></td>
<td><img src="Plots/dragon_landing_perch.gif" alt="Landing approach and perching dragon path states" /></td>
<td><img src="Plots/dragon_takeoff.gif" alt="Dragon takeoff path state" /></td>
</tr>
</table>

These short cuts keep the hero loop compact while preserving a closer look at each transition group.

### Seeded trajectory ensemble

![Accumulated dragon approach trajectories](Plots/dragon_trajectory_ensemble.gif)

*Figure 2. Four hundred and twenty seeded approaches accumulate into an occupancy field. Recent paths remain visible over the square-root-scaled density.*

This experiment was inspired by Curcuit's one-shot research in Minecraft speedrunning: generate many dragon approaches, study where paths concentrate, then tune the player setup around a repeatable interaction. The repository models the dragon-side path process. It does not claim to simulate the menu manipulation, arrow momentum, multipart hitbox interaction, or damage calculation of the damageless one-shot technique.

## End dimension

![End dimension overview](Plots/end_dimension_overview.png)

*Figure 3. (a) Seed-derived outer-End simplex samples. (b) Central island geometry, shuffled spikes, cages, and the 20 gateway directions. (c) A schematic End City piece assembly.*

Panel (a) keeps End stone as the dominant visual material. Its sites satisfy the Java 1.16 End source's radial and simplex-noise branch, but the plotted point size is a deterministic visual encoding rather than a block-perfect island boundary.

The central geometry uses ten spikes on radius 42. For shuffled value $v \in \{0,\ldots,9\}$,

$$
r = 2 + \left\lfloor\frac{v}{3}\right\rfloor, \qquad h = 76 + 3v.
$$

Values 1 and 2 are caged. The shuffle is seeded from the low 16 bits of the first Java `nextLong` drawn from the world seed. Twenty post-fight gateway directions lie on radius 96 with floored coordinates. The End City panel is a clean structural schematic of recursive towers, bridges, and ship branching, not a claim that this exact city belongs to the displayed seed.

## Chunk generation

![Chunk status dependency wave](Plots/seed_loading.gif)

*Figure 4. The official Java 1.16.1 status order moves across a chunk dependency wave. Color texture comes from exact population-seed mixing.*

The displayed sequence is `EMPTY`, `STRUCTURE_STARTS`, `STRUCTURE_REFERENCES`, `BIOMES`, `NOISE`, `SURFACE`, `CARVERS`, `LIQUID_CARVERS`, `FEATURES`, `LIGHT`, `SPAWN`, `HEIGHTMAPS`, and `FULL`. Labels are abbreviated in the GIF only to keep the visual clean. Wall-clock concurrency, task scheduling, disk access, and server load are intentionally not modeled.

## Village candidate placement

![Village candidate placement](Plots/structure_placement.gif)

*Figure 5. One exact village candidate per 32 x 32 chunk region. The inset isolates the current pair of `nextInt(24)` offsets.*

Java 1.16.1 villages use spacing 32, separation 8, and salt 10387312. A candidate is therefore selected inside a 24 x 24 chunk window:

$$
c_x = 32R_x + J_x, \qquad c_z = 32R_z + J_z, \qquad J_x,J_z \sim \texttt{nextInt}(24).
$$

Independent candidate points are deliberately unconnected. A later biome and structure-start check decides whether a village actually generates.

## Nether structures

![Nether fortress, bastion, and ruined portal candidates](Plots/multi_structure_generation.gif)

*Figure 6. Fortress and bastion candidates share one 27-chunk grid. Ruined portals use an independent 25-chunk grid. The inset relates the central Nether portal candidate to Overworld scale and first-ring strongholds.*

Fortresses and bastions share spacing 27, separation 4, and salt 30084232. After the two candidate offsets, `nextInt(5) < 2` selects a fortress; otherwise it selects a bastion. Ruined portals use spacing 25, separation 10, and salt 34222645. These are exact candidate-stage rules. Nether biome and piece-generation checks are not presented as completed structures.

*Same seed. Different random stream. Different fate.*

## Structure analysis

![Six-panel structure analysis](Plots/structure_analysis.png)

*Figure 7. (a) Village offset coverage. (b) The shared fortress and bastion candidate field. (c) Stronghold rings. (d) Empirical nearest-candidate distance. (e) Salt independence. (f) Two-throw and three-throw triangulation error under bearing noise.*

The seeds differ between analyses so the dashboard does not accidentally teach one aesthetically convenient seed as a general rule. Panel (e) reports an empirical correlation between village and ruined-portal offsets under their independent salts. Panel (f) is a Monte Carlo observation model, with bearing noise stated on the horizontal axis.

## Stronghold rings and triangulation

![Stronghold ring distribution and triangulation](Plots/stronghold_rings.png)

*Figure 8. (a) All 128 seeded candidates across eight rings. (b) Two bearing rays toward a first-ring target. (c) The distribution of noisy line intersections near that target.*

The ring populations are

$$
3,\ 6,\ 10,\ 15,\ 21,\ 28,\ 36,\ 9.
$$

For ring index $i$, the candidate radius in chunks is sampled around

$$
128 + 192i + \left(U - \frac{1}{2}\right)80,
$$

with an even angular step within each ring and a seeded rotation between rings. The first-ring candidate range is 1,408 to 2,688 blocks from world origin. Vanilla then searches around each candidate for an allowed biome, so the points are not portal-room coordinates.

*The ring is deterministic. The throws are measurements.*

## Reproduce the figures

```powershell
git clone https://github.com/IsolatedSingularity/Minecraft-Generation.git
cd Minecraft-Generation
py -3 -m pip install -r requirements.txt
py -3 Code/render_all.py
py -3 -m unittest discover -s tests -v
```

All active outputs use deterministic seeds. GIFs are rendered once, quantized to compact adaptive palettes, and stored as single loops. The hero retains a 1600 x 900 canvas while dropping from roughly 51 MB to under 3 MB.

## Primary technical references

1. [OpenJDK 8 `java.util.Random` source](https://github.com/openjdk/jdk8u/blob/master/jdk/src/share/classes/java/util/Random.java)
2. [Fabric Yarn 1.16.1 `EnderDragonEntity`](https://maven.fabricmc.net/docs/yarn-1.16.1%2Bbuild.10/net/minecraft/entity/boss/dragon/EnderDragonEntity.html)
3. [Fabric Yarn 1.16.1 `ChunkStatus`](https://maven.fabricmc.net/docs/yarn-1.16.1%2Bbuild.10/net/minecraft/world/chunk/ChunkStatus.html)
4. [Fabric Yarn 1.16.1 `EndSpikeFeature`](https://maven.fabricmc.net/docs/yarn-1.16.1%2Bbuild.10/net/minecraft/world/gen/feature/EndSpikeFeature.html)
5. [FeatureUtils fortress source](https://github.com/KaptainWutax/FeatureUtils/blob/a271711f58e283547634ca31e466b9b8b0e5d825/src/main/java/kaptainwutax/featureutils/structure/Fortress.java), [bastion source](https://github.com/KaptainWutax/FeatureUtils/blob/a271711f58e283547634ca31e466b9b8b0e5d825/src/main/java/kaptainwutax/featureutils/structure/BastionRemnant.java), and [ruined portal source](https://github.com/KaptainWutax/FeatureUtils/blob/a271711f58e283547634ca31e466b9b8b0e5d825/src/main/java/kaptainwutax/featureutils/structure/RuinedPortal.java)
6. [BiomeUtils End source](https://github.com/KaptainWutax/BiomeUtils/blob/166f1757be3e1e036f0b25f9c063df3f863a1c49/src/main/java/kaptainwutax/biomeutils/source/EndBiomeSource.java) and [NoiseUtils simplex sampler](https://github.com/KaptainWutax/NoiseUtils/blob/2cf64e1d2e7e674fbf5b7247f16e8dc56ae2a31c/src/main/java/kaptainwutax/noiseutils/simplex/SimplexNoiseSampler.java)
7. [Curcuit's dragon one-shot research notes](https://docs.google.com/document/d/11qCI-BhZftr7tA1qvEp3VScY3ki2aVsC52eJy2Psp4Q)

The Yarn pages provide names and API structure for Mojang classes. The KaptainWutax projects are readable community source ports used to cross-check constants and random-call order.

---

*Author: Jeffrey Morais*

> [!TIP]
> First-ring strongholds begin 1,408 blocks from origin. Two Eye of Ender bearings define an intersection; a third provides a useful consistency check when measurement noise is nontrivial.

> [!NOTE]
> Active figures target Java Edition 1.16.1. Bedrock Edition and newer structure-set behavior are outside scope.

<details>
<summary>&#128220; The Scroll of Forbidden Knowledge</summary>

```text
The ancient texts speak of seeds most cursed:

Seed 164311266871034  Where villages fear to spawn
Seed 1785852800490    The stronghold that wasn't
Seed 27594263         Portal room behind bedrock

Some seeds are best left unplanted.

Also, did you know Herobrine's removal was never actually implemented?
The changelog lies. He watches through the noise field.
Always 3 chunks behind. Always listening for footsteps.

The generation is deterministic.
Your survival is not.

The dragon has circled 2^48 times before.
It will circle 2^48 times again.
You are merely the current observer.

Some say that when the LCG state equals your world seed,
you can hear the algorithm thinking.

But that is just superstition.

Isn't it?

From the Ender Tongue archives, circa 2011
```

</details>
