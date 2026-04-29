# Minecraft 1.16 Dimension Details

Source: minecraft.wiki. Target version: **Java Edition 1.16** (Nether Update).

## The Nether (1.16: Nether Update)

### Terrain
- Height: 128 blocks (Y=0 to Y=127)
- Bedrock floor: Y=0-4 (random pattern)
- Bedrock ceiling: Y=123-127 (random pattern)
- Lava sea level: Y=31
- Build limit: Y=256 (can build above ceiling)
- Terrain shape is **independent of biome distribution**; biomes alter surface material and add terrain features to the base netherrack terrain
- 1:8 coordinate ratio with Overworld (X/Z divided by 8)
- Cave-like terrain with no sky, constant dim ambient light
- Fog color varies by biome (red in wastes, blue in warped forest, etc.)

### Biomes (5 total, added/renamed in 1.16)

| Biome | Distribution | Description | Mobs |
|-------|-------------|-------------|------|
| Nether Wastes | 36.30% | Original Nether biome (renamed from "Nether"). Netherrack terrain, glowstone ceiling clusters, lava springs, gravel/soul sand shores | Ghasts, Zombified Piglins, Magma Cubes, Piglins, Endermen, Striders |
| Crimson Forest | 22.22% | Dense huge crimson fungi, nether wart block canopy, weeping vines, crimson nylium floor, shroomlights | Hoglins, Piglins, Zombified Piglins, Striders |
| Soul Sand Valley | 17.08% | Vast grottos of soul sand/soul soil, exposed nether fossils (bone blocks), basalt pillars, soul fire, blue fog | Skeletons, Ghasts, Endermen, Striders |
| Basalt Deltas | 15.86% | Volcanic remnants. Basalt + blackstone terrain, chaotic/uneven surface, magma block deltas, constrained lava pools, gray fog, ash particles | Magma Cubes (high rate), Ghasts |
| Warped Forest | 8.54% | Blue variant of crimson forest. Huge warped fungi, warped nylium floor, twisting vines, nether sprouts, shroomlights, magenta-purple fog. No music plays | Endermen, Striders |

### Nether Structures (1.16)

| Structure | Biomes | Description |
|-----------|--------|-------------|
| Nether Fortress | All biomes | Nether brick castle, bridges over lava. Blaze spawners, nether wart farms. Only source of blazes + wither skeletons |
| Bastion Remnant | All except Basalt Deltas | Blackstone/basalt fortifications, 4 variants (Bridge, Hoglin Stable, Housing Units, Treasure Room). Piglins, piglin brutes, hoglins. Gold blocks, loot |
| Nether Fossil | Soul Sand Valley only | Bone block structures, 14 designs |
| Ruined Portal | All biomes | Shattered portal frames with crying obsidian, gold blocks, loot chest |

### Nether Terrain Features

| Feature | Description |
|---------|-------------|
| Lava Sea | At and below Y=31 in all biomes. Striders spawn here |
| Glowstone Blob | Ceiling clusters in all biomes |
| Basalt Pillar | Floor-to-ceiling pillars in soul sand valleys |
| Delta | Constrained lava + magma block sheets in basalt deltas |
| Hidden Lava | Single lava sources buried in netherrack, all biomes |

### Nether Ores (1.16)
| Ore | Y Range | Notes |
|-----|---------|-------|
| Ancient Debris | All altitudes, peak at Y=8-22 | Max 2 veins/chunk (size 1-3) + bonus vein (size 1-2). Same blast resistance as obsidian. Always buried (no exposed faces) |
| Nether Gold Ore | 10-117 | 10 attempts/chunk, vein size 10 |
| Nether Quartz Ore | 10-117 | 16 attempts/chunk, vein size 14 |

## The End (1.16)

### Terrain
- Composed entirely of End stone
- No daylight cycle, constant dim light
- No weather
- Beds and respawn anchors explode when used
- Void below the islands

### Structure
- **Central Island**: Large End stone landmass centered near (0, 0)
  - Exit portal at (0, 64, 0) with bedrock frame
  - 10 obsidian pillars (End spikes) arranged in 42-block radius circle
  - End crystals atop each pillar (some caged in iron bars)
  - Obsidian platform at (100, 49, 0) where players spawn
- **Void Gap**: ~1000 blocks of empty void between central and outer islands
- **Outer Islands**: Scattered End stone landmasses beyond the gap
  - Vary in size from tiny to several hundred blocks wide
  - Chorus trees on some islands
  - End cities (with optional End ships) generate in End Highlands

### End Biomes (5 in Java Edition)
| Biome | Location | Description |
|-------|----------|-------------|
| The End | Central island | Main boss arena |
| Small End Islands | Near outer edge | Tiny scattered islets |
| End Midlands | Outer islands | Medium terrain, no cities |
| End Highlands | Outer islands | Tallest terrain, End cities generate here |
| End Barrens | Outer island edges | Thin, barren island edges |

### End Terrain Generation
- Central island: simplex noise-based with erosion factor
- Outer islands: generated chunk-by-chunk using noise threshold
  - `if (noise * 100 - sqrt(x^2 + z^2) > -100)` determines island placement
- Overflow bug creates concentric ring gaps:
  - First void ring at X/Z = +/-370,720
  - Second ring starts at +/-524,288
  - Pattern continues with diminishing gaps to world boundary
  - Condition: `sin(X^2 + Z^2) / 43748131634 > 0` AND `X^2 + Z^2 - 1,000,000 > 0`
- 20 End gateways generated (one per dragon kill), at radius 96 from center

### End Structures
| Structure | Description |
|-----------|-------------|
| Exit Portal | Bedrock frame at (0, 64, 0), activated on dragon death |
| End Spike | 10 obsidian pillars at radius 42, varying heights |
| Obsidian Platform | 5x5 obsidian at (100, 49, 0), regenerated on entry |
| End Gateway | Bedrock portals linking central to outer islands |
| End City | Tower structures in End Highlands, purpur blocks |
| End Ship | Floating ship attached to some End Cities, contains elytra |
| Chorus Tree | Chorus plant + flower, grows on End stone |

### End Spike Details (Obsidian Pillars)
- 10 pillars on a 42-block radius circle centered at (0, 0)
- Absolute XZ coordinates (identical for all seeds): (42,0), (33,24), (12,39), (-12,39), (-33,24), (-42,0), (-33,-24), (-12,-39), (12,-39), (33,-24)
- Pillar radii: 3 to 6 blocks
- Top Y values: 76 to 103
- Total obsidian blocks across all pillars: ~40,499
- Each has bedrock block at top with End crystal
- Some pillars have iron bar cages around the crystal
- On dragon respawn: 10-block radius cube centered on top bedrock (down to Y=66) is wiped, pillars/crystals restored

### Exit Portal (End Fountain)
- Always at (0, 0) in the End
- 7-block wide bedrock bowl with central bedrock pillar
- 4 torches on the pillar
- 16 end stone blocks beneath outer bedrock ring
- Frame generates on first End entry; portal blocks activate on dragon death
- First kill: dragon egg placed atop central pillar
- Y-level set to highest block at (0, 0); may clip at height limit

### End Gateway Details
- Up to 20 gateways, one generated per dragon kill
- All 20 form a circle at Y=75, radius 96 from (0, 0)
- Placement order randomized per world seed
- Each central gateway links to outer island location with paired return gateway
- Tracked in `level.dat` under `DimensionData -> 1 -> DragonFight -> Gateways`
- Bug MC-104736: incorrect DragonFight data regeneration could duplicate gateways

## Ender Dragon AI (1.16)

### Stats
- Health: 200 HP
- Hitbox: Height 8 blocks, Width 16 blocks
- 8 damageable sub-hitboxes (head, body, tail segments, wings)
- Head takes full damage; other parts reduce damage by ~75% (effective 800 HP if never hit on head)
- Immune to: fire, lava, status effects, suffocation, arrows while perched
- Healed by End crystals: 1 HP per 0.5 seconds per crystal

### Dragon Phases (DragonPhase NBT tag)

| Phase | ID | Description |
|-------|----|-------------|
| Circling | 0 | Default flight pattern around arena |
| Strafing | 1 | Preparing to shoot fireball at player |
| Flying to Portal | 2 | Transitioning to land (approach) |
| Landing | 3 | Touch-down animation on exit portal |
| Taking Off | 4 | Launching from portal back to circling |
| Breath Attack | 5 | Perched, breathing purple particle cloud (Harming) |
| Searching | 6 | Perched, looking for player to target breath |
| Roaring | 7 | Perched, roar animation before breath attack |
| Charging | 8 | Dive-bombing at a player |
| Flying to Die | 9 | Flying to exit portal for death animation |
| Hovering | 10 | Default /summon state, flapping in place |

### Behavioral State Machine
Four main states with transitions:

1. **Guarding** (Phase 0): Circles the arena at Y~100. Transitions to Strafing or Landing based on proximity/crystal state.

2. **Strafing** (Phase 1): Flies toward player, fires dragon fireball. Fireball creates lingering Harming effect cloud on impact. One fireball per destroyed crystal (retaliation). Returns to Circling after attack.

3. **Perching** (Phases 2->3->6->7->5->4): 
   - Flies to portal (2), lands (3)
   - Searches for player within 20 blocks (6)
   - Roars for 1.25 seconds (7)
   - Breath attack for 3 seconds, creating damage cloud (5)
   - Takes off (4), returns to Circling
   - Perch probability: `P = 1 / (3 + activeCrystals)`

4. **Charging** (Phase 8): Dives at player. Deals 6/10/15 HP damage (Easy/Normal/Hard). Flings player into air. Returns to Circling.

### Dragon Fireball
- Projectile entity fired during Strafing
- Creates area effect cloud on impact (lingering Harming)
- Cloud: horizontal disc, ~5-6 blocks diameter visible, ~3-4 blocks damage zone
- Does NOT shrink when affecting mobs (unlike lingering potions)
- Can be collected with glass bottle for dragon's breath

### Combat Damage Values
| Attack | Easy | Normal | Hard |
|--------|------|--------|------|
| Head contact | 6 HP | 10 HP | 15 HP |
| Wing contact | 3.5 HP | 5 HP | 7.5 HP |
| Dragon's breath | 3 HP/sec | 3 HP/sec | 3 HP/sec |

### Death Sequence
1. Dragon flies to exit portal (Phase 9)
2. Rises into sky, bright light beams from body
3. Explodes after 200 ticks
4. Drops 12,000 XP (first kill) or 500 XP (subsequent)
5. Activates exit portal
6. Generates one End gateway (first 20 kills only)
7. Dragon egg appears atop portal (first kill only in JE)

### Known 1.16 Bug
MC-272431 / MC-271337: Dragon's vertical velocity calculation has a typo introduced in 19w08b (1.14 snapshot), causing erratic flight behavior. The dragon no longer dives directly to the fountain; instead it slowly descends. This bug persists through 1.16.

### Dragon Java Architecture
- Main class: `net.minecraft.world.entity.boss.enderdragon.EnderDragon` (extends Mob, Enemy)
- Fight controller: `net.minecraft.world.level.dimension.end.EndDragonFight`
- Phase classes: `net.minecraft.world.entity.boss.enderdragon.phases.*`
- Phase manager: `EnderDragonPhaseManager` holds current phase, calls `doServerTick()` each tick
- Network-synced phase ID: `EntityDataAccessor<Integer> DATA_PHASE`

### Dragon Hitbox Sub-Entities
- `EnderDragonPart` instances: **head**, **neck**, **body**, **tail** (3 segments), **wings** (2)
- Each is a separate collision entity for localized damage and knockback
- Head: full damage. Other parts: ~25% damage (effective 800 HP if head never hit)

### Dragon Pathfinding System
- 24 static pathfinding nodes hardcoded around central End island
- Nodes aligned with obsidian pillar coordinates (42/0, 33/24, 12/39, etc.) and exit portal
- NOT dynamic: nodes do not adapt to terrain changes or player-placed blocks
- A*-like pathfinding between nodes using `BinaryHeap openSet`
- Flight trajectory: smoothed spline between current position and destination node
- Target altitude: randomized 0-20 blocks above destination node (organic flight variation)
- Node Y-level: engine scans downward from build limit to highest block at that XZ

### Pathfinding Exploits and Bugs
- Blocks at Y=255 on node XZ coordinates corrupt spline trajectory; dragon enters infinite stutter loop
- Velocity bug detail (MC-272431): `ydd * 0.01D` should be `ydd * 0.1D` in `aiStep` method
  - Missing magnitude in vertical velocity damping
  - Dragon cannot shed altitude during LAND_ON_PORTAL approach
  - Causes horizontal gliding instead of smooth descent

### Segment Circular Buffer
- `double[][] positions` stores past dragon positions in circular buffer
- Smooths body segment animation: neck, tail follow head with tick delay
- Rendering interpolates segment positions from buffer history
