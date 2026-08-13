"""Exact candidate-stage structure placement for Java 1.16.1."""

from dataclasses import dataclass

from .constants import (
    CHUNK_SIZE,
    DESERT_PYRAMID_SALT,
    END_CITY_SALT,
    END_CITY_SEPARATION,
    END_CITY_SPACING,
    IGLOO_SALT,
    IGLOO_SEPARATION,
    IGLOO_SPACING,
    MANSION_SALT,
    MANSION_SEPARATION,
    MANSION_SPACING,
    MONUMENT_SALT,
    MONUMENT_SEPARATION,
    MONUMENT_SPACING,
    JUNGLE_PYRAMID_SALT,
    NETHER_RUINED_PORTAL_SEPARATION,
    NETHER_RUINED_PORTAL_SPACING,
    NETHER_STRUCTURE_SALT,
    NETHER_STRUCTURE_SEPARATION,
    NETHER_STRUCTURE_SPACING,
    PILLAGER_OUTPOST_SALT,
    OCEAN_RUIN_SALT,
    OCEAN_RUIN_SEPARATION,
    OCEAN_RUIN_SPACING,
    OVERWORLD_RUINED_PORTAL_SEPARATION,
    OVERWORLD_RUINED_PORTAL_SPACING,
    RUINED_PORTAL_SALT,
    SHIPWRECK_SALT,
    SHIPWRECK_SEPARATION,
    SHIPWRECK_SPACING,
    SWAMP_HUT_SALT,
    VILLAGE_SALT,
    VILLAGE_SEPARATION,
    VILLAGE_SPACING,
)
from .lcg import MinecraftLCG, generate_region_seed


@dataclass(frozen=True)
class StructureConfig:
    name: str
    spacing: int
    separation: int
    salt: int
    uniform: bool = False


VILLAGE = StructureConfig(
    'village', VILLAGE_SPACING, VILLAGE_SEPARATION, VILLAGE_SALT,
)
DESERT_PYRAMID = StructureConfig(
    'desert_pyramid', VILLAGE_SPACING, VILLAGE_SEPARATION,
    DESERT_PYRAMID_SALT,
)
JUNGLE_PYRAMID = StructureConfig(
    'jungle_pyramid', VILLAGE_SPACING, VILLAGE_SEPARATION,
    JUNGLE_PYRAMID_SALT,
)
SWAMP_HUT = StructureConfig(
    'swamp_hut', VILLAGE_SPACING, VILLAGE_SEPARATION, SWAMP_HUT_SALT,
)
PILLAGER_OUTPOST = StructureConfig(
    'pillager_outpost', VILLAGE_SPACING, VILLAGE_SEPARATION,
    PILLAGER_OUTPOST_SALT,
)
IGLOO = StructureConfig(
    'igloo', IGLOO_SPACING, IGLOO_SEPARATION, IGLOO_SALT,
)
WOODLAND_MANSION = StructureConfig(
    'woodland_mansion', MANSION_SPACING, MANSION_SEPARATION, MANSION_SALT,
)
OCEAN_MONUMENT = StructureConfig(
    'ocean_monument', MONUMENT_SPACING, MONUMENT_SEPARATION,
    MONUMENT_SALT, uniform=True,
)
SHIPWRECK = StructureConfig(
    'shipwreck', SHIPWRECK_SPACING, SHIPWRECK_SEPARATION, SHIPWRECK_SALT,
)
OCEAN_RUIN = StructureConfig(
    'ocean_ruin', OCEAN_RUIN_SPACING, OCEAN_RUIN_SEPARATION, OCEAN_RUIN_SALT,
)
OVERWORLD_RUINED_PORTAL = StructureConfig(
    'ruined_portal', OVERWORLD_RUINED_PORTAL_SPACING,
    OVERWORLD_RUINED_PORTAL_SEPARATION, RUINED_PORTAL_SALT,
)

OVERWORLD_STRUCTURES = (
    VILLAGE,
    DESERT_PYRAMID,
    JUNGLE_PYRAMID,
    SWAMP_HUT,
    PILLAGER_OUTPOST,
    IGLOO,
    WOODLAND_MANSION,
    OCEAN_MONUMENT,
    SHIPWRECK,
    OCEAN_RUIN,
    OVERWORLD_RUINED_PORTAL,
)

OVERWORLD_STRUCTURE_BIOMES = {
    'village': frozenset({
        'plains', 'desert', 'savanna', 'taiga', 'snowy_tundra',
    }),
    'desert_pyramid': frozenset({'desert'}),
    'jungle_pyramid': frozenset({'jungle'}),
    'swamp_hut': frozenset({'swamp'}),
    'pillager_outpost': frozenset({
        'plains', 'desert', 'savanna', 'taiga', 'snowy_tundra',
    }),
    'igloo': frozenset({'snowy_tundra'}),
    'woodland_mansion': frozenset({'dark_forest'}),
    'ocean_monument': frozenset({'water', 'deep_water'}),
    'shipwreck': frozenset({'shore', 'water', 'deep_water'}),
    'ocean_ruin': frozenset({'water', 'deep_water'}),
    'ruined_portal': frozenset({
        'shore', 'plains', 'forest', 'dark_forest', 'desert', 'savanna',
        'jungle', 'swamp', 'taiga', 'snowy_tundra', 'mountains',
        'badlands', 'mushroom_fields', 'water', 'deep_water',
    }),
}
NETHER_SHARED = StructureConfig(
    'nether_shared', NETHER_STRUCTURE_SPACING,
    NETHER_STRUCTURE_SEPARATION, NETHER_STRUCTURE_SALT,
)
NETHER_RUINED_PORTAL = StructureConfig(
    'nether_ruined_portal', NETHER_RUINED_PORTAL_SPACING,
    NETHER_RUINED_PORTAL_SEPARATION, RUINED_PORTAL_SALT,
)
END_CITY = StructureConfig(
    'end_city', END_CITY_SPACING, END_CITY_SEPARATION,
    END_CITY_SALT, uniform=True,
)


def candidate_in_region(world_seed, region_x, region_z, config):
    """Return an exact random-spread candidate before biome validation."""
    region_seed = generate_region_seed(
        world_seed, region_x, region_z, config.salt,
    )
    random = MinecraftLCG(region_seed)
    window = config.spacing - config.separation
    if config.uniform:
        offset_x = random.next_int(window)
        offset_z = random.next_int(window)
    else:
        offset_x = (random.next_int(window) + random.next_int(window)) // 2
        offset_z = (random.next_int(window) + random.next_int(window)) // 2
    chunk_x = region_x * config.spacing + offset_x
    chunk_z = region_z * config.spacing + offset_z
    return {
        'name': config.name,
        'region_x': region_x,
        'region_z': region_z,
        'region_seed': region_seed,
        'offset_x': offset_x,
        'offset_z': offset_z,
        'chunk_x': chunk_x,
        'chunk_z': chunk_z,
        'block_x': chunk_x * CHUNK_SIZE + CHUNK_SIZE // 2,
        'block_z': chunk_z * CHUNK_SIZE + CHUNK_SIZE // 2,
        'window': window,
    }


def structure_biome_compatible(structure_name, biome_name):
    """Return whether the illustrative biome category can host a structure."""
    return biome_name in OVERWORLD_STRUCTURE_BIOMES[structure_name]


def pillager_outpost_source_gate(world_seed, chunk_x, chunk_z):
    """Apply the Java 1.16.1 outpost 1/5 roll and village exclusion.

    This is the structure-specific gate in ``PillagerOutpostFeature``. Biome
    qualification remains a separate generator/configuration concern.
    """
    section_x = int(chunk_x) >> 4
    section_z = int(chunk_z) >> 4
    random = MinecraftLCG(section_x ^ (section_z << 4) ^ int(world_seed))
    random.next_int()
    if random.next_int(5) != 0:
        return False

    minimum_x = int(chunk_x) - 10
    maximum_x = int(chunk_x) + 10
    minimum_z = int(chunk_z) - 10
    maximum_z = int(chunk_z) + 10
    first_region_x = minimum_x // VILLAGE.spacing - 1
    last_region_x = maximum_x // VILLAGE.spacing + 1
    first_region_z = minimum_z // VILLAGE.spacing - 1
    last_region_z = maximum_z // VILLAGE.spacing + 1
    for region_x in range(first_region_x, last_region_x + 1):
        for region_z in range(first_region_z, last_region_z + 1):
            village = candidate_in_region(
                world_seed, region_x, region_z, VILLAGE,
            )
            if (
                minimum_x <= village['chunk_x'] <= maximum_x
                and minimum_z <= village['chunk_z'] <= maximum_z
            ):
                return False
    return True


def nether_shared_candidate(world_seed, region_x, region_z):
    """Return the shared fortress or bastion candidate and its exact split."""
    region_seed = generate_region_seed(
        world_seed, region_x, region_z, NETHER_SHARED.salt,
    )
    random = MinecraftLCG(region_seed)
    window = NETHER_SHARED.spacing - NETHER_SHARED.separation
    offset_x = random.next_int(window)
    offset_z = random.next_int(window)
    type_roll = random.next_int(5)
    structure_type = 'fortress' if type_roll < 2 else 'bastion'
    chunk_x = region_x * NETHER_SHARED.spacing + offset_x
    chunk_z = region_z * NETHER_SHARED.spacing + offset_z
    return {
        'name': structure_type,
        'region_x': region_x,
        'region_z': region_z,
        'region_seed': region_seed,
        'offset_x': offset_x,
        'offset_z': offset_z,
        'type_roll': type_roll,
        'chunk_x': chunk_x,
        'chunk_z': chunk_z,
        'block_x': chunk_x * CHUNK_SIZE + CHUNK_SIZE // 2,
        'block_z': chunk_z * CHUNK_SIZE + CHUNK_SIZE // 2,
        'window': window,
    }


def region_grid(world_seed, region_radius, config):
    """Generate a square grid of exact random-spread candidates."""
    return [
        candidate_in_region(world_seed, region_x, region_z, config)
        for region_x in range(-region_radius, region_radius + 1)
        for region_z in range(-region_radius, region_radius + 1)
    ]
