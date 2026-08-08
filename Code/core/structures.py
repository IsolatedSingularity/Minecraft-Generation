"""Exact candidate-stage structure placement for Java 1.16.1."""

from dataclasses import dataclass

from .constants import (
    CHUNK_SIZE,
    DESERT_PYRAMID_SALT,
    JUNGLE_PYRAMID_SALT,
    NETHER_RUINED_PORTAL_SEPARATION,
    NETHER_RUINED_PORTAL_SPACING,
    NETHER_STRUCTURE_SALT,
    NETHER_STRUCTURE_SEPARATION,
    NETHER_STRUCTURE_SPACING,
    PILLAGER_OUTPOST_SALT,
    RUINED_PORTAL_SALT,
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

OVERWORLD_STRUCTURES = (
    VILLAGE,
    DESERT_PYRAMID,
    JUNGLE_PYRAMID,
    SWAMP_HUT,
    PILLAGER_OUTPOST,
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
}
NETHER_SHARED = StructureConfig(
    'nether_shared', NETHER_STRUCTURE_SPACING,
    NETHER_STRUCTURE_SEPARATION, NETHER_STRUCTURE_SALT,
)
NETHER_RUINED_PORTAL = StructureConfig(
    'nether_ruined_portal', NETHER_RUINED_PORTAL_SPACING,
    NETHER_RUINED_PORTAL_SEPARATION, RUINED_PORTAL_SALT,
)


def candidate_in_region(world_seed, region_x, region_z, config):
    """Return the exact uniform-grid candidate before biome validation."""
    region_seed = generate_region_seed(
        world_seed, region_x, region_z, config.salt,
    )
    random = MinecraftLCG(region_seed)
    window = config.spacing - config.separation
    offset_x = random.next_int(window)
    offset_z = random.next_int(window)
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
    """Generate a square grid of exact uniform candidates."""
    return [
        candidate_in_region(world_seed, region_x, region_z, config)
        for region_x in range(-region_radius, region_radius + 1)
        for region_z in range(-region_radius, region_radius + 1)
    ]
