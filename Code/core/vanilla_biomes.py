"""Source-faithful Java 1.16.1 biome selection.

The Overworld sampler ports the complete ``BiomeLayers.build`` graph.  The
Nether sampler ports ``MultiNoiseBiomeSource`` and its four Double-Perlin
fields.  Coordinates passed to these samplers are biome-noise coordinates,
which are block coordinates divided by four.
"""

from functools import lru_cache

import numpy as np

from .lcg import MinecraftLCG, to_signed_long
from .vanilla_noise import DoublePerlinNoiseSampler, PerlinNoiseSampler


BIOME_NAMES = {
    0: 'ocean', 1: 'plains', 2: 'desert', 3: 'mountains', 4: 'forest',
    5: 'taiga', 6: 'swamp', 7: 'river', 8: 'nether_wastes', 9: 'the_end',
    10: 'frozen_ocean', 11: 'frozen_river', 12: 'snowy_tundra',
    13: 'snowy_mountains', 14: 'mushroom_fields',
    15: 'mushroom_field_shore', 16: 'beach', 17: 'desert_hills',
    18: 'wooded_hills', 19: 'taiga_hills', 20: 'mountain_edge',
    21: 'jungle', 22: 'jungle_hills', 23: 'jungle_edge', 24: 'deep_ocean',
    25: 'stone_shore', 26: 'snowy_beach', 27: 'birch_forest',
    28: 'birch_forest_hills', 29: 'dark_forest', 30: 'snowy_taiga',
    31: 'snowy_taiga_hills', 32: 'giant_tree_taiga',
    33: 'giant_tree_taiga_hills', 34: 'wooded_mountains', 35: 'savanna',
    36: 'savanna_plateau', 37: 'badlands', 38: 'wooded_badlands_plateau',
    39: 'badlands_plateau', 40: 'small_end_islands', 41: 'end_midlands',
    42: 'end_highlands', 43: 'end_barrens', 44: 'warm_ocean',
    45: 'lukewarm_ocean', 46: 'cold_ocean', 47: 'deep_warm_ocean',
    48: 'deep_lukewarm_ocean', 49: 'deep_cold_ocean',
    50: 'deep_frozen_ocean', 127: 'the_void', 129: 'sunflower_plains',
    130: 'desert_lakes', 131: 'gravelly_mountains', 132: 'flower_forest',
    133: 'taiga_mountains', 134: 'swamp_hills', 140: 'ice_spikes',
    149: 'modified_jungle', 151: 'modified_jungle_edge',
    155: 'tall_birch_forest', 156: 'tall_birch_hills',
    157: 'dark_forest_hills', 158: 'snowy_taiga_mountains',
    160: 'giant_spruce_taiga', 161: 'giant_spruce_taiga_hills',
    162: 'modified_gravelly_mountains', 163: 'shattered_savanna',
    164: 'shattered_savanna_plateau', 165: 'eroded_badlands',
    166: 'modified_wooded_badlands_plateau',
    167: 'modified_badlands_plateau', 168: 'bamboo_jungle',
    169: 'bamboo_jungle_hills', 170: 'soul_sand_valley',
    171: 'crimson_forest', 172: 'warped_forest', 173: 'basalt_deltas',
}

BIOME_DEPTH_SCALE = {
    0: (-1.0, 0.1), 1: (0.125, 0.05), 2: (0.125, 0.05),
    3: (1.0, 0.5), 4: (0.1, 0.2), 5: (0.2, 0.2),
    6: (-0.2, 0.1), 7: (-0.5, 0.0), 10: (-1.0, 0.1),
    11: (-0.5, 0.0), 12: (0.125, 0.05), 13: (0.45, 0.3),
    14: (0.2, 0.3), 15: (0.0, 0.025), 16: (0.0, 0.025),
    17: (0.45, 0.3), 18: (0.45, 0.3), 19: (0.45, 0.3),
    20: (0.8, 0.3), 21: (0.1, 0.2), 22: (0.45, 0.3),
    23: (0.1, 0.2), 24: (-1.8, 0.1), 25: (0.1, 0.8),
    26: (0.0, 0.025), 27: (0.1, 0.2), 28: (0.45, 0.3),
    29: (0.1, 0.2), 30: (0.2, 0.2), 31: (0.45, 0.3),
    32: (0.2, 0.2), 33: (0.45, 0.3), 34: (1.0, 0.5),
    35: (0.125, 0.05), 36: (1.5, 0.025), 37: (0.1, 0.2),
    38: (1.5, 0.025), 39: (1.5, 0.025), 44: (-1.0, 0.1),
    45: (-1.0, 0.1), 46: (-1.0, 0.1), 47: (-1.8, 0.1),
    48: (-1.8, 0.1), 49: (-1.8, 0.1), 50: (-1.8, 0.1),
    129: (0.125, 0.05), 130: (0.225, 0.25), 131: (1.0, 0.5),
    132: (0.1, 0.4), 133: (0.3, 0.4), 134: (-0.1, 0.3),
    140: (0.425, 0.45000002), 149: (0.2, 0.4), 151: (0.2, 0.4),
    155: (0.2, 0.4), 156: (0.55, 0.5), 157: (0.2, 0.4),
    158: (0.3, 0.4), 160: (0.2, 0.2), 161: (0.2, 0.2),
    162: (1.0, 0.5), 163: (0.3625, 1.225),
    164: (1.05, 1.2125001), 165: (0.1, 0.2), 166: (0.45, 0.3),
    167: (0.45, 0.3), 168: (0.1, 0.2), 169: (0.45, 0.3),
}

OVERWORLD_FAMILY = {
    0: 'water', 1: 'plains', 2: 'desert', 3: 'mountains', 4: 'forest',
    5: 'taiga', 6: 'swamp', 7: 'water', 10: 'water', 11: 'water',
    12: 'snowy_tundra', 13: 'snowy_tundra', 14: 'mushroom_fields',
    15: 'mushroom_fields', 16: 'shore', 17: 'desert', 18: 'forest',
    19: 'taiga', 20: 'mountains', 21: 'jungle', 22: 'jungle',
    23: 'jungle', 24: 'deep_water', 25: 'shore', 26: 'shore',
    27: 'forest', 28: 'forest', 29: 'dark_forest', 30: 'taiga',
    31: 'taiga', 32: 'taiga', 33: 'taiga', 34: 'mountains',
    35: 'savanna', 36: 'savanna', 37: 'badlands', 38: 'badlands',
    39: 'badlands', 44: 'water', 45: 'water', 46: 'water',
    47: 'deep_water', 48: 'deep_water', 49: 'deep_water', 50: 'deep_water',
    129: 'plains', 130: 'desert', 131: 'mountains', 132: 'forest',
    133: 'taiga', 134: 'swamp', 140: 'snowy_tundra', 149: 'jungle',
    151: 'jungle', 155: 'forest', 156: 'forest', 157: 'dark_forest',
    158: 'taiga', 160: 'taiga', 161: 'taiga', 162: 'mountains',
    163: 'savanna', 164: 'savanna', 165: 'badlands', 166: 'badlands',
    167: 'badlands', 168: 'jungle', 169: 'jungle',
}

OCEANS = {0, 10, 24, 44, 45, 46, 47, 48, 49, 50}
SHALLOW_OCEANS = {0, 10, 44, 45, 46}
DEEP_OCEANS = {24, 47, 48, 49, 50}
JUNGLES = {21, 22, 23, 149, 151, 168, 169}
BADLANDS = {37, 38, 39, 165, 166, 167}
SNOW_BIOMES = {10, 11, 12, 13, 26, 30, 31, 140, 158}

BIOME_CATEGORY = {
    **{value: 'ocean' for value in OCEANS},
    1: 'plains', 129: 'plains', 2: 'desert', 17: 'desert', 130: 'desert',
    3: 'mountains', 20: 'mountains', 25: 'none', 34: 'mountains',
    131: 'mountains', 162: 'mountains', 4: 'forest', 18: 'forest',
    27: 'forest', 28: 'forest', 29: 'forest', 132: 'forest',
    155: 'forest', 156: 'forest', 157: 'forest', 5: 'taiga',
    19: 'taiga', 30: 'taiga', 31: 'taiga', 32: 'taiga', 33: 'taiga',
    133: 'taiga', 158: 'taiga', 160: 'taiga', 161: 'taiga',
    6: 'swamp', 134: 'swamp', 7: 'river', 11: 'river',
    12: 'icy', 13: 'icy', 140: 'icy', 14: 'mushroom', 15: 'mushroom',
    16: 'beach', 26: 'beach', 21: 'jungle', 22: 'jungle', 23: 'jungle',
    149: 'jungle', 151: 'jungle', 168: 'jungle', 169: 'jungle',
    35: 'savanna', 36: 'savanna', 163: 'savanna', 164: 'savanna',
    **{value: 'badlands' for value in BADLANDS},
}
BIOME_TEMPERATURE = {
    10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 26: 0.05,
    30: -0.5, 31: -0.5, 140: 0.0, 158: -0.5,
    2: 2.0, 17: 2.0, 35: 1.2, 36: 1.0,
    37: 2.0, 38: 2.0, 39: 2.0, 163: 1.1, 164: 1.0,
    165: 2.0, 166: 2.0, 167: 2.0,
}
MODIFIED = {
    1: 129, 2: 130, 3: 131, 4: 132, 5: 133, 6: 134, 12: 140,
    21: 149, 23: 151, 27: 155, 28: 156, 29: 157, 30: 158,
    32: 160, 33: 161, 34: 162, 35: 163, 36: 164, 37: 165,
    38: 166, 39: 167,
}


def _mix_seed(seed, salt):
    seed = to_signed_long(seed)
    product = to_signed_long(seed * 6364136223846793005)
    seed = to_signed_long(seed * to_signed_long(product + 1442695040888963407))
    return to_signed_long(seed + salt)


class _Context:
    def __init__(self, seed, salt):
        mixed_salt = _mix_seed(salt, salt)
        mixed_salt = _mix_seed(mixed_salt, salt)
        mixed_salt = _mix_seed(mixed_salt, salt)
        world = _mix_seed(seed, mixed_salt)
        world = _mix_seed(world, mixed_salt)
        self.world_seed = _mix_seed(world, mixed_salt)
        self.local_seed = 0
        self.noise = PerlinNoiseSampler(MinecraftLCG(seed))

    def init(self, x, z):
        value = self.world_seed
        value = _mix_seed(value, x)
        value = _mix_seed(value, z)
        value = _mix_seed(value, x)
        self.local_seed = _mix_seed(value, z)

    def next_int(self, bound):
        result = ((self.local_seed >> 24) % int(bound) + int(bound)) % int(bound)
        self.local_seed = _mix_seed(self.local_seed, self.world_seed)
        return int(result)

    def choose(self, *values):
        return values[self.next_int(len(values))]


def _layer(seed, salt, function):
    context = _Context(seed, salt)

    @lru_cache(maxsize=262144)
    def sample(x, z):
        x, z = int(x), int(z)
        context.init(x, z)
        return int(function(context, x, z))
    return sample


def _identity(seed, salt, parent, operation):
    return _layer(seed, salt, lambda context, x, z: operation(context, parent(x, z)))


def _south_east(seed, salt, parent, operation):
    return _layer(seed, salt, lambda context, x, z: operation(context, parent(x, z)))


def _cross(seed, salt, parent, operation):
    def sample(context, x, z):
        return operation(
            context, parent(x, z - 1), parent(x + 1, z),
            parent(x, z + 1), parent(x - 1, z), parent(x, z),
        )
    return _layer(seed, salt, sample)


def _diagonal(seed, salt, parent, operation):
    def sample(context, x, z):
        return operation(
            context, parent(x - 1, z + 1), parent(x + 1, z + 1),
            parent(x + 1, z - 1), parent(x - 1, z - 1), parent(x, z),
        )
    return _layer(seed, salt, sample)


def _merge(seed, salt, first, second, operation):
    return _layer(seed, salt, lambda context, x, z: operation(context, first, second, x, z))


def _scale(seed, salt, parent, fuzzy=False):
    context = _Context(seed, salt)

    @lru_cache(maxsize=262144)
    def sample(x, z):
        x, z = int(x), int(z)
        center = parent(x >> 1, z >> 1)
        context.init((x >> 1) << 1, (z >> 1) << 1)
        odd_x, odd_z = x & 1, z & 1
        if odd_x == 0 and odd_z == 0:
            return center
        south = parent(x >> 1, (z + 1) >> 1)
        mixed_south = context.choose(center, south)
        if odd_x == 0:
            return mixed_south
        east = parent((x + 1) >> 1, z >> 1)
        mixed_east = context.choose(center, east)
        if odd_z == 0:
            return mixed_east
        southeast = parent((x + 1) >> 1, (z + 1) >> 1)
        if fuzzy:
            return context.choose(center, east, south, southeast)
        i, j, k, l = center, east, south, southeast
        if j == k == l:
            return j
        if i == j == k or i == j == l or i == k == l:
            return i
        if i == j and k != l:
            return i
        if i == k and j != l:
            return i
        if i == l and j != k:
            return i
        if j == k and i != l:
            return j
        if j == l and i != k:
            return j
        if k == l and i != j:
            return k
        return context.choose(i, j, k, l)
    return sample


def _similar(first, second):
    if first == second:
        return True
    if first in {38, 39}:
        return second in {38, 39}
    first_category = BIOME_CATEGORY.get(first, 'none')
    second_category = BIOME_CATEGORY.get(second, 'none')
    return first_category != 'none' and first_category == second_category


def _temperature_group(value):
    if value in OCEANS:
        return 'ocean'
    temperature = BIOME_TEMPERATURE.get(value, 0.7)
    return 'cold' if temperature < 0.2 else ('medium' if temperature < 1.0 else 'warm')


def _increase_edge(context, sw, se, ne, nw, center):
    if center not in SHALLOW_OCEANS or all(value in SHALLOW_OCEANS for value in (nw, ne, sw, se)):
        if center not in SHALLOW_OCEANS and any(value in SHALLOW_OCEANS for value in (nw, sw, ne, se)) and context.next_int(5) == 0:
            for value in (nw, sw, ne, se):
                if value in SHALLOW_OCEANS:
                    return 4 if center == 4 else value
        return center
    selected, count = 1, 1
    for value in (nw, ne, sw, se):
        if value not in SHALLOW_OCEANS:
            if context.next_int(count) == 0:
                selected = value
            count += 1
    return selected if context.next_int(3) == 0 else (4 if selected == 4 else center)


def _set_base(context, value):
    special = (value & 3840) >> 8
    value &= -3841
    if value in OCEANS or value == 14:
        return value
    groups = {
        1: (2, 2, 2, 35, 35, 1),
        2: (4, 29, 3, 1, 27, 6),
        3: (4, 3, 5, 1),
        4: (12, 12, 12, 30),
    }
    if special:
        if value == 1:
            return 39 if context.next_int(3) == 0 else 38
        if value == 2:
            return 21
        if value == 3:
            return 32
    return context.choose(*groups.get(value, (14,)))


def _ease_edge(context, n, e, s, w, center):
    def replace_similar(source, edge, replacement):
        if not _similar(center, source):
            return None
        return center if all(_similar(value, source) or _temperature_group(value) == 'medium' or _temperature_group(source) == 'medium' for value in (n, e, s, w)) else edge

    for source, edge in ((3, 20),):
        result = replace_similar(source, edge, edge)
        if result is not None:
            return result
    for source, edge in ((38, 37), (39, 37), (32, 5)):
        if center == source:
            return center if all(_similar(value, source) for value in (n, e, s, w)) else edge
    if center == 2 and 12 in (n, e, s, w):
        return 34
    if center == 6:
        if any(value in {2, 12, 30} for value in (n, e, s, w)):
            return 1
        if any(value in {21, 168} for value in (n, e, s, w)):
            return 23
    return center


def _add_hills(context, biome_sampler, noise_sampler, x, z):
    biome = biome_sampler(x, z)
    noise = noise_sampler(x, z)
    remainder = (noise - 2) % 29
    if biome not in SHALLOW_OCEANS and noise >= 2 and remainder == 1:
        return MODIFIED.get(biome, biome)
    if context.next_int(3) != 0 and remainder != 0:
        return biome
    hills = {
        2: 17, 4: 18, 27: 28, 29: 1, 5: 19, 32: 33, 30: 31,
        12: 13, 21: 22, 168: 169, 0: 24, 45: 48, 46: 49,
        10: 50, 3: 34, 35: 36,
    }
    selected = hills.get(biome, biome)
    if biome == 1:
        selected = 18 if context.next_int(3) == 0 else 4
    elif _similar(biome, 38):
        selected = 37
    elif biome in DEEP_OCEANS and context.next_int(3) == 0:
        selected = 1 if context.next_int(2) == 0 else 4
    if remainder == 0 and selected != biome:
        selected = MODIFIED.get(selected, biome)
    if selected != biome:
        neighbors = (
            biome_sampler(x, z - 1), biome_sampler(x + 1, z),
            biome_sampler(x - 1, z), biome_sampler(x, z + 1),
        )
        if sum(_similar(value, biome) for value in neighbors) >= 3:
            return selected
    return biome


def _add_edge_biomes(context, n, e, s, w, center):
    neighbors = (n, e, s, w)
    if center == 14 and any(value in SHALLOW_OCEANS for value in neighbors):
        return 15
    if center in JUNGLES:
        wooded = lambda value: value in JUNGLES or value in {4, 5} or value in OCEANS
        if not all(wooded(value) for value in neighbors):
            return 23
        if any(value in OCEANS for value in neighbors):
            return 16
    elif center in {3, 34, 20}:
        if any(value in OCEANS for value in neighbors):
            return 25
    elif center in SNOW_BIOMES:
        if center not in OCEANS and any(value in OCEANS for value in neighbors):
            return 26
    elif center not in {37, 38}:
        if center not in OCEANS | {7, 6} and any(value in OCEANS for value in neighbors):
            return 16
    elif all(value not in OCEANS for value in neighbors) and not all(value in BADLANDS for value in neighbors):
        return 2
    return center


def _build_overworld(seed):
    continent = _layer(seed, 1, lambda context, x, z: 1 if (x == 0 and z == 0) else (1 if context.next_int(10) == 0 else 0))
    land = _scale(seed, 2000, continent, fuzzy=True)
    land = _diagonal(seed, 1, land, _increase_edge)
    land = _scale(seed, 2001, land)
    for salt in (2, 50, 70):
        land = _diagonal(seed, salt, land, _increase_edge)
    land = _cross(seed, 2, land, lambda context, n, e, s, w, c: 1 if c in SHALLOW_OCEANS and all(value in SHALLOW_OCEANS for value in (n, e, s, w)) and context.next_int(2) == 0 else c)

    def ocean_temperature(context, x, z):
        value = context.noise.sample(x / 8.0, z / 8.0, 0.0)
        if value > 0.4:
            return 44
        if value > 0.2:
            return 45
        if value < -0.4:
            return 10
        return 46 if value < -0.2 else 0
    ocean = _layer(seed, 2, ocean_temperature)
    for index in range(6):
        ocean = _scale(seed, 2001 + index, ocean)

    land = _south_east(seed, 2, land, lambda context, value: value if value in SHALLOW_OCEANS else (4 if (roll := context.next_int(6)) == 0 else (3 if roll == 1 else 1)))
    land = _diagonal(seed, 3, land, _increase_edge)
    land = _cross(seed, 2, land, lambda context, n, e, s, w, c: 2 if c == 1 and any(value in {3, 4} for value in (n, e, s, w)) else c)
    land = _cross(seed, 2, land, lambda context, n, e, s, w, c: 3 if c == 4 and any(value in {1, 2} for value in (n, e, s, w)) else c)
    land = _identity(seed, 3, land, lambda context, value: value if value in SHALLOW_OCEANS else value | ((1 + context.next_int(15)) << 8 & 3840) if context.next_int(13) == 0 else value)
    land = _scale(seed, 2002, land)
    land = _scale(seed, 2003, land)
    land = _diagonal(seed, 4, land, _increase_edge)
    land = _diagonal(seed, 5, land, lambda context, sw, se, ne, nw, c: 14 if c in SHALLOW_OCEANS and all(value in SHALLOW_OCEANS for value in (sw, se, ne, nw)) and context.next_int(100) == 0 else c)
    land = _cross(seed, 4, land, lambda context, n, e, s, w, c: ({44: 47, 45: 48, 0: 24, 46: 49, 10: 50}.get(c, 24) if c in SHALLOW_OCEANS and all(value in SHALLOW_OCEANS for value in (n, e, s, w)) else c))

    noise = _identity(seed, 100, land, lambda context, value: value if value in SHALLOW_OCEANS else context.next_int(299999) + 2)
    biomes = _identity(seed, 200, land, _set_base)
    biomes = _south_east(seed, 1001, biomes, lambda context, value: 168 if value == 21 and context.next_int(10) == 0 else value)
    for index in range(2):
        biomes = _scale(seed, 1000 + index, biomes)
    biomes = _cross(seed, 1000, biomes, _ease_edge)
    base_noise = noise
    hills_noise = base_noise
    for index in range(2):
        hills_noise = _scale(seed, 1000 + index, hills_noise)
    biomes = _merge(seed, 1000, biomes, hills_noise, _add_hills)
    river_noise = base_noise
    for index in range(2):
        river_noise = _scale(seed, 1000 + index, river_noise)
    for index in range(4):
        river_noise = _scale(seed, 1000 + index, river_noise)
    river_noise = _cross(seed, 1, river_noise, lambda context, n, e, s, w, c: -1 if all((value if value < 2 else 2 + (value & 1)) == (c if c < 2 else 2 + (c & 1)) for value in (n, e, s, w)) else 7)
    river_noise = _cross(seed, 1000, river_noise, lambda context, n, e, s, w, c: context.choose(w, n) if e == w and n == s else (w if e == w else (n if n == s else c)))
    biomes = _south_east(seed, 1001, biomes, lambda context, value: 129 if value == 1 and context.next_int(57) == 0 else value)
    for index in range(4):
        biomes = _scale(seed, 1000 + index, biomes)
        if index == 0:
            biomes = _diagonal(seed, 3, biomes, _increase_edge)
        if index == 1:
            biomes = _cross(seed, 1000, biomes, _add_edge_biomes)
    biomes = _cross(seed, 1000, biomes, lambda context, n, e, s, w, c: context.choose(w, n) if e == w and n == s else (w if e == w else (n if n == s else c)))

    def add_rivers(context, first, second, x, z):
        biome, river = first(x, z), second(x, z)
        if biome in OCEANS:
            return biome
        if river == 7:
            return 11 if biome == 12 else (15 if biome in {14, 15} else 7)
        return biome
    biomes = _merge(seed, 100, biomes, river_noise, add_rivers)

    def apply_ocean(context, first, second, x, z):
        biome, temperature = first(x, z), second(x, z)
        if biome not in OCEANS:
            return biome
        for dx in range(-8, 9, 4):
            for dz in range(-8, 9, 4):
                if first(x + dx, z + dz) not in OCEANS:
                    if temperature == 44:
                        return 45
                    if temperature == 10:
                        return 46
        if biome == 24:
            return {45: 48, 0: 24, 46: 49, 10: 50}.get(temperature, temperature)
        return temperature
    return _merge(seed, 100, biomes, ocean, apply_ocean)


class OverworldBiomeSource:
    def __init__(self, seed):
        self.seed = int(seed)
        self.sampler = _build_overworld(self.seed)

    def sample(self, biome_x, biome_z):
        return self.sampler(int(biome_x), int(biome_z))

    def sample_grid(self, biome_x, biome_z):
        x, z = np.broadcast_arrays(np.asarray(biome_x), np.asarray(biome_z))
        output = np.empty(x.shape, dtype=np.int16)
        for index in np.ndindex(x.shape):
            output[index] = self.sample(x[index], z[index])
        return output


NETHER_PROTOTYPES = np.asarray([
    (0.0, 0.0, 0.0, 0.0, 0.0), (0.0, -0.5, 0.0, 0.0, 0.0),
    (0.4, 0.0, 0.0, 0.0, 0.0), (0.0, 0.5, 0.0, 0.0, 0.375),
    (-0.5, 0.0, 0.0, 0.0, 0.175),
])
NETHER_IDS = np.asarray([8, 170, 171, 172, 173], dtype=np.int16)


class NetherBiomeSource:
    def __init__(self, seed):
        seed = int(seed)
        self.samplers = tuple(DoublePerlinNoiseSampler(seed + offset) for offset in range(4))

    def sample_grid(self, biome_x, biome_z):
        x, z = np.broadcast_arrays(np.asarray(biome_x, dtype=float), np.asarray(biome_z, dtype=float))
        fields = np.stack([sampler.sample(x, 0.0, z) for sampler in self.samplers], axis=-1)
        points = np.concatenate((fields, np.zeros((*fields.shape[:-1], 1))), axis=-1)
        distances = np.sum((points[..., None, :] - NETHER_PROTOTYPES) ** 2, axis=-1)
        return NETHER_IDS[np.argmin(distances, axis=-1)]

    def sample(self, biome_x, biome_z):
        return int(self.sample_grid(float(biome_x), float(biome_z)))


def biome_names(ids):
    ids = np.asarray(ids)
    names = np.empty(ids.shape, dtype='<U40')
    for value in np.unique(ids):
        names[ids == value] = BIOME_NAMES.get(int(value), f'biome_{int(value)}')
    return names
