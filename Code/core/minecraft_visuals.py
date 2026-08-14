"""Source-faithful Minecraft 1.16.1 terrain backdrops.

The biome samplers and density columns are Python ports of the Java 1.16.1
generator.  Large fixed-seed README views use compact samples exported by
``Audits/VanillaTerrainCache.java`` from the original server JAR.  The sample
pixels store raw biome IDs and ``WORLD_SURFACE_WG`` heights, while the visible
surface is assembled from the small, provenance-tracked vanilla texture set
under ``Assets/minecraft_1_16_1``.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from PIL import Image

from .vanilla_biomes import NetherBiomeSource, OVERWORLD_FAMILY, OverworldBiomeSource
from .vanilla_terrain import VanillaTerrainSampler


@dataclass(frozen=True)
class BiomeDefinition:
    key: str
    label: str
    dimension: str
    base_color: str
    accent_color: str
    texture: str


OVERWORLD_BIOMES = {
    item.key: item for item in (
        BiomeDefinition('deep_water', 'Deep ocean', 'overworld', '#173B63', '#102A4A', 'water_still.png'),
        BiomeDefinition('water', 'Ocean and river', 'overworld', '#2B6694', '#4A86AE', 'water_still.png'),
        BiomeDefinition('shore', 'Beach and shore', 'overworld', '#D7C07A', '#B99D5E', 'sand.png'),
        BiomeDefinition('plains', 'Plains', 'overworld', '#79A84B', '#9AC25F', 'grass_block_top.png'),
        BiomeDefinition('forest', 'Forest', 'overworld', '#3D7838', '#245A2C', 'grass_block_top.png'),
        BiomeDefinition('dark_forest', 'Dark forest', 'overworld', '#285B32', '#183E27', 'podzol_top.png'),
        BiomeDefinition('desert', 'Desert', 'overworld', '#D8BE72', '#E7D18F', 'sand.png'),
        BiomeDefinition('savanna', 'Savanna', 'overworld', '#A8A85A', '#777D3B', 'grass_block_top.png'),
        BiomeDefinition('jungle', 'Jungle', 'overworld', '#2E7136', '#174D2A', 'grass_block_top.png'),
        BiomeDefinition('swamp', 'Swamp', 'overworld', '#586B45', '#334D3E', 'grass_block_top.png'),
        BiomeDefinition('taiga', 'Taiga', 'overworld', '#5D8068', '#34594D', 'podzol_top.png'),
        BiomeDefinition('snowy_tundra', 'Snowy and frozen', 'overworld', '#DDE5E4', '#AFC9D3', 'snow.png'),
        BiomeDefinition('mountains', 'Mountains', 'overworld', '#85857D', '#B0B0A8', 'stone.png'),
        BiomeDefinition('badlands', 'Badlands', 'overworld', '#B96B45', '#E39A55', 'red_sand.png'),
        BiomeDefinition('mushroom_fields', 'Mushroom fields', 'overworld', '#9A6A9E', '#D6A4B6', 'mycelium_top.png'),
    )
}

NETHER_BIOMES = {
    item.key: item for item in (
        BiomeDefinition('nether_wastes', 'Nether wastes', 'nether', '#682D38', '#873A43', 'netherrack.png'),
        BiomeDefinition('crimson_forest', 'Crimson forest', 'nether', '#8E2F43', '#C34658', 'crimson_nylium.png'),
        BiomeDefinition('warped_forest', 'Warped forest', 'nether', '#176C68', '#35A69A', 'warped_nylium.png'),
        BiomeDefinition('soul_sand_valley', 'Soul sand valley', 'nether', '#5B443D', '#88705D', 'soul_sand.png'),
        BiomeDefinition('basalt_deltas', 'Basalt deltas', 'nether', '#36323E', '#686573', 'basalt_top.png'),
    )
}
NETHER_TERRAIN_CLASSES = {
    **NETHER_BIOMES,
    'lava': BiomeDefinition('lava', 'Lava at Y=31', 'nether', '#F08A32', '#FFD15C', 'lava_still.png'),
}

OVERWORLD_BLOCKS = {key: value.base_color for key, value in OVERWORLD_BIOMES.items()}
NETHER_BLOCKS = {
    'netherrack': NETHER_BIOMES['nether_wastes'].base_color,
    'crimson': NETHER_BIOMES['crimson_forest'].base_color,
    'warped': NETHER_BIOMES['warped_forest'].base_color,
    'soul_sand': NETHER_BIOMES['soul_sand_valley'].base_color,
    'basalt': NETHER_BIOMES['basalt_deltas'].base_color,
    'lava': NETHER_TERRAIN_CLASSES['lava'].base_color,
}

_ROOT = Path(__file__).resolve().parents[2]
_TEXTURES = _ROOT / 'Assets' / 'minecraft_1_16_1' / 'textures' / 'block'
_SAMPLES = _ROOT / 'Assets' / 'minecraft_1_16_1' / 'terrain_samples'

# These are generated data, not hand-painted stand-ins.  Extents are blocks.
_OVERWORLD_CACHES = (
    (42, -1536, 1536, -1536, 1536, 'overworld_seed_42_center_3072.png'),
    (42, -26240, 26240, -26240, 26240, 'overworld_seed_42_52480.png'),
    (-4172144997902289642, -168, 168, -168, 168,
     'overworld_spawn_seed_neg4172144997902289642.png'),
)
_NETHER_CACHES = (
    (42, -26240, 26240, -26240, 26240, 'nether_seed_42_52480.png'),
)

_NETHER_NAME = {
    8: 'nether_wastes', 170: 'soul_sand_valley', 171: 'crimson_forest',
    172: 'warped_forest', 173: 'basalt_deltas',
}

_TEXTURE_TINT = {
    'deep_water': '#2B5F91', 'water': '#3E78AE', 'plains': '#7FAE55',
    'forest': '#4C8A43', 'savanna': '#A8A85A', 'jungle': '#3C863D',
    'swamp': '#66784A',
}


def _coordinate_grid(resolution, x_extent, z_extent, coordinate_scale):
    x_values = np.linspace(float(x_extent[0]), float(x_extent[1]), int(resolution))
    z_values = np.linspace(float(z_extent[0]), float(z_extent[1]), int(resolution))
    return np.meshgrid(x_values * float(coordinate_scale), z_values * float(coordinate_scale))


@lru_cache(maxsize=None)
def _load_cache(filename):
    pixels = np.asarray(Image.open(_SAMPLES / filename).convert('RGBA'))
    # PNG row zero is north/top.  All plotting arrays use row zero at minimum Z.
    pixels = np.flipud(pixels)
    if not np.all(pixels[..., 2] == 1):
        raise ValueError(f'Unsupported terrain-cache format: {filename}')
    return pixels[..., 0].astype(np.int16), pixels[..., 1].astype(np.int16)


def _best_cache(seed, block_x_extent, block_z_extent):
    choices = []
    for item in _OVERWORLD_CACHES:
        cache_seed, min_x, max_x, min_z, max_z, filename = item
        if int(seed) != cache_seed:
            continue
        if (min_x <= block_x_extent[0] and max_x >= block_x_extent[1]
                and min_z <= block_z_extent[0] and max_z >= block_z_extent[1]):
            ids, _ = _load_cache(filename)
            step = max((max_x - min_x) / max(ids.shape[1] - 1, 1),
                       (max_z - min_z) / max(ids.shape[0] - 1, 1))
            choices.append((step, item))
    return min(choices, default=(None, None), key=lambda value: value[0])[1]


def _sample_cache(cache, block_x, block_z):
    _, min_x, max_x, min_z, max_z, filename = cache
    ids, heights = _load_cache(filename)
    columns = np.rint((block_x - min_x) * (ids.shape[1] - 1) / (max_x - min_x))
    rows = np.rint((block_z - min_z) * (ids.shape[0] - 1) / (max_z - min_z))
    columns = np.clip(columns.astype(int), 0, ids.shape[1] - 1)
    rows = np.clip(rows.astype(int), 0, ids.shape[0] - 1)
    return ids[rows, columns], heights[rows, columns]


@lru_cache(maxsize=None)
def _load_nether_cache(filename):
    pixels = np.flipud(np.asarray(Image.open(_SAMPLES / filename).convert('RGBA')))
    if not np.all(pixels[..., 2] == 1):
        raise ValueError(f'Unsupported Nether terrain-cache format: {filename}')
    return pixels[..., 0].astype(np.int16), pixels[..., 1].astype(bool)


def _best_nether_cache(seed, block_x_extent, block_z_extent):
    for item in _NETHER_CACHES:
        cache_seed, min_x, max_x, min_z, max_z, _ = item
        if (int(seed) == cache_seed and min_x <= block_x_extent[0]
                and max_x >= block_x_extent[1] and min_z <= block_z_extent[0]
                and max_z >= block_z_extent[1]):
            return item
    return None


def _sample_nether_cache(cache, block_x, block_z):
    _, min_x, max_x, min_z, max_z, filename = cache
    ids, lava = _load_nether_cache(filename)
    columns = np.rint((block_x - min_x) * (ids.shape[1] - 1) / (max_x - min_x))
    rows = np.rint((block_z - min_z) * (ids.shape[0] - 1) / (max_z - min_z))
    columns = np.clip(columns.astype(int), 0, ids.shape[1] - 1)
    rows = np.clip(rows.astype(int), 0, ids.shape[0] - 1)
    return ids[rows, columns], lava[rows, columns]


def overworld_surface_sample(
    seed, resolution=256, x_extent=(-4096.0, 4096.0), z_extent=None,
    coordinate_scale=1.0,
):
    """Return raw biome IDs and WORLD_SURFACE_WG heights on a plot grid."""
    if z_extent is None:
        z_extent = x_extent
    block_x, block_z = _coordinate_grid(resolution, x_extent, z_extent, coordinate_scale)
    block_x_extent = (float(np.min(block_x)), float(np.max(block_x)))
    block_z_extent = (float(np.min(block_z)), float(np.max(block_z)))
    cache = _best_cache(seed, block_x_extent, block_z_extent)
    if cache is not None:
        return _sample_cache(cache, block_x, block_z)

    # Exact fallback for arbitrary seeds and extents.  Large fixed figures
    # should add a cache instead because the scalar layer graph is expensive.
    biome_source = OverworldBiomeSource(int(seed))
    biome_ids = biome_source.sample_grid(
        np.floor_divide(np.rint(block_x).astype(np.int64), 4),
        np.floor_divide(np.rint(block_z).astype(np.int64), 4),
    )
    terrain = VanillaTerrainSampler(int(seed), 'overworld')
    heights = terrain.height_points(
        np.rint(block_x).astype(np.int64), np.rint(block_z).astype(np.int64),
    )
    return biome_ids, heights


def _family_grid(biome_ids, dimension):
    output = np.empty(np.asarray(biome_ids).shape, dtype='<U20')
    mapping = OVERWORLD_FAMILY if dimension == 'overworld' else _NETHER_NAME
    fallback = 'plains' if dimension == 'overworld' else 'nether_wastes'
    for value in np.unique(biome_ids):
        output[biome_ids == value] = mapping.get(int(value), fallback)
    return output


def minecraft_biome_grid(
    seed, resolution=256, x_extent=(-4096.0, 4096.0), z_extent=None,
    coordinate_scale=1.0, showcase=False,
):
    """Return exact Java 1.16.1 Overworld biome families.

    ``showcase`` is accepted for compatibility but never injects biomes.
    """
    biome_ids, _ = overworld_surface_sample(
        seed, resolution, x_extent, z_extent, coordinate_scale,
    )
    return _family_grid(biome_ids, 'overworld')


def minecraft_nether_biome_grid(
    seed, resolution=256, x_extent=(-2048.0, 2048.0), z_extent=None,
    coordinate_scale=1.0, showcase=False,
):
    """Return the exact 1.16.1 four-field MultiNoise classification."""
    if z_extent is None:
        z_extent = x_extent
    block_x, block_z = _coordinate_grid(resolution, x_extent, z_extent, coordinate_scale)
    cache = _best_nether_cache(
        seed, (float(np.min(block_x)), float(np.max(block_x))),
        (float(np.min(block_z)), float(np.max(block_z))),
    )
    if cache is not None:
        biome_ids, _ = _sample_nether_cache(cache, block_x, block_z)
    else:
        source = NetherBiomeSource(int(seed))
        biome_ids = source.sample_grid(block_x / 4.0, block_z / 4.0)
    return _family_grid(biome_ids, 'nether')


@lru_cache(maxsize=None)
def _texture(filename):
    pixels = np.asarray(Image.open(_TEXTURES / filename).convert('RGBA'), dtype=float) / 255.0
    return pixels[:16, :16]


def _texture_rgb(definition, block_x, block_z):
    texture = _texture(definition.texture)
    columns = np.mod(np.floor(block_x).astype(np.int64), 16)
    rows = np.mod(np.floor(block_z).astype(np.int64), 16)
    sampled = texture[rows, columns, :3]
    tint = _TEXTURE_TINT.get(definition.key)
    if tint is not None:
        luminance = np.mean(sampled, axis=-1, keepdims=True)
        sampled = np.clip((0.42 + 0.78 * luminance) * np.asarray(to_rgb(tint)), 0.0, 1.0)
    return sampled


def terrain_rgba_from_sample(
    biome_ids, heights, block_x, block_z, dimension='overworld', lava=None,
    flat=False,
):
    """Turn exact generator samples into a textured or stage-flat RGBA map."""
    families = _family_grid(biome_ids, dimension)
    definitions = OVERWORLD_BIOMES if dimension == 'overworld' else NETHER_BIOMES
    output = np.zeros((*families.shape, 4), dtype=float)
    for name, definition in definitions.items():
        selected = families == name
        if flat:
            output[selected, :3] = to_rgb(definition.base_color)
        else:
            texture = _texture_rgb(definition, block_x, block_z)
            output[selected, :3] = texture[selected]

    if not flat and heights is not None:
        gradient_z, gradient_x = np.gradient(np.asarray(heights, dtype=float))
        shade = np.clip(0.94 + 0.028 * gradient_x - 0.032 * gradient_z, 0.72, 1.10)
        output[..., :3] = np.clip(output[..., :3] * shade[..., None], 0.0, 1.0)

    if lava is not None:
        definition = NETHER_TERRAIN_CLASSES['lava']
        texture = _texture_rgb(definition, block_x, block_z)
        selected = np.asarray(lava, dtype=bool)
        output[selected, :3] = (
            0.52 * output[selected, :3] + 0.48 * texture[selected]
        )
    output[..., 3] = 1.0
    return output


def minecraft_terrain_rgba(
    seed, resolution=256, dimension='overworld',
    x_extent=(-4096.0, 4096.0), z_extent=None,
    coordinate_scale=1.0, showcase=False,
):
    """Return a source-faithful textured terrain raster."""
    if z_extent is None:
        z_extent = x_extent
    block_x, block_z = _coordinate_grid(resolution, x_extent, z_extent, coordinate_scale)
    if dimension == 'overworld':
        biome_ids, heights = overworld_surface_sample(
            seed, resolution, x_extent, z_extent, coordinate_scale,
        )
        return terrain_rgba_from_sample(
            biome_ids, heights, block_x, block_z, 'overworld',
        )
    if dimension == 'nether':
        cache = _best_nether_cache(
            seed, (float(np.min(block_x)), float(np.max(block_x))),
            (float(np.min(block_z)), float(np.max(block_z))),
        )
        if cache is not None:
            biome_ids, lava = _sample_nether_cache(cache, block_x, block_z)
            return terrain_rgba_from_sample(
                biome_ids, None, block_x, block_z, 'nether', lava=lava,
            )
        source = NetherBiomeSource(int(seed))
        biome_ids = source.sample_grid(block_x / 4.0, block_z / 4.0)
        # Y=31 is the last full level below the lava sea.  Negative density is
        # therefore an actual lava-filled cavity, not a decorative sixth biome.
        coarse_resolution = min(int(resolution), 97)
        coarse_x, coarse_z = _coordinate_grid(
            coarse_resolution, x_extent, z_extent, coordinate_scale,
        )
        density = VanillaTerrainSampler(int(seed), 'nether').density_points(
            np.rint(coarse_x).astype(np.int64), 31,
            np.rint(coarse_z).astype(np.int64),
        )
        rows = np.rint(np.linspace(0, coarse_resolution - 1, int(resolution))).astype(int)
        lava = density[np.ix_(rows, rows)] <= 0.0
        return terrain_rgba_from_sample(
            biome_ids, None, block_x, block_z, 'nether', lava=lava,
        )
    raise ValueError(f'Unsupported dimension: {dimension}')


def biome_texture_swatch(name, size=24):
    definitions = {**OVERWORLD_BIOMES, **NETHER_TERRAIN_CLASSES}
    definition = definitions[name]
    coordinates = np.arange(int(size), dtype=float)
    block_x, block_z = np.meshgrid(coordinates, coordinates)
    output = np.zeros((int(size), int(size), 4), dtype=float)
    output[..., :3] = _texture_rgb(definition, block_x, block_z)
    output[..., 3] = 1.0
    return output


def draw_minecraft_terrain(
    ax, extent, seed, dimension='overworld', resolution=256,
    alpha=0.92, zorder=0, coordinate_scale=1.0, showcase=False,
):
    rgba = minecraft_terrain_rgba(
        seed, resolution=resolution, dimension=dimension,
        x_extent=(extent[0], extent[1]), z_extent=(extent[2], extent[3]),
        coordinate_scale=coordinate_scale, showcase=showcase,
    )
    rgba[..., 3] *= float(alpha)
    return ax.imshow(
        rgba, extent=extent, origin='lower', interpolation='nearest',
        zorder=zorder, aspect='auto',
    )
