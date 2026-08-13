"""Coordinate-aware Minecraft-style terrain backdrops.

The biome catalog, climate classification, and texture renderers in this
module are deliberately reusable across every active figure.  The surfaces
are source-informed explanatory models, not block-perfect vanilla biome
exports.  Exact structure and pathing mathematics remain separate overlays.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import to_rgb

from .end_generation import SimplexNoise2D


@dataclass(frozen=True)
class BiomeDefinition:
    """Visual and semantic information for one biome family."""

    key: str
    label: str
    dimension: str
    base_color: str
    accent_color: str
    texture: str


OVERWORLD_BIOMES = {
    item.key: item for item in (
        BiomeDefinition('deep_water', 'Deep ocean', 'overworld', '#173B63', '#102A4A', 'waves'),
        BiomeDefinition('water', 'Ocean', 'overworld', '#2B6694', '#4A86AE', 'waves'),
        BiomeDefinition('shore', 'Beach', 'overworld', '#D7C07A', '#B99D5E', 'sand'),
        BiomeDefinition('plains', 'Plains', 'overworld', '#79A84B', '#9AC25F', 'grass'),
        BiomeDefinition('forest', 'Forest', 'overworld', '#3D7838', '#245A2C', 'canopy'),
        BiomeDefinition('dark_forest', 'Dark forest', 'overworld', '#285B32', '#183E27', 'dense_canopy'),
        BiomeDefinition('desert', 'Desert', 'overworld', '#D8BE72', '#E7D18F', 'dunes'),
        BiomeDefinition('savanna', 'Savanna', 'overworld', '#A8A85A', '#777D3B', 'dry_grass'),
        BiomeDefinition('jungle', 'Jungle', 'overworld', '#2E7136', '#174D2A', 'jungle'),
        BiomeDefinition('swamp', 'Swamp', 'overworld', '#586B45', '#334D3E', 'marsh'),
        BiomeDefinition('taiga', 'Taiga', 'overworld', '#5D8068', '#34594D', 'spruce'),
        BiomeDefinition('snowy_tundra', 'Snowy tundra', 'overworld', '#DDE5E4', '#AFC9D3', 'snow'),
        BiomeDefinition('mountains', 'Mountains', 'overworld', '#85857D', '#B0B0A8', 'rock'),
        BiomeDefinition('badlands', 'Badlands', 'overworld', '#B96B45', '#E39A55', 'terracotta'),
        BiomeDefinition('mushroom_fields', 'Mushroom fields', 'overworld', '#9A6A9E', '#D6A4B6', 'mushrooms'),
    )
}

NETHER_BIOMES = {
    item.key: item for item in (
        BiomeDefinition('nether_wastes', 'Nether wastes', 'nether', '#682D38', '#873A43', 'netherrack'),
        BiomeDefinition('crimson_forest', 'Crimson forest', 'nether', '#8E2F43', '#C34658', 'crimson'),
        BiomeDefinition('warped_forest', 'Warped forest', 'nether', '#176C68', '#35A69A', 'warped'),
        BiomeDefinition('soul_sand_valley', 'Soul sand valley', 'nether', '#5B443D', '#88705D', 'soul_sand'),
        BiomeDefinition('basalt_deltas', 'Basalt deltas', 'nether', '#36323E', '#686573', 'basalt'),
    )
}
NETHER_TERRAIN_CLASSES = {
    **NETHER_BIOMES,
    'lava': BiomeDefinition(
        'lava', 'Lava terrain (not a biome)', 'nether',
        '#F08A32', '#FFD15C', 'lava',
    ),
}

# Compatibility palettes retained for older modules.
OVERWORLD_BLOCKS = {
    key: biome.base_color for key, biome in OVERWORLD_BIOMES.items()
}
NETHER_BLOCKS = {
    'netherrack': NETHER_BIOMES['nether_wastes'].base_color,
    'crimson': NETHER_BIOMES['crimson_forest'].base_color,
    'warped': NETHER_BIOMES['warped_forest'].base_color,
    'soul_sand': NETHER_BIOMES['soul_sand_valley'].base_color,
    'basalt': NETHER_BIOMES['basalt_deltas'].base_color,
    'lava': NETHER_TERRAIN_CLASSES['lava'].base_color,
}


def _coordinate_grid(resolution, x_extent, z_extent, coordinate_scale):
    x_values = np.linspace(float(x_extent[0]), float(x_extent[1]), int(resolution))
    z_values = np.linspace(float(z_extent[0]), float(z_extent[1]), int(resolution))
    x, z = np.meshgrid(x_values, z_values)
    return x * float(coordinate_scale), z * float(coordinate_scale)


def _noise_layers(
    seed, resolution, x_extent, z_extent, coordinate_scale=1.0,
):
    """Sample coherent fields in world coordinates rather than image space."""
    x, z = _coordinate_grid(
        resolution, x_extent, z_extent, coordinate_scale,
    )
    broad = SimplexNoise2D(seed).sample_grid(x / 2800.0, z / 2800.0)
    medium = SimplexNoise2D(seed + 104729).sample_grid(
        x / 1100.0 + 17.0, z / 1100.0 - 11.0,
    )
    detail = SimplexNoise2D(seed - 130363).sample_grid(
        x / 320.0 - 29.0, z / 320.0 + 23.0,
    )
    climate = SimplexNoise2D(seed + 32452843).sample_grid(
        x / 3600.0 + 41.0, z / 3600.0 - 37.0,
    )
    moisture = SimplexNoise2D(seed - 49979687).sample_grid(
        x / 2400.0 - 13.0, z / 2400.0 + 31.0,
    )
    return x, z, broad, medium, detail, climate, moisture


def _showcase_overworld_biomes(biomes):
    """Compress rare-biome spacing so one small README map shows every class."""
    rows, columns = np.indices(biomes.shape)
    nx = 2.0 * columns / max(biomes.shape[1] - 1, 1) - 1.0
    nz = 2.0 * rows / max(biomes.shape[0] - 1, 1) - 1.0
    showcase = (
        ('mushroom_fields', -0.73, 0.63, 0.13),
        ('badlands', 0.70, 0.61, 0.16),
        ('snowy_tundra', -0.54, -0.70, 0.17),
        ('taiga', -0.18, -0.65, 0.16),
        ('jungle', 0.56, -0.58, 0.16),
        ('swamp', 0.73, -0.13, 0.14),
        ('desert', 0.42, 0.25, 0.16),
        ('savanna', 0.04, 0.67, 0.15),
        ('dark_forest', -0.45, 0.26, 0.24),
    )
    for name, center_x, center_z, radius in showcase:
        distance = ((nx - center_x) / radius) ** 2 + (
            (nz - center_z) / (radius * 0.76)
        ) ** 2
        biomes[distance <= 1.0] = name
    return biomes


def minecraft_biome_grid(
    seed, resolution=256, x_extent=(-4096.0, 4096.0), z_extent=None,
    coordinate_scale=1.0, showcase=False,
):
    """Return a coordinate-aware illustrative Overworld biome grid.

    ``showcase`` deliberately compresses rare-biome spacing for small README
    panels.  It must be described as a rare-biome demonstration, not a
    frequency-faithful seed export.
    """
    if z_extent is None:
        z_extent = x_extent
    _, _, broad, medium, detail, climate, moisture = _noise_layers(
        int(seed), int(resolution), x_extent, z_extent, coordinate_scale,
    )
    elevation = 0.64 * broad + 0.25 * medium + 0.11 * detail
    biomes = np.full(elevation.shape, 'plains', dtype='<U20')

    land = elevation >= -0.23
    biomes[land & (moisture > 0.05)] = 'forest'
    biomes[land & (moisture > 0.43) & (climate < 0.16)] = 'dark_forest'
    biomes[land & (climate > 0.20) & (moisture < -0.06)] = 'desert'
    biomes[
        land & (climate > 0.14) & (moisture >= -0.06) & (moisture < 0.24)
    ] = 'savanna'
    biomes[land & (climate > 0.16) & (moisture > 0.31)] = 'jungle'
    biomes[
        land & (elevation < 0.19) & (climate > -0.16) & (moisture > 0.36)
    ] = 'swamp'
    biomes[land & (climate < -0.15) & (elevation > -0.20)] = 'taiga'
    biomes[land & (climate < -0.39) & (elevation > -0.12)] = 'snowy_tundra'
    biomes[land & (elevation > 0.49)] = 'mountains'
    biomes[
        land & ((elevation > 0.59) | ((elevation > 0.45) & (climate < -0.08)))
    ] = 'snowy_tundra'
    biomes[
        land & (climate > 0.38) & (moisture < -0.25) & (medium > -0.03)
    ] = 'badlands'
    biomes[
        land & (broad > 0.28) & (medium < -0.24) & (moisture > 0.11)
    ] = 'mushroom_fields'

    biomes[elevation < -0.52] = 'deep_water'
    biomes[(elevation >= -0.52) & (elevation < -0.31)] = 'water'
    biomes[(elevation >= -0.31) & (elevation < -0.23)] = 'shore'
    if showcase:
        _showcase_overworld_biomes(biomes)
    return biomes


def minecraft_nether_biome_grid(
    seed, resolution=256, x_extent=(-2048.0, 2048.0), z_extent=None,
    coordinate_scale=1.0, showcase=False,
):
    """Return a source-shaped proxy for the five Java 1.16 Nether biomes.

    Java samples four Double-Perlin fields and chooses the closest of five
    ``MixedNoisePoint`` prototypes. The classification and exact prototype
    coordinates are preserved here, while seeded simplex fields stand in for
    the four Double-Perlin rasters. Lava is terrain, not a sixth biome.
    """
    if z_extent is None:
        z_extent = x_extent
    x, z = _coordinate_grid(
        int(resolution), x_extent, z_extent, coordinate_scale,
    )
    fields = np.stack([
        SimplexNoise2D(int(seed) + offset).sample_grid(
            x / 2350.0 + offset * 7.0,
            z / 2350.0 - offset * 11.0,
        )
        for offset in range(4)
    ], axis=-1)
    names = np.array([
        'nether_wastes', 'soul_sand_valley', 'crimson_forest',
        'warped_forest', 'basalt_deltas',
    ])
    prototypes = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.5, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.375],
        [-0.5, 0.0, 0.0, 0.0, 0.175],
    ])
    points = np.concatenate(
        (fields, np.zeros((*fields.shape[:2], 1))), axis=-1,
    )
    distances = np.sum(
        (points[..., None, :] - prototypes[None, None, :, :]) ** 2,
        axis=-1,
    )
    biomes = names[np.argmin(distances, axis=-1)]

    if showcase:
        rows, columns = np.indices(biomes.shape)
        nx = 2.0 * columns / max(biomes.shape[1] - 1, 1) - 1.0
        nz = 2.0 * rows / max(biomes.shape[0] - 1, 1) - 1.0
        required = (
            ('crimson_forest', -0.53, 0.42),
            ('warped_forest', 0.48, 0.43),
            ('soul_sand_valley', -0.50, -0.48),
            ('basalt_deltas', 0.50, -0.48),
        )
        for name, center_x, center_z in required:
            mask = ((nx - center_x) / 0.25) ** 2 + ((nz - center_z) / 0.20) ** 2 <= 1
            biomes[mask] = name
    return biomes


def _texture_pattern(texture, x, z, detail, medium):
    rows, columns = np.indices(detail.shape)
    if texture == 'waves':
        return ((rows + 2 * columns) % 17) < 2
    if texture in {'sand', 'dunes'}:
        return np.sin(x / 75.0 + z / 180.0 + medium * 3.0) > 0.76
    if texture in {'canopy', 'dense_canopy', 'jungle', 'spruce', 'grass', 'dry_grass'}:
        threshold = {
            'grass': 0.50, 'dry_grass': 0.43, 'canopy': 0.20,
            'dense_canopy': -0.02, 'jungle': -0.10, 'spruce': 0.16,
        }[texture]
        return detail > threshold
    if texture == 'marsh':
        return (detail < -0.24) | (((rows + columns) % 23) == 0)
    if texture == 'snow':
        return (detail > 0.22) | (((rows - columns) % 19) == 0)
    if texture == 'rock':
        return np.abs(detail) < 0.09
    if texture == 'terracotta':
        return (np.floor((z + 110.0 * medium) / 95.0).astype(int) % 3) == 0
    if texture == 'mushrooms':
        return ((rows * 7 + columns * 11) % 37) < 3
    if texture in {'netherrack', 'crimson', 'warped', 'soul_sand', 'basalt'}:
        modulus = {
            'netherrack': 17, 'crimson': 13, 'warped': 14,
            'soul_sand': 19, 'basalt': 9,
        }[texture]
        return ((rows * 5 + columns * 7) % modulus < 2) | (detail > 0.46)
    if texture == 'lava':
        return np.sin(x / 58.0 - z / 43.0 + detail * 2.0) > 0.35
    return detail > 0.42


def _terrain_rgba_from_biomes(biomes, definitions, x, z, detail, medium):
    output = np.zeros((*biomes.shape, 4), dtype=float)
    for name, definition in definitions.items():
        selected = biomes == name
        output[selected, :3] = to_rgb(definition.base_color)
        pattern = selected & _texture_pattern(
            definition.texture, x, z, detail, medium,
        )
        output[pattern, :3] = to_rgb(definition.accent_color)
    shade = np.clip(0.90 + 0.11 * detail, 0.74, 1.06)
    shade = np.floor(shade * 14.0) / 14.0
    output[..., :3] = np.clip(output[..., :3] * shade[..., None], 0.0, 1.0)
    output[..., 3] = 1.0
    return output


def minecraft_terrain_rgba(
    seed, resolution=256, dimension='overworld',
    x_extent=(-4096.0, 4096.0), z_extent=None,
    coordinate_scale=1.0, showcase=False,
):
    """Return a deterministic textured biome surface in plot coordinates."""
    if z_extent is None:
        z_extent = x_extent
    x, z, _, medium, detail, _, _ = _noise_layers(
        int(seed), int(resolution), x_extent, z_extent, coordinate_scale,
    )
    if dimension == 'overworld':
        biomes = minecraft_biome_grid(
            seed, resolution, x_extent, z_extent, coordinate_scale, showcase,
        )
        definitions = OVERWORLD_BIOMES
    elif dimension == 'nether':
        biomes = minecraft_nether_biome_grid(
            seed, resolution, x_extent, z_extent, coordinate_scale, showcase,
        )
        definitions = NETHER_BIOMES
    else:
        raise ValueError(f'Unsupported dimension: {dimension}')
    output = _terrain_rgba_from_biomes(
        biomes, definitions, x, z, detail, medium,
    )
    if dimension == 'nether':
        # Lava remains an independent terrain overlay, never a biome label.
        lava_field = 0.62 * medium + 0.38 * detail
        lava = (lava_field < -0.58) | (
            (np.abs(medium) < 0.016) & (detail < -0.22)
        )
        lava_definition = NETHER_TERRAIN_CLASSES['lava']
        output[lava, :3] = to_rgb(lava_definition.base_color)
        accent = lava & _texture_pattern('lava', x, z, detail, medium)
        output[accent, :3] = to_rgb(lava_definition.accent_color)
    return output


def biome_texture_swatch(name, size=24):
    """Return a large textured legend swatch for one biome definition."""
    definitions = {**OVERWORLD_BIOMES, **NETHER_TERRAIN_CLASSES}
    definition = definitions[name]
    coordinates = np.linspace(-240.0, 240.0, int(size))
    x, z = np.meshgrid(coordinates, coordinates)
    detail = SimplexNoise2D(sum(map(ord, name))).sample_grid(x / 90.0, z / 90.0)
    medium = SimplexNoise2D(sum(map(ord, name)) + 37).sample_grid(x / 180.0, z / 180.0)
    biomes = np.full((int(size), int(size)), name, dtype='<U20')
    return _terrain_rgba_from_biomes(
        biomes, {name: definition}, x, z, detail, medium,
    )


def draw_minecraft_terrain(
    ax, extent, seed, dimension='overworld', resolution=256,
    alpha=0.92, zorder=0, coordinate_scale=1.0, showcase=False,
):
    """Draw a coordinate-aware pixelated terrain backdrop."""
    x_extent = (extent[0], extent[1])
    z_extent = (extent[2], extent[3])
    rgba = minecraft_terrain_rgba(
        seed, resolution=resolution, dimension=dimension,
        x_extent=x_extent, z_extent=z_extent,
        coordinate_scale=coordinate_scale, showcase=showcase,
    )
    rgba[..., 3] *= float(alpha)
    return ax.imshow(
        rgba, extent=extent, origin='lower', interpolation='nearest',
        zorder=zorder, aspect='auto',
    )
