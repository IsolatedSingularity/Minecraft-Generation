"""Minecraft-style terrain backdrops for mathematical visualizations.

The backdrops are deterministic explanatory textures, not block-perfect world
exports. Exact algorithm overlays remain separate from these visual surfaces.
"""

import numpy as np
from matplotlib.colors import to_rgb

from .end_generation import SimplexNoise2D


OVERWORLD_BLOCKS = {
    'deep_water': '#173B63',
    'water': '#2B6694',
    'shore': '#D7C07A',
    'plains': '#79A84B',
    'forest': '#3D7838',
    'dark_forest': '#285B32',
    'desert': '#D8BE72',
    'taiga': '#5D8068',
    'snow': '#DDE5E4',
    'stone': '#85857D',
}

NETHER_BLOCKS = {
    'netherrack': '#682D38',
    'crimson': '#94334E',
    'warped': '#4E4A78',
    'soul_sand': '#5B443D',
    'basalt': '#36323E',
    'lava': '#F08A32',
}


def _noise_layers(seed, resolution):
    coordinates = np.linspace(-160.0, 160.0, int(resolution))
    x, z = np.meshgrid(coordinates, coordinates)
    broad = SimplexNoise2D(seed).sample_grid(x / 43.0, z / 43.0)
    medium = SimplexNoise2D(seed + 104729).sample_grid(
        x / 18.0 + 17.0, z / 18.0 - 11.0,
    )
    detail = SimplexNoise2D(seed - 130363).sample_grid(
        x / 7.0 - 29.0, z / 7.0 + 23.0,
    )
    climate = SimplexNoise2D(seed + 32452843).sample_grid(
        x / 55.0 + 41.0, z / 55.0 - 37.0,
    )
    moisture = SimplexNoise2D(seed - 49979687).sample_grid(
        x / 38.0 - 13.0, z / 38.0 + 31.0,
    )
    return broad, medium, detail, climate, moisture


def _paint(output, mask, color):
    output[mask, :3] = to_rgb(color)


def minecraft_terrain_rgba(seed, resolution=256, dimension='overworld'):
    """Return a deterministic pixel-art terrain texture.

    The texture supplies visual Minecraft context only. It deliberately does
    not stand in for the complete Java 1.16.1 biome and terrain generators.
    """
    broad, medium, detail, climate, moisture = _noise_layers(
        int(seed), int(resolution),
    )
    output = np.zeros((int(resolution), int(resolution), 4), dtype=float)

    if dimension == 'overworld':
        elevation = 0.67 * broad + 0.24 * medium + 0.09 * detail
        _paint(output, np.ones_like(elevation, dtype=bool), OVERWORLD_BLOCKS['plains'])
        _paint(output, moisture > 0.10, OVERWORLD_BLOCKS['forest'])
        _paint(output, moisture > 0.47, OVERWORLD_BLOCKS['dark_forest'])
        _paint(
            output,
            (climate > 0.22) & (moisture < -0.08),
            OVERWORLD_BLOCKS['desert'],
        )
        _paint(
            output,
            (climate < -0.18) & (elevation > -0.15),
            OVERWORLD_BLOCKS['taiga'],
        )
        _paint(
            output,
            (climate < -0.42) & (elevation > -0.08),
            OVERWORLD_BLOCKS['snow'],
        )
        _paint(output, elevation > 0.50, OVERWORLD_BLOCKS['stone'])
        _paint(
            output,
            (elevation > 0.62) | ((elevation > 0.48) & (climate < -0.10)),
            OVERWORLD_BLOCKS['snow'],
        )
        _paint(
            output,
            (elevation >= -0.27) & (elevation < -0.17),
            OVERWORLD_BLOCKS['shore'],
        )
        _paint(output, elevation < -0.17, OVERWORLD_BLOCKS['water'])
        _paint(output, elevation < -0.47, OVERWORLD_BLOCKS['deep_water'])
    elif dimension == 'nether':
        terrain = 0.62 * broad + 0.27 * medium + 0.11 * detail
        _paint(output, np.ones_like(terrain, dtype=bool), NETHER_BLOCKS['netherrack'])
        _paint(
            output,
            (climate > 0.24) & (moisture > -0.16),
            NETHER_BLOCKS['crimson'],
        )
        _paint(
            output,
            (climate < -0.26) & (moisture > 0.02),
            NETHER_BLOCKS['warped'],
        )
        _paint(output, moisture < -0.38, NETHER_BLOCKS['soul_sand'])
        _paint(output, terrain > 0.53, NETHER_BLOCKS['basalt'])
        lava = (terrain < -0.52) | (
            (np.abs(medium) < 0.025) & (broad < -0.05)
        )
        _paint(output, lava, NETHER_BLOCKS['lava'])
    else:
        raise ValueError(f'Unsupported dimension: {dimension}')

    shade = np.clip(0.88 + 0.13 * detail, 0.72, 1.05)
    shade = np.floor(shade * 12.0) / 12.0
    output[..., :3] = np.clip(output[..., :3] * shade[..., None], 0.0, 1.0)
    output[..., 3] = 1.0
    return output


def draw_minecraft_terrain(
    ax, extent, seed, dimension='overworld', resolution=256,
    alpha=0.92, zorder=0,
):
    """Draw a pixelated terrain backdrop and return its image artist."""
    rgba = minecraft_terrain_rgba(
        seed, resolution=resolution, dimension=dimension,
    )
    rgba[..., 3] *= float(alpha)
    return ax.imshow(
        rgba, extent=extent, origin='lower', interpolation='nearest',
        zorder=zorder, aspect='auto',
    )
