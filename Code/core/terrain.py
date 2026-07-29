"""Deterministic terrain context layers for placement visualizations.

These fields are visual context, not claims of bit-exact vanilla terrain.
Structure candidate coordinates remain source-faithful and are drawn above them.
"""

import matplotlib.colors as colors
import numpy as np


def terrainField(limit, seed, resolution=280, dimension='overworld'):
    """Return a deterministic, smoothly varying context field and color map."""
    coordinates = np.linspace(-limit, limit, resolution)
    xValues, zValues = np.meshgrid(coordinates, coordinates)
    seedPhase = (int(seed) % 104729) / 104729.0 * 2.0 * np.pi
    scale = max(float(limit), 1.0)

    broad = (
        np.sin(2.2 * np.pi * xValues / scale + seedPhase)
        + 0.86 * np.cos(2.7 * np.pi * zValues / scale - seedPhase * 0.61)
    )
    detail = (
        0.55 * np.sin(6.3 * np.pi * (xValues + 0.32 * zValues) / scale)
        + 0.34 * np.cos(9.1 * np.pi * (zValues - 0.21 * xValues) / scale)
        + 0.18 * np.sin(15.7 * np.pi * (xValues - zValues) / scale + seedPhase)
    )
    field = broad + detail
    field = (field - field.min()) / max(float(np.ptp(field)), 1e-9)

    if dimension == 'nether':
        colorMap = colors.LinearSegmentedColormap.from_list(
            'iosNetherTerrain',
            [
                (0.00, '#C9C9CE'),
                (0.22, '#D6B997'),
                (0.43, '#E4A4A0'),
                (0.62, '#D9868C'),
                (0.80, '#8FCBC2'),
                (1.00, '#F3BE66'),
            ],
        )
    else:
        colorMap = colors.LinearSegmentedColormap.from_list(
            'iosOverworldTerrain',
            [
                (0.00, '#9DD7F5'),
                (0.30, '#C4E5F4'),
                (0.42, '#E9D9A8'),
                (0.54, '#B9DDA7'),
                (0.76, '#86C991'),
                (1.00, '#A7A9AF'),
            ],
        )
    return field, colorMap


def addTerrainBackdrop(ax, limit, seed, dimension='overworld', alpha=0.28):
    """Draw a faint contextual terrain field across an existing coordinate axis."""
    field, colorMap = terrainField(limit, seed, dimension=dimension)
    return ax.imshow(
        field,
        origin='lower',
        extent=(-limit, limit, -limit, limit),
        cmap=colorMap,
        interpolation='bilinear',
        alpha=alpha,
        zorder=-5,
    )
