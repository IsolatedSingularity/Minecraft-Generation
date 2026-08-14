"""Export the fixed-seed Nether README raster from the exact Python port.

Pixel layout matches the Overworld cache convention where possible:
red stores the raw biome ID, green stores whether Y=31 is a lava-filled
negative-density cell, blue stores format version 1, and alpha is opaque.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Code'))

from core.vanilla_biomes import NetherBiomeSource
from core.vanilla_terrain import VanillaTerrainSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--extent', type=int, default=26240)
    parser.add_argument('--resolution', type=int, default=161)
    arguments = parser.parse_args()

    coordinates = np.rint(np.linspace(
        -arguments.extent, arguments.extent, arguments.resolution,
    )).astype(np.int64)
    block_x, block_z = np.meshgrid(coordinates, coordinates)
    biomes = NetherBiomeSource(arguments.seed).sample_grid(
        block_x / 4.0, block_z / 4.0,
    )
    density = VanillaTerrainSampler(arguments.seed, 'nether').density_points(
        block_x, 31, block_z,
    )
    pixels = np.zeros((*biomes.shape, 4), dtype=np.uint8)
    pixels[..., 0] = biomes
    pixels[..., 1] = density <= 0.0
    pixels[..., 2] = 1
    pixels[..., 3] = 255
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.flipud(pixels), 'RGBA').save(arguments.output)
    print(
        f'wrote {arguments.output} ({arguments.resolution}x{arguments.resolution}), '
        f'seed={arguments.seed}, blocks=-{arguments.extent}..{arguments.extent}'
    )


if __name__ == '__main__':
    main()
