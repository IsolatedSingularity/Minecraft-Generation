"""Source-faithful Java 1.16.1 density terrain sampling.

The renderer needs top-down samples rather than full chunk objects.  This
module ports the numerical path behind ``SurfaceChunkGenerator.getHeight``:
the three interpolated octave stacks, biome depth/scale blending, density
slides, four-corner horizontal interpolation, and End island override.
Sampling is batched for figures, but every returned column follows the same
equations and RNG construction order as the game.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .end_generation import SimplexNoise2D
from .lcg import MinecraftLCG
from .vanilla_biomes import BIOME_DEPTH_SCALE, NetherBiomeSource, OverworldBiomeSource
from .vanilla_noise import (
    OctavePerlinNoiseSampler,
    OctaveSimplexNoiseSampler,
    clamped_lerp,
    maintain_precision,
)


@dataclass(frozen=True)
class NoiseSettings:
    height: int
    xz_scale: float
    y_scale: float
    xz_factor: float
    y_factor: float
    top_target: float
    top_size: float
    top_offset: float
    bottom_target: float
    bottom_size: float
    bottom_offset: float
    size_horizontal: int
    size_vertical: int
    density_factor: float
    density_offset: float
    random_density_offset: bool
    simplex_surface: bool
    island_override: bool
    sea_level: int
    has_fluid: bool


SETTINGS = {
    'overworld': NoiseSettings(
        256, 0.9999999814507745, 0.9999999814507745, 80.0, 160.0,
        -10.0, 3.0, 0.0, -30.0, 0.0, 0.0, 1, 2, 1.0, -0.46875,
        True, True, False, 63, True,
    ),
    'nether': NoiseSettings(
        128, 1.0, 3.0, 80.0, 60.0,
        120.0, 3.0, 0.0, 320.0, 4.0, -1.0, 1, 2, 0.0,
        0.019921875, False, False, False, 32, True,
    ),
    'end': NoiseSettings(
        128, 2.0, 1.0, 80.0, 160.0,
        -3000.0, 64.0, -46.0, -30.0, 7.0, 1.0, 2, 1, 0.0, 0.0,
        True, False, True, 0, False,
    ),
}


BIOME_WEIGHT = np.asarray([
    10.0 / np.sqrt(dx * dx + dz * dz + 0.2)
    for dz in range(-2, 3) for dx in range(-2, 3)
], dtype=float).reshape(5, 5)


def _java_divide_by_two(value):
    return int(int(value) / 2)


def _end_noise_at(simplex, x, z):
    x, z = int(x), int(z)
    half_x = _java_divide_by_two(x)
    half_z = _java_divide_by_two(z)
    remainder_x = x - half_x * 2
    remainder_z = z - half_z * 2
    value = np.clip(100.0 - np.sqrt(float(x * x + z * z)) * 8.0, -100.0, 80.0)
    for offset_x in range(-12, 13):
        for offset_z in range(-12, 13):
            site_x = half_x + offset_x
            site_z = half_z + offset_z
            if (
                site_x * site_x + site_z * site_z > 4096
                and simplex.sample(site_x, site_z) < -0.9
            ):
                falloff = (
                    (abs(float(site_x)) * 3439.0 + abs(float(site_z)) * 147.0)
                    % 13.0 + 9.0
                )
                dx = remainder_x - offset_x * 2
                dz = remainder_z - offset_z * 2
                influence = np.clip(
                    100.0 - np.sqrt(float(dx * dx + dz * dz)) * falloff,
                    -100.0, 80.0,
                )
                value = max(value, influence)
    return float(value)


class VanillaTerrainSampler:
    """Query vanilla base terrain columns for one seed and dimension."""

    def __init__(self, seed, dimension='overworld'):
        if dimension not in SETTINGS:
            raise ValueError(f'Unsupported dimension: {dimension}')
        self.seed = int(seed)
        self.dimension = dimension
        self.settings = SETTINGS[dimension]
        self.horizontal_resolution = self.settings.size_horizontal * 4
        self.vertical_resolution = self.settings.size_vertical * 4
        self.noise_size_y = self.settings.height // self.vertical_resolution
        self.biome_source = (
            OverworldBiomeSource(seed) if dimension == 'overworld'
            else NetherBiomeSource(seed) if dimension == 'nether'
            else None
        )

        random = MinecraftLCG(seed)
        self.lower = OctavePerlinNoiseSampler(random, range(-15, 1))
        self.upper = OctavePerlinNoiseSampler(random, range(-15, 1))
        self.interpolation = OctavePerlinNoiseSampler(random, range(-7, 1))
        if self.settings.simplex_surface:
            self.surface_depth = OctaveSimplexNoiseSampler(random, range(-3, 1))
        else:
            self.surface_depth = OctavePerlinNoiseSampler(random, range(-3, 1))
        random.advance(2620)
        self.density_offset_noise = OctavePerlinNoiseSampler(random, range(-15, 1))
        self.end_simplex = SimplexNoise2D(seed) if self.settings.island_override else None

    @lru_cache(maxsize=262144)
    def biome_id(self, biome_x, biome_z):
        if self.biome_source is None:
            return 9
        return int(self.biome_source.sample(int(biome_x), int(biome_z)))

    @lru_cache(maxsize=262144)
    def _shape_at(self, noise_x, noise_z):
        if self.dimension == 'end':
            terrain = _end_noise_at(self.end_simplex, noise_x, noise_z) - 8.0
            return terrain, 0.25 if terrain > 0.0 else 1.0

        if self.dimension == 'nether':
            center_depth = 0.1
        else:
            center_depth = BIOME_DEPTH_SCALE[self.biome_id(noise_x, noise_z)][0]
        weighted_scale = 0.0
        weighted_depth = 0.0
        total_weight = 0.0
        for dz in range(-2, 3):
            for dx in range(-2, 3):
                if self.dimension == 'nether':
                    depth, scale = 0.1, 0.2
                else:
                    depth, scale = BIOME_DEPTH_SCALE[
                        self.biome_id(noise_x + dx, noise_z + dz)
                    ]
                adjusted_weight = (0.5 if depth > center_depth else 1.0)
                adjusted_weight *= BIOME_WEIGHT[dz + 2, dx + 2] / (depth + 2.0)
                weighted_scale += scale * adjusted_weight
                weighted_depth += depth * adjusted_weight
                total_weight += adjusted_weight
        mean_depth = weighted_depth / total_weight
        mean_scale = weighted_scale / total_weight
        terrain = (mean_depth * 0.5 - 0.125) * 0.265625
        stretch = 96.0 / (mean_scale * 0.9 + 0.1)
        return terrain, stretch

    def _sample_interpolated_noise(self, x, y, z):
        settings = self.settings
        horizontal_scale = 684.412 * settings.xz_scale
        vertical_scale = 684.412 * settings.y_scale
        horizontal_stretch = horizontal_scale / settings.xz_factor
        vertical_stretch = vertical_scale / settings.y_factor
        lower = 0.0
        upper = 0.0
        interpolation = 0.0
        frequency = 1.0
        for octave in range(16):
            sx = maintain_precision(np.asarray(x) * horizontal_scale * frequency)
            sy = maintain_precision(np.asarray(y) * vertical_scale * frequency)
            sz = maintain_precision(np.asarray(z) * horizontal_scale * frequency)
            y_step = vertical_scale * frequency
            lower_sampler = self.lower.get_octave(octave)
            upper_sampler = self.upper.get_octave(octave)
            if lower_sampler is not None:
                lower = lower + lower_sampler.sample(
                    sx, sy, sz, y_step, np.asarray(y) * y_step,
                ) / frequency
            if upper_sampler is not None:
                upper = upper + upper_sampler.sample(
                    sx, sy, sz, y_step, np.asarray(y) * y_step,
                ) / frequency
            if octave < 8:
                interpolation_sampler = self.interpolation.get_octave(octave)
                if interpolation_sampler is not None:
                    interpolation = interpolation + interpolation_sampler.sample(
                        maintain_precision(np.asarray(x) * horizontal_stretch * frequency),
                        maintain_precision(np.asarray(y) * vertical_stretch * frequency),
                        maintain_precision(np.asarray(z) * horizontal_stretch * frequency),
                        vertical_stretch * frequency,
                        np.asarray(y) * vertical_stretch * frequency,
                    ) / frequency
            frequency /= 2.0
        return clamped_lerp(
            lower / 512.0,
            upper / 512.0,
            (interpolation / 10.0 + 1.0) / 2.0,
        )

    def _random_density(self, x, z):
        if not self.settings.random_density_offset:
            return np.zeros(np.broadcast(np.asarray(x), np.asarray(z)).shape)
        value = self.density_offset_noise.sample(
            np.asarray(x) * 200.0, 10.0, np.asarray(z) * 200.0,
            1.0, 0.0, True,
        )
        adjusted = np.where(value < 0.0, -value * 0.3, value)
        result = adjusted * 24.575625 - 2.0
        return np.where(
            result < 0.0,
            result * 0.009486607142857142,
            np.minimum(result, 1.0) * 0.006640625,
        )

    def _sample_noise_columns(self, noise_x, noise_z):
        noise_x, noise_z = np.broadcast_arrays(
            np.asarray(noise_x, dtype=np.int64),
            np.asarray(noise_z, dtype=np.int64),
        )
        flat_x = noise_x.ravel()
        flat_z = noise_z.ravel()
        y = np.arange(self.noise_size_y + 1, dtype=float)[None, :]
        x = flat_x[:, None]
        z = flat_z[:, None]
        noise = self._sample_interpolated_noise(x, y, z)
        shapes = np.asarray([
            self._shape_at(int(px), int(pz))
            for px, pz in zip(flat_x, flat_z)
        ], dtype=float)
        random_density = self._random_density(flat_x, flat_z)[:, None]
        vertical = 1.0 - y * 2.0 / self.noise_size_y + random_density
        shaped = vertical * self.settings.density_factor + self.settings.density_offset
        shaped = (shaped + shapes[:, 0, None]) * shapes[:, 1, None]
        noise = noise + np.where(shaped > 0.0, shaped * 4.0, shaped)

        if self.settings.top_size > 0.0:
            delta = (
                self.noise_size_y - y - self.settings.top_offset
            ) / self.settings.top_size
            noise = clamped_lerp(self.settings.top_target, noise, delta)
        if self.settings.bottom_size > 0.0:
            delta = (y - self.settings.bottom_offset) / self.settings.bottom_size
            noise = clamped_lerp(self.settings.bottom_target, noise, delta)
        return noise.reshape((*noise_x.shape, self.noise_size_y + 1))

    def _horizontal_columns(self, block_x, block_z):
        block_x, block_z = np.broadcast_arrays(
            np.asarray(block_x, dtype=np.int64), np.asarray(block_z, dtype=np.int64),
        )
        noise_x = np.floor_divide(block_x, self.horizontal_resolution)
        noise_z = np.floor_divide(block_z, self.horizontal_resolution)
        fraction_x = np.mod(block_x, self.horizontal_resolution) / self.horizontal_resolution
        fraction_z = np.mod(block_z, self.horizontal_resolution) / self.horizontal_resolution
        c00 = self._sample_noise_columns(noise_x, noise_z)
        c01 = self._sample_noise_columns(noise_x, noise_z + 1)
        c10 = self._sample_noise_columns(noise_x + 1, noise_z)
        c11 = self._sample_noise_columns(noise_x + 1, noise_z + 1)
        low_z = c00 + fraction_x[..., None] * (c10 - c00)
        high_z = c01 + fraction_x[..., None] * (c11 - c01)
        return low_z + fraction_z[..., None] * (high_z - low_z)

    def height_points(self, block_x, block_z, batch_size=512):
        block_x, block_z = np.broadcast_arrays(
            np.asarray(block_x, dtype=np.int64), np.asarray(block_z, dtype=np.int64),
        )
        shape = block_x.shape
        flat_x, flat_z = block_x.ravel(), block_z.ravel()
        output = np.empty(flat_x.size, dtype=np.int16)
        for start in range(0, flat_x.size, int(batch_size)):
            stop = min(start + int(batch_size), flat_x.size)
            columns = self._horizontal_columns(flat_x[start:stop], flat_z[start:stop])
            heights = np.full(
                stop - start,
                self.settings.sea_level if self.settings.has_fluid else 0,
                dtype=np.int16,
            )
            unresolved = np.ones(stop - start, dtype=bool)
            for cell in range(self.noise_size_y - 1, -1, -1):
                lower = columns[:, cell]
                upper = columns[:, cell + 1]
                for step in range(self.vertical_resolution - 1, -1, -1):
                    fraction = step / self.vertical_resolution
                    density = lower + fraction * (upper - lower)
                    solid = unresolved & (density > 0.0)
                    if np.any(solid):
                        heights[solid] = cell * self.vertical_resolution + step + 1
                        unresolved[solid] = False
                    if not np.any(unresolved):
                        break
                if not np.any(unresolved):
                    break
            output[start:stop] = heights
        return output.reshape(shape)

    def density_points(self, block_x, y, block_z, batch_size=1024):
        block_x, block_z, y = np.broadcast_arrays(
            np.asarray(block_x, dtype=np.int64),
            np.asarray(block_z, dtype=np.int64),
            np.asarray(y, dtype=float),
        )
        shape = block_x.shape
        flat_x, flat_z, flat_y = block_x.ravel(), block_z.ravel(), y.ravel()
        output = np.empty(flat_x.size, dtype=float)
        for start in range(0, flat_x.size, int(batch_size)):
            stop = min(start + int(batch_size), flat_x.size)
            columns = self._horizontal_columns(flat_x[start:stop], flat_z[start:stop])
            cell = np.clip(
                np.floor_divide(flat_y[start:stop].astype(int), self.vertical_resolution),
                0, self.noise_size_y - 1,
            )
            fraction = np.mod(flat_y[start:stop], self.vertical_resolution) / self.vertical_resolution
            row = np.arange(stop - start)
            lower = columns[row, cell]
            upper = columns[row, cell + 1]
            output[start:stop] = lower + fraction * (upper - lower)
        return output.reshape(shape)


def sampled_height_grid(seed, dimension, resolution, x_extent, z_extent=None):
    if z_extent is None:
        z_extent = x_extent
    x_values = np.rint(np.linspace(x_extent[0], x_extent[1], int(resolution))).astype(np.int64)
    z_values = np.rint(np.linspace(z_extent[0], z_extent[1], int(resolution))).astype(np.int64)
    x, z = np.meshgrid(x_values, z_values)
    sampler = VanillaTerrainSampler(seed, dimension)
    return x_values, z_values, sampler.height_points(x, z)
