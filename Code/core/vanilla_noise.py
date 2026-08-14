"""Java 1.16.1 random and noise samplers used by world generation.

This is a direct numerical port of ``PerlinNoiseSampler``,
``OctavePerlinNoiseSampler``, and ``DoublePerlinNoiseSampler``.  The classes
accept either scalars or NumPy arrays so publication maps can sample the same
algorithms without replacing them with image-space noise.
"""

import math

import numpy as np

from .lcg import MinecraftLCG, to_signed_long


PERLIN_GRADIENTS = np.asarray([
    (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
    (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
    (1, 1, 0), (0, -1, 1), (-1, 1, 0), (0, -1, -1),
], dtype=float)


class SimplexNoiseSampler:
    """Two-dimensional simplex sampler constructed from an existing RNG."""

    skew = 0.5 * (math.sqrt(3.0) - 1.0)
    unskew = (3.0 - math.sqrt(3.0)) / 6.0

    def __init__(self, random):
        self.origin_x = random.next_double() * 256.0
        self.origin_y = random.next_double() * 256.0
        self.origin_z = random.next_double() * 256.0
        permutations = np.arange(256, dtype=np.int16)
        for index in range(256):
            selected = index + random.next_int(256 - index)
            permutations[index], permutations[selected] = (
                permutations[selected], permutations[index],
            )
        self.permutations = permutations

    def sample(self, x, z):
        x, z = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(z, dtype=float))
        hairy = (x + z) * self.skew
        section_x = np.floor(x + hairy).astype(np.int64)
        section_z = np.floor(z + hairy).astype(np.int64)
        mixed = (section_x + section_z) * self.unskew
        x0 = x - (section_x - mixed)
        z0 = z - (section_z - mixed)
        step_x = (x0 > z0).astype(np.int64)
        step_z = 1 - step_x
        x1 = x0 - step_x + self.unskew
        z1 = z0 - step_z + self.unskew
        x2 = x0 - 1.0 + 2.0 * self.unskew
        z2 = z0 - 1.0 + 2.0 * self.unskew
        permutations = self.permutations
        section_x &= 255
        section_z &= 255
        h0 = permutations[(section_x + permutations[section_z]) & 255] % 12
        h1 = permutations[
            (section_x + step_x + permutations[(section_z + step_z) & 255]) & 255
        ] % 12
        h2 = permutations[
            (section_x + 1 + permutations[(section_z + 1) & 255]) & 255
        ] % 12

        def corner(hash_value, cx, cz):
            contribution = 0.5 - cx * cx - cz * cz
            active = contribution >= 0.0
            squared = np.where(active, contribution * contribution, 0.0)
            gradient = PERLIN_GRADIENTS[hash_value]
            return squared * squared * (gradient[..., 0] * cx + gradient[..., 1] * cz)

        result = 70.0 * (
            corner(h0, x0, z0) + corner(h1, x1, z1) + corner(h2, x2, z2)
        )
        return float(result) if result.ndim == 0 else result


class OctaveSimplexNoiseSampler:
    """The simplex octave stack used by the Overworld surface builder."""

    def __init__(self, random, octaves):
        octaves = sorted(set(int(value) for value in octaves))
        first_index = -octaves[0]
        last_octave = octaves[-1]
        count = first_index + last_octave + 1
        initial = SimplexNoiseSampler(random)
        self.samplers = [None] * count
        if 0 <= last_octave < count and 0 in octaves:
            self.samplers[last_octave] = initial
        for sampler_index in range(last_octave + 1, count):
            octave = last_octave - sampler_index
            if sampler_index >= 0 and octave in octaves:
                self.samplers[sampler_index] = SimplexNoiseSampler(random)
            else:
                random.advance(262)
        if last_octave > 0:
            raise NotImplementedError('Positive simplex octaves are not used by 1.16.1 terrain')
        self.frequency = 2.0 ** last_octave
        self.persistence = 1.0 / (2.0 ** count - 1.0)

    def sample(self, x, z, use_origin=True):
        result = 0.0
        frequency = self.frequency
        persistence = self.persistence
        for sampler in self.samplers:
            if sampler is not None:
                offset_x = sampler.origin_x if use_origin else 0.0
                offset_z = sampler.origin_y if use_origin else 0.0
                result = result + sampler.sample(
                    np.asarray(x) * frequency + offset_x,
                    np.asarray(z) * frequency + offset_z,
                ) * persistence
            frequency /= 2.0
            persistence *= 2.0
        return result


def _fade(value):
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _lerp(delta, start, end):
    return start + delta * (end - start)


def _lerp3(dx, dy, dz, n000, n100, n010, n110, n001, n101, n011, n111):
    return _lerp(
        dz,
        _lerp(dy, _lerp(dx, n000, n100), _lerp(dx, n010, n110)),
        _lerp(dy, _lerp(dx, n001, n101), _lerp(dx, n011, n111)),
    )


def maintain_precision(value):
    value = np.asarray(value, dtype=float)
    return value - np.floor(value / 33_554_432.0 + 0.5) * 33_554_432.0


class PerlinNoiseSampler:
    """Bit-compatible construction and source-equivalent Perlin sampling."""

    def __init__(self, random):
        self.origin_x = random.next_double() * 256.0
        self.origin_y = random.next_double() * 256.0
        self.origin_z = random.next_double() * 256.0
        permutations = np.arange(256, dtype=np.int16)
        for index in range(256):
            selected = index + random.next_int(256 - index)
            permutations[index], permutations[selected] = (
                permutations[selected], permutations[index],
            )
        self.permutations = permutations

    def _gradient(self, value):
        return self.permutations[np.asarray(value, dtype=np.int64) & 255]

    def sample(self, x, y, z, y_scale=0.0, y_max=0.0):
        x, y, z = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
            np.asarray(z, dtype=float),
        )
        shifted_x = x + self.origin_x
        shifted_y = y + self.origin_y
        shifted_z = z + self.origin_z
        section_x = np.floor(shifted_x).astype(np.int64)
        section_y = np.floor(shifted_y).astype(np.int64)
        section_z = np.floor(shifted_z).astype(np.int64)
        local_x = shifted_x - section_x
        local_y = shifted_y - section_y
        local_z = shifted_z - section_z
        fade_x = _fade(local_x)
        fade_y = _fade(local_y)
        fade_z = _fade(local_z)
        if y_scale != 0.0:
            snapped = np.floor(
                np.minimum(np.asarray(y_max, dtype=float), local_y) / y_scale
            ) * y_scale
        else:
            snapped = 0.0
        local_y_sample = local_y - snapped

        gx0 = self._gradient(section_x)
        gx1 = self._gradient(section_x + 1)
        g000 = self._gradient(gx0 + section_y)
        g010 = self._gradient(gx0 + section_y + 1)
        g100 = self._gradient(gx1 + section_y)
        g110 = self._gradient(gx1 + section_y + 1)

        hashes = (
            self._gradient(g000 + section_z),
            self._gradient(g100 + section_z),
            self._gradient(g010 + section_z),
            self._gradient(g110 + section_z),
            self._gradient(g000 + section_z + 1),
            self._gradient(g100 + section_z + 1),
            self._gradient(g010 + section_z + 1),
            self._gradient(g110 + section_z + 1),
        )
        offsets = (
            (local_x, local_y_sample, local_z),
            (local_x - 1.0, local_y_sample, local_z),
            (local_x, local_y_sample - 1.0, local_z),
            (local_x - 1.0, local_y_sample - 1.0, local_z),
            (local_x, local_y_sample, local_z - 1.0),
            (local_x - 1.0, local_y_sample, local_z - 1.0),
            (local_x, local_y_sample - 1.0, local_z - 1.0),
            (local_x - 1.0, local_y_sample - 1.0, local_z - 1.0),
        )
        values = []
        for hash_value, (ox, oy, oz) in zip(hashes, offsets):
            gradient = PERLIN_GRADIENTS[np.asarray(hash_value) & 15]
            values.append(
                gradient[..., 0] * ox
                + gradient[..., 1] * oy
                + gradient[..., 2] * oz
            )
        result = _lerp3(fade_x, fade_y, fade_z, *values)
        return float(result) if result.ndim == 0 else result


class OctavePerlinNoiseSampler:
    """Source-order octave stack, including discarded RNG consumption."""

    def __init__(self, random, octaves):
        octaves = sorted(set(int(value) for value in octaves))
        if not octaves:
            raise ValueError('Need some octaves')
        first_index = -octaves[0]
        last_octave = octaves[-1]
        count = first_index + last_octave + 1
        if count < 1:
            raise ValueError('Total number of octaves must be positive')

        initial = PerlinNoiseSampler(random)
        self.samplers = [None] * count
        if 0 <= last_octave < count and 0 in octaves:
            self.samplers[last_octave] = initial
        for sampler_index in range(last_octave + 1, count):
            octave = last_octave - sampler_index
            if sampler_index >= 0 and octave in octaves:
                self.samplers[sampler_index] = PerlinNoiseSampler(random)
            else:
                random.advance(262)
        if last_octave > 0:
            derived_seed = int(initial.sample(0.0, 0.0, 0.0) * 9.223372e18)
            derived = MinecraftLCG(to_signed_long(derived_seed))
            for sampler_index in range(last_octave - 1, -1, -1):
                octave = last_octave - sampler_index
                if sampler_index < count and octave in octaves:
                    self.samplers[sampler_index] = PerlinNoiseSampler(derived)
                else:
                    derived.advance(262)
        self.frequency = 2.0 ** last_octave
        self.persistence = 1.0 / (2.0 ** count - 1.0)

    def sample(self, x, y, z, y_scale=0.0, y_max=0.0, use_origin_y=False):
        result = 0.0
        frequency = self.frequency
        persistence = self.persistence
        for sampler in self.samplers:
            if sampler is not None:
                sample_y = -sampler.origin_y if use_origin_y else maintain_precision(np.asarray(y) * frequency)
                result = result + sampler.sample(
                    maintain_precision(np.asarray(x) * frequency),
                    sample_y,
                    maintain_precision(np.asarray(z) * frequency),
                    y_scale * frequency,
                    y_max * frequency,
                ) * persistence
            frequency /= 2.0
            persistence *= 2.0
        return result

    def get_octave(self, index):
        return self.samplers[int(index)]


class DoublePerlinNoiseSampler:
    """The paired, slightly offset Perlin stack used by Nether climates."""

    def __init__(self, seed, octaves=(-7, -6)):
        random = MinecraftLCG(seed)
        octaves = tuple(int(value) for value in octaves)
        self.first = OctavePerlinNoiseSampler(random, octaves)
        self.second = OctavePerlinNoiseSampler(random, octaves)
        span = max(octaves) - min(octaves)
        scale = 0.1 * (1.0 + 1.0 / (span + 1.0))
        self.amplitude = (1.0 / 6.0) / scale

    def sample(self, x, y, z):
        multiplier = 1.0181268882175227
        return (
            self.first.sample(x, y, z)
            + self.second.sample(
                np.asarray(x) * multiplier,
                np.asarray(y) * multiplier,
                np.asarray(z) * multiplier,
            )
        ) * self.amplitude


def clamped_lerp(start, end, delta):
    return np.where(
        np.asarray(delta) < 0.0,
        start,
        np.where(np.asarray(delta) > 1.0, end, _lerp(delta, start, end)),
    )
