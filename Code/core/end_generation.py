"""Java 1.16.1 End geometry and seed-derived outer-island sampling."""

import math

import numpy as np

from .lcg import MinecraftLCG


GRADIENTS = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
], dtype=float)


class SimplexNoise2D:
    """Port of the simplex sampler used by Java 1.16 End biomes."""

    skew = 0.5 * (math.sqrt(3.0) - 1.0)
    unskew = (3.0 - math.sqrt(3.0)) / 6.0

    def __init__(self, world_seed):
        random = MinecraftLCG(world_seed)
        random.advance(17292)
        self.origin_x = random.next_double() * 256.0
        self.origin_y = random.next_double() * 256.0
        self.origin_z = random.next_double() * 256.0
        self.permutations = list(range(256))
        for index in range(256):
            random_index = random.next_int(256 - index) + index
            self.permutations[index], self.permutations[random_index] = (
                self.permutations[random_index], self.permutations[index]
            )

    def lookup(self, value):
        return self.permutations[value & 255]

    @staticmethod
    def corner(hash_value, x, y):
        contribution = 0.5 - x * x - y * y
        if contribution < 0.0:
            return 0.0
        contribution *= contribution
        gradient = GRADIENTS[hash_value % 12]
        return contribution * contribution * (gradient[0] * x + gradient[1] * y)

    def sample(self, x, z):
        hairy = (x + z) * self.skew
        hairy_x = math.floor(x + hairy)
        hairy_z = math.floor(z + hairy)
        mixed = (hairy_x + hairy_z) * self.unskew
        x0 = x - (hairy_x - mixed)
        z0 = z - (hairy_z - mixed)
        if x0 > z0:
            second_x, second_z = 1, 0
        else:
            second_x, second_z = 0, 1
        x1 = x0 - second_x + self.unskew
        z1 = z0 - second_z + self.unskew
        x2 = x0 - 1.0 + 2.0 * self.unskew
        z2 = z0 - 1.0 + 2.0 * self.unskew
        ii = hairy_x & 255
        jj = hairy_z & 255
        grad0 = self.lookup(ii + self.lookup(jj)) % 12
        grad1 = self.lookup(
            ii + second_x + self.lookup(jj + second_z)
        ) % 12
        grad2 = self.lookup(ii + 1 + self.lookup(jj + 1)) % 12
        return 70.0 * (
            self.corner(grad0, x0, z0)
            + self.corner(grad1, x1, z1)
            + self.corner(grad2, x2, z2)
        )


def sample_outer_island_sites(world_seed, count=2600, max_radius_blocks=18000):
    """Sample sites that qualify the End source's simplex-noise branch.

    The qualification and seed path are source-faithful. Point size is a
    deterministic visual encoding, not a claim about complete island shape.
    """
    simplex = SimplexNoise2D(world_seed)
    max_chunk = max_radius_blocks // 16
    proposals = np.random.default_rng(world_seed & 0xFFFFFFFF)
    accepted = {}
    batch_size = max(12000, count * 8)
    while len(accepted) < count:
        values = proposals.integers(-max_chunk, max_chunk + 1, size=(batch_size, 2))
        for chunk_x, chunk_z in values:
            key = (int(chunk_x), int(chunk_z))
            if key in accepted:
                continue
            radius_squared = key[0] * key[0] + key[1] * key[1]
            if radius_squared <= 4096:
                continue
            if simplex.sample(key[0], key[1]) >= -0.9:
                continue
            elevation = (
                (abs(float(key[0])) * 3439.0 + abs(float(key[1])) * 147.0)
                % 13.0 + 9.0
            )
            accepted[key] = elevation
            if len(accepted) >= count:
                break
    return [
        {
            'chunk_x': chunk_x,
            'chunk_z': chunk_z,
            'block_x': chunk_x * 16,
            'block_z': chunk_z * 16,
            'elevation': elevation,
        }
        for (chunk_x, chunk_z), elevation in accepted.items()
    ]


def gateway_positions():
    """Return the exact 20 post-fight gateway ring positions."""
    values = []
    for index in range(20):
        angle = 2.0 * math.pi * index / 20.0
        values.append({
            'index': index,
            'x': math.floor(96.0 * math.cos(angle)),
            'z': math.floor(96.0 * math.sin(angle)),
        })
    return values


def pillar_seed(world_seed):
    """Derive the 16-bit End spike seed from the 64-bit world seed."""
    return MinecraftLCG(world_seed).next_long() & 0xFFFF


def spike_layout(world_seed):
    """Return the shuffled 10-spike layout used by Java 1.16.1."""
    random = MinecraftLCG(pillar_seed(world_seed))
    order = list(range(10))
    for size in range(len(order), 1, -1):
        swap_index = random.next_int(size)
        order[size - 1], order[swap_index] = order[swap_index], order[size - 1]

    spikes = []
    for index, value in enumerate(order):
        angle = 2.0 * (-math.pi + math.pi * index / 10.0)
        spikes.append({
            'index': index,
            'x': math.floor(42.0 * math.cos(angle)),
            'z': math.floor(42.0 * math.sin(angle)),
            'radius': 2 + value // 3,
            'height': 76 + value * 3,
            'caged': value in (1, 2),
        })
    return spikes
