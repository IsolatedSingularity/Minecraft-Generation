"""Java 1.16.1 End geometry and seed-derived outer-island sampling."""

import math

import numpy as np
from scipy.ndimage import distance_transform_edt

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

    def sample_grid(self, x, z):
        """Vectorized equivalent of :meth:`sample` for NumPy arrays."""
        x, z = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(z, dtype=float),
        )
        permutations = np.asarray(self.permutations, dtype=np.int64)
        hairy = (x + z) * self.skew
        hairy_x = np.floor(x + hairy).astype(np.int64)
        hairy_z = np.floor(z + hairy).astype(np.int64)
        mixed = (hairy_x + hairy_z) * self.unskew
        x0 = x - (hairy_x - mixed)
        z0 = z - (hairy_z - mixed)
        second_x = (x0 > z0).astype(np.int64)
        second_z = 1 - second_x
        x1 = x0 - second_x + self.unskew
        z1 = z0 - second_z + self.unskew
        x2 = x0 - 1.0 + 2.0 * self.unskew
        z2 = z0 - 1.0 + 2.0 * self.unskew
        ii = hairy_x & 255
        jj = hairy_z & 255
        grad0 = permutations[(ii + permutations[jj]) & 255] % 12
        grad1 = permutations[
            (ii + second_x + permutations[(jj + second_z) & 255]) & 255
        ] % 12
        grad2 = permutations[
            (ii + 1 + permutations[(jj + 1) & 255]) & 255
        ] % 12

        def corner(hash_values, corner_x, corner_z):
            contribution = 0.5 - corner_x * corner_x - corner_z * corner_z
            active = contribution > 0.0
            squared = np.where(active, contribution * contribution, 0.0)
            gradients = GRADIENTS[hash_values]
            dot = gradients[..., 0] * corner_x + gradients[..., 1] * corner_z
            return np.where(active, squared * squared * dot, 0.0)

        return 70.0 * (
            corner(grad0, x0, z0)
            + corner(grad1, x1, z1)
            + corner(grad2, x2, z2)
        )


def outer_island_seed_field(
    world_seed, max_coordinate_blocks=18000, batch_rows=96,
):
    """Enumerate every qualifying Java 1.16.1 outer-island seed site.

    Minecraft evaluates the simplex branch on the complete chunk lattice.
    Returning the full field avoids the spatial bias of a random accepted-site
    sample while keeping the data compact enough for a rasterized overview.
    The result uses a square coordinate window, matching the plotted X/Z view.
    """
    simplex = SimplexNoise2D(world_seed)
    max_chunk = int(max_coordinate_blocks) // 16
    chunk_x = np.arange(-max_chunk, max_chunk + 1, dtype=np.int64)
    found_x = []
    found_z = []
    for start_z in range(-max_chunk, max_chunk + 1, int(batch_rows)):
        stop_z = min(start_z + int(batch_rows), max_chunk + 1)
        chunk_z = np.arange(start_z, stop_z, dtype=np.int64)[:, None]
        x_grid = chunk_x[None, :]
        noise = simplex.sample_grid(x_grid, chunk_z)
        outside_central_void = x_grid * x_grid + chunk_z * chunk_z > 4096
        row_indices, column_indices = np.nonzero(
            outside_central_void & (noise < -0.9)
        )
        if len(row_indices):
            found_x.append(chunk_x[column_indices])
            found_z.append(chunk_z[:, 0][row_indices])

    if not found_x:
        empty = np.array([], dtype=np.int64)
        return {
            'chunk_x': empty,
            'chunk_z': empty,
            'block_x': empty,
            'block_z': empty,
            'falloff': np.array([], dtype=float),
            'visual_radius': np.array([], dtype=float),
        }

    site_x = np.concatenate(found_x)
    site_z = np.concatenate(found_z)
    falloff = (
        (np.abs(site_x) * 3439 + np.abs(site_z) * 147) % 13 + 9
    ).astype(float)
    return {
        'chunk_x': site_x,
        'chunk_z': site_z,
        'block_x': site_x * 16,
        'block_z': site_z * 16,
        'falloff': falloff,
        # Radius of the source cone at a representative visible-density cut.
        'visual_radius': 8.0 * 180.0 / falloff,
    }


def outer_island_projection(
    world_seed, max_coordinate_blocks=18000, resolution=901,
):
    """Project the complete outer-island seed field into visible footprints."""
    sites = outer_island_seed_field(
        world_seed, max_coordinate_blocks=max_coordinate_blocks,
    )
    resolution = int(resolution)
    coordinates = np.linspace(
        -float(max_coordinate_blocks), float(max_coordinate_blocks), resolution,
    )
    pixel_step = float(coordinates[1] - coordinates[0])
    projection = np.zeros((resolution, resolution), dtype=np.float32)
    for falloff in range(9, 22):
        selected = sites['falloff'] == float(falloff)
        if not np.any(selected):
            continue
        columns = np.rint(
            (sites['block_x'][selected] + max_coordinate_blocks) / pixel_step
        ).astype(np.int64)
        rows = np.rint(
            (sites['block_z'][selected] + max_coordinate_blocks) / pixel_step
        ).astype(np.int64)
        columns = np.clip(columns, 0, resolution - 1)
        rows = np.clip(rows, 0, resolution - 1)
        impulses = np.zeros((resolution, resolution), dtype=bool)
        impulses[rows, columns] = True
        distance = distance_transform_edt(~impulses, sampling=pixel_step)
        influence = np.clip(
            (180.0 - distance * float(falloff) / 8.0) / 180.0,
            0.0, 1.0,
        )
        projection = np.maximum(projection, influence.astype(np.float32))
    terraces = np.floor(projection * 7.0) / 7.0
    return coordinates, coordinates, np.ma.masked_where(
        projection <= 0.0, terraces,
    )


def central_island_projection(
    world_seed, extent_blocks=88.0, resolution=241,
):
    """Return a deterministic top-down central-island visual projection.

    The radial envelope follows the Java 1.16.1 End island-density term. A
    seeded simplex modulation supplies the terraced, irregular surface edge
    visible in top-down gameplay. This is a source-shaped projection rather
    than a claim of block-perfect chunk generation.
    """
    coordinates = np.linspace(
        -float(extent_blocks), float(extent_blocks), int(resolution),
    )
    x, z = np.meshgrid(coordinates, coordinates)
    simplex = SimplexNoise2D(world_seed)
    broad = simplex.sample_grid(x / 46.0 + 11.0, z / 46.0 - 7.0)
    detail = simplex.sample_grid(x / 19.0 - 23.0, z / 19.0 + 17.0)
    source_radius = 80.0 + 7.0 * broad + 3.5 * detail
    radial_distance = np.hypot(x, z)
    surface = np.clip((source_radius - radial_distance) / 34.0, 0.0, 1.0)
    terraces = np.floor(surface * 8.0) / 8.0
    return coordinates, coordinates, np.ma.masked_where(surface <= 0.0, terraces)


def end_overflow_ring_boundaries(max_radius_blocks):
    """Return exact axis boundaries of the Java End overflow rings.

    Odd entries mark the first affected eight-block cell of a NaN void band,
    while even entries resume normal terrain. Coordinates are aligned to the
    eight-block sampling lattice used by the End island-density function.
    """
    boundaries = []
    index = 1
    while True:
        raw = 8.0 * math.sqrt(index * (2 ** 31))
        if index % 2:
            boundary = math.floor(raw / 8.0) * 8
            kind = 'void'
        else:
            boundary = math.ceil(raw / 8.0) * 8
            kind = 'terrain'
        if boundary > int(max_radius_blocks):
            break
        boundaries.append({
            'index': index,
            'radius': int(boundary),
            'kind': kind,
        })
        index += 1
    return boundaries


def end_overflow_generation_mask(block_x, block_z):
    """Return where the Java 32-bit radial term remains non-negative."""
    x, z = np.broadcast_arrays(
        np.asarray(block_x, dtype=float), np.asarray(block_z, dtype=float),
    )
    sample_x = np.trunc(x / 8.0).astype(np.int64)
    sample_z = np.trunc(z / 8.0).astype(np.int64)
    unsigned = (sample_x * sample_x + sample_z * sample_z) & 0xFFFFFFFF
    signed = np.where(unsigned >= 0x80000000, unsigned - 0x100000000, unsigned)
    return signed >= 0


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
