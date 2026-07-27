"""Java Edition 1.16.1 stronghold candidate generation."""

import math

from .constants import STRONGHOLD_RINGS
from .lcg import MinecraftLCG


def java_round(value):
    """Match Java Math.round for the positive values used here."""
    return math.floor(value + 0.5)


def generate_stronghold_candidates(seed):
    """Generate the 128 Java 1.16.1 stronghold candidates.

    This follows the pre-1.19.3 ring iterator: the first candidate uses a
    random angle and radius, each ring advances by an even angular step, and
    the next ring receives a random angular rotation. The game then searches
    within 112 blocks for a valid biome, so these are approximate candidates,
    not claims of exact portal-room coordinates.
    """
    rng = MinecraftLCG(seed)
    candidates = []

    ring_index = 0
    ring_position = 0
    ring_count = STRONGHOLD_RINGS[ring_index]['count']
    angle = 2.0 * math.pi * rng.next_double()
    ring_number = 0
    distance_chunks = 4.0 * 32.0 + (rng.next_double() - 0.5) * 32.0 * 2.5

    total_count = sum(ring['count'] for ring in STRONGHOLD_RINGS)
    for index in range(total_count):
        x = java_round(math.cos(angle) * distance_chunks) * 16 + 8
        z = java_round(math.sin(angle) * distance_chunks) * 16 + 8
        ring = STRONGHOLD_RINGS[ring_index]
        candidates.append({
            'index': index + 1,
            'ring': ring_index + 1,
            'ring_index': ring_position + 1,
            'x': x,
            'z': z,
            'radius': math.hypot(x, z),
            'angle': angle,
            'distance_chunks': distance_chunks,
            'min_radius': ring['min_radius'],
            'max_radius': ring['max_radius'],
            'color': ring['color'],
        })

        ring_position += 1
        angle += 2.0 * math.pi / ring_count

        if ring_position == ring_count:
            ring_number += 1
            ring_position = 0
            if ring_index + 1 < len(STRONGHOLD_RINGS):
                ring_index += 1
                ring_count = STRONGHOLD_RINGS[ring_index]['count']
            angle += rng.next_double() * 2.0 * math.pi

        distance_chunks = (
            4.0 * 32.0
            + 6.0 * ring_number * 32.0
            + (rng.next_double() - 0.5) * 32.0 * 2.5
        )

    return candidates

