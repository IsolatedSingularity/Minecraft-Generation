"""Numerical invariants for active Java 1.16.1 visualizations."""

import math
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Code'))

from core.constants import STRONGHOLD_RINGS
from core.dragon import (
    DRAGON_EDGES, DRAGON_NODES, perch_probability, shortest_path,
)
from core.end_generation import (
    SimplexNoise2D,
    central_island_projection,
    end_overflow_generation_mask,
    end_overflow_ring_boundaries,
    gateway_positions,
    outer_island_seed_field,
    pillar_seed,
    spike_layout,
)
from core.lcg import MinecraftLCG, generate_population_seed, generate_region_seed
from core.strongholds import generate_stronghold_candidates
from core.structures import NETHER_RUINED_PORTAL, VILLAGE, candidate_in_region, nether_shared_candidate
from redstone_quasi_connectivity import bud_animation_state


class JavaRandomTests(unittest.TestCase):
    def test_known_java_random_values(self):
        self.assertEqual(MinecraftLCG(0).next_int(), -1155484576)
        self.assertEqual(MinecraftLCG(0).next_int(100), 60)
        self.assertAlmostEqual(
            MinecraftLCG(0).next_double(), 0.730967787376657,
        )
        self.assertEqual(
            MinecraftLCG(0).next_long(), -4962768465676381896,
        )

    def test_region_and_population_seed_are_deterministic(self):
        self.assertEqual(generate_region_seed(42, 0, 0, 10387312), 10387354)
        self.assertEqual(
            generate_population_seed(42, 16, -32),
            generate_population_seed(42, 16, -32),
        )


class DragonTopologyTests(unittest.TestCase):
    def test_node_rings_match_source_radii(self):
        radii = np.linalg.norm(DRAGON_NODES, axis=1)
        np.testing.assert_allclose(radii[:12], 60.0, atol=1.4)
        np.testing.assert_allclose(radii[12:20], 40.0, atol=1.1)
        np.testing.assert_allclose(radii[20:], 20.0, atol=1.1)
        self.assertEqual(len(DRAGON_EDGES), len(set(DRAGON_EDGES)))

    def test_path_and_perch_probability(self):
        path = shortest_path(0, 20, crystals_alive=10)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 20)
        self.assertAlmostEqual(perch_probability(10), 1.0 / 13.0)
        self.assertAlmostEqual(perch_probability(0), 1.0 / 3.0)


class StructureTests(unittest.TestCase):
    def test_uniform_candidates_stay_inside_window(self):
        for config in (VILLAGE, NETHER_RUINED_PORTAL):
            item = candidate_in_region(42, -3, 4, config)
            self.assertGreaterEqual(item['offset_x'], 0)
            self.assertLess(item['offset_x'], config.spacing - config.separation)
            self.assertGreaterEqual(item['offset_z'], 0)
            self.assertLess(item['offset_z'], config.spacing - config.separation)

    def test_nether_shared_split_converges_to_two_fifths(self):
        candidates = [
            nether_shared_candidate(seed, 0, 0)
            for seed in range(2000)
        ]
        for candidate in candidates:
            self.assertIn(candidate['type_roll'], range(5))
            expected_name = (
                'fortress' if candidate['type_roll'] < 2 else 'bastion'
            )
            self.assertEqual(candidate['name'], expected_name)
        fortress_count = sum(
            candidate['name'] == 'fortress' for candidate in candidates
        )
        ratio = fortress_count / 2000.0
        self.assertGreater(ratio, 0.36)
        self.assertLess(ratio, 0.44)

    def test_stronghold_ring_population(self):
        values = generate_stronghold_candidates(42)
        self.assertEqual(len(values), 128)
        counts = [
            sum(value['ring'] == index for value in values)
            for index in range(1, 9)
        ]
        self.assertEqual(counts, [ring['count'] for ring in STRONGHOLD_RINGS])


class EndGeometryTests(unittest.TestCase):
    def test_end_overflow_ring_boundaries_and_point_samples(self):
        boundaries = end_overflow_ring_boundaries(1_100_000)
        self.assertEqual(
            [(item['radius'], item['kind']) for item in boundaries],
            [
                (370_720, 'void'),
                (524_288, 'terrain'),
                (642_112, 'void'),
                (741_456, 'terrain'),
                (828_968, 'void'),
                (908_096, 'terrain'),
                (980_848, 'void'),
                (1_048_576, 'terrain'),
            ],
        )
        samples = end_overflow_generation_mask(
            np.array([370_720, 370_728, 524_280, 524_288]),
            np.zeros(4),
        )
        np.testing.assert_array_equal(
            samples, np.array([True, False, False, True]),
        )

    def test_gateway_and_spike_geometry(self):
        gateways = gateway_positions()
        self.assertEqual(len(gateways), 20)
        self.assertTrue(all(95.0 <= math.hypot(g['x'], g['z']) <= 97.0 for g in gateways))
        self.assertEqual(pillar_seed(42), 35575)
        spikes = spike_layout(42)
        self.assertEqual(len(spikes), 10)
        self.assertEqual(sum(spike['caged'] for spike in spikes), 2)
        self.assertEqual(sorted(spike['height'] for spike in spikes), list(range(76, 104, 3)))

    def test_vectorized_simplex_and_island_projection(self):
        simplex = SimplexNoise2D(42)
        x = np.array([0.0, 123.0, -456.0, 78.5])
        z = np.array([0.0, -456.0, 78.0, -31.25])
        expected = np.array([
            simplex.sample(sample_x, sample_z)
            for sample_x, sample_z in zip(x, z)
        ])
        np.testing.assert_allclose(simplex.sample_grid(x, z), expected, atol=1e-12)

        field = outer_island_seed_field(42, max_coordinate_blocks=2048)
        radii_squared = field['chunk_x'] ** 2 + field['chunk_z'] ** 2
        self.assertGreater(len(radii_squared), 0)
        self.assertTrue(np.all(radii_squared > 4096))
        expected_sites = {
            (chunk_x, chunk_z)
            for chunk_x in range(-128, 129)
            for chunk_z in range(-128, 129)
            if chunk_x * chunk_x + chunk_z * chunk_z > 4096
            and simplex.sample(chunk_x, chunk_z) < -0.9
        }
        self.assertEqual(
            set(zip(field['chunk_x'], field['chunk_z'])), expected_sites,
        )

        _, _, projection = central_island_projection(42, resolution=65)
        self.assertFalse(np.ma.getmaskarray(projection)[32, 32])
        self.assertTrue(np.ma.getmaskarray(projection)[0, 0])


class RedstoneTests(unittest.TestCase):
    def test_bud_cycle_requires_updates_for_both_edges(self):
        waiting_to_extend = bud_animation_state(20)
        self.assertTrue(waiting_to_extend.source_on)
        self.assertEqual(waiting_to_extend.extension, 0.0)

        extended_without_power = bud_animation_state(90)
        self.assertFalse(extended_without_power.source_on)
        self.assertEqual(extended_without_power.extension, 1.0)

        retracting = bud_animation_state(118)
        self.assertFalse(retracting.source_on)
        self.assertGreater(retracting.extension, 0.0)
        self.assertLess(retracting.extension, 1.0)


if __name__ == '__main__':
    unittest.main()
