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
from core.end_generation import gateway_positions, pillar_seed, spike_layout
from core.lcg import MinecraftLCG, generate_population_seed, generate_region_seed
from core.strongholds import generate_stronghold_candidates
from core.structures import NETHER_RUINED_PORTAL, VILLAGE, candidate_in_region, nether_shared_candidate


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
        fortress_count = sum(
            nether_shared_candidate(seed, 0, 0)['name'] == 'fortress'
            for seed in range(2000)
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
    def test_gateway_and_spike_geometry(self):
        gateways = gateway_positions()
        self.assertEqual(len(gateways), 20)
        self.assertTrue(all(95.0 <= math.hypot(g['x'], g['z']) <= 97.0 for g in gateways))
        self.assertEqual(pillar_seed(42), 35575)
        spikes = spike_layout(42)
        self.assertEqual(len(spikes), 10)
        self.assertEqual(sum(spike['caged'] for spike in spikes), 2)
        self.assertEqual(sorted(spike['height'] for spike in spikes), list(range(76, 104, 3)))


if __name__ == '__main__':
    unittest.main()
