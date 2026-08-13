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
    DRAGON_EDGES, DRAGON_NODES, SOURCE_PHASE_TRANSITIONS, STATE_ORDER,
    perch_probability, scripted_showcase, shortest_path,
    simulate_perch_trajectory,
)
from core.end_generation import (
    SimplexNoise2D,
    central_island_projection,
    end_overflow_generation_mask,
    end_overflow_ring_boundaries,
    gateway_positions,
    end_city_candidates,
    end_city_height_candidates,
    end_city_qualification_probability,
    outer_gateway_positions,
    outer_island_seed_field,
    pillar_seed,
    spike_layout,
)
from core.lcg import MinecraftLCG, generate_population_seed, generate_region_seed
from core.minecraft_visuals import (
    NETHER_BIOMES,
    OVERWORLD_BIOMES,
    minecraft_biome_grid,
    minecraft_nether_biome_grid,
)
from core.strongholds import generate_stronghold_candidates
from core.structures import (
    NETHER_RUINED_PORTAL,
    END_CITY,
    OCEAN_MONUMENT,
    OVERWORLD_STRUCTURES,
    VILLAGE,
    candidate_in_region,
    nether_shared_candidate,
    pillager_outpost_source_gate,
)
from dragon_pathfinding import trajectory_animation_state
from multi_structure_generation import nether_structure_candidates
from redstone_quasi_connectivity import bud_animation_state
from seed_loading import chunk_status_snapshot
from structure_placement import overworld_structure_candidates


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
        self.assertEqual(len(DRAGON_NODES), 24)
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

    def test_phase_taxonomy_matches_java_1_16_1(self):
        self.assertEqual(
            STATE_ORDER,
            [
                'holding', 'strafing', 'landing_approach', 'landing',
                'takeoff', 'sitting_flaming', 'sitting_scanning',
                'sitting_attacking', 'charging_player', 'dying', 'hover',
            ],
        )
        self.assertIn(
            ('sitting_scanning', 'charging_player'),
            SOURCE_PHASE_TRANSITIONS,
        )
        self.assertNotIn(
            ('holding', 'charging_player'), SOURCE_PHASE_TRANSITIONS,
        )

    def test_seeded_approaches_only_traverse_legal_graph_edges(self):
        legal_edges = set(DRAGON_EDGES)
        sampled_coordinates = []
        for index in range(40):
            coordinates, nodes = simulate_perch_trajectory(
                12031 + index * 7919,
                crystals_alive=10 - (index % 6),
                player_position=(34.0, -18.0),
            )
            self.assertTrue(np.all(np.isfinite(coordinates)))
            sampled_coordinates.append(coordinates)
            for left, right in zip(nodes, nodes[1:]):
                self.assertIn(tuple(sorted((left, right))), legal_edges)
        radii = np.linalg.norm(np.vstack(sampled_coordinates), axis=1)
        self.assertLess(float(np.mean(radii >= 50.0)), 0.45)

    def test_showcase_delays_and_varies_crystal_destruction(self):
        frames = scripted_showcase()
        self.assertGreaterEqual(len(frames), 260)
        first_explosion = next(
            index for index, frame in enumerate(frames)
            if frame.explosion_index is not None
        )
        self.assertGreater(first_explosion, len(frames) * 0.35)
        order = []
        for frame in frames:
            if (
                frame.explosion_index is not None
                and frame.explosion_index not in order
            ):
                order.append(frame.explosion_index)
        self.assertEqual(order, [7, 2, 9, 4])
        self.assertTrue(any(frame.fireball_position is not None for frame in frames))
        shown_states = {frame.state for frame in frames}
        self.assertEqual(shown_states, set(STATE_ORDER))

    def test_trajectory_batches_and_exact_final_hold(self):
        active_last = trajectory_animation_state(127)
        hold_first = trajectory_animation_state(128)
        hold_last = trajectory_animation_state(151)
        self.assertEqual(active_last, (240, 239, 1.0))
        self.assertEqual(hold_first, (240, 239, 1.0))
        self.assertEqual(hold_last, (240, 239, 1.0))
        shown = [trajectory_animation_state(index)[0] for index in range(152)]
        self.assertTrue(all(left <= right for left, right in zip(shown, shown[1:])))


class StructureTests(unittest.TestCase):
    def test_uniform_candidates_stay_inside_window(self):
        for config in (*OVERWORLD_STRUCTURES, NETHER_RUINED_PORTAL):
            item = candidate_in_region(42, -3, 4, config)
            self.assertGreaterEqual(item['offset_x'], 0)
            self.assertLess(item['offset_x'], config.spacing - config.separation)
            self.assertGreaterEqual(item['offset_z'], 0)
            self.assertLess(item['offset_z'], config.spacing - config.separation)

    def test_overworld_candidate_view_is_inclusive_not_biome_gated(self):
        candidates, biomes, _ = overworld_structure_candidates(
            42, region_radius=5, resolution=256,
        )
        self.assertGreater(len(candidates), 0)
        self.assertEqual(
            {item['name'] for item in candidates},
            {config.name for config in OVERWORLD_STRUCTURES},
        )
        self.assertTrue(any(
            not item['illustrative_biome_match'] for item in candidates
        ))
        self.assertEqual(biomes.shape, (256, 256))
        self.assertGreater(
            sum(item['name'] == 'village' for item in candidates), 16,
        )

    def test_outpost_direct_source_gate_and_nether_extent(self):
        outcomes = [
            pillager_outpost_source_gate(42, chunk_x, chunk_z)
            for chunk_x in range(-96, 97, 8)
            for chunk_z in range(-96, 97, 8)
        ]
        self.assertTrue(any(outcomes))
        self.assertTrue(any(not value for value in outcomes))

        shared, portals, _, (minimum, maximum) = nether_structure_candidates(
            42, region_radius=3, resolution=128,
        )
        self.assertTrue(shared)
        self.assertTrue(portals)
        self.assertTrue(all(
            minimum <= item['chunk_x'] <= maximum
            and minimum <= item['chunk_z'] <= maximum
            for item in (*shared, *portals)
        ))

    def test_expanded_structure_catalog_and_distribution_examples(self):
        self.assertEqual(
            {config.name for config in OVERWORLD_STRUCTURES},
            {
                'village', 'desert_pyramid', 'jungle_pyramid', 'swamp_hut',
                'pillager_outpost', 'igloo', 'woodland_mansion',
                'ocean_monument', 'shipwreck', 'ocean_ruin', 'ruined_portal',
            },
        )
        village = candidate_in_region(42, 0, 0, VILLAGE)
        monument = candidate_in_region(42, 0, 0, OCEAN_MONUMENT)
        end_city = candidate_in_region(42, 0, 0, END_CITY)
        self.assertEqual((village['offset_x'], village['offset_z']), (1, 20))
        self.assertEqual((monument['offset_x'], monument['offset_z']), (8, 16))
        self.assertEqual((end_city['offset_x'], end_city['offset_z']), (8, 7))

    def test_registered_biomes_are_visible_in_showcase_maps(self):
        overworld = minecraft_biome_grid(
            42, 384, (-168, 168), coordinate_scale=16.0, showcase=True,
        )
        nether = minecraft_nether_biome_grid(
            42, 384, (-140, 140), coordinate_scale=16.0, showcase=True,
        )
        self.assertEqual(set(overworld.ravel()), set(OVERWORLD_BIOMES))
        self.assertEqual(set(nether.ravel()), set(NETHER_BIOMES))

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


class ChunkStatusTests(unittest.TestCase):
    def test_dependency_wave_reaches_source_target_status_rings(self):
        stages, growth, hidden = chunk_status_snapshot(0)
        center = 50
        self.assertEqual(stages[center, center], 0)
        self.assertEqual(growth[center, center], 1.0)
        self.assertFalse(hidden[center, center])
        self.assertTrue(hidden[center, center + 1])
        self.assertTrue(np.all(hidden[:40]))

        final_generation = chunk_status_snapshot(79)
        first_hold = chunk_status_snapshot(80)
        for snapshot in (final_generation, first_hold):
            stages, growth, hidden = snapshot
            self.assertEqual(stages[center, center], 12)
            self.assertEqual(stages[center, center + 1], 8)
            self.assertEqual(stages[center, center + 2], 7)
            self.assertEqual(stages[center, center + 3], 1)
            self.assertEqual(stages[center, center + 10], 1)
            self.assertEqual(stages[center, center + 11], 0)
            self.assertEqual(growth[center, center + 10], 1.0)
            self.assertEqual(growth[center, center + 11], 0.0)
            self.assertFalse(hidden[center, center + 10])
            self.assertTrue(hidden[center, center + 11])


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

    def test_outer_gateway_pairing_and_end_city_gate(self):
        gateways = outer_gateway_positions(42)
        self.assertEqual(len(gateways), 20)
        self.assertEqual(len({(item['x'], item['z']) for item in gateways}), 20)
        self.assertTrue(all(
            1023.0 <= math.hypot(item['ideal_x'], item['ideal_z']) <= 1025.0
            for item in gateways
        ))
        cities = end_city_candidates(42, max_coordinate_blocks=2400)
        self.assertGreater(len(cities), 20)
        self.assertTrue(all(
            math.hypot(item['block_x'], item['block_z']) > 1024.0
            for item in cities
        ))

        evaluated, _, _ = end_city_height_candidates(
            42, max_coordinate_blocks=2400, resolution=401,
        )
        self.assertGreater(len(evaluated), len(cities))
        self.assertEqual(
            {item['rotation'] for item in evaluated},
            {
                'NONE', 'CLOCKWISE_90', 'CLOCKWISE_180',
                'COUNTERCLOCKWISE_90',
            },
        )
        self.assertTrue(all(len(item['sample_heights']) == 4 for item in evaluated))
        self.assertTrue(all(
            item['model_min_height'] == min(item['sample_heights'])
            for item in evaluated
        ))
        self.assertTrue(all(
            item['qualified'] == (item['model_min_height'] >= 60.0)
            for item in evaluated
        ))

        x, z, probability = end_city_qualification_probability(
            42, max_coordinate_blocks=1600,
        )
        self.assertEqual(probability.shape, (len(z), len(x)))
        self.assertAlmostEqual(float(probability.max()), 1.0 / 81.0)
        self.assertTrue(np.all(probability[np.hypot(*np.meshgrid(x, z)) <= 1024.0] == 0.0))

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
