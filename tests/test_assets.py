"""Integrity checks for the retained README animations."""

from pathlib import Path
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / 'Plots'

RENDERED_GIFS = {
    'dragon_pathfinding_hero.gif': (1200, 700, 420),
    'dragon_holding_strafe.gif': (900, 500, 88),
    'dragon_landing_perch.gif': (900, 500, 88),
    'dragon_takeoff.gif': (900, 500, 80),
    # The 3 s final hold is coalesced into the final optimized GIF frame.
    'dragon_trajectory_ensemble.gif': (1400, 800, 180),
    # Unchanged wave cells are coalesced while their timing is retained.
    'seed_loading.gif': (1200, 700, 20),
    'structure_placement.gif': (1000, 550, 90),
    'multi_structure_generation.gif': (1000, 550, 80),
    'redstone_quasi_connectivity.gif': (900, 550, 45),
}

README_ASSETS = (
    'dragon_pathfinding.gif',
    'dragon_pathfinding_hero.gif',
    'dragon_holding_strafe.gif',
    'dragon_landing_perch.gif',
    'dragon_takeoff.gif',
    'dragon_trajectory_ensemble.gif',
    'lcg_bit_extraction.png',
    'brownian_noise_composition.png',
    'end_dimension_overview.png',
    'end_structure_generation.png',
    'seed_loading.gif',
    'structure_placement.gif',
    'multi_structure_generation.gif',
    'stronghold_rings.png',
    'world_generation_flow.svg',
    'noise_composition_flow.svg',
    'dragon_navigation_flow.svg',
    'structure_candidate_flow.svg',
)


class AssetIntegrityTests(unittest.TestCase):
    def test_retained_new_gifs_decode_and_are_bounded(self):
        maximum_sizes = {
            'dragon_pathfinding_hero.gif': 32 * 1024 * 1024,
            'dragon_trajectory_ensemble.gif': 36 * 1024 * 1024,
            'structure_placement.gif': 15 * 1024 * 1024,
            'multi_structure_generation.gif': 17 * 1024 * 1024,
        }
        for name, dimensions in RENDERED_GIFS.items():
            minimum_width, minimum_height, minimum_frames = dimensions
            asset = PLOTS / name
            self.assertTrue(asset.is_file(), name)
            with Image.open(asset) as image:
                self.assertGreaterEqual(image.width, minimum_width, name)
                self.assertGreaterEqual(image.height, minimum_height, name)
                self.assertGreaterEqual(image.n_frames, minimum_frames, name)
                image.seek(image.n_frames - 1)
                image.convert('RGB').getpixel((0, 0))
            self.assertLess(
                asset.stat().st_size,
                maximum_sizes.get(name, 8 * 1024 * 1024),
                name,
            )

    def test_new_static_end_structure_figure_is_readable(self):
        for name in (
            'end_structure_generation.png',
            'lcg_bit_extraction.png',
            'brownian_noise_composition.png',
        ):
            with Image.open(PLOTS / name) as image:
                self.assertGreaterEqual(image.width, 2200, name)
                self.assertGreaterEqual(image.height, 1000, name)

    def test_deliberately_slow_animations_retain_their_timing(self):
        expected_duration_ranges = {
            'dragon_pathfinding_hero.gif': (34_000, 38_000),
            'dragon_holding_strafe.gif': (6_000, 8_000),
            'dragon_landing_perch.gif': (6_000, 8_000),
            'dragon_takeoff.gif': (6_000, 8_000),
            'dragon_trajectory_ensemble.gif': (21_000, 25_000),
            'seed_loading.gif': (7_000, 7_500),
            'structure_placement.gif': (10_000, 13_500),
            'multi_structure_generation.gif': (10_000, 14_000),
            'redstone_quasi_connectivity.gif': (13_000, 16_000),
        }
        for name, (minimum_ms, maximum_ms) in expected_duration_ranges.items():
            with Image.open(PLOTS / name) as image:
                total_ms = 0
                for frame_index in range(image.n_frames):
                    image.seek(frame_index)
                    total_ms += image.info.get('duration', 0)
            self.assertGreaterEqual(total_ms, minimum_ms, name)
            self.assertLessEqual(total_ms, maximum_ms, name)

    def test_trajectory_final_state_lingers(self):
        with Image.open(PLOTS / 'dragon_trajectory_ensemble.gif') as image:
            final_dwell_ms = 0
            for frame_index in range(max(0, image.n_frames - 25), image.n_frames):
                image.seek(frame_index)
                final_dwell_ms += image.info.get('duration', 0)
            self.assertGreaterEqual(final_dwell_ms, 2_000)

    def test_readme_references_restored_and_retained_assets(self):
        root_text = (ROOT / 'README.md').read_text(encoding='utf-8')
        for name in README_ASSETS:
            self.assertTrue((PLOTS / name).is_file(), name)
            self.assertIn(f'Plots/{name}', root_text, name)
        self.assertIn('## Legacy Simulations', root_text)
        self.assertTrue(root_text.rstrip().endswith('</details>'))

    def test_active_readmes_exclude_redstone_content(self):
        for relative in ('README.md', 'Code/README.md', 'Plots/README.md'):
            text = (ROOT / relative).read_text(encoding='utf-8').lower()
            self.assertNotIn('redstone', text, relative)
            self.assertNotIn('quasi-connectivity', text, relative)
            self.assertNotIn('mc-108', text, relative)

    def test_active_documentation_has_no_em_dash(self):
        for relative in ('README.md', 'Code/README.md', 'Plots/README.md'):
            text = (ROOT / relative).read_text(encoding='utf-8')
            self.assertNotIn('\N{EM DASH}', text, relative)

    def test_root_readme_uses_static_flow_figures(self):
        text = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertNotIn('```mermaid', text)
        for name in (
            'world_generation_flow.svg', 'noise_composition_flow.svg',
            'dragon_navigation_flow.svg', 'structure_candidate_flow.svg',
        ):
            asset = PLOTS / name
            self.assertTrue(asset.is_file(), name)
            self.assertIn(f'Plots/{name}', text, name)

    def test_root_readme_uses_human_section_titles(self):
        text = (ROOT / 'README.md').read_text(encoding='utf-8')
        for mechanical_title in (
            '#### The idea', '#### The matching code',
            '#### The mathematics', '#### The animation',
        ):
            self.assertNotIn(mechanical_title, text)


if __name__ == '__main__':
    unittest.main()
