"""Integrity checks for the four retained new README animations."""

from pathlib import Path
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / 'Plots'

NEW_GIFS = {
    'dragon_holding_strafe.gif': (900, 500, 20),
    'dragon_landing_perch.gif': (900, 500, 20),
    'dragon_takeoff.gif': (900, 500, 20),
    'dragon_trajectory_ensemble.gif': (900, 600, 80),
}

README_ASSETS = (
    'dragon_pathfinding.gif',
    'end_dimension_overview.png',
    'seed_loading.gif',
    'structure_placement.gif',
    'multi_structure_generation.gif',
    'structure_analysis.png',
    'stronghold_rings.png',
    *NEW_GIFS,
)


class AssetIntegrityTests(unittest.TestCase):
    def test_retained_new_gifs_decode_and_are_bounded(self):
        for name, (minimum_width, minimum_height, minimum_frames) in NEW_GIFS.items():
            asset = PLOTS / name
            self.assertTrue(asset.is_file(), name)
            with Image.open(asset) as image:
                self.assertGreaterEqual(image.width, minimum_width, name)
                self.assertGreaterEqual(image.height, minimum_height, name)
                self.assertGreaterEqual(image.n_frames, minimum_frames, name)
                image.seek(image.n_frames - 1)
                image.convert('RGB').getpixel((0, 0))
            self.assertLess(asset.stat().st_size, 8 * 1024 * 1024, name)

    def test_readme_references_restored_and_retained_assets(self):
        root_text = (ROOT / 'README.md').read_text(encoding='utf-8')
        for name in README_ASSETS:
            self.assertIn(f'Plots/{name}', root_text, name)

    def test_active_documentation_has_no_em_dash(self):
        for relative in ('README.md', 'Code/README.md', 'Plots/README.md'):
            text = (ROOT / relative).read_text(encoding='utf-8')
            self.assertNotIn('\N{EM DASH}', text, relative)


if __name__ == '__main__':
    unittest.main()
