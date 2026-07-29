"""Integrity checks for the active README visualization bundle."""

from pathlib import Path
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / 'Plots'

GIFS = {
    'dragon_pathfinding.gif': (1400, 800, 80),
    'dragon_holding_strafe.gif': (900, 500, 20),
    'dragon_landing_perch.gif': (900, 500, 20),
    'dragon_takeoff.gif': (900, 500, 20),
    'dragon_trajectory_ensemble.gif': (900, 600, 80),
    'seed_loading.gif': (1400, 800, 50),
    'structure_placement.gif': (1400, 800, 80),
    'multi_structure_generation.gif': (1400, 800, 80),
}

PNGS = {
    'end_dimension_overview.png': (3000, 1800),
    'structure_analysis.png': (2500, 1600),
    'stronghold_rings.png': (3000, 1800),
}


class AssetIntegrityTests(unittest.TestCase):
    def test_active_gifs_decode_and_are_bounded(self):
        for name, (minimum_width, minimum_height, minimum_frames) in GIFS.items():
            path = PLOTS / name
            self.assertTrue(path.is_file(), name)
            with Image.open(path) as image:
                self.assertGreaterEqual(image.width, minimum_width, name)
                self.assertGreaterEqual(image.height, minimum_height, name)
                self.assertGreaterEqual(image.n_frames, minimum_frames, name)
                image.seek(image.n_frames - 1)
                image.convert('RGB').getpixel((0, 0))
            self.assertLess(path.stat().st_size, 8 * 1024 * 1024, name)

    def test_hero_is_github_sized(self):
        hero = PLOTS / 'dragon_pathfinding.gif'
        self.assertLess(hero.stat().st_size, 5 * 1024 * 1024)

    def test_active_pngs_decode(self):
        for name, (minimum_width, minimum_height) in PNGS.items():
            path = PLOTS / name
            self.assertTrue(path.is_file(), name)
            with Image.open(path) as image:
                self.assertGreaterEqual(image.width, minimum_width, name)
                self.assertGreaterEqual(image.height, minimum_height, name)
                image.verify()

    def test_readmes_reference_all_active_assets(self):
        root_text = (ROOT / 'README.md').read_text(encoding='utf-8')
        for name in (*GIFS, *PNGS):
            self.assertIn(f'Plots/{name}', root_text, name)

    def test_active_documentation_has_no_em_dash(self):
        for relative in ('README.md', 'Code/README.md', 'Plots/README.md'):
            text = (ROOT / relative).read_text(encoding='utf-8')
            self.assertNotIn('\N{EM DASH}', text, relative)


if __name__ == '__main__':
    unittest.main()
