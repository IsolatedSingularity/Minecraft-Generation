"""Regression tests for animation output helpers."""

from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageSequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.rendering import optimize_gif


class OptimizeGifTests(unittest.TestCase):
    def testPreservesPerFrameDurations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'timing.gif'
            frames = [
                Image.new('RGB', (12, 12), '#4A90E2'),
                Image.new('RGB', (12, 12), '#5CCB73'),
            ]
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=[40, 120],
                loop=0,
            )

            optimize_gif(path, colors=16)

            with Image.open(path) as result:
                durations = [
                    frame.info.get('duration', 0)
                    for frame in ImageSequence.Iterator(result)
                ]
            self.assertEqual(durations, [40, 120])


if __name__ == '__main__':
    unittest.main()
