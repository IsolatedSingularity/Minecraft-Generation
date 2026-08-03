"""Animation output helpers with README-oriented GIF optimization."""

import os
from pathlib import Path
import tempfile
import time

from PIL import Image, ImageSequence


def optimize_gif(path, colors=128):
    """Quantize a GIF through an atomic sibling file."""
    target = Path(path)
    with Image.open(target) as source:
        default_duration = source.info.get('duration', 80)
        loop = source.info.get('loop', 0)
        frames = []
        durations = []
        for frame in ImageSequence.Iterator(source):
            durations.append(frame.info.get('duration', default_duration))
            quantized = frame.convert('RGBA').convert(
                'P',
                palette=Image.Palette.ADAPTIVE,
                colors=colors,
                dither=Image.Dither.NONE,
            )
            frames.append(quantized)

    handle = tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f'.{target.stem}-',
        suffix='.gif', delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        frames[0].save(
            temporary,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=loop,
            optimize=True,
            disposal=2,
        )
        for attempt in range(4):
            try:
                os.replace(temporary, target)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(0.2 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)
    return target


def gif_metadata(path):
    """Return basic GIF metadata used by tests and render summaries."""
    target = Path(path)
    with Image.open(target) as image:
        return {
            'width': image.width,
            'height': image.height,
            'frames': getattr(image, 'n_frames', 1),
            'duration_ms': image.info.get('duration', 0),
            'bytes': target.stat().st_size,
        }
