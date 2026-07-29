"""Java 1.16.1 chunk-status and population-seed animation.

The status order and population-seed mixing are exact. Wall-clock scheduling is
represented as a deterministic dependency wave rather than a profiler trace.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import numpy as np

from core.lcg import MinecraftLCG, generate_population_seed
from core.rendering import optimize_gif
from core.style import COLORS, addSoftShadow, apply_style, style_axis


apply_style()

STATUS_NAMES = [
    'EMPTY', 'STR. STARTS', 'STR. REFS', 'BIOMES', 'NOISE',
    'SURFACE', 'CARVERS', 'LIQ. CARVERS', 'FEATURES', 'LIGHT',
    'SPAWN', 'HEIGHTMAPS', 'FULL',
]
STATUS_COLORS = [
    '#E8E8ED', '#DCEAF7', '#CFE5F8', '#C1E0F7', '#B0D9F5',
    '#9BCFF1', '#A9C8F3', '#B9C2F1', '#C9BCEE', '#DAB9E8',
    '#E9BFD8', '#B9E1C2', '#5CCB73',
]


def _chunk_texture(seed, radius):
    values = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=float)
    for row, chunk_z in enumerate(range(-radius, radius + 1)):
        for column, chunk_x in enumerate(range(-radius, radius + 1)):
            population_seed = generate_population_seed(
                seed, chunk_x * 16, chunk_z * 16,
            )
            random = MinecraftLCG(population_seed)
            values[row, column] = 0.88 + 0.12 * random.next_int(256) / 255.0
    return values


def create_seed_loading_animation(
    save_path, seed=-4172144997902289642, fps=12, duration=9,
):
    total_frames = int(fps * duration)
    radius = 10
    size = 2 * radius + 1
    distances = np.fromfunction(
        lambda row, column: np.maximum(
            np.abs(row - radius), np.abs(column - radius)
        ),
        (size, size), dtype=float,
    )
    texture = _chunk_texture(seed, radius)

    figure, axis = plt.subplots(figsize=(12.8, 7.2), facecolor=COLORS['background'])
    figure.subplots_adjust(left=0.13, right=0.87, top=0.94, bottom=0.19)
    axis.set_xlim(-radius - 0.5, radius + 0.5)
    axis.set_ylim(-radius - 0.5, radius + 0.5)
    axis.set_xticks(range(-radius, radius + 1, 2))
    axis.set_yticks(range(-radius, radius + 1, 2))
    axis.set_xlabel('Chunk X')
    axis.set_ylabel('Chunk Z')
    style_axis(axis, equal=True, grid=False)

    rgba = np.zeros((size, size, 4), dtype=float)
    image = axis.imshow(
        rgba, origin='lower', interpolation='nearest',
        extent=(-radius - 0.5, radius + 0.5, -radius - 0.5, radius + 0.5),
        zorder=1,
    )
    for value in np.arange(-radius - 0.5, radius + 1.0, 1.0):
        axis.axvline(
            value, color=COLORS['grid'], linewidth=0.32, alpha=0.70, zorder=2,
        )
        axis.axhline(
            value, color=COLORS['grid'], linewidth=0.32, alpha=0.70, zorder=2,
        )
    axis.scatter(
        [0], [0], marker='+', s=90, c=COLORS['text'],
        linewidths=1.2, zorder=5,
    )
    frontier = Rectangle(
        (-0.5, -0.5), 1, 1, fill=False,
        edgecolor=COLORS['blue'], linewidth=1.45, alpha=0.0, zorder=4,
    )
    axis.add_patch(frontier)

    legend_axis = figure.add_axes([0.105, 0.045, 0.79, 0.085])
    legend_axis.set_xlim(0, len(STATUS_NAMES))
    legend_axis.set_ylim(0, 1)
    legend_axis.axis('off')
    for index, (name, color) in enumerate(zip(STATUS_NAMES, STATUS_COLORS)):
        button = FancyBboxPatch(
            (index + 0.08, 0.50), 0.68, 0.22,
            boxstyle='round,pad=0.02,rounding_size=0.07',
            facecolor=color, edgecolor=COLORS['panel'], linewidth=0.55,
        )
        legend_axis.add_patch(button)
        addSoftShadow(button, offset=(0.8, -0.8), alpha=0.14)
        legend_axis.text(
            index + 0.42, 0.28, name, ha='center', va='center',
            color=COLORS['muted'], fontsize=6.4, fontweight='bold',
        )
    stage_marker = Circle(
        (0.42, 0.61), 0.15, facecolor='none',
        edgecolor=COLORS['blue'], linewidth=1.15,
    )
    legend_axis.add_patch(stage_marker)

    def update(frame_index):
        progress = frame_index / max(total_frames - 1, 1)
        wave = progress * (len(STATUS_NAMES) + radius + 4.0)
        stage_values = np.floor(wave - distances * 0.74).astype(int)
        stage_values = np.clip(stage_values, 0, len(STATUS_NAMES) - 1)
        output = np.empty((size, size, 4), dtype=float)
        for stage, color in enumerate(STATUS_COLORS):
            mask = stage_values == stage
            base = np.array(to_rgba(color))
            output[mask, :3] = base[:3] * texture[mask, None]
            output[mask, 3] = 0.98
        image.set_data(output)

        active_stage = int(np.clip(
            np.floor(progress * len(STATUS_NAMES)),
            0, len(STATUS_NAMES) - 1,
        ))
        stage_marker.center = (active_stage + 0.42, 0.61)
        current_radius = min(
            radius, int(max(0.0, wave - len(STATUS_NAMES) + 1.0)),
        )
        frontier.set_xy((-current_radius - 0.5, -current_radius - 0.5))
        frontier.set_width(2 * current_radius + 1)
        frontier.set_height(2 * current_radius + 1)
        frontier.set_alpha(0.82 if current_radius > 0 else 0.0)
        return []

    animation = FuncAnimation(
        figure, update, frames=total_frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=125)
    plt.close(figure)
    optimize_gif(save_path, colors=128)
    return str(save_path)


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)
    create_seed_loading_animation(plots / 'seed_loading.gif')


if __name__ == '__main__':
    main()
