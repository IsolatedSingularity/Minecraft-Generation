"""Java 1.16.1 chunk-status and population-seed animation.

The status order and population-seed mixing are exact. Wall-clock scheduling is
represented as a deterministic dependency wave rather than a profiler trace.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Rectangle
import numpy as np

from core.lcg import MinecraftLCG, generate_population_seed
from core.rendering import optimize_gif
from core.style import COLORS, apply_style


apply_style()

STATUS_NAMES = [
    'EMPTY', 'STR. STARTS', 'STR. REFS', 'BIOMES', 'NOISE',
    'SURFACE', 'CARVERS', 'LIQ. CARVERS', 'FEATURES', 'LIGHT',
    'SPAWN', 'HEIGHTMAPS', 'FULL',
]
STATUS_COLORS = [
    '#121722', '#222A3D', '#2D3953', '#39466B', '#4E5E91',
    '#6678B5', '#7B74C8', '#8975CF', '#9A76D6', '#B276D0',
    '#C27BC8', '#E391A8', '#73D49B',
]


def _chunk_texture(seed, radius):
    values = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=float)
    for row, chunk_z in enumerate(range(-radius, radius + 1)):
        for column, chunk_x in enumerate(range(-radius, radius + 1)):
            population_seed = generate_population_seed(
                seed, chunk_x * 16, chunk_z * 16,
            )
            random = MinecraftLCG(population_seed)
            values[row, column] = 0.78 + 0.22 * random.next_int(256) / 255.0
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
    figure.subplots_adjust(left=0.11, right=0.89, top=0.965, bottom=0.16)
    axis.set_aspect('equal')
    axis.set_xlim(-radius - 0.5, radius + 0.5)
    axis.set_ylim(-radius - 0.5, radius + 0.5)
    axis.set_xticks(range(-radius, radius + 1, 2))
    axis.set_yticks(range(-radius, radius + 1, 2))
    axis.set_xlabel('Chunk X')
    axis.set_ylabel('Chunk Z')
    axis.tick_params(colors=COLORS['muted'], labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(COLORS['grid'])

    rgba = np.zeros((size, size, 4), dtype=float)
    image = axis.imshow(
        rgba, origin='lower', interpolation='nearest',
        extent=(-radius - 0.5, radius + 0.5, -radius - 0.5, radius + 0.5),
        zorder=1,
    )
    for value in np.arange(-radius - 0.5, radius + 1.0, 1.0):
        axis.axvline(value, color=COLORS['grid'], linewidth=0.28, alpha=0.44, zorder=2)
        axis.axhline(value, color=COLORS['grid'], linewidth=0.28, alpha=0.44, zorder=2)
    axis.scatter(
        [0], [0], marker='+', s=90, c=COLORS['text'],
        linewidths=1.2, zorder=5,
    )
    frontier = Rectangle(
        (-0.5, -0.5), 1, 1, fill=False,
        edgecolor=COLORS['cyan'], linewidth=1.3, alpha=0.0, zorder=4,
    )
    axis.add_patch(frontier)

    legend_axis = figure.add_axes([0.115, 0.045, 0.77, 0.065])
    legend_axis.set_xlim(0, len(STATUS_NAMES))
    legend_axis.set_ylim(0, 1)
    legend_axis.axis('off')
    for index, (name, color) in enumerate(zip(STATUS_NAMES, STATUS_COLORS)):
        legend_axis.add_patch(Rectangle(
            (index + 0.06, 0.50), 0.72, 0.18,
            facecolor=color, edgecolor='none',
        ))
        legend_axis.text(
            index + 0.42, 0.28, name, ha='center', va='center',
            color=COLORS['muted'], fontsize=6.6,
        )
    stage_marker = Circle(
        (0.42, 0.59), 0.09, facecolor='none',
        edgecolor=COLORS['text'], linewidth=0.8,
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
            output[mask, 3] = 0.96
        image.set_data(output)

        active_stage = int(np.clip(np.floor(progress * len(STATUS_NAMES)), 0, len(STATUS_NAMES) - 1))
        stage_marker.center = (active_stage + 0.42, 0.59)
        current_radius = min(radius, int(max(0.0, wave - len(STATUS_NAMES) + 1.0)))
        frontier.set_xy((-current_radius - 0.5, -current_radius - 0.5))
        frontier.set_width(2 * current_radius + 1)
        frontier.set_height(2 * current_radius + 1)
        frontier.set_alpha(0.75 if current_radius > 0 else 0.0)
        return []

    animation = FuncAnimation(
        figure, update, frames=total_frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=125)
    plt.close(figure)
    optimize_gif(save_path, colors=96)
    return str(save_path)


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)
    create_seed_loading_animation(plots / 'seed_loading.gif')


if __name__ == '__main__':
    main()
