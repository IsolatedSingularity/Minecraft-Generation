"""Java 1.16.1 chunk-status dependency animation.

The status order is exact. Wall-clock scheduling is represented as a
deterministic dependency wave rather than a profiler trace.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle
import numpy as np

from core.minecraft_visuals import draw_minecraft_terrain
from core.rendering import optimize_gif
from core.style import COLORS, apply_style


apply_style()

STATUS_NAMES = [
    'EMPTY', 'STR. STARTS', 'STR. REFS', 'BIOMES', 'NOISE',
    'SURFACE', 'CARVERS', 'LIQ. CARVERS', 'FEATURES', 'LIGHT',
    'SPAWN', 'HEIGHTMAPS', 'FULL',
]
STATUS_COLORS = [
    '#141822', '#384152', '#4C5264', '#596C86', '#506C9B',
    '#788657', '#8D7863', '#6D7280', '#7EA65F', '#E0C65B',
    '#C9865E', '#8EC7B0', '#59C985',
]
STATUS_SHORT = [
    'EMPTY', 'STARTS', 'REFS', 'BIOMES', 'NOISE', 'SURFACE', 'CARVE',
    'LIQUID', 'FEATURES', 'LIGHT', 'SPAWN', 'MAPS', 'FULL',
]


def create_seed_loading_animation(
    save_path, seed=-4172144997902289642, fps=8, duration=12,
):
    total_frames = int(fps * duration)
    radius = 7
    size = 2 * radius + 1
    distances = np.fromfunction(
        lambda row, column: np.maximum(
            np.abs(row - radius), np.abs(column - radius)
        ),
        (size, size), dtype=float,
    )
    figure, axis = plt.subplots(figsize=(8.4, 8.4), facecolor=COLORS['background'])
    figure.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.17)
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

    draw_minecraft_terrain(
        axis,
        (-radius - 0.5, radius + 0.5, -radius - 0.5, radius + 0.5),
        seed=seed, dimension='overworld', resolution=240, alpha=0.96,
    )
    rgba = np.zeros((size, size, 4), dtype=float)
    image = axis.imshow(
        rgba, origin='lower', interpolation='nearest',
        extent=(-radius - 0.5, radius + 0.5, -radius - 0.5, radius + 0.5),
        zorder=1,
    )
    for value in np.arange(-radius - 0.5, radius + 1.0, 1.0):
        axis.axvline(value, color='#07090E', linewidth=0.52, alpha=0.62, zorder=2)
        axis.axhline(value, color='#07090E', linewidth=0.52, alpha=0.62, zorder=2)
    axis.scatter(
        [0], [0], marker='+', s=90, c=COLORS['text'],
        linewidths=1.2, zorder=5,
    )
    target_outline = Rectangle(
        (-0.5, -0.5), 1, 1, fill=False,
        edgecolor=COLORS['gold'], linewidth=2.1, zorder=6,
    )
    axis.add_patch(target_outline)
    axis.text(
        0.018, 0.976, 'CHUNK STATUS DEPENDENCY WAVE',
        transform=axis.transAxes, ha='left', va='top',
        fontsize=15, fontweight='black', color=COLORS['text'],
        bbox=dict(
            boxstyle='square,pad=0.35', facecolor=COLORS['background'],
            edgecolor='none', alpha=0.82,
        ), zorder=8,
    )
    center_label = axis.text(
        0.982, 0.035, '', transform=axis.transAxes,
        ha='right', va='bottom', fontsize=12.5, fontweight='bold',
        family='monospace', color=COLORS['text'], zorder=8,
        bbox=dict(
            boxstyle='round,pad=0.38', facecolor=COLORS['panel'],
            edgecolor=COLORS['gold'], alpha=0.92,
        ),
    )

    legend_axis = figure.add_axes([0.04, 0.045, 0.92, 0.075])
    legend_axis.set_xlim(0, len(STATUS_NAMES))
    legend_axis.set_ylim(0, 1)
    legend_axis.axis('off')
    for index, (name, color) in enumerate(zip(STATUS_SHORT, STATUS_COLORS)):
        legend_axis.add_patch(Rectangle(
            (index + 0.04, 0.40), 0.90, 0.30,
            facecolor=color, edgecolor='#07090E', linewidth=0.45,
        ))
        legend_axis.text(
            index + 0.49, 0.19, name, ha='center', va='center',
            color=COLORS['muted'], fontsize=6.8, fontweight='bold',
        )
    stage_marker = Rectangle(
        (0.04, 0.40), 0.90, 0.30, fill=False,
        edgecolor=COLORS['text'], linewidth=1.55,
    )
    legend_axis.add_patch(stage_marker)
    figure.text(
        0.50, 0.938, 'JAVA 1.16.1 CHUNK GENERATION',
        ha='center', va='center', color=COLORS['text'],
        fontsize=17, fontweight='black',
    )
    figure.text(
        0.50, 0.905,
        'Exact status order shown as a deterministic dependency model',
        ha='center', va='center', color=COLORS['muted'], fontsize=9.5,
    )

    def update(frame_index):
        progress = frame_index / max(total_frames - 1, 1)
        center_progress = progress * (len(STATUS_NAMES) - 1 + 0.999)
        stage_values = np.floor(center_progress - distances * 0.72).astype(int)
        stage_values = np.clip(stage_values, 0, len(STATUS_NAMES) - 1)
        output = np.empty((size, size, 4), dtype=float)
        for stage, color in enumerate(STATUS_COLORS):
            mask = stage_values == stage
            base = np.array(to_rgba(color))
            output[mask, :3] = base[:3]
            output[mask, 3] = 0.78 - 0.52 * stage / (len(STATUS_NAMES) - 1)
        image.set_data(output)

        active_stage = int(stage_values[radius, radius])
        stage_marker.set_x(active_stage + 0.04)
        center_label.set_text(
            f'CENTER CHUNK  {STATUS_NAMES[active_stage]}  '
            f'{active_stage + 1:02d}/{len(STATUS_NAMES):02d}'
        )
        return []

    animation = FuncAnimation(
        figure, update, frames=total_frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    optimize_gif(save_path, colors=80)
    return str(save_path)


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)
    create_seed_loading_animation(plots / 'seed_loading.gif')


if __name__ == '__main__':
    main()
