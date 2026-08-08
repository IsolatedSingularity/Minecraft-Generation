"""Java 1.16.1 chunk-status dependency animation.

The status order is exact. Wall-clock scheduling is represented as a
deterministic dependency wave rather than a profiler trace.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
import numpy as np

from core.minecraft_visuals import minecraft_terrain_rgba
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


def chunk_status_snapshot(
    frame_index, fps=8, duration=12, full_hold=2, radius=7,
):
    """Return stage indices and reveal growth for one animation frame."""
    total_frames = int(round(float(fps) * float(duration)))
    hold_frames = int(round(float(fps) * float(full_hold)))
    generation_frames = total_frames - hold_frames
    if generation_frames < 2:
        raise ValueError('duration must leave at least two generation frames')

    distances = np.fromfunction(
        lambda row, column: np.maximum(
            np.abs(row - radius), np.abs(column - radius)
        ),
        (2 * radius + 1, 2 * radius + 1), dtype=float,
    )
    if int(frame_index) >= generation_frames:
        progress = 1.0
    else:
        progress = int(frame_index) / max(generation_frames - 1, 1)

    dependency_lag = 0.72
    wave_extent = len(STATUS_NAMES) + radius * dependency_lag
    phase = progress * wave_extent - distances * dependency_lag
    stages = np.floor(phase).astype(int)
    hidden = phase < 0.0
    stages = np.clip(stages, 0, len(STATUS_NAMES) - 1)
    growth = np.clip(phase, 0.0, 1.0)
    if progress >= 1.0:
        stages.fill(len(STATUS_NAMES) - 1)
        growth.fill(1.0)
        hidden.fill(False)
    return stages, growth, hidden


def _effect_positions(radius, modulus, remainder, offset):
    values = []
    size = 2 * radius + 1
    for row in range(size):
        for column in range(size):
            code = (column * 37 + row * 61 + 17) % modulus
            if code != remainder:
                continue
            values.append((
                column - radius + offset[0],
                row - radius + offset[1],
                row,
                column,
            ))
    return values


def create_seed_loading_animation(
    save_path, seed=-4172144997902289642, fps=8, duration=12,
    full_hold=2,
):
    total_frames = int(round(fps * duration))
    radius = 7
    size = 2 * radius + 1
    pixels_per_chunk = 12
    terrain = minecraft_terrain_rgba(
        seed, resolution=size * pixels_per_chunk, dimension='overworld',
        x_extent=(-radius - 0.5, radius + 0.5),
        z_extent=(-radius - 0.5, radius + 0.5),
        coordinate_scale=16.0, showcase=True,
    )[..., :3]
    background_rgb = np.asarray(to_rgb(COLORS['background']))

    figure, axis = plt.subplots(
        figsize=(8.4, 8.4), facecolor=COLORS['background'],
    )
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

    output = np.empty((*terrain.shape[:2], 4), dtype=float)
    output[..., :3] = background_rgb
    output[..., 3] = 1.0
    image = axis.imshow(
        output, origin='lower', interpolation='nearest',
        extent=(-radius - 0.5, radius + 0.5, -radius - 0.5, radius + 0.5),
        zorder=1,
    )
    for value in np.arange(-radius - 0.5, radius + 1.0, 1.0):
        axis.axvline(
            value, color='#07090E', linewidth=0.52, alpha=0.70, zorder=3,
        )
        axis.axhline(
            value, color='#07090E', linewidth=0.52, alpha=0.70, zorder=3,
        )
    axis.scatter(
        [0], [0], marker='+', s=90, c=COLORS['text'],
        linewidths=1.2, zorder=7,
    )
    target_outline = Rectangle(
        (-0.5, -0.5), 1, 1, fill=False,
        edgecolor=COLORS['gold'], linewidth=2.1, zorder=8,
    )
    axis.add_patch(target_outline)
    feature_data = _effect_positions(radius, 3, 0, (-0.12, 0.10))
    light_data = _effect_positions(radius, 7, 0, (0.18, 0.16))
    spawn_data = _effect_positions(radius, 11, 0, (-0.18, -0.16))
    heightmap_data = _effect_positions(radius, 13, 0, (0.16, -0.17))
    feature_marks = axis.scatter(
        [], [], s=23, marker='^', c='#E9CE70',
        edgecolors='#3E3215', linewidths=0.48, zorder=6,
    )
    light_marks = axis.scatter(
        [], [], s=24, marker='*', c=STATUS_COLORS[9],
        edgecolors='#5A4916', linewidths=0.36, zorder=6.2,
    )
    spawn_marks = axis.scatter(
        [], [], s=14, marker='o', c=STATUS_COLORS[10],
        edgecolors=COLORS['text'], linewidths=0.32, zorder=6.3,
    )
    heightmap_marks = axis.scatter(
        [], [], s=30, marker='s', facecolors='none',
        edgecolors=STATUS_COLORS[11], linewidths=0.62, zorder=6.4,
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
        0.50, 0.938, 'RADIAL WORLD GENERATION   JAVA 1.16.1',
        ha='center', va='center', color=COLORS['text'],
        fontsize=17, fontweight='black',
    )

    def _visible_offsets(data, stages, threshold):
        visible = [
            (x, z) for x, z, row, column in data
            if stages[row, column] >= threshold
        ]
        return np.asarray(visible) if visible else np.empty((0, 2))

    def update(frame_index):
        stages, growth, hidden = chunk_status_snapshot(
            frame_index, fps=fps, duration=duration,
            full_hold=full_hold, radius=radius,
        )
        output[..., :3] = background_rgb
        output[..., 3] = 1.0
        for row in range(size):
            for column in range(size):
                if hidden[row, column] or growth[row, column] <= 0.0:
                    continue
                stage = int(stages[row, column])
                pixel_row = row * pixels_per_chunk
                pixel_column = column * pixels_per_chunk
                tile = terrain[
                    pixel_row:pixel_row + pixels_per_chunk,
                    pixel_column:pixel_column + pixels_per_chunk,
                ].copy()
                stage_color = np.asarray(to_rgb(STATUS_COLORS[stage]))
                if stage <= 3:
                    reveal = 0.08 + 0.08 * stage
                    tile = (1.0 - reveal) * stage_color + reveal * tile
                elif stage == 4:
                    luminance = np.mean(tile, axis=2, keepdims=True)
                    tile = 0.58 * luminance + 0.42 * stage_color
                else:
                    tint = max(0.04, 0.24 - 0.026 * (stage - 5))
                    tile = (1.0 - tint) * tile + tint * stage_color

                if stage >= 6 and (row * 13 + column * 7) % 9 == 0:
                    center = pixels_per_chunk // 2
                    tile[center - 1:center + 1, center - 1:center + 1] *= 0.28
                if stage >= 7 and (row * 11 + column * 5) % 13 == 0:
                    tile[2:4, -4:-2] = np.asarray(to_rgb('#4D88B5'))

                side = max(1, int(round(pixels_per_chunk * growth[row, column])))
                start = (pixels_per_chunk - side) // 2
                stop = start + side
                target_rows = slice(pixel_row + start, pixel_row + stop)
                target_columns = slice(pixel_column + start, pixel_column + stop)
                output[target_rows, target_columns, :3] = tile[
                    start:stop, start:stop,
                ]
        image.set_data(output)

        feature_marks.set_offsets(_visible_offsets(feature_data, stages, 8))
        light_marks.set_offsets(_visible_offsets(light_data, stages, 9))
        spawn_marks.set_offsets(_visible_offsets(spawn_data, stages, 10))
        heightmap_marks.set_offsets(
            _visible_offsets(heightmap_data, stages, 11)
        )

        active_stage = int(stages[radius, radius])
        stage_marker.set_x(active_stage + 0.04)
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
