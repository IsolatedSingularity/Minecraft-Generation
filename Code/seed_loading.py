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

# ChunkStatus.DISTANCE_TO_TARGET_GENERATION_STATUS in Java 1.16.1:
# target, one-chunk ring, two-chunk ring, then radii 3 through 10.
TARGET_STATUS_BY_DISTANCE = {
    0: 12,  # FULL
    1: 8,   # FEATURES
    2: 7,   # LIQUID_CARVERS
}


def chunk_status_snapshot(
    frame_index, fps=8, duration=12, full_hold=2, radius=15,
):
    """Return the source-backed target-status dependency snapshot."""
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

    target = np.zeros(distances.shape, dtype=int)
    target[distances <= 10] = 1
    for distance, status in TARGET_STATUS_BY_DISTANCE.items():
        target[distances == distance] = status

    phase = progress * (len(STATUS_NAMES) - 1)
    stages = np.minimum(np.floor(phase).astype(int), target)
    growth = np.where(
        stages >= target, 1.0, np.clip(phase - stages, 0.0, 1.0),
    )
    hidden = np.zeros(distances.shape, dtype=bool)
    if progress >= 1.0:
        stages = target
        growth.fill(1.0)
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
    radius = 15
    size = 2 * radius + 1
    pixels_per_chunk = 10
    terrain = minecraft_terrain_rgba(
        seed, resolution=size * pixels_per_chunk, dimension='overworld',
        x_extent=(-radius - 0.5, radius + 0.5),
        z_extent=(-radius - 0.5, radius + 0.5),
        coordinate_scale=16.0, showcase=False,
    )[..., :3]
    background_rgb = np.asarray(to_rgb(COLORS['background']))

    figure = plt.figure(
        figsize=(12.8, 7.2), facecolor=COLORS['background'],
    )
    grid = figure.add_gridspec(
        1, 2, width_ratios=[2.18, 0.92],
        left=0.055, right=0.985, top=0.90, bottom=0.095, wspace=0.075,
    )
    axis = figure.add_subplot(grid[0, 0])
    side = figure.add_subplot(grid[0, 1])
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.axis('off')
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
            value, color='#07090E', linewidth=0.36, alpha=0.62, zorder=3,
        )
        axis.axhline(
            value, color='#07090E', linewidth=0.36, alpha=0.62, zorder=3,
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
    detail_axis = side.inset_axes([0.04, 0.61, 0.92, 0.34])
    detail_axis.set_xlim(-3.5, 3.5)
    detail_axis.set_ylim(-3.5, 3.5)
    detail_axis.set_aspect('equal')
    detail_axis.set_xticks(range(-3, 4))
    detail_axis.set_yticks(range(-3, 4))
    detail_axis.tick_params(colors=COLORS['muted'], labelsize=5.8, pad=1)
    detail_axis.set_title('CENTRAL 7x7 DEPENDENCY DETAIL', fontsize=7.5, pad=4)
    crop_start = (radius - 3) * pixels_per_chunk
    crop_stop = (radius + 4) * pixels_per_chunk
    detail_image = detail_axis.imshow(
        output[crop_start:crop_stop, crop_start:crop_stop],
        origin='lower', interpolation='nearest',
        extent=(-3.5, 3.5, -3.5, 3.5), zorder=1,
    )
    for value in np.arange(-3.5, 4.0, 1.0):
        detail_axis.axvline(value, color='#07090E', linewidth=0.58, alpha=0.78, zorder=3)
        detail_axis.axhline(value, color='#07090E', linewidth=0.58, alpha=0.78, zorder=3)
    detail_axis.add_patch(Rectangle(
        (-0.5, -0.5), 1, 1, fill=False,
        edgecolor=COLORS['gold'], linewidth=1.7, zorder=8,
    ))

    legend_axis = side.inset_axes([0.0, 0.03, 1.0, 0.52])
    legend_axis.set_xlim(0, 2)
    legend_axis.set_ylim(0, 7.4)
    legend_axis.axis('off')
    legend_axis.text(
        0.0, 7.18, 'CENTER-CHUNK STATUS', color=COLORS['text'],
        fontsize=8.6, fontweight='black', va='center',
    )
    stage_boxes = []
    for index, (name, color) in enumerate(zip(STATUS_SHORT, STATUS_COLORS)):
        column = index // 7
        row = index % 7
        x = column + 0.04
        y = 6.55 - row * 0.77
        box = Rectangle(
            (x, y), 0.22, 0.36,
            facecolor=color, edgecolor='#07090E', linewidth=0.45,
        )
        legend_axis.add_patch(box)
        stage_boxes.append(box)
        legend_axis.text(
            x + 0.29, y + 0.18, name, ha='left', va='center',
            color=COLORS['muted'], fontsize=7.0, fontweight='bold',
        )
    legend_axis.text(
        0.02, 0.10,
        'Final source target by Chebyshev radius:\n'
        '0 FULL | 1 FEATURES | 2 LIQUID CARVERS | 3-10 STRUCTURE STARTS',
        color=COLORS['muted'], fontsize=6.6, va='bottom', linespacing=1.35,
    )
    figure.text(
        0.50, 0.953, 'CHUNK-STATUS DEPENDENCY WAVE   JAVA 1.16.1',
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

        active_stage = int(stages[radius, radius])
        for index, box in enumerate(stage_boxes):
            box.set_edgecolor(COLORS['text'] if index == active_stage else '#07090E')
            box.set_linewidth(1.55 if index == active_stage else 0.45)
        detail_image.set_data(output[
            crop_start:crop_stop, crop_start:crop_stop,
        ])
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
