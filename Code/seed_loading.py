"""Java 1.16.1 chunk-status dependency animation.

The status order and terminal dependency footprint are source-exact. The
large radial request wave is an explicitly illustrative view of world loading,
not a profiler trace or a claim about scheduler timing.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Circle, Rectangle
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
    frame_index, fps=10, duration=6, full_hold=1, radius=10,
):
    """Return one staged view of the exact terminal dependency footprint.

    The animation advances every dependency cell through the ordered status
    taxonomy and caps it at the source-required terminal status for its
    Chebyshev distance. The ordering is exact; relative wall-clock scheduling
    remains illustrative.
    """
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

    active_stage = min(
        len(STATUS_NAMES) - 1,
        int(np.floor(progress * len(STATUS_NAMES))),
    )
    stages = np.minimum(active_stage, target)
    hidden = distances > 10.0
    growth = np.where(hidden, 0.0, 1.0)
    if progress >= 1.0:
        stages = target
    return stages, growth, hidden


def create_seed_loading_animation(
    save_path, seed=-4172144997902289642, fps=10, duration=6,
    full_hold=1,
):
    """Render a broad radial request wave and exact local dependency inset."""
    total_frames = int(round(fps * duration))
    hold_frames = int(round(fps * full_hold))
    generation_frames = total_frames - hold_frames
    dependency_radius = 10
    display_radius = 360
    resolution = 721

    terrain = minecraft_terrain_rgba(
        seed, resolution=resolution, dimension='overworld',
        x_extent=(-display_radius - 0.5, display_radius + 0.5),
        z_extent=(-display_radius - 0.5, display_radius + 0.5),
        coordinate_scale=16.0, showcase=False,
    )[..., :3]
    background_rgb = np.asarray(to_rgb(COLORS['background']))
    context = 0.12 * terrain + 0.88 * background_rgb
    output = np.empty((*terrain.shape[:2], 4), dtype=float)
    output[..., :3] = context
    output[..., 3] = 1.0

    coordinates = np.linspace(
        -display_radius - 0.5, display_radius + 0.5, resolution,
    )
    pixel_x, pixel_z = np.meshgrid(coordinates, coordinates)
    radial_distance = np.hypot(pixel_x, pixel_z)
    maximum_request_radius = float(np.hypot(display_radius, display_radius))

    figure = plt.figure(figsize=(12.8, 7.2), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[2.34, 0.96],
        left=0.052, right=0.985, top=0.90, bottom=0.095, wspace=0.07,
    )
    axis = figure.add_subplot(grid[0, 0])
    side = figure.add_subplot(grid[0, 1])
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.axis('off')

    axis.set_aspect('equal')
    axis.set_xlim(-display_radius - 0.5, display_radius + 0.5)
    axis.set_ylim(-display_radius - 0.5, display_radius + 0.5)
    axis.set_xticks(range(-display_radius, display_radius + 1, 120))
    axis.set_yticks(range(-display_radius, display_radius + 1, 120))
    axis.set_xlabel('Chunk X')
    axis.set_ylabel('Chunk Z')
    axis.set_title(
        'Illustrative radial chunk requests across a broad world view',
        fontsize=10.5, pad=7,
    )
    axis.tick_params(colors=COLORS['muted'], labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(COLORS['grid'])

    image = axis.imshow(
        output, origin='lower', interpolation='nearest',
        extent=(
            -display_radius - 0.5, display_radius + 0.5,
            -display_radius - 0.5, display_radius + 0.5,
        ),
        zorder=1,
    )
    for value in range(-display_radius, display_radius + 1, 60):
        axis.axvline(value, color='#59677E', linewidth=0.34, alpha=0.20, zorder=3)
        axis.axhline(value, color='#59677E', linewidth=0.34, alpha=0.20, zorder=3)
    axis.scatter(
        [0], [0], marker='+', s=95, c=COLORS['text'],
        linewidths=1.25, zorder=8,
    )
    axis.add_patch(Rectangle(
        (-10.5, -10.5), 21, 21, fill=False,
        edgecolor=COLORS['violet'], linewidth=1.35,
        linestyle='--', alpha=0.95, zorder=7,
    ))
    request_front = Circle(
        (0, 0), 0.1, fill=False, edgecolor=COLORS['cyan'],
        linewidth=2.0, alpha=0.0, zorder=9,
    )
    axis.add_patch(request_front)
    request_text = axis.text(
        0.018, 0.982, '', transform=axis.transAxes,
        ha='left', va='top', color=COLORS['text'], fontsize=8.2,
        fontweight='bold', family='monospace', zorder=10,
    )

    detail_axis = side.inset_axes([0.025, 0.47, 0.95, 0.49])
    detail_axis.set_xlim(-10.5, 10.5)
    detail_axis.set_ylim(-10.5, 10.5)
    detail_axis.set_aspect('equal')
    detail_axis.set_xticks(range(-10, 11, 2))
    detail_axis.set_yticks(range(-10, 11, 2))
    detail_axis.tick_params(colors=COLORS['muted'], labelsize=5.5, pad=1)
    detail_axis.set_title(
        'EXACT FULL-CHUNK DEPENDENCY FOOTPRINT  21 x 21',
        fontsize=7.1, pad=4,
    )
    initial_stages, _, initial_hidden = chunk_status_snapshot(
        0, fps=fps, duration=duration, full_hold=full_hold,
        radius=dependency_radius,
    )
    dependency_image = detail_axis.imshow(
        np.ma.masked_where(initial_hidden, initial_stages),
        origin='lower', interpolation='nearest',
        extent=(-10.5, 10.5, -10.5, 10.5),
        cmap=ListedColormap(STATUS_COLORS), vmin=0, vmax=len(STATUS_COLORS) - 1,
        zorder=1,
    )
    for value in np.arange(-10.5, 11.0, 1.0):
        detail_axis.axvline(
            value, color='#07090E', linewidth=0.42, alpha=0.72, zorder=3,
        )
        detail_axis.axhline(
            value, color='#07090E', linewidth=0.42, alpha=0.72, zorder=3,
        )
    detail_axis.add_patch(Rectangle(
        (-0.5, -0.5), 1, 1, fill=False,
        edgecolor=COLORS['gold'], linewidth=1.8, zorder=8,
    ))

    legend_axis = side.inset_axes([0.0, 0.015, 1.0, 0.40])
    legend_axis.set_xlim(0, 2)
    legend_axis.set_ylim(0, 7.4)
    legend_axis.axis('off')
    legend_axis.text(
        0.0, 7.18, 'CENTER STATUS AND SOURCE TARGETS', color=COLORS['text'],
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
        'Exact terminal target by Chebyshev radius:\n'
        '0 FULL | 1 FEATURES | 2 LIQUID CARVERS | 3-10 STRUCTURE STARTS',
        color=COLORS['muted'], fontsize=6.6, va='bottom', linespacing=1.35,
    )
    figure.text(
        0.50, 0.953, 'CHUNK-STATUS DEPENDENCY WAVE',
        ha='center', va='center', color=COLORS['text'],
        fontsize=17, fontweight='black',
    )

    def update(frame_index):
        if frame_index >= generation_frames:
            progress = 1.0
        else:
            progress = frame_index / max(generation_frames - 1, 1)
        eased = progress * progress * (3.0 - 2.0 * progress)
        wave_radius = maximum_request_radius * eased
        reveal = np.clip((wave_radius - radial_distance + 8.0) / 16.0, 0.0, 1.0)
        output[..., :3] = (
            context * (1.0 - reveal[..., None])
            + terrain * reveal[..., None]
        )
        image.set_data(output)
        request_front.set_radius(max(wave_radius, 0.1))
        request_front.set_alpha(0.90 if progress < 1.0 else 0.0)
        request_text.set_text(
            f'REQUEST RADIUS  {min(wave_radius, maximum_request_radius):5.0f} CHUNKS'
        )

        stages, _, hidden = chunk_status_snapshot(
            frame_index, fps=fps, duration=duration,
            full_hold=full_hold, radius=dependency_radius,
        )
        dependency_image.set_data(np.ma.masked_where(hidden, stages))
        active_stage = int(stages[dependency_radius, dependency_radius])
        for index, box in enumerate(stage_boxes):
            box.set_edgecolor(COLORS['text'] if index == active_stage else '#07090E')
            box.set_linewidth(1.55 if index == active_stage else 0.45)
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
