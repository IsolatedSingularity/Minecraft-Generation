"""Java 1.16.1 spawn preparation and chunk-status visualization.

The status order, fully generated 21 by 21 spawn region, surrounding dependency
shells, and vanilla loading-screen colour mapping are source-backed. Relative
task timing is an explanatory schedule.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.patches import Rectangle
import numpy as np

from core.minecraft_visuals import (
    overworld_surface_sample,
    terrain_rgba_from_sample,
)
from core.rendering import optimize_gif
from core.style import COLORS, apply_style


apply_style()


STATUS_NAMES = [
    'EMPTY', 'STR. STARTS', 'STR. REFS', 'BIOMES', 'NOISE',
    'SURFACE', 'CARVERS', 'LIQ. CARVERS', 'FEATURES', 'LIGHT',
    'SPAWN', 'HEIGHTMAPS', 'FULL',
]
STATUS_SHORT = [
    'EMPTY', 'STARTS', 'REFS', 'BIOMES', 'NOISE', 'SURFACE', 'CARVE',
    'LIQUID', 'FEATURES', 'LIGHT', 'SPAWN', 'MAPS', 'FULL',
]

# LevelLoadingScreen.STATUS_TO_COLOR, rendered in a slightly softened dark UI.
VANILLA_STATUS_COLORS = [
    '#545454', '#999999', '#5F6251', '#80B252', '#D1D1D1', '#726C49',
    '#6D6A5C', '#303692', '#21C600', '#CCCCCC', '#F26760', '#EEEEEE', '#FFFFFF',
]
STATUS_COLORS = [
    '#1A1E28', '#5B606C', '#4F554A', '#587A43', '#8B94A6', '#716B4D',
    '#6B6359', '#414B8D', '#4D9C56', '#B7B7A7', '#C87362', '#BBD0C5', '#73D49B',
]

# ChunkStatus.DISTANCE_TO_TARGET_GENERATION_STATUS in Java 1.16.1. These
# statuses describe the neighbours required around a FULL chunk. They do not
# describe the terminal state of the 441 chunks held by the START ticket.
DEPENDENCY_STATUS_BY_DISTANCE = {
    1: 8,
    2: 7,
    **{distance: 1 for distance in range(3, 11)},
}
FULL_STATUS = len(STATUS_NAMES) - 1
SPAWN_RADIUS = 10
TRACKER_RADIUS = 22


def _distance_grid(radius):
    return np.fromfunction(
        lambda row, column: np.maximum(
            np.abs(row - radius), np.abs(column - radius),
        ),
        (2 * radius + 1, 2 * radius + 1), dtype=float,
    )


def _animation_progress(frame_index, fps, duration, full_hold):
    total_frames = int(round(float(fps) * float(duration)))
    hold_frames = int(round(float(fps) * float(full_hold)))
    generation_frames = total_frames - hold_frames
    if generation_frames < 2:
        raise ValueError('duration must leave at least two generation frames')
    return (
        1.0 if int(frame_index) >= generation_frames
        else int(frame_index) / max(generation_frames - 1, 1)
    )


def chunk_status_snapshot(frame_index, fps=10, duration=6, full_hold=1, radius=10):
    """Return an explanatory center-out schedule for the 21 by 21 spawn region."""
    progress = _animation_progress(frame_index, fps, duration, full_hold)
    distances = _distance_grid(radius)
    # A radius-11 START ticket gives every chunk at Chebyshev distance 10 or
    # less a ticket level below 33, whose target generation status is FULL.
    # Java's concurrent scheduler does not promise a visual order, so the
    # animation uses a deterministic center-out wave.
    work = progress * (FULL_STATUS + radius)
    stages = np.floor(work - distances).astype(int)
    stages = np.clip(stages, 0, FULL_STATUS)
    hidden = np.zeros(distances.shape, dtype=bool)
    if progress >= 1.0:
        stages.fill(FULL_STATUS)
    return stages, np.where(hidden, 0.0, 1.0), hidden


def loading_tracker_snapshot(
    frame_index, fps=10, duration=6, full_hold=1,
    spawn_radius=SPAWN_RADIUS, tracker_radius=TRACKER_RADIUS,
):
    """Return the spawn region plus its lower-status generation dependencies."""
    progress = _animation_progress(frame_index, fps, duration, full_hold)
    distances = _distance_grid(tracker_radius)
    dependency_distance = distances - spawn_radius
    target = np.zeros(distances.shape, dtype=int)
    target[distances <= spawn_radius] = FULL_STATUS
    for distance, status in DEPENDENCY_STATUS_BY_DISTANCE.items():
        target[dependency_distance == distance] = status

    work = progress * (FULL_STATUS + spawn_radius)
    stages = np.floor(work - distances).astype(int)
    stages = np.maximum(stages, 0)
    stages = np.minimum(stages, target)
    hidden = target == 0
    if progress >= 1.0:
        stages = target
    return stages, hidden


def _world_stage_rgba(biome_layer, terrain, heights, chunk_stages):
    """Reveal source-derived terrain only when each chunk reaches that stage."""
    height, width, _ = terrain.shape
    background = np.asarray(to_rgb(COLORS['background']))
    output = np.empty((height, width, 4), dtype=float)
    output[..., :3] = background
    output[..., 3] = 1.0

    rows = np.floor(np.linspace(0, 21, height, endpoint=False)).astype(int)
    columns = np.floor(np.linspace(0, 21, width, endpoint=False)).astype(int)
    pixel_stages = chunk_stages[np.ix_(rows, columns)]

    early = (pixel_stages >= 1) & (pixel_stages < 3)
    output[early, :3] = np.asarray(to_rgb('#242A33'))
    biome_mask = pixel_stages == 3
    output[biome_mask, :3] = biome_layer[biome_mask, :3]

    height_values = np.asarray(heights, dtype=float)
    height_normalized = np.clip((height_values - 48.0) / 48.0, 0.0, 1.0)
    noise_layer = np.repeat(height_normalized[..., None], 3, axis=2)
    noise_layer *= np.asarray([0.69, 0.78, 0.91])
    noise_mask = pixel_stages == 4
    output[noise_mask, :3] = noise_layer[noise_mask]

    surface_mask = pixel_stages >= 5
    output[surface_mask, :3] = terrain[surface_mask, :3]
    return output


def create_seed_loading_animation(
    save_path, seed=-4172144997902289642, fps=10, duration=7.2, full_hold=1.2,
):
    """Render progressive terrain meaning beside the vanilla-style tracker."""
    total_frames = int(round(fps * duration))
    radius = SPAWN_RADIUS
    tracker_radius = TRACKER_RADIUS
    display_radius = 10.5
    resolution = 169
    block_x_values = np.linspace(-168, 168, resolution)
    block_z_values = np.linspace(-168, 168, resolution)
    block_x, block_z = np.meshgrid(block_x_values, block_z_values)
    biome_ids, heights = overworld_surface_sample(
        seed, resolution=resolution,
        x_extent=(-168, 168), z_extent=(-168, 168), coordinate_scale=1.0,
    )
    terrain = terrain_rgba_from_sample(
        biome_ids, heights, block_x, block_z, 'overworld',
    )
    biome_layer = terrain_rgba_from_sample(
        biome_ids, heights, block_x, block_z, 'overworld', flat=True,
    )
    initial_stages, _, _ = chunk_status_snapshot(
        0, fps=fps, duration=duration, full_hold=full_hold, radius=radius,
    )
    initial_world = _world_stage_rgba(
        biome_layer, terrain, heights, initial_stages,
    )

    figure = plt.figure(figsize=(12.8, 7.2), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[2.18, 1.0],
        left=0.055, right=0.98, top=0.88, bottom=0.10, wspace=0.10,
    )
    axis = figure.add_subplot(grid[0, 0])
    side = figure.add_subplot(grid[0, 1])
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.axis('off')

    world_image = axis.imshow(
        initial_world, origin='lower', interpolation='nearest',
        extent=(-display_radius, display_radius, -display_radius, display_radius),
    )
    axis.set_aspect('equal')
    axis.set_xlim(-display_radius, display_radius)
    axis.set_ylim(-display_radius, display_radius)
    axis.set_xlabel('Chunk X')
    axis.set_ylabel('Chunk Z')
    axis.set_title('21 x 21 SPAWN REGION  |  441 CHUNKS', fontsize=11, pad=8)
    axis.scatter([0], [0], marker='+', s=90, c=COLORS['text'], linewidths=1.2, zorder=8)
    for spine in axis.spines.values():
        spine.set_color(COLORS['grid'])
    axis.tick_params(colors=COLORS['muted'], labelsize=8)
    stage_text = axis.text(
        0.02, 0.98, '', transform=axis.transAxes, ha='left', va='top',
        color=COLORS['text'], fontsize=9.3, fontweight='bold', family='monospace',
        bbox=dict(boxstyle='round,pad=0.35', facecolor=COLORS['panel'],
                  edgecolor=COLORS['grid'], alpha=0.94), zorder=9,
    )

    for boundary in np.arange(-10.5, 11.5, 1.0):
        axis.axvline(boundary, color='#11151D', linewidth=0.28, alpha=0.42, zorder=4)
        axis.axhline(boundary, color='#11151D', linewidth=0.28, alpha=0.42, zorder=4)

    tracker = side.inset_axes([0.02, 0.35, 0.96, 0.62])
    tracker.set_title('SPAWN REGION + GENERATION DEPENDENCIES', fontsize=8.1, pad=5)
    tracker.set_aspect('equal')
    tracker.set_xticks([])
    tracker.set_yticks([])
    tracker.set_facecolor('#080808')
    tracker_stages, tracker_hidden = loading_tracker_snapshot(
        0, fps=fps, duration=duration, full_hold=full_hold,
        spawn_radius=radius, tracker_radius=tracker_radius,
    )
    tracker_image = tracker.imshow(
        np.ma.masked_where(tracker_hidden | (tracker_stages == 0), tracker_stages),
        origin='lower',
        interpolation='nearest', cmap=ListedColormap(VANILLA_STATUS_COLORS),
        vmin=0, vmax=len(VANILLA_STATUS_COLORS) - 1,
    )
    tracker.add_patch(Rectangle(
        (-0.5, -0.5), 2 * tracker_radius + 1, 2 * tracker_radius + 1,
        fill=False, edgecolor='#FFEEFF', linewidth=0.8, alpha=0.48,
    ))
    tracker.add_patch(Rectangle(
        (tracker_radius - radius - 0.5, tracker_radius - radius - 0.5),
        2 * radius + 1, 2 * radius + 1, fill=False,
        edgecolor=COLORS['cyan'], linewidth=1.2, alpha=0.92,
    ))
    tracker.add_patch(Rectangle(
        (tracker_radius - 0.5, tracker_radius - 0.5), 1, 1, fill=False,
        edgecolor=COLORS['gold'], linewidth=1.8,
    ))

    progress_text = side.text(
        0.50, 0.305, '0%', ha='center', va='center',
        color=COLORS['text'], fontsize=19, fontweight='black',
    )
    strip = side.inset_axes([0.04, 0.045, 0.92, 0.18])
    strip.set_xlim(0, 7)
    strip.set_ylim(0, 2)
    strip.axis('off')
    legend_indices = (1, 3, 4, 5, 8, 9, 12)
    stage_boxes = []
    for position, index in enumerate(legend_indices):
        column = position % 4
        row = position // 4
        x = column * 1.75
        y = 1.45 - row * 0.85
        box = Rectangle(
            (x, y), 0.36, 0.30, facecolor=STATUS_COLORS[index],
            edgecolor='#07090E', linewidth=0.5,
        )
        strip.add_patch(box)
        strip.text(x + 0.45, y + 0.15, STATUS_SHORT[index], va='center',
                   color=COLORS['muted'], fontsize=6.6, fontweight='bold')
        stage_boxes.append((index, box))

    figure.suptitle(
        'PREPARING THE SPAWN REGION', color=COLORS['text'],
        fontsize=17, fontweight='black', y=0.96,
    )

    def update(frame_index):
        stages, _, hidden = chunk_status_snapshot(
            frame_index, fps=fps, duration=duration,
            full_hold=full_hold, radius=radius,
        )
        world_image.set_data(_world_stage_rgba(
            biome_layer, terrain, heights, stages,
        ))
        tracker_stages, tracker_hidden = loading_tracker_snapshot(
            frame_index, fps=fps, duration=duration,
            full_hold=full_hold, spawn_radius=radius,
            tracker_radius=tracker_radius,
        )
        tracker_image.set_data(np.ma.masked_where(
            tracker_hidden | (tracker_stages == 0), tracker_stages,
        ))
        full_chunks = int(np.count_nonzero(stages == FULL_STATUS))
        pipeline_progress = np.sum(stages) / (stages.size * FULL_STATUS)
        visible_stage = int(np.max(stages))
        stage_text.set_text(f'FULL CHUNKS  {full_chunks:3d} / {stages.size}')
        progress_text.set_text(f'{round(100.0 * pipeline_progress):d}% REGION PIPELINE')
        for index, box in stage_boxes:
            box.set_edgecolor(COLORS['text'] if index == visible_stage else '#07090E')
            box.set_linewidth(1.5 if index == visible_stage else 0.5)
        return []

    animation = FuncAnimation(
        figure, update, frames=total_frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
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
