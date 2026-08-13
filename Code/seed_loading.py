"""Java 1.16.1 spawn preparation and chunk-status visualization.

The status order, 21 by 21 target footprint, and vanilla loading-screen colour
mapping are source-backed. Relative task timing is an explanatory schedule.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap, to_rgb
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

# ChunkStatus.DISTANCE_TO_TARGET_GENERATION_STATUS in Java 1.16.1.
TARGET_STATUS_BY_DISTANCE = {0: 12, 1: 8, 2: 7}


def _distance_grid(radius):
    return np.fromfunction(
        lambda row, column: np.maximum(
            np.abs(row - radius), np.abs(column - radius),
        ),
        (2 * radius + 1, 2 * radius + 1), dtype=float,
    )


def chunk_status_snapshot(frame_index, fps=10, duration=6, full_hold=1, radius=10):
    """Return a staged view of the exact terminal dependency footprint."""
    total_frames = int(round(float(fps) * float(duration)))
    hold_frames = int(round(float(fps) * float(full_hold)))
    generation_frames = total_frames - hold_frames
    if generation_frames < 2:
        raise ValueError('duration must leave at least two generation frames')
    progress = (
        1.0 if int(frame_index) >= generation_frames
        else int(frame_index) / max(generation_frames - 1, 1)
    )
    distances = _distance_grid(radius)
    target = np.zeros(distances.shape, dtype=int)
    target[distances <= 10] = 1
    for distance, status in TARGET_STATUS_BY_DISTANCE.items():
        target[distances == distance] = status
    active_stage = min(len(STATUS_NAMES) - 1, int(np.floor(progress * len(STATUS_NAMES))))
    stages = np.minimum(active_stage, target)
    hidden = distances > 10.0
    if progress >= 1.0:
        stages = target
    return stages, np.where(hidden, 0.0, 1.0), hidden


def _world_stage_rgba(terrain, progress):
    """Build a visible terrain proxy instead of tinting a finished surface."""
    height, width, _ = terrain.shape
    background = np.asarray(to_rgb(COLORS['background']))
    output = np.empty((height, width, 4), dtype=float)
    output[..., :3] = background
    output[..., 3] = 1.0

    luminance = np.mean(terrain, axis=2, keepdims=True)
    noise_layer = np.repeat(luminance, 3, axis=2) * np.array([0.70, 0.78, 0.92])
    biome_layer = 0.38 * terrain + 0.62 * np.array([0.19, 0.25, 0.34])
    surface_layer = 0.78 * terrain + 0.22 * luminance
    features_layer = np.clip(terrain * 1.10 + 0.035, 0.0, 1.0)

    stages = (
        (0.10, 0.24, biome_layer),
        (0.22, 0.42, noise_layer),
        (0.38, 0.60, surface_layer),
        (0.56, 0.79, features_layer),
        (0.74, 0.96, terrain),
    )
    for start, end, layer in stages:
        blend = np.clip((progress - start) / (end - start), 0.0, 1.0)
        output[..., :3] = output[..., :3] * (1.0 - blend) + layer * blend
    return output


def create_seed_loading_animation(
    save_path, seed=-4172144997902289642, fps=10, duration=7.2, full_hold=1.2,
):
    """Render progressive terrain meaning beside the vanilla-style tracker."""
    total_frames = int(round(fps * duration))
    hold_frames = int(round(fps * full_hold))
    generation_frames = total_frames - hold_frames
    radius = 10
    display_radius = 180
    resolution = 361
    terrain = minecraft_terrain_rgba(
        seed, resolution=resolution, dimension='overworld',
        x_extent=(-display_radius - 0.5, display_radius + 0.5),
        z_extent=(-display_radius - 0.5, display_radius + 0.5),
        coordinate_scale=16.0, showcase=False,
    )[..., :3]
    initial_world = _world_stage_rgba(terrain, 0.0)

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
    axis.set_title('WHAT EACH GENERATION STAGE ADDS', fontsize=11, pad=8)
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

    tracker = side.inset_axes([0.08, 0.38, 0.84, 0.56])
    tracker.set_title('VANILLA-STYLE SPAWN REGION TRACKER', fontsize=8.4, pad=5)
    tracker.set_aspect('equal')
    tracker.set_xticks([])
    tracker.set_yticks([])
    tracker.set_facecolor('#080808')
    initial_stages, _, initial_hidden = chunk_status_snapshot(
        0, fps=fps, duration=duration, full_hold=full_hold, radius=radius,
    )
    tracker_image = tracker.imshow(
        np.ma.masked_where(initial_hidden | (initial_stages == 0), initial_stages),
        origin='lower',
        interpolation='nearest', cmap=ListedColormap(VANILLA_STATUS_COLORS),
        vmin=0, vmax=len(VANILLA_STATUS_COLORS) - 1,
    )
    tracker.add_patch(Rectangle(
        (-0.5, -0.5), 21, 21, fill=False,
        edgecolor='#FFEEFF', linewidth=1.0, alpha=0.72,
    ))
    tracker.add_patch(Rectangle(
        (radius - 0.5, radius - 0.5), 1, 1, fill=False,
        edgecolor=COLORS['gold'], linewidth=1.8,
    ))

    progress_text = side.text(
        0.50, 0.335, '0%', ha='center', va='center',
        color=COLORS['text'], fontsize=19, fontweight='black',
    )
    strip = side.inset_axes([0.04, 0.08, 0.92, 0.18])
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
        'BUILDING THE SPAWN REGION', color=COLORS['text'],
        fontsize=17, fontweight='black', y=0.96,
    )

    def update(frame_index):
        progress = (
            1.0 if frame_index >= generation_frames
            else frame_index / max(generation_frames - 1, 1)
        )
        eased = progress * progress * (3.0 - 2.0 * progress)
        world_image.set_data(_world_stage_rgba(terrain, eased))
        stages, _, hidden = chunk_status_snapshot(
            frame_index, fps=fps, duration=duration,
            full_hold=full_hold, radius=radius,
        )
        tracker_image.set_data(np.ma.masked_where(hidden | (stages == 0), stages))
        center_stage = int(stages[radius, radius])
        visible_stage = min(center_stage, 12)
        stage_text.set_text(f'CENTER CHUNK  {STATUS_NAMES[visible_stage]}')
        progress_text.set_text(f'{round(100.0 * center_stage / 12):d}% PIPELINE')
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
