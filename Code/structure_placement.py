"""Exact Java 1.16.1 village candidate placement animation."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path as MarkerPath
import numpy as np

from core.constants import VILLAGE_SPACING
from core.rendering import optimize_gif
from core.structures import VILLAGE, candidate_in_region
from core.style import COLORS, addSoftShadow, apply_style, style_axis
from core.terrain import addTerrainBackdrop


apply_style()

HOUSE_MARKER = MarkerPath(
    [
        (-0.78, -0.72), (0.78, -0.72), (0.78, 0.08),
        (0.0, 0.82), (-0.78, 0.08), (-0.78, -0.72),
    ],
    [
        MarkerPath.MOVETO, MarkerPath.LINETO, MarkerPath.LINETO,
        MarkerPath.LINETO, MarkerPath.LINETO, MarkerPath.CLOSEPOLY,
    ],
)


def spiral_regions(radius):
    regions = [
        (x, z) for x in range(-radius, radius + 1)
        for z in range(-radius, radius + 1)
    ]
    return sorted(regions, key=lambda item: (
        max(abs(item[0]), abs(item[1])),
        np.arctan2(item[1], item[0]),
    ))


def create_structure_placement_animation(
    save_path, seed=42, region_radius=5, fps=12, duration=9,
):
    regions = spiral_regions(region_radius)
    candidates = [
        candidate_in_region(seed, region_x, region_z, VILLAGE)
        for region_x, region_z in regions
    ]
    total_frames = int(fps * duration)
    spacing = VILLAGE_SPACING
    limit = (region_radius + 0.65) * spacing

    figure, axis = plt.subplots(figsize=(12.8, 7.2), facecolor=COLORS['background'])
    figure.subplots_adjust(left=0.10, right=0.90, top=0.95, bottom=0.11)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel('Chunk X')
    axis.set_ylabel('Chunk Z')
    style_axis(axis, equal=True, grid=False)
    addTerrainBackdrop(axis, limit, seed, dimension='overworld', alpha=0.34)

    for region_x in range(-region_radius, region_radius + 1):
        for region_z in range(-region_radius, region_radius + 1):
            axis.add_patch(Rectangle(
                (region_x * spacing, region_z * spacing), spacing, spacing,
                facecolor=COLORS['panel'], edgecolor=COLORS['grid'],
                linewidth=0.58, alpha=0.16, zorder=0,
            ))

    axis.axhline(0, color=COLORS['muted'], linewidth=0.55, alpha=0.42)
    axis.axvline(0, color=COLORS['muted'], linewidth=0.55, alpha=0.42)
    axis.scatter(
        [0], [0], marker='+', s=78, c=COLORS['text'],
        linewidths=1.0, zorder=8,
    )

    points = axis.scatter(
        [], [], s=58, marker=HOUSE_MARKER, c=[], cmap='Blues',
        norm=Normalize(0, np.sqrt(2) * 23),
        edgecolors=COLORS['panel'], linewidths=0.65,
        alpha=0.92, zorder=5,
    )
    current_region = Rectangle(
        (0, 0), spacing, spacing, fill=False,
        edgecolor=COLORS['blue'], linewidth=1.6, alpha=0.0, zorder=7,
    )
    current_window = Rectangle(
        (0, 0), 24, 24, facecolor=COLORS['cyan'],
        edgecolor=COLORS['blue'], linewidth=0.95,
        linestyle='--', alpha=0.0, zorder=2,
    )
    axis.add_patch(current_window)
    axis.add_patch(current_region)
    current_point = axis.scatter(
        [], [], s=190, marker=HOUSE_MARKER, facecolors='none',
        edgecolors=COLORS['blue'], linewidths=1.55, zorder=9,
    )

    legendHandles = [
        Patch(
            facecolor='#B9DDA7', edgecolor='none', alpha=0.55,
            label='Terrain context',
        ),
        Patch(
            facecolor='none', edgecolor=COLORS['grid'],
            label='32 x 32 chunk region',
        ),
        Patch(
            facecolor=COLORS['cyan'], edgecolor=COLORS['blue'],
            alpha=0.30, label='24 x 24 candidate window',
        ),
        Line2D(
            [0], [0], marker=MarkerStyle(HOUSE_MARKER), linestyle='None',
            markerfacecolor=COLORS['blue'], markeredgecolor=COLORS['panel'],
            markersize=8.5, label='Java RNG village candidate',
        ),
    ]
    legend = axis.legend(
        handles=legendHandles, loc='upper right', title='Placement key',
        borderpad=0.9, labelspacing=0.75, fontsize=8.0, title_fontsize=8.8,
    )
    addSoftShadow(legend.get_frame(), offset=(1.8, -1.8), alpha=0.20)

    def update(frame_index):
        progress = frame_index / max(total_frames - 1, 1)
        shown = min(len(candidates), max(1, round(progress * len(candidates))))
        visible = candidates[:shown]
        offsets = np.array([
            [item['chunk_x'], item['chunk_z']] for item in visible
        ])
        offset_magnitude = np.array([
            np.hypot(item['offset_x'], item['offset_z']) for item in visible
        ])
        points.set_offsets(offsets)
        points.set_array(offset_magnitude)
        item = visible[-1]
        region_origin = (
            item['region_x'] * spacing,
            item['region_z'] * spacing,
        )
        current_region.set_xy(region_origin)
        current_region.set_alpha(0.95)
        current_window.set_xy(region_origin)
        current_window.set_alpha(0.18)
        current_point.set_offsets(np.array([[item['chunk_x'], item['chunk_z']]]))
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
    create_structure_placement_animation(plots / 'structure_placement.gif')


if __name__ == '__main__':
    main()
