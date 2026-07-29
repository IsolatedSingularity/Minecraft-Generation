"""Java 1.16.1 Nether structure candidate animation."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path as MarkerPath
import numpy as np

from core.constants import (
    NETHER_RUINED_PORTAL_SPACING,
    NETHER_STRUCTURE_SPACING,
)
from core.rendering import optimize_gif
from core.structures import (
    NETHER_RUINED_PORTAL,
    candidate_in_region,
    nether_shared_candidate,
)
from core.style import COLORS, addSoftShadow, apply_style, style_axis
from core.terrain import addTerrainBackdrop


apply_style()

FORTRESS_MARKER = MarkerPath(
    [
        (-0.82, -0.72), (0.82, -0.72), (0.82, 0.74),
        (0.42, 0.74), (0.42, 0.30), (0.12, 0.30),
        (0.12, 0.74), (-0.18, 0.74), (-0.18, 0.30),
        (-0.48, 0.30), (-0.48, 0.74), (-0.82, 0.74),
        (-0.82, -0.72),
    ],
    [MarkerPath.MOVETO] + [MarkerPath.LINETO] * 11 + [MarkerPath.CLOSEPOLY],
)
BASTION_MARKER = MarkerPath(
    [
        (-0.80, 0.72), (0.80, 0.72), (0.68, -0.18),
        (0.0, -0.86), (-0.68, -0.18), (-0.80, 0.72),
    ],
    [
        MarkerPath.MOVETO, MarkerPath.LINETO, MarkerPath.LINETO,
        MarkerPath.LINETO, MarkerPath.LINETO, MarkerPath.CLOSEPOLY,
    ],
)
PORTAL_MARKER = MarkerPath(
    [
        (-0.58, -0.82), (0.58, -0.82), (0.58, 0.40),
        (0.30, 0.82), (-0.30, 0.82), (-0.58, 0.40), (-0.58, -0.82),
    ],
    [
        MarkerPath.MOVETO, MarkerPath.LINETO, MarkerPath.LINETO,
        MarkerPath.LINETO, MarkerPath.LINETO, MarkerPath.LINETO,
        MarkerPath.CLOSEPOLY,
    ],
)


def _spiral_regions(radius):
    values = [
        (x, z) for x in range(-radius, radius + 1)
        for z in range(-radius, radius + 1)
    ]
    return sorted(values, key=lambda item: (
        max(abs(item[0]), abs(item[1])),
        np.arctan2(item[1], item[0]),
    ))


def create_multi_structure_animation(
    save_path, seed=42, region_radius=5, fps=12, duration=9,
):
    regions = _spiral_regions(region_radius)
    shared = [
        nether_shared_candidate(seed, region_x, region_z)
        for region_x, region_z in regions
    ]
    portals = [
        candidate_in_region(seed, region_x, region_z, NETHER_RUINED_PORTAL)
        for region_x, region_z in regions
    ]
    total_frames = int(fps * duration)
    limit = (region_radius + 0.75) * NETHER_STRUCTURE_SPACING

    figure, axis = plt.subplots(figsize=(12.8, 7.2), facecolor=COLORS['background'])
    figure.subplots_adjust(left=0.075, right=0.75, top=0.95, bottom=0.11)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel('Nether chunk X')
    axis.set_ylabel('Nether chunk Z')
    style_axis(axis, equal=True, grid=False)
    addTerrainBackdrop(axis, limit, seed, dimension='nether', alpha=0.42)

    grid_extent = int(limit // NETHER_STRUCTURE_SPACING + 1)
    for coordinate in range(-grid_extent, grid_extent + 1):
        value = coordinate * NETHER_STRUCTURE_SPACING
        axis.axvline(value, color=COLORS['fortress'], linewidth=0.48, alpha=0.28)
        axis.axhline(value, color=COLORS['fortress'], linewidth=0.48, alpha=0.28)
    portal_extent = int(limit // NETHER_RUINED_PORTAL_SPACING + 1)
    for coordinate in range(-portal_extent, portal_extent + 1):
        value = coordinate * NETHER_RUINED_PORTAL_SPACING
        axis.axvline(
            value, color=COLORS['ruined_portal'], linewidth=0.42,
            alpha=0.20, linestyle=':',
        )
        axis.axhline(
            value, color=COLORS['ruined_portal'], linewidth=0.42,
            alpha=0.20, linestyle=':',
        )

    axis.scatter(
        [0], [0], marker='+', s=85, c=COLORS['text'],
        linewidths=1.1, zorder=8,
    )
    fortress_points = axis.scatter(
        [], [], s=72, marker=FORTRESS_MARKER, c=COLORS['fortress'],
        edgecolors=COLORS['panel'], linewidths=0.75, zorder=6,
    )
    bastion_points = axis.scatter(
        [], [], s=76, marker=BASTION_MARKER, c=COLORS['bastion'],
        edgecolors=COLORS['panel'], linewidths=0.75, zorder=6,
    )
    portal_points = axis.scatter(
        [], [], s=68, marker=PORTAL_MARKER, c=COLORS['ruined_portal'],
        edgecolors=COLORS['panel'], linewidths=0.75, zorder=7,
    )
    active_region = Rectangle(
        (0, 0), NETHER_STRUCTURE_SPACING, NETHER_STRUCTURE_SPACING,
        fill=False, edgecolor=COLORS['blue'], linewidth=1.35,
        alpha=0.0, zorder=8,
    )
    axis.add_patch(active_region)

    legendHandles = [
        Patch(
            facecolor='#E4A4A0', edgecolor='none', alpha=0.62,
            label='Nether terrain context',
        ),
        Line2D(
            [0], [0], marker=MarkerStyle(FORTRESS_MARKER), linestyle='None',
            markerfacecolor=COLORS['fortress'], markeredgecolor=COLORS['panel'],
            markersize=9.5, label='Nether fortress',
        ),
        Line2D(
            [0], [0], marker=MarkerStyle(BASTION_MARKER), linestyle='None',
            markerfacecolor=COLORS['bastion'], markeredgecolor=COLORS['panel'],
            markersize=9.5, label='Bastion remnant',
        ),
        Line2D(
            [0], [0], marker=MarkerStyle(PORTAL_MARKER), linestyle='None',
            markerfacecolor=COLORS['ruined_portal'], markeredgecolor=COLORS['panel'],
            markersize=9.5, label='Ruined portal',
        ),
        Patch(
            facecolor='none', edgecolor=COLORS['blue'],
            label='Active candidate region',
        ),
    ]
    legend = figure.legend(
        handles=legendHandles, loc='center right', bbox_to_anchor=(0.985, 0.50),
        title='Nether placement key', borderpad=1.0, labelspacing=1.0,
        fontsize=8.4, title_fontsize=9.3,
    )
    addSoftShadow(legend.get_frame(), offset=(2.0, -2.0), alpha=0.20)

    def update(frame_index):
        progress = frame_index / max(total_frames - 1, 1)
        shared_progress = np.clip(progress / 0.72, 0.0, 1.0)
        portal_progress = np.clip((progress - 0.24) / 0.68, 0.0, 1.0)
        shared_count = max(1, round(shared_progress * len(shared)))
        portal_count = max(0, round(portal_progress * len(portals)))
        visible_shared = shared[:shared_count]
        visible_portals = portals[:portal_count]
        fortresses = [item for item in visible_shared if item['name'] == 'fortress']
        bastions = [item for item in visible_shared if item['name'] == 'bastion']
        fortress_points.set_offsets(
            np.array([[item['chunk_x'], item['chunk_z']] for item in fortresses])
            if fortresses else np.empty((0, 2))
        )
        bastion_points.set_offsets(
            np.array([[item['chunk_x'], item['chunk_z']] for item in bastions])
            if bastions else np.empty((0, 2))
        )
        portal_points.set_offsets(
            np.array([[item['chunk_x'], item['chunk_z']] for item in visible_portals])
            if visible_portals else np.empty((0, 2))
        )
        current = visible_portals[-1] if visible_portals else visible_shared[-1]
        spacing = (
            NETHER_RUINED_PORTAL_SPACING
            if visible_portals else NETHER_STRUCTURE_SPACING
        )
        active_region.set_xy((
            current['region_x'] * spacing,
            current['region_z'] * spacing,
        ))
        active_region.set_width(spacing)
        active_region.set_height(spacing)
        active_region.set_alpha(0.88)
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
    create_multi_structure_animation(plots / 'multi_structure_generation.gif')


if __name__ == '__main__':
    main()
