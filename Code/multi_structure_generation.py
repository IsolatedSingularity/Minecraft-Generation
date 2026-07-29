"""Java 1.16.1 Nether structure candidate animation."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
import numpy as np

from core.constants import (
    NETHER_RUINED_PORTAL_SPACING,
    NETHER_STRUCTURE_SPACING,
)
from core.rendering import optimize_gif
from core.strongholds import generate_stronghold_candidates
from core.structures import (
    NETHER_RUINED_PORTAL,
    candidate_in_region,
    nether_shared_candidate,
)
from core.style import COLORS, apply_style


apply_style()


def _spiral_regions(radius):
    values = [(x, z) for x in range(-radius, radius + 1)
              for z in range(-radius, radius + 1)]
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
        candidate_in_region(
            seed, region_x, region_z, NETHER_RUINED_PORTAL,
        )
        for region_x, region_z in regions
    ]
    strongholds = generate_stronghold_candidates(seed)[:3]
    total_frames = int(fps * duration)
    limit = (region_radius + 0.75) * NETHER_STRUCTURE_SPACING

    figure, axis = plt.subplots(figsize=(12.8, 7.2), facecolor=COLORS['background'])
    figure.subplots_adjust(left=0.085, right=0.925, top=0.965, bottom=0.105)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect('equal')
    axis.set_xlabel('Nether chunk X')
    axis.set_ylabel('Nether chunk Z')
    axis.set_facecolor('#140E12')
    axis.tick_params(colors=COLORS['muted'], labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(COLORS['grid'])

    grid_extent = int(limit // NETHER_STRUCTURE_SPACING + 1)
    for coordinate in range(-grid_extent, grid_extent + 1):
        value = coordinate * NETHER_STRUCTURE_SPACING
        axis.axvline(value, color=COLORS['fortress'], linewidth=0.45, alpha=0.19)
        axis.axhline(value, color=COLORS['fortress'], linewidth=0.45, alpha=0.19)
    portal_extent = int(limit // NETHER_RUINED_PORTAL_SPACING + 1)
    for coordinate in range(-portal_extent, portal_extent + 1):
        value = coordinate * NETHER_RUINED_PORTAL_SPACING
        axis.axvline(value, color=COLORS['ruined_portal'], linewidth=0.4,
                     alpha=0.14, linestyle=':')
        axis.axhline(value, color=COLORS['ruined_portal'], linewidth=0.4,
                     alpha=0.14, linestyle=':')

    axis.scatter([0], [0], marker='+', s=85, c=COLORS['text'],
                 linewidths=1.1, zorder=8)
    fortress_points = axis.scatter(
        [], [], s=48, marker='o', c=COLORS['fortress'],
        edgecolors=COLORS['text'], linewidths=0.35, zorder=6,
    )
    bastion_points = axis.scatter(
        [], [], s=52, marker='s', c=COLORS['bastion'],
        edgecolors=COLORS['text'], linewidths=0.35, zorder=6,
    )
    portal_points = axis.scatter(
        [], [], s=42, marker='D', c=COLORS['ruined_portal'],
        edgecolors=COLORS['text'], linewidths=0.35, zorder=7,
    )
    active_region = Rectangle(
        (0, 0), NETHER_STRUCTURE_SPACING, NETHER_STRUCTURE_SPACING,
        fill=False, edgecolor=COLORS['cyan'], linewidth=1.1,
        alpha=0.0, zorder=8,
    )
    axis.add_patch(active_region)

    inset = axis.inset_axes([0.735, 0.045, 0.23, 0.23])
    inset.set_facecolor(COLORS['panel'])
    inset.set_aspect('equal')
    inset.set_xlim(-3200, 3200)
    inset.set_ylim(-3200, 3200)
    inset.tick_params(colors=COLORS['muted'], labelsize=5.5)
    inset.set_xticks([-2048, 0, 2048])
    inset.set_yticks([-2048, 0, 2048])
    for spine in inset.spines.values():
        spine.set_color(COLORS['grid'])
    inset.add_patch(Circle(
        (0, 0), 2048, fill=False, edgecolor=COLORS['stronghold'],
        linestyle='--', linewidth=0.7, alpha=0.6,
    ))
    inset.scatter([0], [0], marker='+', s=35, c=COLORS['text'], linewidths=0.8)
    stronghold_points = inset.scatter(
        [item['x'] for item in strongholds],
        [item['z'] for item in strongholds],
        s=35, c=COLORS['stronghold'], marker='o',
        edgecolors=COLORS['text'], linewidths=0.35, alpha=0.0,
    )
    linked_portal = portals[0]
    linked_point = inset.scatter(
        [linked_portal['chunk_x'] * 16 * 8],
        [linked_portal['chunk_z'] * 16 * 8],
        s=30, c=COLORS['ruined_portal'], marker='D',
        edgecolors=COLORS['text'], linewidths=0.35, alpha=0.0,
    )

    legend = figure.add_axes([0.18, 0.025, 0.64, 0.04])
    legend.set_xlim(0, 1)
    legend.set_ylim(0, 1)
    legend.axis('off')
    entries = [
        (0.03, 'o', COLORS['fortress'], 'fortress candidate'),
        (0.34, 's', COLORS['bastion'], 'bastion candidate'),
        (0.64, 'D', COLORS['ruined_portal'], 'ruined portal candidate'),
    ]
    for x, marker, color, label in entries:
        legend.scatter([x], [0.5], s=38, marker=marker, c=color,
                       edgecolors=COLORS['text'], linewidths=0.3)
        legend.text(x + 0.035, 0.5, label, va='center',
                    color=COLORS['muted'], fontsize=7.2)

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
        spacing = NETHER_RUINED_PORTAL_SPACING if visible_portals else NETHER_STRUCTURE_SPACING
        active_region.set_xy((current['region_x'] * spacing, current['region_z'] * spacing))
        active_region.set_width(spacing)
        active_region.set_height(spacing)
        active_region.set_alpha(0.85)

        inset_alpha = float(np.clip((progress - 0.72) / 0.20, 0.0, 1.0))
        stronghold_points.set_alpha(inset_alpha)
        if visible_portals:
            linked_point.set_alpha(inset_alpha)
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
    create_multi_structure_animation(plots / 'multi_structure_generation.gif')


if __name__ == '__main__':
    main()
