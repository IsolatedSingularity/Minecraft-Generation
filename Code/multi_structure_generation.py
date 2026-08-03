"""Java 1.16.1 Nether structure candidate animation."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import numpy as np

from core.constants import (
    NETHER_RUINED_PORTAL_SPACING,
    NETHER_STRUCTURE_SPACING,
)
from core.minecraft_visuals import draw_minecraft_terrain
from core.rendering import optimize_gif
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
    save_path, seed=42, region_radius=4, fps=8, duration=12,
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
    total_frames = int(fps * duration)
    minimum = -region_radius * NETHER_STRUCTURE_SPACING - 5
    maximum = (region_radius + 1) * NETHER_STRUCTURE_SPACING + 5

    figure, axis = plt.subplots(figsize=(12.0, 6.75), facecolor=COLORS['background'])
    figure.subplots_adjust(left=0.075, right=0.98, top=0.875, bottom=0.10)
    axis.set_xlim(minimum, maximum)
    axis.set_ylim(minimum, maximum)
    axis.set_aspect('equal')
    axis.set_xlabel('Nether chunk X')
    axis.set_ylabel('Nether chunk Z')
    axis.set_facecolor('#140E12')
    axis.tick_params(colors=COLORS['muted'], labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(COLORS['grid'])

    draw_minecraft_terrain(
        axis, (minimum, maximum, minimum, maximum), seed=seed,
        dimension='nether', resolution=256, alpha=0.90,
    )

    grid_extent = region_radius + 2
    for coordinate in range(-grid_extent, grid_extent + 1):
        value = coordinate * NETHER_STRUCTURE_SPACING
        axis.axvline(value, color=COLORS['coral'], linewidth=0.75, alpha=0.30)
        axis.axhline(value, color=COLORS['coral'], linewidth=0.75, alpha=0.30)
    portal_extent = region_radius + 2
    for coordinate in range(-portal_extent, portal_extent + 1):
        value = coordinate * NETHER_RUINED_PORTAL_SPACING
        axis.axvline(value, color=COLORS['violet'], linewidth=0.62,
                     alpha=0.24, linestyle=':')
        axis.axhline(value, color=COLORS['violet'], linewidth=0.62,
                     alpha=0.24, linestyle=':')

    axis.scatter([0], [0], marker='+', s=85, c=COLORS['text'],
                 linewidths=1.1, zorder=8)
    fortress_points = axis.scatter(
        [], [], s=72, marker='P', c=COLORS['fortress'],
        edgecolors=COLORS['text'], linewidths=0.48, zorder=6,
    )
    bastion_points = axis.scatter(
        [], [], s=68, marker='s', c=COLORS['bastion'],
        edgecolors=COLORS['text'], linewidths=0.48, zorder=6,
    )
    portal_points = axis.scatter(
        [], [], s=58, marker='D', c=COLORS['ruined_portal'],
        edgecolors=COLORS['text'], linewidths=0.48, zorder=7,
    )
    active_shared = Rectangle(
        (0, 0), NETHER_STRUCTURE_SPACING, NETHER_STRUCTURE_SPACING,
        fill=False, edgecolor=COLORS['coral'], linewidth=1.5,
        alpha=0.0, zorder=8,
    )
    active_portal = Rectangle(
        (0, 0), NETHER_RUINED_PORTAL_SPACING, NETHER_RUINED_PORTAL_SPACING,
        fill=False, edgecolor=COLORS['violet'], linewidth=1.5,
        linestyle='--', alpha=0.0, zorder=8,
    )
    axis.add_patch(active_shared)
    axis.add_patch(active_portal)
    axis.text(
        0.018, 0.978, '27 x 27 SHARED GRID   25 x 25 PORTAL GRID',
        transform=axis.transAxes, ha='left', va='top',
        color=COLORS['text'], fontsize=13.5, fontweight='black', zorder=10,
        bbox=dict(
            boxstyle='square,pad=0.32', facecolor=COLORS['background'],
            edgecolor='none', alpha=0.80,
        ),
    )
    axis.text(
        0.985, 0.975,
        'FORTRESS  roll 0-1 / 5\nBASTION   roll 2-4 / 5\nPORTAL    independent salt',
        transform=axis.transAxes, ha='right', va='top',
        color=COLORS['text'], fontsize=8.7, fontweight='bold', zorder=10,
        bbox=dict(
            boxstyle='round,pad=0.34', facecolor=COLORS['panel'],
            edgecolor=COLORS['grid'], alpha=0.90,
        ),
    )
    trace_text = axis.text(
        0.50, 0.025, '', transform=axis.transAxes,
        ha='center', va='bottom', color=COLORS['text'],
        fontsize=9.7, fontweight='bold', family='monospace', zorder=10,
        bbox=dict(
            boxstyle='round,pad=0.42', facecolor=COLORS['panel'],
            edgecolor=COLORS['violet'], alpha=0.94,
        ),
    )
    figure.text(
        0.50, 0.936, 'NETHER STRUCTURE CANDIDATE LAYERS   JAVA 1.16.1',
        ha='center', va='center', color=COLORS['text'],
        fontsize=17, fontweight='black',
    )

    def update(frame_index):
        progress = frame_index / max(total_frames - 1, 1)
        shared_progress = np.clip(progress / 0.90, 0.0, 1.0)
        portal_progress = np.clip((progress - 0.08) / 0.84, 0.0, 1.0)
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
        shared_item = visible_shared[-1]
        active_shared.set_xy((
            shared_item['region_x'] * NETHER_STRUCTURE_SPACING,
            shared_item['region_z'] * NETHER_STRUCTURE_SPACING,
        ))
        active_shared.set_alpha(0.92)
        if visible_portals:
            portal_item = visible_portals[-1]
            active_portal.set_xy((
                portal_item['region_x'] * NETHER_RUINED_PORTAL_SPACING,
                portal_item['region_z'] * NETHER_RUINED_PORTAL_SPACING,
            ))
            active_portal.set_alpha(0.92)
            portal_text = (
                f"PORTAL ({portal_item['chunk_x']:+04d},"
                f"{portal_item['chunk_z']:+04d})"
            )
        else:
            portal_text = 'PORTAL pending'
        trace_text.set_text(
            f"SHARED ROLL {shared_item['type_roll']} -> "
            f"{shared_item['name'].upper()} "
            f"({shared_item['chunk_x']:+04d},{shared_item['chunk_z']:+04d})   "
            f"{portal_text}"
        )
        return []

    animation = FuncAnimation(
        figure, update, frames=total_frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(figure)
    optimize_gif(save_path, colors=80)
    return str(save_path)


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)
    create_multi_structure_animation(plots / 'multi_structure_generation.gif')


if __name__ == '__main__':
    main()
