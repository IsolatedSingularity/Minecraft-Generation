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
from core.minecraft_visuals import (
    NETHER_BIOMES,
    biome_texture_swatch,
    draw_minecraft_terrain,
    minecraft_nether_biome_grid,
)
from core.rendering import optimize_gif
from core.structure_visuals import STRUCTURE_SCHEMATICS, draw_structure_schematic
from core.structures import (
    NETHER_RUINED_PORTAL,
    candidate_in_region,
    nether_shared_candidate,
)
from core.style import COLORS, apply_style


apply_style()


def _spiral_regions(radius):
    values = [
        (x, z) for x in range(-radius, radius + 1)
        for z in range(-radius, radius + 1)
    ]
    return sorted(values, key=lambda item: (
        max(abs(item[0]), abs(item[1])),
        np.arctan2(item[1], item[0]),
    ))


def _biome_at_chunk(biomes, chunk_x, chunk_z, minimum, maximum):
    scale = (biomes.shape[0] - 1) / float(maximum - minimum)
    column = int(np.clip(round((chunk_x - minimum) * scale), 0, biomes.shape[1] - 1))
    row = int(np.clip(round((chunk_z - minimum) * scale), 0, biomes.shape[0] - 1))
    return str(biomes[row, column])


def _draw_legend(axis):
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis('off')
    axis.text(
        0.0, 0.98, 'NETHER BIOME TEXTURES',
        color=COLORS['text'], fontsize=10, fontweight='black', va='top',
    )
    for index, (name, biome) in enumerate(NETHER_BIOMES.items()):
        y = 0.88 - index * 0.105
        axis.imshow(
            biome_texture_swatch(name, 24),
            extent=(0.0, 0.19, y - 0.045, y + 0.045),
            origin='lower', interpolation='nearest', aspect='auto', zorder=2,
        )
        axis.add_patch(Rectangle(
            (0.0, y - 0.045), 0.19, 0.09,
            facecolor='none', edgecolor=COLORS['muted'],
            linewidth=0.5, zorder=3,
        ))
        axis.text(
            0.24, y, biome.label, color=COLORS['text'],
            fontsize=8.0, va='center', ha='left',
        )

    axis.text(
        0.0, 0.22, 'STRUCTURE PLANS',
        color=COLORS['text'], fontsize=10, fontweight='black', va='top',
    )
    for index, name in enumerate(('fortress', 'bastion', 'ruined_portal')):
        y = 0.145 - index * 0.066
        draw_structure_schematic(axis, name, 0.075, y, size=0.036, zorder=5)
        axis.text(
            0.17, y, STRUCTURE_SCHEMATICS[name].label,
            color=COLORS['text'], fontsize=7.6, va='center', ha='left',
        )


def nether_structure_candidates(seed=42, region_radius=4, resolution=384):
    """Return displayed candidate layers and their illustrative biome gates."""
    minimum = -region_radius * NETHER_STRUCTURE_SPACING - 5
    maximum = (region_radius + 1) * NETHER_STRUCTURE_SPACING + 5
    biomes = minecraft_nether_biome_grid(
        seed, resolution=resolution,
        x_extent=(minimum, maximum), z_extent=(minimum, maximum),
        coordinate_scale=16.0, showcase=True,
    )
    regions = _spiral_regions(region_radius)
    shared = []
    portals = []
    for region_x, region_z in regions:
        shared_item = nether_shared_candidate(seed, region_x, region_z)
        shared_item['biome'] = _biome_at_chunk(
            biomes, shared_item['chunk_x'], shared_item['chunk_z'],
            minimum, maximum,
        )
        # Bastions do not start in basalt deltas. Fortresses remain valid.
        if not (
            shared_item['name'] == 'bastion'
            and shared_item['biome'] == 'basalt_deltas'
        ):
            shared.append(shared_item)

        portal = candidate_in_region(
            seed, region_x, region_z, NETHER_RUINED_PORTAL,
        )
        portal['biome'] = _biome_at_chunk(
            biomes, portal['chunk_x'], portal['chunk_z'], minimum, maximum,
        )
        portals.append(portal)
    return shared, portals, biomes, (minimum, maximum)


def create_multi_structure_animation(
    save_path, seed=42, region_radius=4, fps=8, duration=14,
):
    shared, portals, _, (minimum, maximum) = nether_structure_candidates(
        seed, region_radius,
    )
    total_frames = int(fps * duration)

    figure = plt.figure(figsize=(16.0, 8.6), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[3.8, 1.05],
        left=0.055, right=0.98, top=0.89, bottom=0.11, wspace=0.08,
    )
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])
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
        dimension='nether', resolution=384, alpha=0.92,
        coordinate_scale=16.0, showcase=True,
    )
    grid_extent = region_radius + 2
    for coordinate in range(-grid_extent, grid_extent + 1):
        value = coordinate * NETHER_STRUCTURE_SPACING
        axis.axvline(value, color=COLORS['coral'], linewidth=0.70, alpha=0.27)
        axis.axhline(value, color=COLORS['coral'], linewidth=0.70, alpha=0.27)
    for coordinate in range(-grid_extent, grid_extent + 1):
        value = coordinate * NETHER_RUINED_PORTAL_SPACING
        axis.axvline(value, color=COLORS['violet'], linewidth=0.58, alpha=0.22, linestyle=':')
        axis.axhline(value, color=COLORS['violet'], linewidth=0.58, alpha=0.22, linestyle=':')
    axis.scatter([0], [0], marker='+', s=85, c=COLORS['text'], linewidths=1.1, zorder=12)

    shared_groups = []
    for item in shared:
        group = draw_structure_schematic(
            axis, item['name'], item['chunk_x'], item['chunk_z'],
            size=4.1, zorder=7,
        )
        group.set_visible(False)
        shared_groups.append(group)
    portal_groups = []
    for item in portals:
        group = draw_structure_schematic(
            axis, 'ruined_portal', item['chunk_x'], item['chunk_z'],
            size=3.6, zorder=8,
        )
        group.set_visible(False)
        portal_groups.append(group)

    active_shared = Rectangle(
        (0, 0), NETHER_STRUCTURE_SPACING, NETHER_STRUCTURE_SPACING,
        fill=False, edgecolor=COLORS['coral'], linewidth=1.55,
        alpha=0.0, zorder=11,
    )
    active_portal = Rectangle(
        (0, 0), NETHER_RUINED_PORTAL_SPACING, NETHER_RUINED_PORTAL_SPACING,
        fill=False, edgecolor=COLORS['violet'], linewidth=1.55,
        linestyle='--', alpha=0.0, zorder=11,
    )
    axis.add_patch(active_shared)
    axis.add_patch(active_portal)
    trace_text = figure.text(
        0.405, 0.052, '', ha='center', va='center',
        color=COLORS['text'], fontsize=8.9, fontweight='bold',
        family='monospace',
        bbox=dict(
            boxstyle='round,pad=0.42', facecolor=COLORS['panel'],
            edgecolor=COLORS['violet'], alpha=0.95,
        ),
    )
    _draw_legend(legend_axis)
    figure.suptitle(
        'NETHER STRUCTURE GENERATION   JAVA 1.16.1',
        color=COLORS['text'], fontsize=18, fontweight='black', y=0.96,
    )

    def update(frame_index):
        progress = frame_index / max(total_frames - 1, 1)
        shared_progress = np.clip(progress / 0.90, 0.0, 1.0)
        portal_progress = np.clip((progress - 0.08) / 0.84, 0.0, 1.0)
        shared_count = max(1, round(shared_progress * len(shared)))
        portal_count = max(0, round(portal_progress * len(portals)))
        for index, group in enumerate(shared_groups):
            group.set_visible(index < shared_count)
        for index, group in enumerate(portal_groups):
            group.set_visible(index < portal_count)

        shared_item = shared[shared_count - 1]
        active_shared.set_xy((
            shared_item['region_x'] * NETHER_STRUCTURE_SPACING,
            shared_item['region_z'] * NETHER_STRUCTURE_SPACING,
        ))
        active_shared.set_alpha(0.92)
        if portal_count:
            portal_item = portals[portal_count - 1]
            active_portal.set_xy((
                portal_item['region_x'] * NETHER_RUINED_PORTAL_SPACING,
                portal_item['region_z'] * NETHER_RUINED_PORTAL_SPACING,
            ))
            active_portal.set_alpha(0.92)
            portal_text = (
                f"PORTAL ({portal_item['chunk_x']:+04d},{portal_item['chunk_z']:+04d}) "
                f"{NETHER_BIOMES[portal_item['biome']].label.upper()}"
            )
        else:
            portal_text = 'PORTAL PENDING'
        trace_text.set_text(
            f"SHARED ROLL {shared_item['type_roll']} -> {shared_item['name'].upper()} "
            f"({shared_item['chunk_x']:+04d},{shared_item['chunk_z']:+04d}) "
            f"{NETHER_BIOMES[shared_item['biome']].label.upper()}   {portal_text}"
        )
        return []

    animation = FuncAnimation(
        figure, update, frames=total_frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=72)
    plt.close(figure)
    optimize_gif(save_path, colors=64)
    return str(save_path)


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)
    create_multi_structure_animation(plots / 'multi_structure_generation.gif')


if __name__ == '__main__':
    main()
