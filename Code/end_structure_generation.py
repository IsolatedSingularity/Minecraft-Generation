"""End-city candidates and paired End-gateway visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from core.end_generation import (
    end_city_candidates,
    gateway_positions,
    outer_gateway_positions,
    outer_island_seed_field,
)
from core.end_visuals import ISLAND_CMAP, draw_central_island
from core.structure_visuals import draw_structure_schematic
from core.style import COLORS, apply_style, style_axis


apply_style()


def create_end_structure_generation(save_path, dpi=210, seed=42):
    """Render End-city placement and both gateway endpoints."""
    limit = 3600.0
    sites = outer_island_seed_field(seed, max_coordinate_blocks=int(limit))
    radii = np.hypot(sites['block_x'], sites['block_z'])
    outer = radii > 1024.0
    strength = 22.0 - sites['falloff'][outer]
    cities = end_city_candidates(seed, max_coordinate_blocks=int(limit))
    outer_gateways = outer_gateway_positions(seed)
    central_gateways = gateway_positions()

    figure = plt.figure(figsize=(16.0, 9.0), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[3.8, 1.18],
        left=0.055, right=0.98, top=0.91, bottom=0.08, wspace=0.10,
    )
    axis = figure.add_subplot(grid[0, 0])
    side = figure.add_subplot(grid[0, 1])

    axis.scatter(
        sites['block_x'][outer], sites['block_z'][outer],
        s=2.0 + 0.75 * strength, c=strength,
        cmap=ISLAND_CMAP, vmin=1.0, vmax=13.0,
        marker='o', linewidths=0, alpha=0.58, zorder=1,
    )
    axis.add_patch(Circle(
        (0, 0), 1024, fill=False, edgecolor=COLORS['end_stone'],
        linewidth=1.2, linestyle='--', alpha=0.82, zorder=3,
    ))
    draw_central_island(axis, seed=seed, extent=180, resolution=121, alpha=0.98, zorder=4)

    for outer_gateway in outer_gateways:
        axis.plot(
            [outer_gateway['central_x'], outer_gateway['x']],
            [outer_gateway['central_z'], outer_gateway['z']],
            color=COLORS['portal'], linewidth=0.48, alpha=0.26, zorder=2,
        )
    axis.scatter(
        [item['x'] for item in outer_gateways],
        [item['z'] for item in outer_gateways],
        s=42, marker='D', c=COLORS['portal'],
        edgecolors=COLORS['text'], linewidths=0.55, zorder=8,
    )
    axis.scatter(
        [item['x'] for item in central_gateways],
        [item['z'] for item in central_gateways],
        s=14, marker='s', c=COLORS['cyan'],
        edgecolors=COLORS['text'], linewidths=0.25, zorder=8,
    )
    for city in cities:
        draw_structure_schematic(
            axis, 'end_city', city['block_x'], city['block_z'],
            size=42.0, zorder=7,
        )

    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel('Block X')
    axis.set_ylabel('Block Z')
    axis.set_title('First outer-island band and qualified End-city candidates', fontsize=12, pad=9)
    style_axis(axis, equal=True, grid=False)

    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.axis('off')
    side.text(
        0.0, 0.98, 'GATEWAY PAIRING', color=COLORS['text'],
        fontsize=10.5, fontweight='black', va='top',
    )
    inset = side.inset_axes([0.05, 0.57, 0.90, 0.35])
    draw_central_island(inset, seed=seed, extent=112, alpha=0.68, zorder=0)
    inset.add_patch(Circle(
        (0, 0), 96, fill=False, edgecolor=COLORS['cyan'],
        linewidth=1.0, linestyle='--', alpha=0.82,
    ))
    inset.scatter(
        [item['x'] for item in central_gateways],
        [item['z'] for item in central_gateways],
        s=24, marker='s', c=COLORS['cyan'],
        edgecolors=COLORS['text'], linewidths=0.4, zorder=4,
    )
    for item in central_gateways[::2]:
        inset.text(
            item['x'] * 1.10, item['z'] * 1.10, str(item['index']),
            color=COLORS['muted'], fontsize=5.5, ha='center', va='center',
        )
    inset.set_xlim(-116, 116)
    inset.set_ylim(-116, 116)
    inset.axis('off')

    side.text(
        0.0, 0.51, 'MAP LEGEND', color=COLORS['text'],
        fontsize=10.5, fontweight='black', va='top',
    )
    legend_handles = [
        Line2D([0], [0], marker='s', linestyle='none', markersize=7,
               markerfacecolor=COLORS['cyan'], markeredgecolor=COLORS['text'],
               label='Central gateway, radius 96'),
        Line2D([0], [0], marker='D', linestyle='none', markersize=7,
               markerfacecolor=COLORS['portal'], markeredgecolor=COLORS['text'],
               label='Outer safe destination'),
        Line2D([0], [0], marker='o', linestyle='none', markersize=6,
               markerfacecolor=COLORS['end_stone'], markeredgecolor='none',
               label='Outer-island source site'),
    ]
    side.legend(
        handles=legend_handles, loc='upper left', bbox_to_anchor=(-0.02, 0.47),
        frameon=False, fontsize=8.1, labelcolor=COLORS['text'],
    )
    draw_structure_schematic(side, 'end_city', 0.10, 0.25, size=0.055, zorder=5)
    side.text(
        0.22, 0.25, f'End city\n{len(cities)} qualified starts shown',
        color=COLORS['text'], fontsize=8.0, va='center', linespacing=1.35,
    )
    side.text(
        0.0, 0.13,
        'City grid: 20 x 20 chunks\nCandidate window: 9 x 9 chunks\nSalt: 10387313\n\nOuter endpoints snap the 1,024-block\nideal vector to this 2D island model.',
        color=COLORS['muted'], fontsize=7.6, va='top', linespacing=1.45,
    )

    figure.suptitle(
        'END STRUCTURE GENERATION   JAVA 1.16.1',
        color=COLORS['text'], fontsize=18, fontweight='black', y=0.975,
    )
    figure.savefig(
        save_path, dpi=dpi, facecolor=COLORS['background'],
        edgecolor='none', bbox_inches='tight',
    )
    plt.close(figure)
    return str(save_path)


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)
    create_end_structure_generation(plots / 'end_structure_generation.png')


if __name__ == '__main__':
    main()
