"""Three-panel Java 1.16.1 End dimension structure visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import numpy as np

from core.end_generation import (
    end_overflow_generation_mask,
    gateway_positions,
    outer_island_seed_field,
)
from core.end_visuals import (
    ISLAND_CMAP,
    draw_central_island,
    draw_end_fountain,
    draw_end_spikes,
)
from core.style import COLORS, apply_style, style_axis


apply_style()


def _panel_label(ax, label):
    ax.text(
        0.018, 0.975, label, transform=ax.transAxes,
        ha='left', va='top', color=COLORS['text'],
        fontsize=11, fontweight='bold', zorder=20,
    )


def _draw_million_scale_overflow(ax):
    """Draw exact point samples of the modular radial overflow field."""
    limit = 6_000_000.0
    resolution = 1401
    coordinates = np.linspace(-limit, limit, resolution)
    x, z = np.meshgrid(coordinates, coordinates)
    generated = end_overflow_generation_mask(x, z)
    colormap = ListedColormap([
        '#05070C',
        '#BFC18A',
    ])
    ax.imshow(
        generated.astype(float),
        extent=(-limit, limit, -limit, limit),
        origin='lower', interpolation='nearest', cmap=colormap,
        vmin=0.0, vmax=1.0, zorder=1,
    )
    ticks = [-6_000_000, -3_000_000, 0, 3_000_000, 6_000_000]
    labels = [f'{value / 1_000_000:.0f}M' for value in ticks]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel('X (blocks)')
    ax.set_ylabel('Z (blocks)')
    ax.set_title('Signed-32 overflow lattice at map scale', fontsize=11, pad=8)
    style_axis(ax, equal=True, grid=False)
    _panel_label(ax, '(a)')


def _draw_central_geometry(ax, seed):
    draw_central_island(ax, seed=seed, extent=112, alpha=0.62, zorder=0)
    ax.add_patch(Circle(
        (0, 0), 42, fill=False, edgecolor=COLORS['muted'],
        linewidth=0.85, linestyle=':', alpha=0.78,
    ))
    ax.add_patch(Circle(
        (0, 0), 96, fill=False, edgecolor=COLORS['cyan'],
        linewidth=0.95, linestyle='--', alpha=0.72,
    ))
    draw_end_spikes(
        ax, seed=seed, crystals_alive=10,
        radius_override=5.0, zorder=5, tower_edgecolor='none',
        cage_linewidth=1.55, cage_extent=3.25,
    )
    draw_end_fountain(ax, active=True, zorder=8)
    ax.add_patch(Circle(
        (0, 0), 8.0, fill=False, edgecolor=COLORS['gold'],
        linewidth=1.35, alpha=0.95, zorder=10,
    ))
    ax.add_patch(Circle(
        (0, 0), 5.6, facecolor=COLORS['portal'], edgecolor='none',
        alpha=0.12, zorder=7,
    ))

    gateways = gateway_positions()
    ax.scatter(
        [item['x'] for item in gateways],
        [item['z'] for item in gateways],
        s=24, c=COLORS['cyan'], marker='s',
        edgecolors=COLORS['text'], linewidths=0.35, zorder=7,
    )
    for item in gateways[::2]:
        ax.text(
            item['x'] * 1.08, item['z'] * 1.08, str(item['index']),
            color=COLORS['muted'], fontsize=5.7,
            ha='center', va='center',
        )
    ax.set_xlim(-116, 116)
    ax.set_ylim(-116, 116)
    ax.set_xticks([-96, -42, 0, 42, 96])
    ax.set_yticks([-96, -42, 0, 42, 96])
    ax.set_xlabel('X (blocks)')
    ax.set_ylabel('Z (blocks)')
    ax.set_title('Central fight geometry', fontsize=11, pad=8)
    style_axis(ax, equal=True, grid=True)
    _panel_label(ax, '(b)')


def _draw_local_outer_islands(ax, seed):
    """Show the first outer-island source ring as separate seed sites."""
    limit = 3_600.0
    sites = outer_island_seed_field(seed, max_coordinate_blocks=int(limit))
    radii = np.hypot(sites['block_x'], sites['block_z'])
    first_ring = (radii > 1_024.0) & (radii <= limit)
    strength = 22.0 - sites['falloff'][first_ring]
    ax.scatter(
        sites['block_x'][first_ring], sites['block_z'][first_ring],
        s=2.5 + 1.15 * strength,
        c=strength, cmap=ISLAND_CMAP, vmin=1.0, vmax=13.0,
        marker='o', linewidths=0, alpha=0.78, zorder=2,
    )
    ax.add_patch(Circle(
        (0, 0), 1_024, fill=False, edgecolor=COLORS['end_stone'],
        linewidth=1.05, linestyle='--', alpha=0.88, zorder=5,
    ))
    draw_central_island(
        ax, seed=seed, extent=180, resolution=121,
        alpha=0.98, zorder=7,
    )
    ax.add_patch(Circle(
        (0, 0), 185, fill=False, edgecolor=COLORS['end_stone'],
        linewidth=0.85, alpha=0.72, zorder=8,
    ))
    ax.scatter(
        [0], [0], s=28, marker='D', c=COLORS['portal'],
        edgecolors=COLORS['text'], linewidths=0.55, zorder=9,
    )
    ax.text(
        0, -300, 'central island', color=COLORS['muted'],
        fontsize=6.8, ha='center', va='top', zorder=9,
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xticks([-3_000, 0, 3_000], ['-3k', '0', '3k'])
    ax.set_yticks([-3_000, 0, 3_000], ['-3k', '0', '3k'])
    ax.set_xlabel('X (blocks)')
    ax.set_ylabel('Z (blocks)')
    ax.set_title('First outer-island seed ring', fontsize=11, pad=8)
    style_axis(ax, equal=True, grid=False)
    _panel_label(ax, '(c)')


def create_end_dimension_overview(save_path, dpi=220, seed=42):
    figure = plt.figure(figsize=(15.5, 10.0), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        2, 2, width_ratios=[1.65, 1.0], height_ratios=[1.0, 0.92],
        left=0.055, right=0.98, top=0.97, bottom=0.07,
        wspace=0.16, hspace=0.18,
    )
    overflow = figure.add_subplot(grid[:, 0])
    geometry = figure.add_subplot(grid[0, 1])
    local_islands = figure.add_subplot(grid[1, 1])
    _draw_million_scale_overflow(overflow)
    _draw_central_geometry(geometry, seed)
    _draw_local_outer_islands(local_islands, seed)
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
    create_end_dimension_overview(plots / 'end_dimension_overview.png')


if __name__ == '__main__':
    main()
