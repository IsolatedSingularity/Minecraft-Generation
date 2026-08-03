"""Three-panel Java 1.16.1 End dimension overview."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import numpy as np

from core.end_generation import (
    end_overflow_generation_mask,
    end_overflow_ring_boundaries,
    gateway_positions,
    outer_island_projection,
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
    boundary_count = len(end_overflow_ring_boundaries(limit))
    ax.text(
        0.025, 0.045,
        f'{boundary_count} land/void transitions by 6.0M blocks\n'
        'regular map sampling exposes the lattice-like moire',
        transform=ax.transAxes, ha='left', va='bottom',
        color=COLORS['text'], fontsize=8.0,
        bbox=dict(
            boxstyle='round,pad=0.34', facecolor=COLORS['panel'],
            edgecolor=COLORS['grid'], alpha=0.90,
        ), zorder=8,
    )
    ax.text(
        0.975, 0.045,
        'first affected cell 370,720\nterrain resumes 524,288',
        transform=ax.transAxes, ha='right', va='bottom',
        color=COLORS['muted'], fontsize=7.8, family='monospace',
        bbox=dict(
            boxstyle='round,pad=0.34', facecolor=COLORS['panel'],
            edgecolor=COLORS['grid'], alpha=0.90,
        ), zorder=8,
    )
    ticks = [-6_000_000, -3_000_000, 0, 3_000_000, 6_000_000]
    labels = [f'{value / 1_000_000:.0f}M' for value in ticks]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel('X (blocks)')
    ax.set_ylabel('Z (blocks)')
    ax.set_title('Million-scale modular End ring field', fontsize=11, pad=8)
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
        radius_override=5.0, zorder=5,
    )
    draw_end_fountain(ax, active=True, zorder=8)

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
    ax.text(
        0.97, 0.04, 'all spike footprints shown at max radius 5',
        transform=ax.transAxes, ha='right', va='bottom',
        color=COLORS['muted'], fontsize=7.2,
        bbox=dict(
            boxstyle='round,pad=0.28', facecolor=COLORS['panel'],
            edgecolor=COLORS['grid'], alpha=0.88,
        ), zorder=12,
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
    """Show the complete local outer-island source projection."""
    limit = 18_000.0
    x, z, projection = outer_island_projection(
        seed, max_coordinate_blocks=int(limit), resolution=801,
    )
    values = projection.filled(0.0)
    rgba = ISLAND_CMAP(values)
    visible = ~np.ma.getmaskarray(projection)
    rgba[..., 3] = np.where(visible, 0.24 + 0.72 * values, 0.0)
    ax.imshow(
        rgba, extent=(x[0], x[-1], z[0], z[-1]),
        origin='lower', interpolation='nearest', zorder=1,
    )
    for radius, label in (
        (1_024, '1,024'),
        (5_000, '5k'),
        (10_000, '10k'),
        (15_000, '15k'),
    ):
        emphasized = radius == 1_024
        ax.add_patch(Circle(
            (0, 0), radius, fill=False,
            edgecolor=COLORS['end_stone'] if emphasized else COLORS['muted'],
            linewidth=1.05 if emphasized else 0.52,
            linestyle='--' if emphasized else ':',
            alpha=0.86 if emphasized else 0.42,
            zorder=5,
        ))
        angle = np.deg2rad(35.0)
        ax.text(
            radius * np.cos(angle), radius * np.sin(angle), label,
            color=COLORS['end_stone'] if emphasized else COLORS['muted'],
            fontsize=6.6, fontweight='bold' if emphasized else 'normal',
            ha='center', va='center', zorder=6,
        )
    draw_central_island(
        ax, seed=seed, extent=150, resolution=101,
        alpha=0.98, zorder=7,
    )
    ax.scatter(
        [0], [0], s=18, marker='D', c=COLORS['portal'],
        edgecolors=COLORS['text'], linewidths=0.4, zorder=9,
    )
    ax.text(
        0.025, 0.045,
        'complete simplex-qualified source field\n'
        'dashed ring marks the central 1,024-block gulf',
        transform=ax.transAxes, ha='left', va='bottom',
        color=COLORS['muted'], fontsize=7.4,
        bbox=dict(
            boxstyle='round,pad=0.30', facecolor=COLORS['panel'],
            edgecolor=COLORS['grid'], alpha=0.88,
        ), zorder=10,
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xticks([-15_000, 0, 15_000], ['-15k', '0', '15k'])
    ax.set_yticks([-15_000, 0, 15_000], ['-15k', '0', '15k'])
    ax.set_xlabel('X (blocks)')
    ax.set_ylabel('Z (blocks)')
    ax.set_title('Local outer-island field and central gulf', fontsize=11, pad=8)
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
