"""Three-panel Java 1.16.1 End dimension overview."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np

from core.end_generation import (
    gateway_positions, outer_island_projection, outer_island_seed_field,
)
from core.end_visuals import (
    ISLAND_CMAP, draw_central_island, draw_end_fountain, draw_end_spikes,
)
from core.style import COLORS, apply_style, style_axis


apply_style()


def _panel_label(ax, label):
    ax.text(
        0.018, 0.975, label, transform=ax.transAxes,
        ha='left', va='top', color=COLORS['text'],
        fontsize=11, fontweight='bold',
    )


def _draw_island_overview(ax, seed):
    sites = outer_island_seed_field(seed, max_coordinate_blocks=18000)
    x, z, projection = outer_island_projection(
        seed, max_coordinate_blocks=18000, resolution=901,
    )
    values = projection.filled(0.0)
    rgba = ISLAND_CMAP(values)
    visible = ~np.ma.getmaskarray(projection)
    rgba[..., 3] = np.where(visible, 0.18 + 0.50 * values, 0.0)
    ax.imshow(
        rgba, extent=(x[0], x[-1], z[0], z[-1]),
        origin='lower', interpolation='nearest', zorder=1,
    )
    texture = slice(None, None, 9)
    ax.scatter(
        sites['block_x'][texture], sites['block_z'][texture],
        s=1.2, c=COLORS['end_stone'], alpha=0.20,
        edgecolors='none', rasterized=True, zorder=2,
    )
    ax.add_patch(Circle(
        (0, 0), 900, facecolor=COLORS['background'],
        edgecolor='none', zorder=4,
    ))
    ax.add_patch(Circle(
        (0, 0), 1024, fill=False, edgecolor=COLORS['grid'],
        linewidth=0.8, linestyle=':', zorder=4,
    ))
    ax.add_patch(Circle(
        (0, 0), 105, facecolor=COLORS['end_stone'],
        edgecolor=COLORS['text'], linewidth=0.7, alpha=0.94, zorder=5,
    ))
    ax.scatter(
        [0], [0], s=26, c=COLORS['portal'], marker='o',
        edgecolors=COLORS['text'], linewidths=0.5, zorder=6,
    )
    for radius in (5000, 10000, 15000):
        ax.add_patch(Circle(
            (0, 0), radius, fill=False, edgecolor=COLORS['grid'],
            linewidth=0.5, alpha=0.45,
        ))
        ax.text(
            radius / np.sqrt(2), radius / np.sqrt(2), f'{radius // 1000}k',
            color=COLORS['muted'], fontsize=6.5, ha='center', va='center',
        )
    ax.set_xlim(-18000, 18000)
    ax.set_ylim(-18000, 18000)
    ax.set_xlabel('X (blocks)')
    ax.set_ylabel('Z (blocks)')
    style_axis(ax, equal=True, grid=False)
    _panel_label(ax, '(a)')


def _draw_central_geometry(ax, seed):
    draw_central_island(ax, seed=seed, extent=112, alpha=0.58, zorder=0)
    ax.add_patch(Circle(
        (0, 0), 42, fill=False, edgecolor=COLORS['muted'],
        linewidth=0.8, linestyle=':', alpha=0.75,
    ))
    ax.add_patch(Circle(
        (0, 0), 96, fill=False, edgecolor=COLORS['cyan'],
        linewidth=0.9, linestyle='--', alpha=0.68,
    ))
    draw_end_spikes(ax, seed=seed, crystals_alive=10, zorder=5)
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
            color=COLORS['muted'], fontsize=5.7, ha='center', va='center',
        )
    ax.text(30, 31, '42', color=COLORS['muted'], fontsize=6.5)
    ax.text(68, 70, '96', color=COLORS['cyan'], fontsize=6.5)
    ax.set_xlim(-116, 116)
    ax.set_ylim(-116, 116)
    ax.set_xticks([-96, -42, 0, 42, 96])
    ax.set_yticks([-96, -42, 0, 42, 96])
    ax.set_xlabel('X (blocks)')
    ax.set_ylabel('Z (blocks)')
    style_axis(ax, equal=True, grid=True)
    _panel_label(ax, '(b)')


def _iso_point(x, y, z):
    return x - 0.56 * z, y + 0.30 * z


def _iso_box(ax, x, y, z, width, height, depth, color, alpha=1.0):
    p000 = _iso_point(x, y, z)
    p100 = _iso_point(x + width, y, z)
    p110 = _iso_point(x + width, y + height, z)
    p010 = _iso_point(x, y + height, z)
    p001 = _iso_point(x, y, z + depth)
    p101 = _iso_point(x + width, y, z + depth)
    p111 = _iso_point(x + width, y + height, z + depth)
    p011 = _iso_point(x, y + height, z + depth)
    top = Polygon([p010, p110, p111, p011], closed=True,
                  facecolor=COLORS['end_stone'], edgecolor=COLORS['grid'],
                  linewidth=0.55, alpha=alpha)
    front = Polygon([p000, p100, p110, p010], closed=True,
                    facecolor=color, edgecolor=COLORS['grid'],
                    linewidth=0.55, alpha=alpha)
    side = Polygon([p100, p101, p111, p110], closed=True,
                   facecolor=COLORS['end_shadow'], edgecolor=COLORS['grid'],
                   linewidth=0.55, alpha=alpha)
    ax.add_patch(side)
    ax.add_patch(front)
    ax.add_patch(top)


def _draw_end_city(ax):
    ax.axis('off')
    ax.set_facecolor(COLORS['background'])

    _iso_box(ax, -2.6, 0.0, -1.6, 5.4, 1.0, 3.2, COLORS['purpur'])
    _iso_box(ax, -2.1, 1.0, -1.25, 4.4, 1.35, 2.5, COLORS['purpur'])
    _iso_box(ax, -1.55, 2.35, -0.95, 3.3, 1.35, 1.9, COLORS['purpur'])
    _iso_box(ax, -0.85, 3.70, -0.60, 1.9, 3.5, 1.2, COLORS['purpur'])
    _iso_box(ax, -1.15, 7.20, -0.85, 2.5, 0.55, 1.7, COLORS['purpur'])

    _iso_box(ax, 0.85, 5.30, -0.35, 4.5, 0.35, 0.70, COLORS['purpur'])
    _iso_box(ax, 4.80, 5.05, -0.75, 1.45, 1.00, 1.50, COLORS['purpur'])
    ship_body = Polygon([
        _iso_point(6.2, 5.0, -0.7), _iso_point(9.2, 5.0, -0.7),
        _iso_point(10.2, 5.55, 0.0), _iso_point(9.2, 6.0, 0.7),
        _iso_point(6.2, 6.0, 0.7), _iso_point(5.4, 5.55, 0.0),
    ], closed=True, facecolor=COLORS['purpur'],
       edgecolor=COLORS['text'], linewidth=0.7, alpha=0.95)
    ax.add_patch(ship_body)
    mast_x, mast_y = _iso_point(7.8, 6.0, 0.0)
    ax.plot([mast_x, mast_x], [mast_y, mast_y + 2.5],
            color=COLORS['end_stone'], linewidth=1.1)
    ax.scatter([mast_x], [mast_y + 2.55], s=24, marker='D',
               c=COLORS['portal'], edgecolors=COLORS['text'], linewidths=0.4)

    labels = [
        (-2.5, 0.0, 'base floors'),
        (-0.2, 5.0, 'tower chain'),
        (3.0, 5.75, 'bridge branch'),
        (7.7, 7.3, 'ship branch'),
    ]
    for x, y, label in labels:
        ax.text(x, y, label, color=COLORS['muted'], fontsize=7,
                ha='center', va='bottom')
    ax.set_xlim(-5.0, 11.2)
    ax.set_ylim(-0.5, 10.0)
    _panel_label(ax, '(c)')


def create_end_dimension_overview(save_path, dpi=220, seed=42):
    figure = plt.figure(figsize=(15.5, 10.0), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        2, 2, width_ratios=[1.65, 1.0], height_ratios=[1.0, 0.92],
        left=0.055, right=0.98, top=0.97, bottom=0.07,
        wspace=0.16, hspace=0.18,
    )
    overview = figure.add_subplot(grid[:, 0])
    geometry = figure.add_subplot(grid[0, 1])
    city = figure.add_subplot(grid[1, 1])
    _draw_island_overview(overview, seed)
    _draw_central_geometry(geometry, seed)
    _draw_end_city(city)
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
