"""Three-panel Java 1.16.1 End dimension overview."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle
import numpy as np

from core.end_generation import gateway_positions, sample_outer_island_sites, spike_layout
from core.style import COLORS, addSoftShadow, apply_style, style_axis


apply_style()


def _panel_label(ax, label):
    badge = ax.text(
        0.025, 0.965, label, transform=ax.transAxes,
        ha='left', va='top', color=COLORS['blue'],
        fontsize=9.0, fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.35', facecolor=COLORS['panel'],
            edgecolor=COLORS['grid'], alpha=0.96,
        ),
        zorder=20,
    )
    addSoftShadow(badge.get_bbox_patch(), offset=(1.0, -1.0), alpha=0.16)


def _draw_island_overview(ax, seed):
    sites = sample_outer_island_sites(seed, count=2800, max_radius_blocks=18000)
    xValues = np.array([site['block_x'] for site in sites])
    zValues = np.array([site['block_z'] for site in sites])
    elevation = np.array([site['elevation'] for site in sites])
    sizes = 2.2 + 0.20 * elevation ** 1.45
    scale = (elevation - elevation.min()) / max(float(np.ptp(elevation)), 1.0)
    rgba = np.tile(np.array(to_rgba(COLORS['end_stone'])), (len(xValues), 1))
    rgba[:, :3] *= (0.72 + 0.28 * scale)[:, None]
    rgba[:, 3] = 0.38 + 0.34 * scale
    ax.scatter(
        xValues, zValues, s=sizes, c=rgba,
        edgecolors='none', rasterized=True,
    )
    ax.add_patch(Circle(
        (0, 0), 1000, facecolor=COLORS['panel'],
        edgecolor=COLORS['grid'], linewidth=0.8, linestyle=':', zorder=4,
    ))
    ax.add_patch(Circle(
        (0, 0), 105, facecolor=COLORS['end_stone'],
        edgecolor=COLORS['text'], linewidth=0.7, alpha=0.94, zorder=5,
    ))
    ax.scatter(
        [0], [0], s=28, c=COLORS['portal'], marker='o',
        edgecolors=COLORS['panel'], linewidths=0.7, zorder=6,
    )
    for radius in (5000, 10000, 15000):
        ax.add_patch(Circle(
            (0, 0), radius, fill=False, edgecolor=COLORS['grid'],
            linewidth=0.6, alpha=0.70,
        ))
        ax.text(
            radius / np.sqrt(2), radius / np.sqrt(2), f'{radius // 1000}k',
            color=COLORS['muted'], fontsize=6.8, ha='center', va='center',
        )
    ax.set_xlim(-18000, 18000)
    ax.set_ylim(-18000, 18000)
    ax.set_xlabel('X (blocks)')
    ax.set_ylabel('Z (blocks)')
    ax.set_title('Outer-island field', loc='left', pad=12, fontsize=11)
    style_axis(ax, equal=True, grid=False)
    _panel_label(ax, 'A')


def _draw_central_geometry(ax, seed):
    ax.add_patch(Circle(
        (0, 0), 100, facecolor=COLORS['end_stone'],
        edgecolor=COLORS['grid'], linewidth=0.8, alpha=0.74,
    ))
    ax.add_patch(Circle(
        (0, 0), 7.5, facecolor=COLORS['obsidian'],
        edgecolor=COLORS['portal'], linewidth=1.2, zorder=8,
    ))
    ax.add_patch(Circle(
        (0, 0), 42, fill=False, edgecolor=COLORS['muted'],
        linewidth=0.85, linestyle=':', alpha=0.76,
    ))
    ax.add_patch(Circle(
        (0, 0), 96, fill=False, edgecolor=COLORS['violet'],
        linewidth=1.0, linestyle='--', alpha=0.78,
    ))

    for spike in spike_layout(seed):
        size = 40 + spike['radius'] * 15
        ax.scatter(
            [spike['x']], [spike['z']], s=size,
            c=COLORS['obsidian'], edgecolors=COLORS['panel'],
            linewidths=0.75, zorder=5,
        )
        ax.scatter(
            [spike['x']], [spike['z']], s=20,
            c=COLORS['green'], marker='D',
            edgecolors=COLORS['panel'], linewidths=0.5, zorder=6,
        )
        if spike['caged']:
            ax.scatter(
                [spike['x']], [spike['z']], s=size + 38,
                facecolors='none', edgecolors=COLORS['gold'],
                marker='s', linewidths=1.0, zorder=7,
            )

    gateways = gateway_positions()
    ax.scatter(
        [item['x'] for item in gateways],
        [item['z'] for item in gateways],
        s=28, c=COLORS['violet'], marker='s',
        edgecolors=COLORS['panel'], linewidths=0.65, zorder=7,
    )
    ax.set_xlim(-116, 116)
    ax.set_ylim(-116, 116)
    ax.set_xticks([-96, -42, 0, 42, 96])
    ax.set_yticks([-96, -42, 0, 42, 96])
    ax.set_xlabel('X (blocks)')
    ax.set_ylabel('Z (blocks)')
    ax.set_title('Central island geometry', loc='left', pad=12, fontsize=11)
    style_axis(ax, equal=True, grid=True)
    _panel_label(ax, 'B')


def _draw_gateway_positions(ax):
    gateways = gateway_positions()
    ax.add_patch(Circle(
        (0, 0), 96, fill=False, edgecolor=COLORS['violet'],
        linestyle='--', linewidth=1.2, alpha=0.86,
    ))
    for item in gateways:
        ax.scatter(
            [item['x']], [item['z']], s=52,
            c=COLORS['violet'], marker='s',
            edgecolors=COLORS['panel'], linewidths=0.85, zorder=6,
        )
        radius = max(np.hypot(item['x'], item['z']), 1.0)
        labelScale = 1.12
        ax.text(
            item['x'] * labelScale, item['z'] * labelScale,
            str(item['index']), color=COLORS['muted'], fontsize=6.8,
            ha='center', va='center',
        )
    ax.scatter(
        [0], [0], marker='+', s=76, c=COLORS['text'],
        linewidths=1.0, zorder=7,
    )
    ax.set_xlim(-124, 124)
    ax.set_ylim(-124, 124)
    ax.set_xlabel('X (blocks)')
    ax.set_ylabel('Z (blocks)')
    ax.set_title('End gateway positions', loc='left', pad=12, fontsize=11)
    style_axis(ax, equal=True, grid=True)
    _panel_label(ax, 'C')


def create_end_dimension_overview(save_path, dpi=220, seed=42):
    figure = plt.figure(figsize=(15.5, 7.4), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 3, width_ratios=[1.34, 1.0, 1.0],
        left=0.055, right=0.98, top=0.86, bottom=0.11, wspace=0.16,
    )
    overview = figure.add_subplot(grid[0, 0])
    geometry = figure.add_subplot(grid[0, 1])
    gateways = figure.add_subplot(grid[0, 2])
    _draw_island_overview(overview, seed)
    _draw_central_geometry(geometry, seed)
    _draw_gateway_positions(gateways)
    figure.suptitle(
        'The End: dimension overview', x=0.055, y=0.955,
        ha='left', va='top', fontsize=17, fontweight='bold',
        color=COLORS['text'],
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
    create_end_dimension_overview(plots / 'end_dimension_overview.png')


if __name__ == '__main__':
    main()
