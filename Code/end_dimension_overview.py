"""Three-panel Java 1.16.1 End dimension overview."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Rectangle
import numpy as np
from scipy.ndimage import maximum_filter

from core.end_generation import (
    SimplexNoise2D,
    end_overflow_generation_mask,
    end_overflow_ring_boundaries,
    gateway_positions,
)
from core.end_visuals import (
    draw_central_island, draw_end_fountain, draw_end_spikes,
)
from core.style import COLORS, apply_style, style_axis


apply_style()


def _panel_label(ax, label):
    ax.text(
        0.018, 0.975, label, transform=ax.transAxes,
        ha='left', va='top', color=COLORS['text'],
        fontsize=11, fontweight='bold',
    )


def _overflow_texture(seed, x, z, generated, dilation=2):
    noise = SimplexNoise2D(seed).sample_grid(x / 16.0, z / 16.0)
    radius = np.hypot(x, z)
    sites = (noise < -0.9) & (radius > 1024.0) & generated
    islands = maximum_filter(sites.astype(float), size=int(dilation))
    color = np.asarray(to_rgba(COLORS['end_stone']))
    rgba = np.zeros((*generated.shape, 4), dtype=float)
    shade = 0.68 + 0.28 * np.clip((-noise - 0.35) / 0.65, 0.0, 1.0)
    rgba[..., :3] = color[:3] * shade[..., None]
    rgba[..., 3] = np.where(generated, 0.055 + 0.62 * islands, 0.0)
    return rgba


def _draw_island_overview(ax, seed):
    limit = 1_100_000.0
    coordinates = np.linspace(-limit, limit, 901)
    x, z = np.meshgrid(coordinates, coordinates)
    generated = end_overflow_generation_mask(x, z)
    rgba = _overflow_texture(seed, x, z, generated, dilation=3)
    ax.imshow(
        rgba, extent=(-limit, limit, -limit, limit),
        origin='lower', interpolation='nearest', zorder=1,
    )
    boundaries = end_overflow_ring_boundaries(limit)
    for item in boundaries:
        color = COLORS['coral'] if item['kind'] == 'void' else COLORS['cyan']
        ax.add_patch(Circle(
            (0, 0), item['radius'], fill=False, edgecolor=color,
            linewidth=0.72, alpha=0.64,
        ))
    for item in boundaries[:4]:
        angle = np.deg2rad(33.0)
        radius = item['radius']
        ax.text(
            radius * np.cos(angle), radius * np.sin(angle),
            f"{radius // 1000}k",
            color=COLORS['coral'] if item['kind'] == 'void' else COLORS['cyan'],
            fontsize=6.8, fontweight='bold', ha='center', va='center',
        )
    ax.scatter(
        [0], [0], s=30, c=COLORS['portal'], marker='D',
        edgecolors=COLORS['text'], linewidths=0.5, zorder=6,
    )
    zoom_bounds = (320_000, 570_000, -150_000, 150_000)
    ax.add_patch(Rectangle(
        (zoom_bounds[0], zoom_bounds[2]),
        zoom_bounds[1] - zoom_bounds[0],
        zoom_bounds[3] - zoom_bounds[2],
        fill=False, edgecolor=COLORS['gold'], linewidth=1.0,
        linestyle=(0, (4, 2)), zorder=7,
    ))
    ax.text(
        0.025, 0.035,
        'End-stone texture shows terrain availability\nindividual islands are below pixel scale',
        transform=ax.transAxes, color=COLORS['muted'], fontsize=7.2,
        ha='left', va='bottom',
        bbox=dict(
            boxstyle='round,pad=0.35', facecolor=COLORS['panel'],
            edgecolor=COLORS['grid'], alpha=0.88,
        ),
    )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ticks = [-1_000_000, -500_000, 0, 500_000, 1_000_000]
    ax.set_xticks(ticks, [f'{value / 1_000_000:.1f}M' for value in ticks])
    ax.set_yticks(ticks, [f'{value / 1_000_000:.1f}M' for value in ticks])
    ax.set_xlabel('X (blocks)')
    ax.set_ylabel('Z (blocks)')
    style_axis(ax, equal=True, grid=False)
    ax.set_title('Long-distance End terrain availability', fontsize=11, pad=8)
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
    ax.set_title('Central fight geometry', fontsize=11, pad=8)
    _panel_label(ax, '(b)')


def _draw_overflow_zoom(ax, seed):
    x_limits = (320_000.0, 570_000.0)
    z_limits = (-150_000.0, 150_000.0)
    x_coordinates = np.linspace(*x_limits, 801)
    z_coordinates = np.linspace(*z_limits, 721)
    x, z = np.meshgrid(x_coordinates, z_coordinates)
    generated = end_overflow_generation_mask(x, z)
    rgba = _overflow_texture(seed, x, z, generated, dilation=2)
    ax.imshow(
        rgba, extent=(*x_limits, *z_limits), origin='lower',
        interpolation='nearest', zorder=1,
    )
    boundaries = end_overflow_ring_boundaries(570_000)
    for item in boundaries:
        color = COLORS['coral'] if item['kind'] == 'void' else COLORS['cyan']
        ax.add_patch(Circle(
            (0, 0), item['radius'], fill=False, edgecolor=color,
            linewidth=1.05, alpha=0.86, zorder=3,
        ))
    ax.text(
        348_000, 118_000, 'terrain', color=COLORS['end_stone'],
        fontsize=9, fontweight='bold', ha='center',
    )
    ax.text(
        444_000, 0, '32-bit overflow void', color=COLORS['coral'],
        fontsize=9.5, fontweight='bold', ha='center', va='center',
        bbox=dict(
            boxstyle='round,pad=0.28', facecolor=COLORS['panel'],
            edgecolor=COLORS['coral'], alpha=0.90,
        ),
    )
    ax.text(
        548_000, 118_000, 'terrain resumes', color=COLORS['cyan'],
        fontsize=9, fontweight='bold', ha='center',
    )
    ax.set_xlim(*x_limits)
    ax.set_ylim(*z_limits)
    ax.set_xticks(
        [320_000, 370_720, 450_000, 524_288, 570_000],
        ['320k', '370,720', '450k', '524,288', '570k'],
    )
    ax.set_yticks([-150_000, 0, 150_000], ['-150k', '0', '150k'])
    ax.set_xlabel('X (blocks)')
    ax.set_ylabel('Z (blocks)')
    style_axis(ax, equal=True, grid=False)
    ax.set_title('First overflow ring detail', fontsize=11, pad=8)
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
    overflow_zoom = figure.add_subplot(grid[1, 1])
    _draw_island_overview(overview, seed)
    _draw_central_geometry(geometry, seed)
    _draw_overflow_zoom(overflow_zoom, seed)
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
