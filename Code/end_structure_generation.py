"""End City candidates and fixed-seed generated-height visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import numpy as np
from PIL import Image

from core.end_generation import (
    end_city_height_candidates,
    outer_gateway_positions,
)
from core.style import COLORS, apply_style, style_axis


apply_style()


def _end_surface_rgba(height, coordinates):
    texture_path = (
        Path(__file__).resolve().parents[1] / 'Assets' / 'minecraft_1_16_1'
        / 'textures' / 'block' / 'end_stone.png'
    )
    texture = np.asarray(Image.open(texture_path).convert('RGB'), dtype=float) / 255.0
    block_x, block_z = np.meshgrid(coordinates, coordinates)
    columns = np.mod(block_x.astype(np.int64), 16)
    rows = np.mod(block_z.astype(np.int64), 16)
    output = np.zeros((*height.shape, 4), dtype=float)
    output[..., :3] = texture[rows, columns]
    values = height.filled(0.0).astype(float)
    gradient_z, gradient_x = np.gradient(values)
    shade = np.clip(0.93 + 0.018 * gradient_x - 0.022 * gradient_z, 0.72, 1.08)
    output[..., :3] *= shade[..., None]
    output[..., 3] = np.where(np.ma.getmaskarray(height), 0.0, 0.94)
    return np.clip(output, 0.0, 1.0)


def create_end_structure_generation(save_path, dpi=210, seed=42):
    """Render exact End base terrain and the End City four-height gate."""
    limit = 3600.0
    all_candidates, height_x, generated_height = end_city_height_candidates(
        seed, max_coordinate_blocks=int(limit), resolution=101,
    )
    cities = [item for item in all_candidates if item['qualified']]
    rejected = [item for item in all_candidates if not item['qualified']]
    outer_gateways = outer_gateway_positions(seed)

    figure = plt.figure(figsize=(16.6, 8.4), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 3, width_ratios=[1.0, 1.0, 0.035],
        left=0.045, right=0.965, top=0.82, bottom=0.075, wspace=0.075,
    )
    axis = figure.add_subplot(grid[0, 0])
    probability_axis = figure.add_subplot(grid[0, 1])
    colorbar_axis = figure.add_subplot(grid[0, 2])

    axis.imshow(
        _end_surface_rgba(generated_height, height_x),
        extent=(height_x[0], height_x[-1], height_x[0], height_x[-1]),
        origin='lower', interpolation='nearest', zorder=1,
    )
    axis.add_patch(Circle(
        (0, 0), 1024, fill=False, edgecolor=COLORS['end_stone'],
        linewidth=1.2, linestyle='--', alpha=0.82, zorder=3,
    ))
    axis.scatter(
        [item['x'] for item in outer_gateways],
        [item['z'] for item in outer_gateways],
        s=21, marker='D', c=COLORS['portal'], alpha=0.72,
        edgecolors=COLORS['text'], linewidths=0.35, zorder=8,
    )
    axis.scatter(
        [item['block_x'] for item in cities],
        [item['block_z'] for item in cities],
        s=22, marker='s', c=COLORS['purpur'],
        edgecolors=COLORS['text'], linewidths=0.36, alpha=0.94, zorder=7,
    )

    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xlabel('Block X')
    axis.set_ylabel('Block Z')
    axis.set_title(
        f'Generated End terrain and {len(cities)} qualified End City starts',
        fontsize=11.5, pad=45,
    )
    style_axis(axis, equal=True, grid=False)
    axis.legend(
        handles=[
            Line2D(
                [], [], marker='D', linestyle='none', markersize=6.5,
                markerfacecolor=COLORS['portal'], markeredgecolor=COLORS['text'],
                label='Modeled outer gateway endpoint',
            ),
            Line2D(
                [], [], marker='s', linestyle='none', markersize=7.0,
                markerfacecolor=COLORS['purpur'], markeredgecolor=COLORS['text'],
                label='Qualified End City start',
            ),
            Patch(
                facecolor=COLORS['end_stone'], edgecolor='#77745F',
                alpha=0.86, label='Generated End-stone surface',
            ),
        ],
        loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=3,
        borderaxespad=0.0, columnspacing=1.0, handletextpad=0.45,
        frameon=True, facecolor=COLORS['background'],
        edgecolor=COLORS['grid'], framealpha=0.96, fontsize=7.2,
    )

    height_image = probability_axis.imshow(
        generated_height,
        extent=(
            height_x[0], height_x[-1], height_x[0], height_x[-1],
        ),
        origin='lower', cmap='viridis',
        vmin=0.0, vmax=84.0,
        interpolation='nearest', zorder=1,
    )
    probability_axis.contour(
        height_x, height_x, generated_height.filled(0.0), levels=[60.0],
        colors=['#F4ECFF'], linewidths=0.72, alpha=0.68, zorder=2,
    )
    probability_axis.add_patch(Circle(
        (0, 0), 1024, fill=False, edgecolor=COLORS['end_stone'],
        linewidth=0.8, linestyle='--', alpha=0.78, zorder=3,
    ))
    probability_axis.scatter(
        [item['block_x'] for item in rejected],
        [item['block_z'] for item in rejected],
        s=12, marker='x', c='#7E8798', linewidths=0.48,
        alpha=0.58, zorder=3,
    )
    probability_axis.scatter(
        [item['block_x'] for item in cities],
        [item['block_z'] for item in cities],
        s=16, marker='s', c=COLORS['purpur'],
        edgecolors=COLORS['text'], linewidths=0.32, alpha=0.92, zorder=4,
    )
    probability_axis.set_xlim(-limit, limit)
    probability_axis.set_ylim(-limit, limit)
    probability_axis.set_xlabel('Block X')
    probability_axis.set_ylabel('Block Z')
    probability_axis.set_title(
        'Generated surface height and four-sample End City gate',
        fontsize=11.5, pad=45,
    )
    style_axis(probability_axis, equal=True, grid=False)
    probability_axis.tick_params(labelsize=8.0)
    probability_axis.legend(
        handles=[
            Line2D(
                [], [], color='#F4ECFF', linewidth=1.5,
                label='Generated height = 60 contour',
            ),
            Line2D(
                [], [], marker='s', linestyle='none', markersize=7.0,
                markerfacecolor=COLORS['purpur'], markeredgecolor=COLORS['text'],
                label='Qualified End City start',
            ),
            Line2D(
                [], [], marker='x', linestyle='none', markersize=6.5,
                color='#9AA3B3', label='Rejected End City start',
            ),
        ],
        loc='lower left', bbox_to_anchor=(0.0, 1.01), ncol=3,
        borderaxespad=0.0, columnspacing=1.0, handletextpad=0.45,
        frameon=True, facecolor=COLORS['background'],
        edgecolor=COLORS['grid'], framealpha=0.96, fontsize=7.2,
    )

    colorbar = figure.colorbar(
        height_image, cax=colorbar_axis, orientation='vertical',
    )
    colorbar.set_label(
        'WORLD_SURFACE_WG height', fontsize=8.2, labelpad=7,
    )
    colorbar.set_ticks((0, 20, 40, 60, 72, 84))
    colorbar.ax.tick_params(labelsize=7.4, pad=2)
    colorbar.outline.set_edgecolor(COLORS['grid'])
    figure.suptitle(
        'END STRUCTURE GENERATION',
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
