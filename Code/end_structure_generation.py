"""End-city candidates and paired End-gateway visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import numpy as np

from core.end_generation import (
    end_city_candidates,
    end_city_qualification_probability,
    gateway_positions,
    outer_gateway_positions,
    outer_island_projection,
)
from core.end_visuals import ISLAND_CMAP, draw_central_island
from core.structure_visuals import draw_structure_schematic
from core.style import COLORS, apply_style, style_axis


apply_style()


CITY_PROBABILITY_CMAP = LinearSegmentedColormap.from_list(
    'end_city_qualification',
    [COLORS['background'], '#31234A', '#7650A5', '#D978B2', '#F4D46A'],
)


def create_end_structure_generation(save_path, dpi=210, seed=42):
    """Render End-city placement and both gateway endpoints."""
    limit = 3600.0
    island_x, island_z, island_projection = outer_island_projection(
        seed, max_coordinate_blocks=int(limit), resolution=901,
    )
    cities = end_city_candidates(seed, max_coordinate_blocks=int(limit))
    probability_x, probability_z, qualification_probability = (
        end_city_qualification_probability(
            seed, max_coordinate_blocks=int(limit),
        )
    )
    outer_gateways = outer_gateway_positions(seed)
    central_gateways = gateway_positions()

    figure = plt.figure(figsize=(16.0, 9.0), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[3.55, 1.30],
        left=0.055, right=0.985, top=0.91, bottom=0.08, wspace=0.025,
    )
    axis = figure.add_subplot(grid[0, 0])
    side = figure.add_subplot(grid[0, 1])

    island_values = island_projection.filled(0.0)
    island_rgba = ISLAND_CMAP(island_values)
    island_visible = ~np.ma.getmaskarray(island_projection)
    island_rgba[..., 3] = np.where(
        island_visible, 0.50 + 0.46 * island_values, 0.0,
    )
    axis.imshow(
        island_rgba,
        extent=(island_x[0], island_x[-1], island_z[0], island_z[-1]),
        origin='lower', interpolation='nearest', zorder=1,
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
            size=76.0, zorder=7,
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
    inset = side.inset_axes([0.04, 0.60, 0.92, 0.34])
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
        0.0, 0.535, 'FIXED-SEED CITY QUALIFICATION PRIOR',
        color=COLORS['text'], fontsize=10.0, fontweight='black', va='top',
    )
    probability_axis = side.inset_axes([0.04, 0.115, 0.92, 0.375])
    probability_image = probability_axis.imshow(
        qualification_probability * 100.0,
        extent=(
            probability_x[0], probability_x[-1],
            probability_z[0], probability_z[-1],
        ),
        origin='lower', cmap=CITY_PROBABILITY_CMAP,
        vmin=0.0, vmax=100.0 / 81.0,
        interpolation='bilinear', zorder=1,
    )
    probability_axis.add_patch(Circle(
        (0, 0), 1024, fill=False, edgecolor=COLORS['end_stone'],
        linewidth=0.8, linestyle='--', alpha=0.78, zorder=3,
    ))
    for city in cities:
        draw_structure_schematic(
            probability_axis, 'end_city', city['block_x'], city['block_z'],
            size=62.0, zorder=4, alpha=0.82,
        )
    probability_axis.set_xlim(-limit, limit)
    probability_axis.set_ylim(-limit, limit)
    probability_axis.set_xticks((-3000, 0, 3000))
    probability_axis.set_yticks((-3000, 0, 3000))
    probability_axis.set_title(
        '1/81 candidate prior masked by modeled island support',
        fontsize=7.2, pad=4,
    )
    style_axis(probability_axis, equal=True, grid=False)
    probability_axis.tick_params(labelsize=5.8, pad=1.5)

    colorbar_axis = side.inset_axes([0.14, 0.060, 0.72, 0.020])
    colorbar = figure.colorbar(
        probability_image, cax=colorbar_axis, orientation='horizontal',
    )
    colorbar.ax.set_title(
        'Modeled qualification probability (%)', fontsize=6.3, pad=3,
    )
    colorbar.ax.tick_params(labelsize=5.7, pad=1.5)
    colorbar.outline.set_edgecolor(COLORS['grid'])
    side.text(
        0.50, 0.010,
        f'{len(cities)} qualified starts | ship glyph is a symbolic End-city marker',
        color=COLORS['muted'], fontsize=6.5, ha='center', va='bottom',
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
