"""End-city candidates and fixed-seed qualification-prior visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

from core.end_generation import (
    end_city_height_candidates,
    outer_gateway_positions,
    outer_island_projection,
)
from core.end_visuals import ISLAND_CMAP, draw_central_island
from core.structure_visuals import draw_structure_schematic
from core.style import COLORS, apply_style, style_axis


apply_style()


def create_end_structure_generation(save_path, dpi=210, seed=42):
    """Render equal-size End-city support and modeled height-gate panels."""
    limit = 3600.0
    island_x, island_z, island_projection = outer_island_projection(
        seed, max_coordinate_blocks=int(limit), resolution=901,
    )
    all_candidates, height_x, modeled_height = end_city_height_candidates(
        seed, max_coordinate_blocks=int(limit), resolution=901,
    )
    cities = [item for item in all_candidates if item['qualified']]
    rejected = [item for item in all_candidates if not item['qualified']]
    outer_gateways = outer_gateway_positions(seed)

    figure = plt.figure(figsize=(16.6, 8.4), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 3, width_ratios=[1.0, 1.0, 0.035],
        left=0.045, right=0.965, top=0.90, bottom=0.105, wspace=0.075,
    )
    axis = figure.add_subplot(grid[0, 0])
    probability_axis = figure.add_subplot(grid[0, 1])
    colorbar_axis = figure.add_subplot(grid[0, 2])

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

    axis.scatter(
        [item['x'] for item in outer_gateways],
        [item['z'] for item in outer_gateways],
        s=21, marker='D', c=COLORS['portal'], alpha=0.72,
        edgecolors=COLORS['text'], linewidths=0.35, zorder=8,
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
    axis.set_title(
        f'Outer-island support and {len(cities)} qualified model starts',
        fontsize=11.5, pad=9,
    )
    style_axis(axis, equal=True, grid=False)
    axis.legend(
        handles=[
            Line2D(
                [], [], marker='D', linestyle='none', markersize=6.5,
                markerfacecolor=COLORS['portal'], markeredgecolor=COLORS['text'],
                label='Modeled outer-gateway destination',
            ),
            Line2D(
                [], [], marker='P', linestyle='none', markersize=7.5,
                markerfacecolor=COLORS['purpur'], markeredgecolor=COLORS['text'],
                label='Qualified End-city model start',
            ),
        ],
        loc='upper left', frameon=True, facecolor=COLORS['background'],
        edgecolor=COLORS['grid'], framealpha=0.94, fontsize=8.2,
    )

    height_image = probability_axis.imshow(
        modeled_height,
        extent=(
            height_x[0], height_x[-1], height_x[0], height_x[-1],
        ),
        origin='lower', cmap='viridis',
        vmin=42.0, vmax=84.0,
        interpolation='nearest', zorder=1,
    )
    probability_axis.contour(
        height_x, height_x, modeled_height.filled(0.0), levels=[60.0],
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
    for city in cities:
        draw_structure_schematic(
            probability_axis, 'end_city', city['block_x'], city['block_z'],
            size=44.0, zorder=4, alpha=0.94,
        )
    probability_axis.set_xlim(-limit, limit)
    probability_axis.set_ylim(-limit, limit)
    probability_axis.set_xlabel('Block X')
    probability_axis.set_ylabel('Block Z')
    probability_axis.set_title(
        'Modeled surface height and exact four-sample city gate',
        fontsize=11.5, pad=9,
    )
    style_axis(probability_axis, equal=True, grid=False)
    probability_axis.tick_params(labelsize=8.0)
    probability_axis.legend(
        handles=[
            Line2D(
                [], [], color='#F4ECFF', linewidth=1.5,
                label='Modeled height = 60 contour',
            ),
            Line2D(
                [], [], marker='P', linestyle='none', markersize=7.5,
                markerfacecolor=COLORS['purpur'], markeredgecolor=COLORS['text'],
                label='Four-sample minimum passes',
            ),
            Line2D(
                [], [], marker='x', linestyle='none', markersize=6.5,
                color='#9AA3B3', label='Four-sample minimum fails',
            ),
        ],
        loc='upper left', frameon=True, facecolor=COLORS['background'],
        edgecolor=COLORS['grid'], framealpha=0.94, fontsize=8.2,
    )

    colorbar = figure.colorbar(
        height_image, cax=colorbar_axis, orientation='vertical',
    )
    colorbar.set_label(
        'Modeled WORLD_SURFACE_WG height', fontsize=8.2, labelpad=7,
    )
    colorbar.set_ticks((42, 48, 54, 60, 66, 72, 78, 84))
    colorbar.ax.tick_params(labelsize=7.4, pad=2)
    colorbar.outline.set_edgecolor(COLORS['grid'])
    figure.text(
        0.50, 0.042,
        f'Fixed world seed 42 | exact 20-chunk candidate grid, Java rotation, and min-of-four >= 60 gate | '
        f'{len(cities)}/{len(all_candidates)} starts qualify in the documented 2D height proxy',
        color=COLORS['muted'], fontsize=7.5, ha='center', va='center',
    )

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
