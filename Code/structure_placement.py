"""Java 1.16.1 Overworld structure-candidate visualization."""

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
import numpy as np

from core.minecraft_visuals import (
    OVERWORLD_BIOMES,
    biome_texture_swatch,
    draw_minecraft_terrain,
    minecraft_biome_grid,
)
from core.rendering import optimize_gif
from core.structure_visuals import (
    STRUCTURE_SCHEMATICS,
    draw_structure_schematic,
)
from core.structures import (
    OVERWORLD_STRUCTURES,
    candidate_in_region,
    pillager_outpost_source_gate,
    structure_biome_compatible,
)
from core.style import COLORS, apply_style


apply_style()


def _biome_at_chunk(biomes, chunk_x, chunk_z, minimum, maximum):
    scale = (biomes.shape[0] - 1) / float(maximum - minimum)
    column = int(np.clip(round((chunk_x - minimum) * scale), 0, biomes.shape[1] - 1))
    row = int(np.clip(round((chunk_z - minimum) * scale), 0, biomes.shape[0] - 1))
    return str(biomes[row, column])


def _regions_covering(minimum, maximum, spacing):
    first = math.floor(minimum / spacing) - 1
    last = math.floor(maximum / spacing) + 1
    values = [
        (region_x, region_z)
        for region_x in range(first, last + 1)
        for region_z in range(first, last + 1)
    ]
    return sorted(values, key=lambda item: (
        max(abs(item[0]), abs(item[1])),
        math.atan2(item[1], item[0]),
    ))


def overworld_structure_candidates(
    seed=42, region_radius=14, resolution=640, max_per_structure=None,
):
    """Return exact candidate-stage starts from every displayed structure grid.

    ``region_radius`` defines the common map half-width in 32-chunk units. The
    individual grids keep their own spacing, separation, salt, and uniform or
    triangular offset rule. The terrain category is context only, not a
    vanilla biome gate. The direct outpost 1/5 roll and village exclusion are
    applied because they are independent source-level start checks.
    """
    half_width = int(region_radius) * 32 + 8
    minimum = -half_width
    maximum = half_width
    biomes = minecraft_biome_grid(
        seed, resolution=resolution,
        x_extent=(minimum, maximum), z_extent=(minimum, maximum),
        coordinate_scale=16.0, showcase=False,
    )
    accepted = []
    for config in OVERWORLD_STRUCTURES:
        compatible = []
        for region_x, region_z in _regions_covering(
            minimum, maximum, config.spacing,
        ):
            item = candidate_in_region(seed, region_x, region_z, config)
            if not (
                minimum <= item['chunk_x'] <= maximum
                and minimum <= item['chunk_z'] <= maximum
            ):
                continue
            biome = _biome_at_chunk(
                biomes, item['chunk_x'], item['chunk_z'], minimum, maximum,
            )
            if (
                config.name == 'pillager_outpost'
                and not pillager_outpost_source_gate(
                    seed, item['chunk_x'], item['chunk_z'],
                )
            ):
                continue
            item['biome'] = biome
            item['illustrative_biome_match'] = structure_biome_compatible(
                config.name, biome,
            )
            item['spacing'] = config.spacing
            item['separation'] = config.separation
            item['uniform'] = config.uniform
            compatible.append(item)
        compatible.sort(key=lambda item: (
            math.hypot(item['chunk_x'], item['chunk_z']),
            math.atan2(item['chunk_z'], item['chunk_x']),
        ))
        if max_per_structure is not None:
            compatible = compatible[:int(max_per_structure)]
        accepted.extend(compatible)
    accepted.sort(key=lambda item: (
        math.hypot(item['chunk_x'], item['chunk_z']), item['name'],
    ))
    return accepted, biomes, (minimum, maximum)


def _draw_legends(axis):
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis('off')
    axis.text(
        0.0, 0.985, 'TOP-DOWN STRUCTURE PLANS',
        color=COLORS['text'], fontsize=9.5, fontweight='black', va='top',
    )
    structure_names = list(STRUCTURE_SCHEMATICS)[:11]
    for index, name in enumerate(structure_names):
        column = index // 6
        row = index % 6
        x = 0.055 + 0.50 * column
        y = 0.925 - row * 0.075
        draw_structure_schematic(
            axis, name, x, y, size=0.028, zorder=5,
        )
        axis.text(
            x + 0.052, y, STRUCTURE_SCHEMATICS[name].label,
            color=COLORS['text'], fontsize=6.9, va='center', ha='left',
        )

    axis.text(
        0.0, 0.505, 'BIOME TEXTURES',
        color=COLORS['text'], fontsize=9.5, fontweight='black', va='top',
    )
    for index, (name, biome) in enumerate(OVERWORLD_BIOMES.items()):
        column = index // 8
        row = index % 8
        x = 0.0 + 0.50 * column
        y = 0.455 - row * 0.052
        axis.imshow(
            biome_texture_swatch(name, 18),
            extent=(x, x + 0.078, y - 0.018, y + 0.018),
            origin='lower', interpolation='nearest', aspect='auto', zorder=3,
        )
        axis.add_patch(Rectangle(
            (x, y - 0.018), 0.078, 0.036,
            facecolor='none', edgecolor=COLORS['muted'],
            linewidth=0.35, zorder=4,
        ))
        axis.text(
            x + 0.092, y, biome.label,
            color=COLORS['muted'], fontsize=6.35, va='center', ha='left',
        )
    axis.text(
        0.0, 0.018,
        'Terrain is source-informed context, not a biome gate.\n'
        'All in-bounds candidates are shown; outpost direct gates apply.',
        color=COLORS['muted'], fontsize=6.6, va='bottom', linespacing=1.35,
    )


def create_structure_placement_animation(
    save_path, seed=42, region_radius=14, fps=8, duration=12,
):
    candidates, _, (minimum, maximum) = overworld_structure_candidates(
        seed=seed, region_radius=region_radius,
    )
    total_frames = int(fps * duration)

    figure = plt.figure(figsize=(16.0, 8.8), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[3.70, 1.34],
        left=0.048, right=0.98, top=0.90, bottom=0.12, wspace=0.08,
    )
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[0, 1])
    axis.set_xlim(minimum, maximum)
    axis.set_ylim(minimum, maximum)
    axis.set_aspect('equal')
    axis.set_xlabel('Chunk X')
    axis.set_ylabel('Chunk Z')
    axis.tick_params(colors=COLORS['muted'], labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(COLORS['grid'])

    draw_minecraft_terrain(
        axis, (minimum, maximum, minimum, maximum), seed=seed,
        dimension='overworld', resolution=640, alpha=0.82,
        coordinate_scale=16.0, showcase=False,
    )
    for coordinate in range(
        math.floor(minimum / 32), math.ceil(maximum / 32) + 1,
    ):
        value = coordinate * 32
        axis.axvline(value, color=COLORS['text'], linewidth=0.42, alpha=0.18, zorder=2)
        axis.axhline(value, color=COLORS['text'], linewidth=0.42, alpha=0.18, zorder=2)
    axis.axhline(0, color=COLORS['muted'], linewidth=0.65, alpha=0.55)
    axis.axvline(0, color=COLORS['muted'], linewidth=0.65, alpha=0.55)
    axis.scatter([0], [0], marker='+', s=80, c=COLORS['text'], linewidths=1.0, zorder=9)

    marker_by_name = {
        'village': 'o', 'desert_pyramid': '^', 'jungle_pyramid': 'v',
        'swamp_hut': 's', 'pillager_outpost': 'P', 'igloo': 'h',
        'woodland_mansion': 'D', 'ocean_monument': 'X',
        'shipwreck': '>', 'ocean_ruin': '<', 'ruined_portal': '*',
    }
    candidate_collections = {}
    for config in OVERWORLD_STRUCTURES:
        style = STRUCTURE_SCHEMATICS[config.name]
        candidate_collections[config.name] = axis.scatter(
            [], [], s=15, marker=marker_by_name[config.name],
            c=style.primary, edgecolors=COLORS['text'], linewidths=0.22,
            alpha=0.88, zorder=6,
        )

    current_region = Rectangle(
        (0, 0), 1, 1, fill=False,
        edgecolor=COLORS['cyan'], linewidth=1.6, alpha=0.0, zorder=11,
    )
    current_window = Rectangle(
        (0, 0), 1, 1, facecolor=COLORS['blue'],
        edgecolor=COLORS['cyan'], linewidth=1.1, linestyle='--',
        alpha=0.0, zorder=3,
    )
    active_outline = Circle(
        (0, 0), 8.5, fill=False, edgecolor=COLORS['text'],
        linewidth=1.2, alpha=0.0, zorder=12,
    )
    axis.add_patch(current_window)
    axis.add_patch(current_region)
    axis.add_patch(active_outline)
    detail = axis.inset_axes([0.715, 0.685, 0.27, 0.29])
    draw_minecraft_terrain(
        detail, (minimum, maximum, minimum, maximum), seed=seed,
        dimension='overworld', resolution=640, alpha=0.96,
        coordinate_scale=16.0, showcase=False,
    )
    detail.set_facecolor(COLORS['panel'])
    detail.tick_params(colors=COLORS['muted'], labelsize=5.8, pad=1)
    for spine in detail.spines.values():
        spine.set_color(COLORS['text'])
        spine.set_linewidth(0.8)
    detail_region = Rectangle(
        (0, 0), 1, 1, fill=False, edgecolor=COLORS['cyan'],
        linewidth=1.4, zorder=8,
    )
    detail_window = Rectangle(
        (0, 0), 1, 1, facecolor=COLORS['blue'],
        edgecolor=COLORS['cyan'], linewidth=0.9, linestyle='--',
        alpha=0.18, zorder=7,
    )
    detail_candidate = detail.scatter(
        [], [], s=48, c=COLORS['gold'], marker='o',
        edgecolors=COLORS['text'], linewidths=0.55, zorder=10,
    )
    detail.add_patch(detail_window)
    detail.add_patch(detail_region)
    detail_window.set_visible(False)
    detail_region.set_visible(False)
    detail_candidate.set_visible(False)
    detail.set_xlim(-96, 96)
    detail.set_ylim(-96, 96)
    detail_collections = {}
    for config in OVERWORLD_STRUCTURES:
        style = STRUCTURE_SCHEMATICS[config.name]
        detail_collections[config.name] = detail.scatter(
            [], [], s=12, marker=marker_by_name[config.name],
            c=style.primary, edgecolors=COLORS['text'], linewidths=0.18,
            alpha=0.9, zorder=9,
        )
    detail.set_title('CENTRAL 192 x 192 CHUNK DETAIL', fontsize=6.8, pad=3)
    _draw_legends(legend_axis)
    figure.suptitle(
        'OVERWORLD STRUCTURE CANDIDATE PLACEMENT',
        color=COLORS['text'], fontsize=18, fontweight='black', y=0.965,
    )

    def update(frame_index):
        progress = frame_index / max(total_frames - 1, 1)
        shown = min(len(candidates), max(1, round(progress * len(candidates))))
        visible = candidates[:shown]
        for config in OVERWORLD_STRUCTURES:
            offsets = np.asarray([
                (entry['chunk_x'], entry['chunk_z'])
                for entry in visible if entry['name'] == config.name
            ], dtype=float)
            if offsets.size == 0:
                offsets = np.empty((0, 2))
            candidate_collections[config.name].set_offsets(offsets)
            detail_collections[config.name].set_offsets(offsets)

        item = candidates[shown - 1]
        spacing = item['spacing']
        window = item['window']
        region_origin = (
            item['region_x'] * spacing,
            item['region_z'] * spacing,
        )
        current_region.set_xy(region_origin)
        current_region.set_width(spacing)
        current_region.set_height(spacing)
        current_region.set_alpha(0.92)
        current_window.set_xy(region_origin)
        current_window.set_width(window)
        current_window.set_height(window)
        current_window.set_alpha(0.14)
        active_outline.center = (item['chunk_x'], item['chunk_z'])
        active_outline.set_alpha(0.92)
        return []

    animation = FuncAnimation(
        figure, update, frames=total_frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=68)
    plt.close(figure)
    optimize_gif(save_path, colors=32)
    return str(save_path)


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)
    create_structure_placement_animation(plots / 'structure_placement.gif')


if __name__ == '__main__':
    main()
