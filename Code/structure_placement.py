"""Java 1.16.1 Overworld structure-candidate visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

from core.constants import VILLAGE_SPACING
from core.minecraft_visuals import (
    OVERWORLD_BLOCKS,
    draw_minecraft_terrain,
    minecraft_biome_grid,
)
from core.rendering import optimize_gif
from core.structures import (
    OVERWORLD_STRUCTURES,
    candidate_in_region,
    structure_biome_compatible,
)
from core.style import COLORS, apply_style


apply_style()


STRUCTURE_STYLES = {
    'village': {'label': 'Village', 'marker': 's', 'color': '#FFD166'},
    'desert_pyramid': {
        'label': 'Desert pyramid', 'marker': '^', 'color': '#F59E42',
    },
    'jungle_pyramid': {
        'label': 'Jungle pyramid', 'marker': 'D', 'color': '#8FD14F',
    },
    'swamp_hut': {'label': 'Swamp hut', 'marker': 'h', 'color': '#58C6A9'},
    'pillager_outpost': {
        'label': 'Pillager outpost', 'marker': 'P', 'color': '#C084FC',
    },
}

BIOME_LABELS = {
    'deep_water': 'Deep ocean',
    'water': 'Ocean',
    'shore': 'Beach',
    'plains': 'Plains',
    'forest': 'Forest',
    'dark_forest': 'Dark forest',
    'desert': 'Desert',
    'savanna': 'Savanna',
    'jungle': 'Jungle',
    'swamp': 'Swamp',
    'taiga': 'Taiga',
    'snowy_tundra': 'Snowy tundra',
    'mountains': 'Mountains',
    'badlands': 'Badlands',
    'mushroom_fields': 'Mushroom fields',
}


def spiral_regions(radius):
    regions = [
        (x, z) for x in range(-radius, radius + 1)
        for z in range(-radius, radius + 1)
    ]
    return sorted(regions, key=lambda item: (
        max(abs(item[0]), abs(item[1])),
        np.arctan2(item[1], item[0]),
    ))


def _biome_at_chunk(biomes, chunk_x, chunk_z, minimum, maximum):
    scale = (biomes.shape[0] - 1) / float(maximum - minimum)
    column = int(np.clip(round((chunk_x - minimum) * scale), 0, biomes.shape[1] - 1))
    row = int(np.clip(round((chunk_z - minimum) * scale), 0, biomes.shape[0] - 1))
    return str(biomes[row, column])


def overworld_structure_candidates(seed=42, region_radius=4, resolution=384):
    """Return exact grid candidates passing the illustrative biome-category gate."""
    spacing = VILLAGE_SPACING
    minimum = -region_radius * spacing - 5
    maximum = (region_radius + 1) * spacing + 5
    biomes = minecraft_biome_grid(seed, resolution=resolution)
    accepted = []
    for region_x, region_z in spiral_regions(region_radius):
        for config in OVERWORLD_STRUCTURES:
            item = candidate_in_region(seed, region_x, region_z, config)
            biome = _biome_at_chunk(
                biomes, item['chunk_x'], item['chunk_z'], minimum, maximum,
            )
            if structure_biome_compatible(config.name, biome):
                item['biome'] = biome
                accepted.append(item)
    return accepted, biomes, (minimum, maximum)


def _add_legends(axis):
    axis.set_axis_off()
    structure_handles = [
        Line2D(
            [0], [0], marker=style['marker'], linestyle='none',
            markerfacecolor=style['color'], markeredgecolor=COLORS['text'],
            markeredgewidth=0.55, markersize=8.5, label=style['label'],
        )
        for style in STRUCTURE_STYLES.values()
    ]
    structure_legend = axis.legend(
        handles=structure_handles, title='Biome-compatible candidates',
        loc='upper left', frameon=False, fontsize=8.6, title_fontsize=9.4,
        labelcolor=COLORS['text'], borderaxespad=0.0, handletextpad=0.75,
    )
    axis.add_artist(structure_legend)

    biome_handles = [
        Patch(facecolor=color, edgecolor='#CED5DF', linewidth=0.35,
              label=BIOME_LABELS[name])
        for name, color in OVERWORLD_BLOCKS.items()
    ]
    axis.legend(
        handles=biome_handles, title='Illustrative biome field',
        loc='lower left', frameon=False, fontsize=7.9, title_fontsize=9.4,
        labelcolor=COLORS['text'], borderaxespad=0.0, handlelength=1.9,
        handleheight=1.05, handletextpad=0.75, ncol=1,
    )
    axis.text(
        0.0, 0.68,
        'Candidate arithmetic is exact.\nBiome boundaries are a deterministic\nexplanatory model, not a vanilla seed export.',
        transform=axis.transAxes, ha='left', va='top',
        color=COLORS['muted'], fontsize=8.2, linespacing=1.35,
    )


def create_structure_placement_animation(
    save_path, seed=42, region_radius=4, fps=8, duration=12,
):
    candidates, _, (minimum, maximum) = overworld_structure_candidates(
        seed=seed, region_radius=region_radius,
    )
    total_frames = int(fps * duration)
    spacing = VILLAGE_SPACING

    figure = plt.figure(figsize=(15.0, 8.4), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[3.75, 1.0],
        left=0.055, right=0.975, top=0.90, bottom=0.12, wspace=0.10,
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
        dimension='overworld', resolution=384, alpha=0.82,
    )
    for region_x in range(-region_radius, region_radius + 1):
        for region_z in range(-region_radius, region_radius + 1):
            axis.add_patch(Rectangle(
                (region_x * spacing, region_z * spacing), spacing, spacing,
                facecolor='none', edgecolor=COLORS['text'],
                linewidth=0.46, alpha=0.28, zorder=2,
            ))
    axis.axhline(0, color=COLORS['muted'], linewidth=0.55, alpha=0.42)
    axis.axvline(0, color=COLORS['muted'], linewidth=0.55, alpha=0.42)
    axis.scatter(
        [0], [0], marker='+', s=75, c=COLORS['text'],
        linewidths=1.0, zorder=6,
    )

    point_artists = {}
    for name, style in STRUCTURE_STYLES.items():
        point_artists[name] = axis.scatter(
            [], [], s=56, c=style['color'], marker=style['marker'],
            edgecolors='#11131A', linewidths=0.55, alpha=0.96, zorder=5,
        )

    current_region = Rectangle(
        (0, 0), spacing, spacing, fill=False,
        edgecolor=COLORS['cyan'], linewidth=1.45, alpha=0.0, zorder=7,
    )
    current_window = Rectangle(
        (0, 0), 24, 24, facecolor=COLORS['blue'],
        edgecolor=COLORS['cyan'], linewidth=1.1,
        linestyle='--', alpha=0.0, zorder=3,
    )
    axis.add_patch(current_window)
    axis.add_patch(current_region)
    current_point = axis.scatter(
        [], [], s=185, facecolors='none', edgecolors=COLORS['text'],
        marker='o', linewidths=1.35, zorder=8,
    )
    trace_text = figure.text(
        0.43, 0.055, '', ha='center', va='center',
        color=COLORS['text'], fontsize=9.6, fontweight='bold',
        family='monospace',
        bbox=dict(
            boxstyle='round,pad=0.45', facecolor=COLORS['panel'],
            edgecolor=COLORS['cyan'], alpha=0.94,
        ),
    )
    _add_legends(legend_axis)
    figure.suptitle(
        'STRUCTURE CANDIDATE PLACEMENT   JAVA 1.16.1',
        color=COLORS['text'], fontsize=18, fontweight='black', y=0.965,
    )

    def update(frame_index):
        progress = frame_index / max(total_frames - 1, 1)
        shown = min(len(candidates), max(1, round(progress * len(candidates))))
        visible = candidates[:shown]
        for name, artist in point_artists.items():
            offsets = np.array([
                [item['chunk_x'], item['chunk_z']]
                for item in visible if item['name'] == name
            ], dtype=float)
            artist.set_offsets(offsets.reshape((-1, 2)))

        item = visible[-1]
        region_origin = (
            item['region_x'] * spacing,
            item['region_z'] * spacing,
        )
        current_region.set_xy(region_origin)
        current_region.set_alpha(0.92)
        current_window.set_xy(region_origin)
        current_window.set_alpha(0.16)
        current_point.set_offsets(np.array([[item['chunk_x'], item['chunk_z']]]))
        label = STRUCTURE_STYLES[item['name']]['label'].upper()
        biome = BIOME_LABELS[item['biome']].upper()
        trace_text.set_text(
            f'{label}   REGION ({item["region_x"]:+03d},{item["region_z"]:+03d})   '
            f'CHUNK ({item["chunk_x"]:+04d},{item["chunk_z"]:+04d})   BIOME {biome}'
        )
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
