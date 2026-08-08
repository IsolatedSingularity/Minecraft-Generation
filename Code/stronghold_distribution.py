"""Java 1.16.1 stronghold candidate-ring visualization."""

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

from core.constants import STRONGHOLD_RINGS
from core.minecraft_visuals import draw_minecraft_terrain
from core.strongholds import generate_stronghold_candidates
from core.style import COLORS, apply_style, style_axis


apply_style()

RING_COLORS = [
    '#FB7185', '#F59E72', '#F6C85F', '#A5D66A',
    '#43D9C2', '#65C7F7', '#A78BFA', '#E879F9',
]


def _panel_label(ax, label):
    ax.text(
        0.018, 0.978, label, transform=ax.transAxes,
        ha='left', va='top', color=COLORS['text'],
        fontsize=12, fontweight='black', zorder=20,
        bbox=dict(
            boxstyle='square,pad=0.22', facecolor=COLORS['background'],
            edgecolor='none', alpha=0.80,
        ),
    )


def _ring_band(ax, ring, color):
    ax.add_patch(Wedge(
        (0, 0), ring['max_radius'], 0, 360,
        width=ring['max_radius'] - ring['min_radius'],
        facecolor=color, edgecolor='none', alpha=0.085, zorder=2,
    ))
    ax.add_patch(Circle(
        (0, 0), ring['min_radius'], fill=False,
        edgecolor=color, linewidth=0.90, linestyle=':', alpha=0.78,
        zorder=3,
    ))
    ax.add_patch(Circle(
        (0, 0), ring['max_radius'], fill=False,
        edgecolor=color, linewidth=1.18, alpha=0.86, zorder=3,
    ))


def create_stronghold_distribution(save_path, dpi=200, seed=42):
    candidates = generate_stronghold_candidates(seed)
    first_ring = candidates[:3]

    figure = plt.figure(figsize=(15.5, 9.4), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        2, 2, width_ratios=[1.72, 1.0], height_ratios=[1.0, 0.82],
        left=0.055, right=0.98, top=0.91, bottom=0.075,
        wspace=0.16, hspace=0.24,
    )
    ring_axis = figure.add_subplot(grid[:, 0])
    first_axis = figure.add_subplot(grid[0, 1])
    count_axis = figure.add_subplot(grid[1, 1])

    maximum = STRONGHOLD_RINGS[-1]['max_radius'] + 1700
    draw_minecraft_terrain(
        ring_axis, (-maximum, maximum, -maximum, maximum),
        seed=seed, dimension='overworld', resolution=384, alpha=0.50,
    )
    for index, (ring, color) in enumerate(zip(STRONGHOLD_RINGS, RING_COLORS)):
        _ring_band(ring_axis, ring, color)
        subset = [item for item in candidates if item['ring'] == index + 1]
        ring_axis.scatter(
            [item['x'] for item in subset],
            [item['z'] for item in subset],
            s=82 if index == 0 else 46,
            marker='o', c=color,
            edgecolors=COLORS['text'] if index == 0 else '#11131A',
            linewidths=0.55 if index == 0 else 0.28,
            alpha=0.96, zorder=6,
        )
        angle = math.radians(17.0 + index * 4.4)
        midpoint = (ring['min_radius'] + ring['max_radius']) / 2.0
        ring_axis.text(
            midpoint * math.cos(angle), midpoint * math.sin(angle),
            f"R{index + 1}  n={ring['count']}",
            color=COLORS['text'], fontsize=7.8, fontweight='bold',
            ha='center', va='center', zorder=8,
            bbox=dict(
                boxstyle='round,pad=0.20', facecolor=COLORS['background'],
                edgecolor=color, alpha=0.88,
            ),
        )
    for item in first_ring:
        ring_axis.plot(
            [0, item['x']], [0, item['z']],
            color=COLORS['gold'], linewidth=0.85, alpha=0.58, zorder=5,
        )
    ring_axis.scatter(
        [0], [0], marker='*', s=150, c=COLORS['text'],
        edgecolors=COLORS['gold'], linewidths=0.8, zorder=9,
    )
    ring_axis.set_xlim(-maximum, maximum)
    ring_axis.set_ylim(-maximum, maximum)
    ring_axis.set_xlabel('Block X')
    ring_axis.set_ylabel('Block Z')
    ring_axis.set_title(
        'All 128 pre-biome-search candidates',
        fontsize=12.5, fontweight='bold', pad=8,
    )
    style_axis(ring_axis, equal=True, grid=False)
    _panel_label(ring_axis, '(a)')

    first_limit = 3100
    draw_minecraft_terrain(
        first_axis, (-first_limit, first_limit, -first_limit, first_limit),
        seed=seed + 73, dimension='overworld', resolution=256, alpha=0.62,
    )
    first_config = STRONGHOLD_RINGS[0]
    first_axis.add_patch(Wedge(
        (0, 0), first_config['max_radius'], 0, 360,
        width=first_config['max_radius'] - first_config['min_radius'],
        facecolor=RING_COLORS[0], edgecolor='none', alpha=0.14, zorder=2,
    ))
    first_axis.add_patch(Circle(
        (0, 0), first_config['min_radius'], fill=False,
        edgecolor=RING_COLORS[0], linewidth=1.0, linestyle=':', zorder=3,
    ))
    first_axis.add_patch(Circle(
        (0, 0), first_config['max_radius'], fill=False,
        edgecolor=RING_COLORS[0], linewidth=1.1, zorder=3,
    ))
    for index, item in enumerate(first_ring, start=1):
        first_axis.add_patch(Circle(
            (item['x'], item['z']), 112, fill=False,
            edgecolor=COLORS['gold'], linewidth=1.15,
            linestyle='--', alpha=0.94, zorder=7,
        ))
        first_axis.plot(
            [0, item['x']], [0, item['z']],
            color=COLORS['text'], linewidth=0.75, alpha=0.48, zorder=5,
        )
        first_axis.scatter(
            [item['x']], [item['z']], s=105, c=RING_COLORS[0],
            edgecolors=COLORS['text'], linewidths=0.8, zorder=8,
        )
        first_axis.text(
            item['x'], item['z'], str(index),
            color=COLORS['text'], fontsize=8, fontweight='bold',
            ha='center', va='center', zorder=9,
        )
    first_axis.scatter(
        [0], [0], marker='*', s=110, c=COLORS['text'],
        edgecolors=COLORS['gold'], linewidths=0.7, zorder=8,
    )
    first_axis.text(
        0.97, 0.05, 'dashed circles = 112-block biome search',
        transform=first_axis.transAxes, ha='right', va='bottom',
        color=COLORS['text'], fontsize=8.8, fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.32', facecolor=COLORS['panel'],
            edgecolor=COLORS['gold'], alpha=0.92,
        ), zorder=10,
    )
    first_axis.set_xlim(-first_limit, first_limit)
    first_axis.set_ylim(-first_limit, first_limit)
    first_axis.set_xlabel('Block X')
    first_axis.set_ylabel('Block Z')
    first_axis.set_title(
        'First ring candidate search neighborhoods',
        fontsize=12.5, fontweight='bold', pad=8,
    )
    style_axis(first_axis, equal=True, grid=False)
    _panel_label(first_axis, '(b)')

    counts = [ring['count'] for ring in STRONGHOLD_RINGS]
    bars = count_axis.bar(
        range(1, 9), counts, color=RING_COLORS,
        edgecolor=COLORS['text'], linewidth=0.55, alpha=0.94,
    )
    for bar, ring in zip(bars, STRONGHOLD_RINGS):
        count_axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.8,
            f"n = {ring['count']}",
            ha='center', va='bottom', color=COLORS['text'],
            fontsize=8.2, fontweight='bold',
        )
    count_axis.text(
        0.02, 0.94, '3 + 6 + 10 + 15 + 21 + 28 + 36 + 9 = 128',
        transform=count_axis.transAxes, ha='left', va='top',
        color=COLORS['muted'], fontsize=9.2, family='monospace',
    )
    count_axis.set_ylim(0, 44)
    count_axis.set_xticks(
        range(1, 9),
        [
            f"R{index}\n{ring['min_radius'] / 1000:.1f}-{ring['max_radius'] / 1000:.1f}"
            for index, ring in enumerate(STRONGHOLD_RINGS, start=1)
        ],
    )
    count_axis.tick_params(axis='x', labelsize=7.2)
    count_axis.set_xlabel('Ring and radial band (thousands of blocks)')
    count_axis.set_ylabel('Candidate count')
    count_axis.set_title(
        'Candidate count and radial search band',
        fontsize=12.5, fontweight='bold', pad=8,
    )
    style_axis(count_axis, grid=True)
    _panel_label(count_axis, '(c)')

    figure.suptitle(
        'STRONGHOLD CANDIDATE RINGS   JAVA 1.16.1',
        color=COLORS['text'], fontsize=18, fontweight='black', y=0.972,
    )
    figure.text(
        0.50, 0.025,
        'Candidate geometry is exact for the seeded ring iterator. Terrain backdrops are illustrative.',
        ha='center', va='center', color=COLORS['muted'], fontsize=9.5,
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
    create_stronghold_distribution(plots / 'stronghold_rings.png')


if __name__ == '__main__':
    main()
