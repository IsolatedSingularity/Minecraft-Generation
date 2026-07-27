"""Java 1.16.1 stronghold candidate-ring visualization.

The ring geometry follows the pre-1.19.3 Java iterator. Points are the
approximate candidates before the vanilla biome search, which can move the
final structure within a 112-block search radius.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyBboxPatch
import numpy as np

from core.constants import MINECRAFT_VERSION, STRONGHOLD_RINGS, TOTAL_STRONGHOLDS
from core.strongholds import generate_stronghold_candidates


plt.style.use('dark_background')

COLORS = {
    'background': '#0B1020',
    'panel': '#111A2E',
    'panel_alt': '#17233B',
    'grid': '#263653',
    'text': '#E8EEF9',
    'muted': '#93A4C3',
    'accent': '#7DD3FC',
    'warning': '#FCD34D',
}

RING_COLORS = [ring['color'] for ring in STRONGHOLD_RINGS]


def add_ring_band(ax, ring, alpha=0.10, linewidth=1.0):
    color = ring['color']
    ax.add_patch(Wedge(
        (0, 0), ring['max_radius'], 0, 360,
        width=ring['max_radius'] - ring['min_radius'],
        facecolor=color, edgecolor='none', alpha=alpha,
    ))
    ax.add_patch(Circle(
        (0, 0), ring['min_radius'], fill=False, color=color,
        alpha=0.55, linewidth=linewidth, linestyle='--',
    ))
    ax.add_patch(Circle(
        (0, 0), ring['max_radius'], fill=False, color=color,
        alpha=0.80, linewidth=linewidth,
    ))


def create_stronghold_distribution(save_path, dpi=300, seed=42):
    """Render the Java 1.16.1 stronghold candidate distribution."""
    candidates = generate_stronghold_candidates(seed)
    max_radius = STRONGHOLD_RINGS[-1]['max_radius'] + 1800

    fig = plt.figure(figsize=(18, 10), facecolor=COLORS['background'])
    gs = fig.add_gridspec(
        10, 12, left=0.04, right=0.97, top=0.88, bottom=0.07,
        hspace=0.70, wspace=0.70,
    )
    ax_map = fig.add_subplot(gs[:, :8])
    ax_info = fig.add_subplot(gs[:7, 8:])
    ax_zoom = fig.add_subplot(gs[7:, 8:])

    for ax in (ax_map, ax_info, ax_zoom):
        ax.set_facecolor(COLORS['panel'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
            spine.set_linewidth(1.0)

    fig.suptitle(
        f'STRONGHOLD CANDIDATE RINGS  |  {MINECRAFT_VERSION}',
        color=COLORS['text'], fontsize=21, fontweight='bold',
        x=0.04, ha='left',
    )
    fig.text(
        0.97, 0.895,
        'Seeded Java ring iterator, shown before the biome search',
        color=COLORS['muted'], fontsize=10, ha='right',
    )

    ax_map.set_xlim(-max_radius, max_radius)
    ax_map.set_ylim(-max_radius, max_radius)
    ax_map.set_aspect('equal')
    ax_map.set_title(
        '128 APPROXIMATE CANDIDATES  /  WORLD ORIGIN (0, 0)',
        color=COLORS['text'], fontsize=12, fontweight='bold',
        loc='left', pad=12,
    )
    ax_map.set_xlabel('Block X', color=COLORS['muted'])
    ax_map.set_ylabel('Block Z', color=COLORS['muted'])
    ax_map.tick_params(colors=COLORS['muted'], labelsize=8)
    ax_map.grid(color=COLORS['grid'], linewidth=0.5, alpha=0.38)

    for ring in STRONGHOLD_RINGS:
        add_ring_band(ax_map, ring)

    for index, ring in enumerate(STRONGHOLD_RINGS):
        mid_radius = (ring['min_radius'] + ring['max_radius']) / 2.0
        label_angle = np.deg2rad(28 + index * 2.4)
        ax_map.text(
            mid_radius * np.cos(label_angle),
            mid_radius * np.sin(label_angle),
            f'R{index + 1}  n={ring["count"]}',
            color=ring['color'], fontsize=8, fontweight='bold',
            ha='center', va='center',
            bbox=dict(
                boxstyle='round,pad=0.25', facecolor=COLORS['panel'],
                edgecolor=ring['color'], alpha=0.92, linewidth=0.8,
            ),
            zorder=6,
        )

    for ring_index, ring in enumerate(STRONGHOLD_RINGS, start=1):
        ring_points = [item for item in candidates if item['ring'] == ring_index]
        ax_map.scatter(
            [item['x'] for item in ring_points],
            [item['z'] for item in ring_points],
            s=38 if ring_index > 1 else 78,
            color=ring['color'],
            edgecolors=COLORS['text'] if ring_index == 1 else 'none',
            linewidths=0.7,
            alpha=0.96,
            zorder=7,
        )

    first_ring = candidates[:STRONGHOLD_RINGS[0]['count']]
    for item in first_ring:
        ax_map.plot(
            [0, item['x']], [0, item['z']],
            color=COLORS['warning'], alpha=0.42, linewidth=0.8, zorder=3,
        )

    ax_map.scatter(
        [0], [0], marker='*', s=220, color=COLORS['warning'],
        edgecolors=COLORS['text'], linewidths=1.2, zorder=9,
    )
    ax_map.text(
        0.98, 0.03, 'colors encode ring  |  highlighted spokes = first ring',
        transform=ax_map.transAxes, ha='right', color=COLORS['muted'],
        fontsize=8,
    )

    ax_info.axis('off')
    ax_info.text(
        0.04, 0.94, 'RING MODEL', transform=ax_info.transAxes,
        color=COLORS['text'], fontsize=12, fontweight='bold',
    )
    info_box = FancyBboxPatch(
        (0.04, 0.72), 0.92, 0.15, transform=ax_info.transAxes,
        boxstyle='round,pad=0.012', facecolor='#203454',
        edgecolor=COLORS['accent'], linewidth=1.2,
    )
    ax_info.add_patch(info_box)
    ax_info.text(
        0.08, 0.81, 'SPEEDRUN RING', transform=ax_info.transAxes,
        color=COLORS['accent'], fontsize=9, fontweight='bold',
    )
    ax_info.text(
        0.08, 0.755, '3 candidates  |  1,408 - 2,688 blocks',
        transform=ax_info.transAxes, color=COLORS['text'], fontsize=10,
        family='monospace',
    )
    ax_info.text(
        0.04, 0.66,
        f'world seed        {seed}\n'
        f'candidate count   {TOTAL_STRONGHOLDS}\n'
        f'candidate search  +/-112 blocks\n'
        f'coordinate frame   origin, not player spawn',
        transform=ax_info.transAxes, color=COLORS['muted'], fontsize=9,
        linespacing=1.6, family='monospace',
    )

    table_y = 0.48
    ax_info.text(
        0.04, table_y + 0.055, 'RING     COUNT     CANDIDATE RANGE',
        transform=ax_info.transAxes, color=COLORS['muted'],
        fontsize=8, family='monospace',
    )
    for index, ring in enumerate(STRONGHOLD_RINGS):
        y = table_y - index * 0.047
        ax_info.text(
            0.04, y, f'{index + 1:>2}       {ring["count"]:>3}',
            transform=ax_info.transAxes, color=ring['color'],
            fontsize=8, family='monospace',
        )
        ax_info.text(
            0.37, y,
            f'{ring["min_radius"]:,} - {ring["max_radius"]:,}',
            transform=ax_info.transAxes, color=COLORS['text'],
            fontsize=8, family='monospace',
        )

    ax_info.text(
        0.04, 0.04,
        'The ring iterator fixes the count and angular spacing.\n'
        'Biome search selects the final valid location.',
        transform=ax_info.transAxes, color=COLORS['muted'], fontsize=8,
        linespacing=1.5,
    )

    ax_zoom.set_xlim(-3200, 3200)
    ax_zoom.set_ylim(-3200, 3200)
    ax_zoom.set_aspect('equal')
    ax_zoom.set_title(
        'FIRST RING DETAIL  /  112-BLOCK SEARCH RADIUS',
        color=COLORS['text'], fontsize=10, fontweight='bold',
        loc='left', pad=10,
    )
    ax_zoom.tick_params(colors=COLORS['muted'], labelsize=7)
    ax_zoom.grid(color=COLORS['grid'], linewidth=0.5, alpha=0.4)
    add_ring_band(ax_zoom, STRONGHOLD_RINGS[0], alpha=0.16, linewidth=0.9)
    for item in first_ring:
        ax_zoom.add_patch(Circle(
            (item['x'], item['z']), 112, fill=False,
            color=COLORS['warning'], alpha=0.48, linewidth=0.8,
            linestyle=':',
        ))
        ax_zoom.scatter(
            [item['x']], [item['z']], s=65, color=item['color'],
            edgecolors=COLORS['text'], linewidths=0.8, zorder=4,
        )
        ax_zoom.text(
            item['x'], item['z'], str(item['ring_index']),
            color=COLORS['text'], fontsize=8, fontweight='bold',
            ha='center', va='center', zorder=5,
        )
    ax_zoom.scatter(
        [0], [0], marker='*', s=90, color=COLORS['warning'],
        edgecolors=COLORS['text'], linewidths=0.8,
    )
    ax_zoom.set_xlabel('Block X', color=COLORS['muted'], fontsize=8)
    ax_zoom.set_ylabel('Block Z', color=COLORS['muted'], fontsize=8)

    fig.savefig(
        save_path, dpi=dpi, facecolor=COLORS['background'],
        edgecolor='none', bbox_inches='tight',
    )
    plt.close(fig)
    return save_path


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(os.path.dirname(script_dir), 'Plots')
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, 'stronghold_rings.png')
    create_stronghold_distribution(output_path)
