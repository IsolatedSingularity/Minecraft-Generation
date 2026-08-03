"""Source-faithful Java 1.16.1 structure placement audit."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from core.constants import NETHER_STRUCTURE_SPACING, VILLAGE_SPACING
from core.minecraft_visuals import draw_minecraft_terrain
from core.structures import (
    NETHER_RUINED_PORTAL,
    VILLAGE,
    candidate_in_region,
    nether_shared_candidate,
)
from core.style import COLORS, apply_style, style_axis


apply_style()

WORLD_SEED = 42


def _panel_label(ax, label, x=0.018, y=0.978, horizontal='left'):
    ax.text(
        x, y, label, transform=ax.transAxes,
        ha=horizontal, va='top', color=COLORS['text'],
        fontsize=12, fontweight='black', zorder=20,
        bbox=dict(
            boxstyle='square,pad=0.22', facecolor=COLORS['background'],
            edgecolor='none', alpha=0.78,
        ),
    )


def _regions(radius):
    return [
        (region_x, region_z)
        for region_x in range(-radius, radius + 1)
        for region_z in range(-radius, radius + 1)
    ]


def _draw_region_grid(ax, radius, spacing, color):
    for region_x, region_z in _regions(radius):
        ax.add_patch(Rectangle(
            (region_x * spacing, region_z * spacing), spacing, spacing,
            fill=False, edgecolor=color, linewidth=0.62,
            alpha=0.48, zorder=3,
        ))


def _candidate_offset_samples(samples=18_432):
    offsets = []
    for sample in range(int(samples)):
        region_x = sample % 17 - 8
        region_z = (sample // 17) % 17 - 8
        village = candidate_in_region(sample, region_x, region_z, VILLAGE)
        offsets.append((village['offset_x'], village['offset_z']))
    return np.asarray(offsets)


def create_structure_analysis(save_path, dpi=200, seed=WORLD_SEED):
    """Render four large panels separating exact math from visual terrain."""
    figure = plt.figure(figsize=(15.5, 9.4), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        2, 2, left=0.055, right=0.98, top=0.91, bottom=0.075,
        wspace=0.15, hspace=0.25,
    )
    village_axis = figure.add_subplot(grid[0, 0])
    region_axis = figure.add_subplot(grid[0, 1])
    nether_axis = figure.add_subplot(grid[1, 0])
    audit_axis = figure.add_subplot(grid[1, 1])

    radius = 3
    village_minimum = -radius * VILLAGE_SPACING - 4
    village_maximum = (radius + 1) * VILLAGE_SPACING + 4
    draw_minecraft_terrain(
        village_axis,
        (village_minimum, village_maximum, village_minimum, village_maximum),
        seed=seed, dimension='overworld', resolution=224, alpha=0.90,
    )
    _draw_region_grid(village_axis, radius, VILLAGE_SPACING, COLORS['text'])
    villages = [
        candidate_in_region(seed, region_x, region_z, VILLAGE)
        for region_x, region_z in _regions(radius)
    ]
    village_axis.scatter(
        [item['chunk_x'] for item in villages],
        [item['chunk_z'] for item in villages],
        s=58, marker='s', c=COLORS['gold'], edgecolors='#47330F',
        linewidths=0.6, zorder=6,
    )
    village_axis.scatter(
        [0], [0], s=100, marker='*', c=COLORS['text'],
        edgecolors=COLORS['gold'], linewidths=0.7, zorder=7,
    )
    village_axis.set_xlim(village_minimum, village_maximum)
    village_axis.set_ylim(village_minimum, village_maximum)
    village_axis.set_xlabel('Chunk X')
    village_axis.set_ylabel('Chunk Z')
    village_axis.set_title(
        'Village candidates over 32 x 32 regions',
        fontsize=12.5, fontweight='bold', pad=8,
    )
    style_axis(village_axis, equal=True, grid=False)
    _panel_label(village_axis, '(a)')

    focus = candidate_in_region(seed, 0, 0, VILLAGE)
    draw_minecraft_terrain(
        region_axis, (0, 32, 0, 32), seed=seed + 11,
        dimension='overworld', resolution=192, alpha=0.90,
    )
    region_axis.add_patch(Rectangle(
        (0, 0), 24, 24, facecolor=COLORS['blue'],
        edgecolor=COLORS['cyan'], linewidth=1.8, alpha=0.20, zorder=3,
    ))
    region_axis.add_patch(Rectangle(
        (24, 0), 8, 32, facecolor=COLORS['coral'],
        edgecolor='none', alpha=0.22, zorder=3,
    ))
    region_axis.add_patch(Rectangle(
        (0, 24), 24, 8, facecolor=COLORS['coral'],
        edgecolor='none', alpha=0.22, zorder=3,
    ))
    for coordinate in range(0, 33, 4):
        region_axis.axvline(
            coordinate, color='#080A10', linewidth=0.48, alpha=0.52, zorder=4,
        )
        region_axis.axhline(
            coordinate, color='#080A10', linewidth=0.48, alpha=0.52, zorder=4,
        )
    region_axis.scatter(
        [focus['offset_x']], [focus['offset_z']], s=190, marker='s',
        c=COLORS['gold'], edgecolors=COLORS['text'], linewidths=1.1, zorder=8,
    )
    region_axis.plot(
        [0, focus['offset_x']], [focus['offset_z'], focus['offset_z']],
        color=COLORS['gold'], linewidth=1.2, zorder=7,
    )
    region_axis.plot(
        [focus['offset_x'], focus['offset_x']], [0, focus['offset_z']],
        color=COLORS['gold'], linewidth=1.2, zorder=7,
    )
    region_axis.text(
        0.97, 0.05,
        f"nextInt(24) = ({focus['offset_x']}, {focus['offset_z']})",
        transform=region_axis.transAxes, ha='right', va='bottom',
        fontsize=11, fontweight='bold', family='monospace',
        bbox=dict(
            boxstyle='round,pad=0.38', facecolor=COLORS['panel'],
            edgecolor=COLORS['gold'], alpha=0.92,
        ), zorder=10,
    )
    region_axis.set_xlim(0, 32)
    region_axis.set_ylim(0, 32)
    region_axis.set_xticks([0, 8, 16, 24, 32])
    region_axis.set_yticks([0, 8, 16, 24, 32])
    region_axis.set_xlabel('Local chunk X')
    region_axis.set_ylabel('Local chunk Z')
    region_axis.set_title(
        'One region and its 24 x 24 candidate window',
        fontsize=12.5, fontweight='bold', pad=8,
    )
    style_axis(region_axis, equal=True, grid=False)
    _panel_label(region_axis, '(b)')

    nether_minimum = -radius * NETHER_STRUCTURE_SPACING - 4
    nether_maximum = (radius + 1) * NETHER_STRUCTURE_SPACING + 4
    draw_minecraft_terrain(
        nether_axis,
        (nether_minimum, nether_maximum, nether_minimum, nether_maximum),
        seed=seed, dimension='nether', resolution=224, alpha=0.91,
    )
    _draw_region_grid(
        nether_axis, radius, NETHER_STRUCTURE_SPACING, COLORS['coral'],
    )
    for coordinate in range(-radius - 1, radius + 2):
        value = coordinate * NETHER_RUINED_PORTAL.spacing
        nether_axis.axvline(
            value, color=COLORS['violet'], linewidth=0.55,
            linestyle=':', alpha=0.40, zorder=4,
        )
        nether_axis.axhline(
            value, color=COLORS['violet'], linewidth=0.55,
            linestyle=':', alpha=0.40, zorder=4,
        )
    shared = [
        nether_shared_candidate(seed, region_x, region_z)
        for region_x, region_z in _regions(radius)
    ]
    portals = [
        candidate_in_region(seed, region_x, region_z, NETHER_RUINED_PORTAL)
        for region_x, region_z in _regions(radius)
    ]
    for name, marker, color, label in (
        ('fortress', 'P', COLORS['fortress'], 'fortress, rolls 0 or 1'),
        ('bastion', 's', COLORS['bastion'], 'bastion, rolls 2 to 4'),
    ):
        subset = [item for item in shared if item['name'] == name]
        nether_axis.scatter(
            [item['chunk_x'] for item in subset],
            [item['chunk_z'] for item in subset],
            s=72, marker=marker, c=color, edgecolors=COLORS['text'],
            linewidths=0.55, label=label, zorder=7,
        )
    nether_axis.scatter(
        [item['chunk_x'] for item in portals],
        [item['chunk_z'] for item in portals],
        s=52, marker='D', c=COLORS['ruined_portal'],
        edgecolors=COLORS['text'], linewidths=0.5,
        label='ruined portal, independent salt', zorder=8,
    )
    nether_axis.set_xlim(nether_minimum, nether_maximum)
    nether_axis.set_ylim(nether_minimum, nether_maximum)
    nether_axis.set_xlabel('Nether chunk X')
    nether_axis.set_ylabel('Nether chunk Z')
    nether_axis.set_title(
        'Shared fortress and bastion grid with portal layer',
        fontsize=12.5, fontweight='bold', pad=8,
    )
    nether_axis.legend(loc='lower right', fontsize=8.3, framealpha=0.92)
    style_axis(nether_axis, equal=True, grid=False)
    _panel_label(nether_axis, '(c)')

    offsets = _candidate_offset_samples()
    heatmap, _, _ = np.histogram2d(
        offsets[:, 1], offsets[:, 0], bins=np.arange(25) - 0.5,
    )
    audit_axis.imshow(
        heatmap, origin='lower', extent=(-0.5, 23.5, -0.5, 23.5),
        interpolation='nearest', cmap='viridis', aspect='equal', zorder=1,
    )
    fortress_share = 2.0 / 5.0
    audit_axis.add_patch(Rectangle(
        (-0.5, 25.2), 24.0 * fortress_share, 1.4,
        facecolor=COLORS['fortress'], edgecolor='none', zorder=3,
    ))
    audit_axis.add_patch(Rectangle(
        (-0.5 + 24.0 * fortress_share, 25.2),
        24.0 * (1.0 - fortress_share), 1.4,
        facecolor=COLORS['bastion'], edgecolor='none', zorder=3,
    ))
    audit_axis.text(
        -0.5 + 12.0 * fortress_share, 25.9,
        f'FORTRESS {fortress_share * 100:.1f}%',
        ha='center', va='center', fontsize=9, fontweight='bold', zorder=4,
    )
    audit_axis.text(
        -0.5 + 24.0 * fortress_share + 12.0 * (1.0 - fortress_share),
        25.9, f'BASTION {(1.0 - fortress_share) * 100:.1f}%',
        ha='center', va='center', fontsize=9, fontweight='bold', zorder=4,
    )
    audit_axis.text(
        0.03, 0.055,
        f'{len(offsets):,} seeded samples\n576 reachable offset pairs',
        transform=audit_axis.transAxes, ha='left', va='bottom',
        color=COLORS['text'], fontsize=9.5, fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.35', facecolor=COLORS['panel'],
            edgecolor=COLORS['grid'], alpha=0.90,
        ), zorder=5,
    )
    audit_axis.set_xlim(-0.5, 23.5)
    audit_axis.set_ylim(-0.5, 27.2)
    audit_axis.set_xticks([0, 4, 8, 12, 16, 20, 23])
    audit_axis.set_yticks([0, 4, 8, 12, 16, 20, 23])
    audit_axis.set_xlabel('Village offset X, nextInt(24)')
    audit_axis.set_ylabel('Village offset Z, nextInt(24)')
    audit_axis.set_title(
        'Uniform offset field and exact 2 to 3 Nether split',
        fontsize=12.5, fontweight='bold', pad=8,
    )
    style_axis(audit_axis, equal=False, grid=False)
    _panel_label(audit_axis, '(d)', x=0.975, y=0.86, horizontal='right')

    figure.suptitle(
        'STRUCTURE PLACEMENT AUDIT   JAVA 1.16.1',
        color=COLORS['text'], fontsize=18, fontweight='black', y=0.972,
    )
    figure.text(
        0.50, 0.025,
        'Candidate coordinates and random rolls are exact. Terrain backdrops are illustrative.',
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
    create_structure_analysis(plots / 'structure_analysis.png')


if __name__ == '__main__':
    main()
