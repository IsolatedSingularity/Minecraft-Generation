"""Six-panel candidate-stage structure analysis for Java 1.16.1."""

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

from core.constants import CHUNK_SIZE, STRONGHOLD_RINGS
from core.strongholds import generate_stronghold_candidates
from core.structures import (
    NETHER_RUINED_PORTAL,
    VILLAGE,
    candidate_in_region,
    nether_shared_candidate,
)
from core.style import COLORS, apply_style, style_axis


apply_style()

SEEDS = {
    'village': 164311266871034,
    'nether': 1785852800490,
    'stronghold': 27594263,
}


def _panel_label(ax, label):
    ax.text(0.02, 0.975, label, transform=ax.transAxes,
            va='top', color=COLORS['text'], fontsize=10.5, fontweight='bold')


def _square_regions(radius):
    return [(x, z) for x in range(-radius, radius + 1)
            for z in range(-radius, radius + 1)]


def _bearing_estimate(throws, target, sigma, random):
    matrix = []
    values = []
    for point in throws:
        delta = target - point
        angle = math.atan2(delta[1], delta[0]) + random.normal(0.0, sigma)
        normal = np.array([-math.sin(angle), math.cos(angle)])
        matrix.append(normal)
        values.append(float(np.dot(normal, point)))
    estimate, *_ = np.linalg.lstsq(np.asarray(matrix), np.asarray(values), rcond=None)
    return estimate


def _nearest_distributions(samples=320):
    output = {'village': [], 'fortress': [], 'bastion': [], 'stronghold': []}
    regions = _square_regions(3)
    for seed in range(samples):
        villages = [candidate_in_region(seed, x, z, VILLAGE) for x, z in regions]
        output['village'].append(min(
            math.hypot(item['block_x'], item['block_z']) for item in villages
        ))
        nether = [nether_shared_candidate(seed, x, z) for x, z in regions]
        for name in ('fortress', 'bastion'):
            subset = [item for item in nether if item['name'] == name]
            if subset:
                output[name].append(min(
                    math.hypot(item['block_x'], item['block_z']) for item in subset
                ))
        strongholds = generate_stronghold_candidates(seed)[:3]
        output['stronghold'].append(min(item['radius'] for item in strongholds))
    return output


def create_structure_analysis(save_path, dpi=220):
    figure, axes = plt.subplots(
        2, 3, figsize=(15.5, 9.8), facecolor=COLORS['background'],
        gridspec_kw={'wspace': 0.30, 'hspace': 0.32},
    )
    ax_offsets, ax_nether, ax_rings, ax_cdf, ax_salts, ax_error = axes.ravel()

    village_regions = _square_regions(10)
    villages = [
        candidate_in_region(SEEDS['village'], x, z, VILLAGE)
        for x, z in village_regions
    ]
    radial_region = np.array([
        math.hypot(item['region_x'], item['region_z']) for item in villages
    ])
    ax_offsets.scatter(
        [item['offset_x'] for item in villages],
        [item['offset_z'] for item in villages],
        s=24, c=radial_region, cmap='viridis', alpha=0.72,
        edgecolors='none',
    )
    ax_offsets.add_patch(Circle((11.5, 11.5), 10.5, fill=False,
                                edgecolor=COLORS['cyan'], linewidth=0.8,
                                linestyle=':', alpha=0.7))
    ax_offsets.set_xlim(-1, 24)
    ax_offsets.set_ylim(-1, 24)
    ax_offsets.set_xlabel('Village offset X (chunks)')
    ax_offsets.set_ylabel('Village offset Z (chunks)')
    style_axis(ax_offsets, equal=True, grid=True)
    _panel_label(ax_offsets, '(a)')

    nether_regions = _square_regions(6)
    nether = [
        nether_shared_candidate(SEEDS['nether'], x, z)
        for x, z in nether_regions
    ]
    for name, marker, color in (
        ('fortress', 'o', COLORS['fortress']),
        ('bastion', 's', COLORS['bastion']),
    ):
        subset = [item for item in nether if item['name'] == name]
        ax_nether.scatter(
            [item['chunk_x'] for item in subset],
            [item['chunk_z'] for item in subset],
            s=30, marker=marker, c=color, edgecolors=COLORS['text'],
            linewidths=0.25, alpha=0.88, label=name,
        )
    ax_nether.scatter([0], [0], marker='+', s=65, c=COLORS['text'], linewidths=0.9)
    ax_nether.set_xlabel('Nether chunk X')
    ax_nether.set_ylabel('Nether chunk Z')
    ax_nether.legend(loc='lower right', fontsize=7, framealpha=0.85)
    style_axis(ax_nether, equal=True, grid=True)
    _panel_label(ax_nether, '(b)')

    strongholds = generate_stronghold_candidates(SEEDS['stronghold'])
    ring_palette = ['#FB7185', '#F59E72', '#F6C85F', '#A5D66A',
                    '#43D9C2', '#65C7F7', '#A78BFA', '#E879F9']
    for ring_index, color in enumerate(ring_palette, start=1):
        subset = [item for item in strongholds if item['ring'] == ring_index]
        ax_rings.scatter(
            [item['x'] for item in subset], [item['z'] for item in subset],
            s=17 if ring_index > 1 else 38, c=color,
            edgecolors=COLORS['text'] if ring_index == 1 else 'none',
            linewidths=0.3, alpha=0.90,
        )
    ax_rings.scatter([0], [0], marker='+', s=65, c=COLORS['text'], linewidths=0.9)
    max_ring = STRONGHOLD_RINGS[-1]['max_radius'] + 1400
    ax_rings.set_xlim(-max_ring, max_ring)
    ax_rings.set_ylim(-max_ring, max_ring)
    ax_rings.set_xlabel('Block X')
    ax_rings.set_ylabel('Block Z')
    style_axis(ax_rings, equal=True, grid=True)
    _panel_label(ax_rings, '(c)')

    distributions = _nearest_distributions()
    for name, color in (
        ('village', COLORS['green']),
        ('fortress', COLORS['fortress']),
        ('bastion', COLORS['bastion']),
        ('stronghold', COLORS['stronghold']),
    ):
        values = np.sort(np.asarray(distributions[name]))
        cdf = np.arange(1, len(values) + 1) / len(values)
        ax_cdf.plot(values, cdf, color=color, linewidth=1.5, label=name)
    ax_cdf.set_xlabel('Nearest candidate distance (native blocks)')
    ax_cdf.set_ylabel('Empirical CDF')
    ax_cdf.set_ylim(0, 1.02)
    ax_cdf.legend(loc='lower right', fontsize=7, framealpha=0.85)
    style_axis(ax_cdf, grid=True)
    _panel_label(ax_cdf, '(d)')

    paired = []
    for region_x, region_z in _square_regions(9):
        village = candidate_in_region(SEEDS['village'], region_x, region_z, VILLAGE)
        portal = candidate_in_region(SEEDS['village'], region_x, region_z, NETHER_RUINED_PORTAL)
        paired.append((village['offset_x'], portal['offset_x'], village['offset_z']))
    paired = np.asarray(paired)
    ax_salts.scatter(
        paired[:, 0], paired[:, 1], s=22, c=paired[:, 2],
        cmap='plasma', alpha=0.58, edgecolors='none',
    )
    correlation = float(np.corrcoef(paired[:, 0], paired[:, 1])[0, 1])
    ax_salts.text(
        0.97, 0.06, f'r = {correlation:+.3f}', transform=ax_salts.transAxes,
        ha='right', color=COLORS['muted'], fontsize=8, family='monospace',
    )
    ax_salts.set_xlabel('Village offset X')
    ax_salts.set_ylabel('Ruined portal offset X')
    style_axis(ax_salts, grid=True)
    _panel_label(ax_salts, '(e)')

    target_item = generate_stronghold_candidates(SEEDS['stronghold'])[0]
    target = np.array([target_item['x'], target_item['z']], dtype=float)
    target_direction = target / np.linalg.norm(target)
    perpendicular = np.array([-target_direction[1], target_direction[0]])
    throws_two = [perpendicular * -240.0, perpendicular * 240.0]
    throws_three = throws_two + [target_direction * -320.0]
    random = np.random.default_rng(2026)
    sigma_degrees = np.linspace(0.05, 1.20, 15)
    for throws, label, color in (
        (throws_two, '2 throws', COLORS['coral']),
        (throws_three, '3 throws', COLORS['cyan']),
    ):
        medians = []
        upper = []
        for value in sigma_degrees:
            sigma = math.radians(value)
            errors = [
                np.linalg.norm(_bearing_estimate(throws, target, sigma, random) - target)
                for _ in range(260)
            ]
            medians.append(np.median(errors))
            upper.append(np.percentile(errors, 90))
        ax_error.plot(sigma_degrees, medians, color=color, linewidth=1.6, label=label)
        ax_error.fill_between(sigma_degrees, medians, upper, color=color, alpha=0.12)
    ax_error.axhline(112, color=COLORS['gold'], linewidth=0.8,
                     linestyle='--', alpha=0.8)
    ax_error.set_xlabel('Bearing noise, sigma (degrees)')
    ax_error.set_ylabel('Intersection error (blocks)')
    ax_error.legend(loc='upper left', fontsize=7, framealpha=0.85)
    style_axis(ax_error, grid=True)
    _panel_label(ax_error, '(f)')

    figure.savefig(save_path, dpi=dpi, facecolor=COLORS['background'],
                   edgecolor='none', bbox_inches='tight')
    plt.close(figure)
    return str(save_path)


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)
    create_structure_analysis(plots / 'structure_analysis.png')


if __name__ == '__main__':
    main()
