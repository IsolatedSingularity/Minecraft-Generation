"""Java 1.16.1 stronghold rings with high-noise triangulation spread."""

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
import numpy as np

from core.constants import STRONGHOLD_RINGS
from core.strongholds import generate_stronghold_candidates
from core.style import COLORS, addSoftShadow, apply_style, style_axis


apply_style()

RING_COLORS = [
    '#FF6B6B', '#FF9F43', '#F4C542', '#77C66E',
    '#45C8B7', '#4A90E2', '#9C8CF2', '#D780D6',
]


def _panel_label(ax, label):
    badge = ax.text(
        0.025, 0.965, label, transform=ax.transAxes,
        va='top', color=COLORS['blue'], fontsize=9.0, fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.35', facecolor=COLORS['panel'],
            edgecolor=COLORS['grid'], alpha=0.96,
        ),
        zorder=20,
    )
    addSoftShadow(badge.get_bbox_patch(), offset=(1.0, -1.0), alpha=0.16)


def _ring_band(ax, ring, color, alpha=0.08):
    ax.add_patch(Wedge(
        (0, 0), ring['max_radius'], 0, 360,
        width=ring['max_radius'] - ring['min_radius'],
        facecolor=color, edgecolor='none', alpha=alpha,
    ))
    ax.add_patch(Circle(
        (0, 0), ring['min_radius'], fill=False,
        edgecolor=color, linewidth=0.5, linestyle=':', alpha=0.48,
    ))
    ax.add_patch(Circle(
        (0, 0), ring['max_radius'], fill=False,
        edgecolor=color, linewidth=0.7, alpha=0.66,
    ))


def _line_intersection(point_a, angle_a, point_b, angle_b):
    direction_a = np.array([math.cos(angle_a), math.sin(angle_a)])
    direction_b = np.array([math.cos(angle_b), math.sin(angle_b)])
    matrix = np.column_stack((direction_a, -direction_b))
    determinant = np.linalg.det(matrix)
    if abs(determinant) < 1e-7:
        return None
    scale = np.linalg.solve(matrix, point_b - point_a)[0]
    return point_a + scale * direction_a


def _triangulation_samples(
    target, seed=42, samples=1800, sigma_degrees=1.2,
):
    target = np.asarray(target, dtype=float)
    perpendicular = np.array([-target[1], target[0]])
    perpendicular /= max(np.linalg.norm(perpendicular), 1.0)
    throw_a = perpendicular * -240.0
    throw_b = perpendicular * 240.0
    angle_a = math.atan2(*(target - throw_a)[::-1])
    angle_b = math.atan2(*(target - throw_b)[::-1])
    random = np.random.default_rng(seed)
    sigma = math.radians(sigma_degrees)
    intersections = []
    for _ in range(samples):
        point = _line_intersection(
            throw_a, angle_a + random.normal(0.0, sigma),
            throw_b, angle_b + random.normal(0.0, sigma),
        )
        if point is not None and np.linalg.norm(point - target) < 2500:
            intersections.append(point)
    return np.asarray(intersections)


def create_stronghold_distribution(save_path, dpi=220, seed=42):
    candidates = generate_stronghold_candidates(seed)
    first_ring = candidates[:3]
    target_item = min(first_ring, key=lambda item: item['radius'])
    target = np.array([target_item['x'], target_item['z']], dtype=float)
    intersections = _triangulation_samples(target, seed)

    figure = plt.figure(figsize=(15.5, 7.8), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[1.15, 1.0],
        left=0.055, right=0.98, top=0.91, bottom=0.10, wspace=0.16,
    )
    ring_axis = figure.add_subplot(grid[0, 0])
    noise_axis = figure.add_subplot(grid[0, 1])

    max_radius = STRONGHOLD_RINGS[-1]['max_radius'] + 1700
    for index, ring in enumerate(STRONGHOLD_RINGS):
        _ring_band(ring_axis, ring, RING_COLORS[index])
        subset = [item for item in candidates if item['ring'] == index + 1]
        ring_axis.scatter(
            [item['x'] for item in subset],
            [item['z'] for item in subset],
            s=30 if index else 56, c=RING_COLORS[index],
            edgecolors=COLORS['panel'] if index == 0 else 'none',
            linewidths=0.65, alpha=0.94, zorder=5,
        )
        angle = math.radians(24 + index * 1.8)
        midpoint = (ring['min_radius'] + ring['max_radius']) / 2.0
        ring_axis.text(
            midpoint * math.cos(angle), midpoint * math.sin(angle),
            f'R{index + 1}', color=RING_COLORS[index], fontsize=7,
            ha='center', va='center', fontweight='bold',
        )
    for item in first_ring:
        ring_axis.plot(
            [0, item['x']], [0, item['z']],
            color=COLORS['gold'], linewidth=0.65, alpha=0.46,
        )
    ring_axis.scatter(
        [0], [0], marker='+', s=82, c=COLORS['text'], linewidths=1.0,
    )
    ring_axis.set_xlim(-max_radius, max_radius)
    ring_axis.set_ylim(-max_radius, max_radius)
    ring_axis.set_xlabel('Block X')
    ring_axis.set_ylabel('Block Z')
    ring_axis.set_title('Stronghold candidate rings', loc='left', pad=12, fontsize=11)
    style_axis(ring_axis, equal=True, grid=True)
    _panel_label(ring_axis, 'A')

    distances = np.linalg.norm(intersections - target, axis=1)
    confidence = np.exp(-distances / max(float(np.percentile(distances, 80)), 1.0))
    point_colors = np.empty((len(intersections), 4))
    blue_rgba = np.array(plt.matplotlib.colors.to_rgba(COLORS['blue']))
    green_rgba = np.array(plt.matplotlib.colors.to_rgba(COLORS['green']))
    point_colors[:, :3] = (
        confidence[:, None] * green_rgba[:3]
        + (1.0 - confidence[:, None]) * blue_rgba[:3]
    )
    point_colors[:, 3] = 0.18 + 0.34 * confidence
    noise_axis.scatter(
        intersections[:, 0], intersections[:, 1], s=12,
        c=point_colors, edgecolors='none', rasterized=True, zorder=3,
    )
    noise_axis.add_patch(Circle(
        target, 112, fill=False, edgecolor=COLORS['gold'],
        linewidth=1.15, linestyle='--', alpha=0.95, zorder=6,
    ))
    noise_axis.scatter(
        [target[0]], [target[1]], s=96, c=COLORS['coral'], marker='*',
        edgecolors=COLORS['panel'], linewidths=0.75, zorder=7,
    )
    median = np.median(intersections, axis=0)
    noise_axis.scatter(
        [median[0]], [median[1]], s=72, marker='X',
        c=COLORS['green'], edgecolors=COLORS['panel'],
        linewidths=0.75, zorder=8,
    )
    zoom_radius = max(520.0, float(np.percentile(distances, 97.5)))
    noise_axis.set_xlim(target[0] - zoom_radius, target[0] + zoom_radius)
    noise_axis.set_ylim(target[1] - zoom_radius, target[1] + zoom_radius)
    noise_axis.set_xlabel('Estimated block X')
    noise_axis.set_ylabel('Estimated block Z')
    noise_axis.set_title(
        'High-noise triangulation spread', loc='left', pad=12, fontsize=11,
    )
    style_axis(noise_axis, equal=True, grid=True)
    _panel_label(noise_axis, 'B')
    noise_axis.text(
        0.975, 0.965, r'$\sigma_\theta = 1.2^\circ$  |  1,800 trials',
        transform=noise_axis.transAxes, ha='right', va='top',
        color=COLORS['muted'], fontsize=7.8,
    )
    legend = noise_axis.legend(
        handles=[
            Line2D(
                [0], [0], marker='o', linestyle='None',
                markerfacecolor=COLORS['blue'], markeredgecolor='none',
                markersize=6, label='Noisy intersections',
            ),
            Line2D(
                [0], [0], marker='*', linestyle='None',
                markerfacecolor=COLORS['coral'], markeredgecolor=COLORS['panel'],
                markersize=9, label='True candidate',
            ),
            Line2D(
                [0], [0], marker='X', linestyle='None',
                markerfacecolor=COLORS['green'], markeredgecolor=COLORS['panel'],
                markersize=7, label='Median estimate',
            ),
            Line2D(
                [0], [0], color=COLORS['gold'], linestyle='--',
                label='112-block biome search radius',
            ),
        ],
        loc='lower right', fontsize=7.8, borderpad=0.8, labelspacing=0.65,
    )
    addSoftShadow(legend.get_frame(), offset=(1.6, -1.6), alpha=0.18)

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
