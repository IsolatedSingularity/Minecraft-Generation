"""Java 1.16.1 stronghold rings with a triangulation simulation."""

from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge
import numpy as np

from core.constants import STRONGHOLD_RINGS
from core.strongholds import generate_stronghold_candidates
from core.style import COLORS, apply_style, style_axis


apply_style()

RING_COLORS = [
    '#FB7185', '#F59E72', '#F6C85F', '#A5D66A',
    '#43D9C2', '#65C7F7', '#A78BFA', '#E879F9',
]


def _panel_label(ax, label):
    ax.text(0.02, 0.975, label, transform=ax.transAxes,
            va='top', color=COLORS['text'], fontsize=11, fontweight='bold')


def _ring_band(ax, ring, color, alpha=0.08):
    ax.add_patch(Wedge(
        (0, 0), ring['max_radius'], 0, 360,
        width=ring['max_radius'] - ring['min_radius'],
        facecolor=color, edgecolor='none', alpha=alpha,
    ))
    ax.add_patch(Circle(
        (0, 0), ring['min_radius'], fill=False,
        edgecolor=color, linewidth=0.5, linestyle=':', alpha=0.45,
    ))
    ax.add_patch(Circle(
        (0, 0), ring['max_radius'], fill=False,
        edgecolor=color, linewidth=0.7, alpha=0.62,
    ))


def _bearing_polygon(origin, angle, length, half_width):
    left = angle - half_width
    right = angle + half_width
    return np.array([
        origin,
        origin + length * np.array([math.cos(left), math.sin(left)]),
        origin + length * np.array([math.cos(right), math.sin(right)]),
    ])


def _line_intersection(point_a, angle_a, point_b, angle_b):
    direction_a = np.array([math.cos(angle_a), math.sin(angle_a)])
    direction_b = np.array([math.cos(angle_b), math.sin(angle_b)])
    matrix = np.column_stack((direction_a, -direction_b))
    determinant = np.linalg.det(matrix)
    if abs(determinant) < 1e-7:
        return None
    scale = np.linalg.solve(matrix, point_b - point_a)[0]
    return point_a + scale * direction_a


def _triangulation_samples(target, seed=42, samples=700, sigma_degrees=0.35):
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
        if point is not None and np.linalg.norm(point - target) < 1500:
            intersections.append(point)
    return throw_a, throw_b, angle_a, angle_b, np.asarray(intersections)


def create_stronghold_distribution(save_path, dpi=220, seed=42):
    candidates = generate_stronghold_candidates(seed)
    first_ring = candidates[:3]
    target_item = min(first_ring, key=lambda item: item['radius'])
    target = np.array([target_item['x'], target_item['z']], dtype=float)
    throw_a, throw_b, angle_a, angle_b, intersections = _triangulation_samples(target, seed)

    figure = plt.figure(figsize=(15.5, 9.4), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        2, 2, width_ratios=[1.72, 1.0], height_ratios=[1.0, 0.86],
        left=0.055, right=0.98, top=0.97, bottom=0.08,
        wspace=0.18, hspace=0.24,
    )
    ring_axis = figure.add_subplot(grid[:, 0])
    bearing_axis = figure.add_subplot(grid[0, 1])
    zoom_axis = figure.add_subplot(grid[1, 1])

    max_radius = STRONGHOLD_RINGS[-1]['max_radius'] + 1700
    for index, ring in enumerate(STRONGHOLD_RINGS):
        _ring_band(ring_axis, ring, RING_COLORS[index])
        subset = [item for item in candidates if item['ring'] == index + 1]
        ring_axis.scatter(
            [item['x'] for item in subset],
            [item['z'] for item in subset],
            s=30 if index else 56, c=RING_COLORS[index],
            edgecolors=COLORS['text'] if index == 0 else 'none',
            linewidths=0.45, alpha=0.92, zorder=5,
        )
        angle = math.radians(24 + index * 1.8)
        midpoint = (ring['min_radius'] + ring['max_radius']) / 2.0
        ring_axis.text(
            midpoint * math.cos(angle), midpoint * math.sin(angle),
            f'R{index + 1}', color=RING_COLORS[index], fontsize=7,
            ha='center', va='center',
        )
    for item in first_ring:
        ring_axis.plot([0, item['x']], [0, item['z']],
                       color=COLORS['gold'], linewidth=0.55, alpha=0.38)
    ring_axis.scatter([0], [0], marker='+', s=80, c=COLORS['text'], linewidths=1.0)
    ring_axis.set_xlim(-max_radius, max_radius)
    ring_axis.set_ylim(-max_radius, max_radius)
    ring_axis.set_xlabel('Block X')
    ring_axis.set_ylabel('Block Z')
    style_axis(ring_axis, equal=True, grid=True)
    _panel_label(ring_axis, '(a)')

    length = float(np.linalg.norm(target) * 1.18)
    half_width = math.radians(0.55)
    bearing_axis.add_patch(Polygon(
        _bearing_polygon(throw_a, angle_a, length, half_width),
        facecolor=COLORS['blue'], edgecolor='none', alpha=0.14,
    ))
    bearing_axis.add_patch(Polygon(
        _bearing_polygon(throw_b, angle_b, length, half_width),
        facecolor=COLORS['violet'], edgecolor='none', alpha=0.14,
    ))
    for point, angle, color in (
        (throw_a, angle_a, COLORS['blue']),
        (throw_b, angle_b, COLORS['violet']),
    ):
        bearing_axis.plot(
            [point[0], point[0] + length * math.cos(angle)],
            [point[1], point[1] + length * math.sin(angle)],
            color=color, linewidth=1.0, alpha=0.85,
        )
        bearing_axis.scatter([point[0]], [point[1]], s=52, c=color,
                             edgecolors=COLORS['text'], linewidths=0.4, zorder=5)
    bearing_axis.scatter([target[0]], [target[1]], s=70, c=COLORS['coral'],
                         edgecolors=COLORS['text'], linewidths=0.6, zorder=6)
    all_points = np.vstack([throw_a, throw_b, target])
    padding = 250
    bearing_axis.set_xlim(all_points[:, 0].min() - padding, all_points[:, 0].max() + padding)
    bearing_axis.set_ylim(all_points[:, 1].min() - padding, all_points[:, 1].max() + padding)
    bearing_axis.set_xlabel('Block X')
    bearing_axis.set_ylabel('Block Z')
    style_axis(bearing_axis, equal=True, grid=True)
    _panel_label(bearing_axis, '(b)')

    distances = np.linalg.norm(intersections - target, axis=1)
    zoom_axis.scatter(
        intersections[:, 0], intersections[:, 1], s=10,
        c=distances, cmap='magma', alpha=0.34, edgecolors='none',
    )
    zoom_axis.add_patch(Circle(
        target, 112, fill=False, edgecolor=COLORS['gold'],
        linewidth=1.0, linestyle='--', alpha=0.9,
    ))
    zoom_axis.scatter([target[0]], [target[1]], s=70, c=COLORS['coral'],
                      edgecolors=COLORS['text'], linewidths=0.6, zorder=5)
    median = np.median(intersections, axis=0)
    zoom_axis.scatter([median[0]], [median[1]], s=55, marker='x',
                      c=COLORS['cyan'], linewidths=1.2, zorder=6)
    zoom_radius = max(260.0, float(np.percentile(distances, 97)))
    zoom_axis.set_xlim(target[0] - zoom_radius, target[0] + zoom_radius)
    zoom_axis.set_ylim(target[1] - zoom_radius, target[1] + zoom_radius)
    zoom_axis.set_xlabel('Block X')
    zoom_axis.set_ylabel('Block Z')
    style_axis(zoom_axis, equal=True, grid=True)
    _panel_label(zoom_axis, '(c)')

    figure.savefig(save_path, dpi=dpi, facecolor=COLORS['background'],
                   edgecolor='none', bbox_inches='tight')
    plt.close(figure)
    return str(save_path)


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)
    create_stronghold_distribution(plots / 'stronghold_rings.png')


if __name__ == '__main__':
    main()
