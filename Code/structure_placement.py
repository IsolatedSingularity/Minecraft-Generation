"""Exact Java 1.16.1 village candidate placement animation."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np

from core.constants import VILLAGE_SPACING
from core.rendering import optimize_gif
from core.structures import VILLAGE, candidate_in_region
from core.style import COLORS, apply_style


apply_style()


def spiral_regions(radius):
    regions = [
        (x, z) for x in range(-radius, radius + 1)
        for z in range(-radius, radius + 1)
    ]
    return sorted(regions, key=lambda item: (
        max(abs(item[0]), abs(item[1])),
        np.arctan2(item[1], item[0]),
    ))


def create_structure_placement_animation(
    save_path, seed=42, region_radius=5, fps=12, duration=9,
):
    regions = spiral_regions(region_radius)
    candidates = [
        candidate_in_region(seed, region_x, region_z, VILLAGE)
        for region_x, region_z in regions
    ]
    total_frames = int(fps * duration)
    spacing = VILLAGE_SPACING
    limit = (region_radius + 0.65) * spacing

    figure, axis = plt.subplots(figsize=(12.8, 7.2), facecolor=COLORS['background'])
    figure.subplots_adjust(left=0.09, right=0.92, top=0.965, bottom=0.10)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_aspect('equal')
    axis.set_xlabel('Chunk X')
    axis.set_ylabel('Chunk Z')
    axis.tick_params(colors=COLORS['muted'], labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(COLORS['grid'])

    for region_x in range(-region_radius, region_radius + 1):
        for region_z in range(-region_radius, region_radius + 1):
            color = COLORS['panel'] if (region_x + region_z) % 2 == 0 else COLORS['panel_alt']
            axis.add_patch(Rectangle(
                (region_x * spacing, region_z * spacing), spacing, spacing,
                facecolor=color, edgecolor=COLORS['grid'],
                linewidth=0.55, alpha=0.66, zorder=0,
            ))

    axis.axhline(0, color=COLORS['muted'], linewidth=0.55, alpha=0.42)
    axis.axvline(0, color=COLORS['muted'], linewidth=0.55, alpha=0.42)
    axis.scatter(
        [0], [0], marker='+', s=75, c=COLORS['text'],
        linewidths=1.0, zorder=6,
    )

    points = axis.scatter(
        [], [], s=32, c=[], cmap='viridis',
        norm=Normalize(0, np.sqrt(2) * 23),
        edgecolors=COLORS['text'], linewidths=0.28,
        alpha=0.88, zorder=5,
    )
    current_region = Rectangle(
        (0, 0), spacing, spacing, fill=False,
        edgecolor=COLORS['cyan'], linewidth=1.55, alpha=0.0, zorder=7,
    )
    current_window = Rectangle(
        (0, 0), 24, 24, facecolor=COLORS['blue'],
        edgecolor=COLORS['cyan'], linewidth=0.8,
        linestyle='--', alpha=0.0, zorder=2,
    )
    axis.add_patch(current_window)
    axis.add_patch(current_region)
    current_point = axis.scatter(
        [], [], s=120, facecolors='none', edgecolors=COLORS['cyan'],
        linewidths=1.2, zorder=8,
    )

    inset = axis.inset_axes([0.735, 0.035, 0.23, 0.23])
    inset.set_xlim(0, 32)
    inset.set_ylim(0, 32)
    inset.set_aspect('equal')
    inset.set_facecolor(COLORS['panel'])
    inset.add_patch(Rectangle(
        (0, 0), 24, 24, facecolor=COLORS['blue'],
        edgecolor=COLORS['cyan'], linewidth=0.9, alpha=0.17,
    ))
    inset.axvline(24, color=COLORS['cyan'], linewidth=0.65, linestyle='--')
    inset.axhline(24, color=COLORS['cyan'], linewidth=0.65, linestyle='--')
    inset.set_xticks([0, 8, 16, 24, 32])
    inset.set_yticks([0, 8, 16, 24, 32])
    inset.tick_params(colors=COLORS['muted'], labelsize=6)
    for spine in inset.spines.values():
        spine.set_color(COLORS['grid'])
    inset_point = inset.scatter(
        [], [], s=70, c=COLORS['gold'],
        edgecolors=COLORS['text'], linewidths=0.55, zorder=4,
    )
    inset_x_line, = inset.plot([], [], color=COLORS['gold'], linewidth=0.7, alpha=0.7)
    inset_z_line, = inset.plot([], [], color=COLORS['gold'], linewidth=0.7, alpha=0.7)

    legend = figure.add_axes([0.145, 0.025, 0.56, 0.035])
    legend.axis('off')
    legend.set_xlim(0, 1)
    legend.set_ylim(0, 1)
    entries = [
        (0.03, COLORS['grid'], '32 x 32 region'),
        (0.38, COLORS['blue'], '24 x 24 candidate window'),
        (0.79, COLORS['gold'], 'Java RNG candidate'),
    ]
    for x, color, label in entries:
        legend.scatter([x], [0.5], s=40, c=color, marker='s')
        legend.text(x + 0.035, 0.5, label, va='center',
                    color=COLORS['muted'], fontsize=7.2)

    def update(frame_index):
        progress = frame_index / max(total_frames - 1, 1)
        shown = min(len(candidates), max(1, round(progress * len(candidates))))
        visible = candidates[:shown]
        offsets = np.array([
            [item['chunk_x'], item['chunk_z']] for item in visible
        ])
        offset_magnitude = np.array([
            np.hypot(item['offset_x'], item['offset_z']) for item in visible
        ])
        points.set_offsets(offsets)
        points.set_array(offset_magnitude)
        item = visible[-1]
        region_origin = (
            item['region_x'] * spacing,
            item['region_z'] * spacing,
        )
        current_region.set_xy(region_origin)
        current_region.set_alpha(0.95)
        current_window.set_xy(region_origin)
        current_window.set_alpha(0.12)
        current_point.set_offsets(np.array([[item['chunk_x'], item['chunk_z']]]))
        inset_point.set_offsets(np.array([[item['offset_x'], item['offset_z']]]))
        inset_x_line.set_data([0, item['offset_x']], [item['offset_z'], item['offset_z']])
        inset_z_line.set_data([item['offset_x'], item['offset_x']], [0, item['offset_z']])
        return []

    animation = FuncAnimation(
        figure, update, frames=total_frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=125)
    plt.close(figure)
    optimize_gif(save_path, colors=96)
    return str(save_path)


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)
    create_structure_placement_animation(plots / 'structure_placement.gif')


if __name__ == '__main__':
    main()
