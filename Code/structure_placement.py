"""Exact Java 1.16.1 village candidate placement animation."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
import numpy as np

from core.constants import VILLAGE_SPACING
from core.minecraft_visuals import draw_minecraft_terrain
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
    save_path, seed=42, region_radius=4, fps=8, duration=12,
):
    regions = spiral_regions(region_radius)
    candidates = [
        candidate_in_region(seed, region_x, region_z, VILLAGE)
        for region_x, region_z in regions
    ]
    total_frames = int(fps * duration)
    spacing = VILLAGE_SPACING
    minimum = -region_radius * spacing - 5
    maximum = (region_radius + 1) * spacing + 5

    figure, axis = plt.subplots(figsize=(12.0, 6.75), facecolor=COLORS['background'])
    figure.subplots_adjust(left=0.075, right=0.98, top=0.875, bottom=0.10)
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
        dimension='overworld', resolution=256, alpha=0.88,
    )

    for region_x in range(-region_radius, region_radius + 1):
        for region_z in range(-region_radius, region_radius + 1):
            axis.add_patch(Rectangle(
                (region_x * spacing, region_z * spacing), spacing, spacing,
                facecolor='none', edgecolor=COLORS['text'],
                linewidth=0.52, alpha=0.34, zorder=2,
            ))

    axis.axhline(0, color=COLORS['muted'], linewidth=0.55, alpha=0.42)
    axis.axvline(0, color=COLORS['muted'], linewidth=0.55, alpha=0.42)
    axis.scatter(
        [0], [0], marker='+', s=75, c=COLORS['text'],
        linewidths=1.0, zorder=6,
    )

    points = axis.scatter(
        [], [], s=47, c=COLORS['gold'], marker='s',
        edgecolors='#4C3512', linewidths=0.55,
        alpha=0.94, zorder=5,
    )
    current_region = Rectangle(
        (0, 0), spacing, spacing, fill=False,
        edgecolor=COLORS['cyan'], linewidth=1.55, alpha=0.0, zorder=7,
    )
    current_window = Rectangle(
        (0, 0), 24, 24, facecolor=COLORS['blue'],
        edgecolor=COLORS['cyan'], linewidth=1.15,
        linestyle='--', alpha=0.0, zorder=3,
    )
    excluded_right = Rectangle(
        (24, 0), 8, 32, facecolor=COLORS['coral'],
        edgecolor='none', alpha=0.0, zorder=3,
    )
    excluded_top = Rectangle(
        (0, 24), 24, 8, facecolor=COLORS['coral'],
        edgecolor='none', alpha=0.0, zorder=3,
    )
    axis.add_patch(current_window)
    axis.add_patch(excluded_right)
    axis.add_patch(excluded_top)
    axis.add_patch(current_region)
    current_point = axis.scatter(
        [], [], s=175, facecolors='none', edgecolors=COLORS['text'],
        marker='s', linewidths=1.3, zorder=8,
    )
    axis.text(
        0.018, 0.978, '32 x 32 REGION   24 x 24 RNG WINDOW',
        transform=axis.transAxes, ha='left', va='top',
        color=COLORS['text'], fontsize=13.5, fontweight='black', zorder=10,
        bbox=dict(
            boxstyle='square,pad=0.32', facecolor=COLORS['background'],
            edgecolor='none', alpha=0.80,
        ),
    )
    trace_text = axis.text(
        0.50, 0.025, '', transform=axis.transAxes,
        ha='center', va='bottom', color=COLORS['text'],
        fontsize=10.5, fontweight='bold', family='monospace', zorder=10,
        bbox=dict(
            boxstyle='round,pad=0.45', facecolor=COLORS['panel'],
            edgecolor=COLORS['cyan'], alpha=0.94,
        ),
    )
    axis.text(
        0.985, 0.975, 'gold = exact candidate\nterrain = illustrative',
        transform=axis.transAxes, ha='right', va='top',
        color=COLORS['muted'], fontsize=8.2, zorder=10,
        bbox=dict(
            boxstyle='round,pad=0.30', facecolor=COLORS['background'],
            edgecolor=COLORS['grid'], alpha=0.78,
        ),
    )
    figure.text(
        0.50, 0.936, 'VILLAGE CANDIDATE PLACEMENT   JAVA 1.16.1',
        ha='center', va='center', color=COLORS['text'],
        fontsize=17, fontweight='black',
    )

    def update(frame_index):
        progress = frame_index / max(total_frames - 1, 1)
        shown = min(len(candidates), max(1, round(progress * len(candidates))))
        visible = candidates[:shown]
        offsets = np.array([
            [item['chunk_x'], item['chunk_z']] for item in visible
        ])
        points.set_offsets(offsets)
        item = visible[-1]
        region_origin = (
            item['region_x'] * spacing,
            item['region_z'] * spacing,
        )
        current_region.set_xy(region_origin)
        current_region.set_alpha(0.95)
        current_window.set_xy(region_origin)
        current_window.set_alpha(0.18)
        excluded_right.set_xy((region_origin[0] + 24, region_origin[1]))
        excluded_top.set_xy((region_origin[0], region_origin[1] + 24))
        excluded_right.set_alpha(0.17)
        excluded_top.set_alpha(0.17)
        current_point.set_offsets(np.array([[item['chunk_x'], item['chunk_z']]]))
        trace_text.set_text(
            f"REGION ({item['region_x']:+03d},{item['region_z']:+03d})   "
            f"OFFSETS ({item['offset_x']:02d},{item['offset_z']:02d})   "
            f"CANDIDATE CHUNK ({item['chunk_x']:+04d},{item['chunk_z']:+04d})"
        )
        return []

    animation = FuncAnimation(
        figure, update, frames=total_frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(figure)
    optimize_gif(save_path, colors=80)
    return str(save_path)


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)
    create_structure_placement_animation(plots / 'structure_placement.gif')


if __name__ == '__main__':
    main()
