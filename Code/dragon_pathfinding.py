"""Java 1.16.1 Ender Dragon pathfinding visualizations.

The source path-node geometry, adjacency masks, and holding-phase probability
rolls are exact. Continuous top-down motion between source targets is a
reduced-order interpolation for legibility.
"""

from pathlib import Path
from itertools import combinations
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, RegularPolygon, Wedge
import numpy as np

from core.constants import END_PILLAR_RADIUS
from core.dragon import (
    DRAGON_EDGES,
    DRAGON_NODES,
    STATE_ORDER,
    perch_probability,
    scripted_showcase,
    simulate_perch_trajectory,
)
from core.end_generation import spike_layout
from core.rendering import optimize_gif
from core.style import COLORS, STATE_COLORS, addSoftShadow, apply_style, style_axis


apply_style()


def _arena_static(ax, seed=42, compact=False):
    ax.set_xlim(-76, 76)
    ax.set_ylim(-76, 76)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['panel'])
    addSoftShadow(ax.patch, offset=(2.0, -2.0), alpha=0.14)

    ax.add_patch(Circle(
        (0, 0), 72, facecolor=COLORS['end_stone'],
        edgecolor='none', alpha=0.13,
    ))
    ax.add_patch(Circle(
        (0, 0), 7.5, facecolor=COLORS['obsidian'],
        edgecolor=COLORS['portal'], linewidth=1.2, alpha=0.95, zorder=7,
    ))

    for start, end in DRAGON_EDGES:
        ax.plot(
            [DRAGON_NODES[start, 0], DRAGON_NODES[end, 0]],
            [DRAGON_NODES[start, 1], DRAGON_NODES[end, 1]],
            color=COLORS['grid'], linewidth=0.55 if compact else 0.7,
            alpha=0.42, zorder=1,
        )

    node_colors = [
        COLORS['blue'] if index < 12
        else COLORS['violet'] if index < 20
        else COLORS['cyan']
        for index in range(24)
    ]
    ax.scatter(
        DRAGON_NODES[:, 0], DRAGON_NODES[:, 1],
        s=9 if compact else 13, c=node_colors, alpha=0.55,
        linewidths=0, zorder=2,
    )

    crystals = []
    for spike in spike_layout(seed):
        marker_size = 26 + 5 * spike['radius']
        ax.scatter(
            [spike['x']], [spike['z']], s=marker_size,
            c=COLORS['obsidian'], edgecolors=COLORS['grid'],
            linewidths=0.7, zorder=4,
        )
        crystal = ax.scatter(
            [spike['x']], [spike['z']], s=18,
            c=COLORS['green'], marker='D', edgecolors=COLORS['text'],
            linewidths=0.35, zorder=5,
        )
        crystals.append(crystal)
    return crystals


def _draw_state_machine(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor(COLORS['panel'])

    card = FancyBboxPatch(
        (0.025, 0.035), 0.95, 0.93,
        boxstyle='round,pad=0.012,rounding_size=0.035',
        transform=ax.transAxes, facecolor=COLORS['panel'],
        edgecolor=COLORS['grid'], linewidth=0.75, zorder=-10,
    )
    ax.add_patch(card)
    addSoftShadow(card, offset=(2.2, -2.2), alpha=0.20)

    center = np.array([0.50, 0.53])
    node_radius = 0.355
    angles = np.linspace(
        np.pi / 2.0, np.pi / 2.0 + 2.0 * np.pi,
        len(STATE_ORDER), endpoint=False,
    )
    positions = {
        state: center + node_radius * np.array([np.cos(angle), np.sin(angle)])
        for state, angle in zip(STATE_ORDER, angles)
    }
    position_array = np.asarray([positions[state] for state in STATE_ORDER])

    ax.add_patch(Circle(
        center, node_radius + 0.042, fill=False,
        edgecolor=COLORS['grid'], linewidth=1.0, alpha=0.75, zorder=0,
    ))

    transitions = [
        ('holding', 'strafing'), ('strafing', 'holding'),
        ('holding', 'charging'), ('charging', 'holding'),
        ('holding', 'landing_approach'),
        ('landing_approach', 'landing'),
        ('landing', 'perching'), ('perching', 'takeoff'),
        ('takeoff', 'holding'),
    ]
    for start, end in transitions:
        arrow = FancyArrowPatch(
            positions[start], positions[end], arrowstyle='-|>',
            mutation_scale=7, color=COLORS['grid'], linewidth=0.8,
            connectionstyle='arc3,rad=0.08', alpha=0.82,
            shrinkA=15, shrinkB=15, zorder=1,
        )
        ax.add_patch(arrow)

    connection_pairs = list(combinations(range(len(STATE_ORDER)), 2))
    connection_segments = [
        [position_array[left], position_array[right]]
        for left, right in connection_pairs
    ]
    energy_links = LineCollection(
        connection_segments, colors=COLORS['blue'], linewidths=1.15,
        alpha=0.36, capstyle='round', zorder=2,
    )
    ax.add_collection(energy_links)

    labels = {
        'holding': 'HOLDING',
        'strafing': 'STRAFE',
        'charging': 'CHARGE',
        'landing_approach': 'APPROACH',
        'landing': 'LAND',
        'perching': 'PERCH',
        'takeoff': 'TAKEOFF',
    }
    nodes = {}
    for state, angle in zip(STATE_ORDER, angles):
        position = positions[state]
        node = Circle(
            position, 0.047, facecolor=COLORS['panel'],
            edgecolor=STATE_COLORS[state], linewidth=1.5, zorder=5,
        )
        ax.add_patch(node)
        addSoftShadow(node, offset=(1.2, -1.2), alpha=0.20)
        label_position = center + (node_radius + 0.082) * np.array([
            np.cos(angle), np.sin(angle),
        ])
        ax.text(
            label_position[0], label_position[1], labels[state],
            ha='center', va='center', color=COLORS['muted'],
            fontsize=6.6, fontweight='bold', zorder=6,
        )
        nodes[state] = node

    crystal_icons = []
    for index in range(10):
        angle = np.pi / 2.0 + 2.0 * np.pi * index / 10.0
        point = center + 0.185 * np.array([np.cos(angle), np.sin(angle)])
        icon = RegularPolygon(
            point, numVertices=4, radius=0.018, orientation=np.pi / 4.0,
            facecolor=COLORS['green'], edgecolor=COLORS['panel'],
            linewidth=0.6, zorder=7,
        )
        ax.add_patch(icon)
        addSoftShadow(icon, offset=(0.8, -0.8), alpha=0.18)
        crystal_icons.append(icon)

    probability_background = Wedge(
        center, 0.088, 90, 450, width=0.014,
        facecolor=COLORS['grid'], edgecolor='none', alpha=0.9, zorder=7,
    )
    probability_wedge = Wedge(
        center, 0.088, 90, 90, width=0.014,
        facecolor=COLORS['green'], edgecolor='none', alpha=0.98, zorder=8,
    )
    ax.add_patch(probability_background)
    ax.add_patch(probability_wedge)
    probability_text = ax.text(
        center[0], center[1] + 0.008, '', ha='center', va='center',
        color=COLORS['text'], fontsize=8.2, fontweight='bold', zorder=9,
    )
    ax.text(
        center[0], center[1] - 0.038, 'PERCH', ha='center', va='center',
        color=COLORS['muted'], fontsize=5.7, fontweight='bold', zorder=9,
    )
    crystal_count_text = ax.text(
        0.50, 0.072, '', ha='center', va='center',
        color=COLORS['muted'], fontsize=7.2, fontweight='bold', zorder=9,
    )
    return (
        nodes, crystal_icons, probability_wedge, probability_text,
        energy_links, position_array, crystal_count_text,
    )

def create_dragon_pathfinding_animation(
    save_path, fps=12, dpi=125, colors=128,
):
    """Create the shorter README hero animation."""
    frames = scripted_showcase()
    figure = plt.figure(figsize=(12.8, 7.2), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[1.72, 0.78],
        left=0.025, right=0.985, top=0.975, bottom=0.025, wspace=0.025,
    )
    arena = figure.add_subplot(grid[0, 0])
    machine = figure.add_subplot(grid[0, 1])
    crystal_markers = _arena_static(arena)
    (
        state_nodes, crystal_icons, probability_wedge, probability_text,
        energy_links, state_positions, crystal_count_text,
    ) = _draw_state_machine(machine)
    connection_pairs = list(combinations(range(len(STATE_ORDER)), 2))

    trail = LineCollection([], linewidths=2.8, capstyle='round', zorder=9)
    arena.add_collection(trail)
    active = arena.scatter(
        [], [], s=105, marker='o', c=COLORS['blue'],
        edgecolors=COLORS['text'], linewidths=1.1, zorder=11,
    )
    direction = arena.scatter(
        [], [], s=45, marker='>', c=COLORS['text'],
        edgecolors='none', zorder=12,
    )

    history = []

    def update(frame_index):
        frame = frames[frame_index]
        history.append(frame.position.copy())
        if len(history) > 34:
            history.pop(0)
        active.set_offsets(frame.position.reshape(1, 2))
        active.set_facecolor(STATE_COLORS[frame.state])
        if len(history) > 1:
            points = np.asarray(history)
            segments = np.stack([points[:-1], points[1:]], axis=1)
            alphas = np.linspace(0.08, 0.88, len(segments))
            base = plt.matplotlib.colors.to_rgba(STATE_COLORS[frame.state])
            colors_value = [(*base[:3], alpha) for alpha in alphas]
            trail.set_segments(segments)
            trail.set_colors(colors_value)
            vector = points[-1] - points[-2]
            angle = np.degrees(np.arctan2(vector[1], vector[0]))
            direction.set_paths([
                plt.matplotlib.markers.MarkerStyle('>').get_path().transformed(
                    plt.matplotlib.transforms.Affine2D().rotate_deg(angle)
                )
            ])
        direction.set_offsets(frame.position.reshape(1, 2))

        for state, node in state_nodes.items():
            is_active = state == frame.state
            node.set_facecolor(STATE_COLORS[state] if is_active else COLORS['panel'])
            node.set_edgecolor(COLORS['text'] if is_active else STATE_COLORS[state])
            node.set_linewidth(2.2 if is_active else 1.3)

        probability = perch_probability(frame.crystals_alive)
        probability_wedge.set_theta2(90 + 360 * probability / (1.0 / 3.0))
        probability_text.set_text(f'{probability * 100:.0f}%')
        crystal_count_text.set_text(f'{frame.crystals_alive} CRYSTALS')
        active_index = STATE_ORDER.index(frame.state)
        ordered_pairs = sorted(
            connection_pairs,
            key=lambda pair: (
                active_index not in pair,
                -min((pair[1] - pair[0]) % len(STATE_ORDER),
                     (pair[0] - pair[1]) % len(STATE_ORDER)),
                pair,
            ),
        )
        connection_count = round(
            len(ordered_pairs) * frame.crystals_alive / 10.0
        )
        energy_links.set_segments([
            [state_positions[left], state_positions[right]]
            for left, right in ordered_pairs[:connection_count]
        ])
        energy_links.set_alpha(
            0.18 + 0.24 * frame.crystals_alive / 10.0
        )
        for index, (icon, marker) in enumerate(zip(crystal_icons, crystal_markers)):
            alive = index < frame.crystals_alive
            color = COLORS['green'] if alive else COLORS['coral']
            alpha = 0.95 if alive else 0.18
            icon.set_facecolor(color)
            icon.set_alpha(alpha)
            marker.set_facecolor(color)
            marker.set_alpha(alpha)
        return []

    animation = FuncAnimation(
        figure, update, frames=len(frames), interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)
    optimize_gif(save_path, colors=colors)
    return str(save_path)


def _clip_ranges(frames):
    approach = next(i for i, frame in enumerate(frames) if frame.state == 'landing_approach')
    takeoff = next(i for i, frame in enumerate(frames) if frame.state == 'takeoff')
    return {
        'dragon_holding_strafe.gif': frames[:approach],
        'dragon_landing_perch.gif': frames[approach:takeoff],
        'dragon_takeoff.gif': frames[takeoff:],
    }


def create_dragon_detail_clips(output_dir, fps=12, dpi=100):
    """Create compact zoomed state clips for the README detail section."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, clip in _clip_ranges(scripted_showcase()).items():
        figure, arena = plt.subplots(figsize=(9.6, 5.4), facecolor=COLORS['background'])
        _arena_static(arena, compact=True)
        trail = LineCollection([], linewidths=3.2, capstyle='round', zorder=9)
        arena.add_collection(trail)
        active = arena.scatter(
            [], [], s=115, c=COLORS['blue'], edgecolors=COLORS['text'],
            linewidths=1.0, zorder=11,
        )
        state_text = arena.text(
            0.035, 0.045, '', transform=arena.transAxes,
            color=COLORS['text'], fontsize=10, fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.35', facecolor=COLORS['panel'],
                edgecolor=COLORS['grid'], alpha=0.92,
            ),
        )
        history = []

        def update(frame_index):
            frame = clip[frame_index]
            history.append(frame.position.copy())
            if len(history) > 28:
                history.pop(0)
            active.set_offsets(frame.position.reshape(1, 2))
            active.set_facecolor(STATE_COLORS[frame.state])
            state_text.set_text(frame.state.replace('_', ' ').upper())
            state_text.get_bbox_patch().set_edgecolor(STATE_COLORS[frame.state])
            if len(history) > 1:
                points = np.asarray(history)
                trail.set_segments(np.stack([points[:-1], points[1:]], axis=1))
                base = plt.matplotlib.colors.to_rgba(STATE_COLORS[frame.state])
                trail.set_colors([
                    (*base[:3], alpha)
                    for alpha in np.linspace(0.08, 0.9, len(points) - 1)
                ])
            return []

        animation = FuncAnimation(
            figure, update, frames=len(clip), interval=1000 / fps, blit=False,
        )
        path = output_dir / name
        animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(figure)
        optimize_gif(path, colors=96)
        outputs.append(str(path))
    return outputs


def create_trajectory_ensemble_animation(
    save_path, seed=12031, trajectories=420, fps=12, frames=192,
):
    """Animate a gradual accumulation of seeded approaches and occupancy."""
    paths = [
        simulate_perch_trajectory(
            seed + index * 7919,
            crystals_alive=10 - (index % 6),
            player_position=(34.0, -18.0),
        )[0]
        for index in range(trajectories)
    ]
    bins = np.linspace(-76, 76, 93)
    contributions = []
    for path in paths:
        histogram, _, _ = np.histogram2d(
            path[:, 1], path[:, 0], bins=(bins, bins),
        )
        contributions.append(histogram)
    cumulative = np.cumsum(np.asarray(contributions), axis=0)

    figure, axis = plt.subplots(
        figsize=(9.6, 6.4), facecolor=COLORS['background'],
    )
    figure.subplots_adjust(left=0.11, right=0.94, top=0.95, bottom=0.11)
    axis.set_xlim(-76, 76)
    axis.set_ylim(-76, 76)
    axis.set_xlabel('X (blocks)')
    axis.set_ylabel('Z (blocks)')
    style_axis(axis, equal=True, grid=True)
    axis.add_patch(Circle(
        (0, 0), 72, facecolor=COLORS['end_stone'],
        edgecolor='none', alpha=0.18, zorder=0,
    ))
    for start, end in DRAGON_EDGES:
        axis.plot(
            [DRAGON_NODES[start, 0], DRAGON_NODES[end, 0]],
            [DRAGON_NODES[start, 1], DRAGON_NODES[end, 1]],
            color=COLORS['grid'], linewidth=0.55, alpha=0.56, zorder=1,
        )
    axis.add_patch(Circle(
        (0, 0), 7.5, fill=False, edgecolor=COLORS['portal'],
        linewidth=1.25, alpha=0.95, zorder=7,
    ))
    axis.add_patch(Circle(
        (0, 0), 2.0, fill=False, edgecolor=COLORS['blue'],
        linewidth=1.0, linestyle=':', alpha=0.95, zorder=8,
    ))

    density_map = LinearSegmentedColormap.from_list(
        'iosTrajectoryDensity',
        [COLORS['panel'], '#CBEAFF', COLORS['cyan'], COLORS['blue'], COLORS['violet']],
    )
    empty_density = np.zeros_like(cumulative[0])
    final_density = np.sqrt(cumulative[-1])
    positive_density = final_density[final_density > 0]
    density_ceiling = float(np.percentile(positive_density, 94))
    image = axis.imshow(
        empty_density, origin='lower',
        extent=(bins[0], bins[-1], bins[0], bins[-1]),
        cmap=density_map,
        norm=PowerNorm(gamma=0.52, vmin=0, vmax=density_ceiling),
        interpolation='bilinear', alpha=0.80, zorder=2,
    )
    history_lines = LineCollection(
        [], linewidths=0.38, colors=COLORS['blue'], alpha=0.055, zorder=4,
    )
    recent_lines = LineCollection(
        [], linewidths=0.90, colors=COLORS['green'], alpha=0.48, zorder=5,
    )
    axis.add_collection(history_lines)
    axis.add_collection(recent_lines)
    count_text = axis.text(
        0.975, 0.035, '', transform=axis.transAxes,
        ha='right', va='bottom', color=COLORS['muted'],
        fontsize=8.2, fontweight='bold',
        bbox=dict(
            boxstyle='round,pad=0.42', facecolor=COLORS['panel'],
            edgecolor=COLORS['grid'], alpha=0.96,
        ),
    )
    addSoftShadow(count_text.get_bbox_patch(), offset=(1.3, -1.3), alpha=0.18)

    def update(frame_index):
        normalized = frame_index / max(frames - 1, 1)
        reveal = np.clip((normalized - 0.04) / 0.86, 0.0, 1.0)
        shown = min(trajectories, round((reveal ** 1.20) * trajectories))
        if shown == 0:
            image.set_data(empty_density)
            history_lines.set_segments([])
            recent_lines.set_segments([])
        else:
            image.set_data(np.sqrt(cumulative[shown - 1]))
            history_start = max(0, shown - 30)
            recent_start = max(0, shown - 4)
            history_lines.set_segments(paths[history_start:recent_start])
            recent_lines.set_segments(paths[recent_start:shown])
        count_text.set_text(f'{shown:03d} SEEDED APPROACHES')
        return []

    animation = FuncAnimation(
        figure, update, frames=frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(figure)
    optimize_gif(save_path, colors=128)
    return str(save_path)


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)
    create_dragon_pathfinding_animation(plots / 'dragon_pathfinding.gif')
    create_dragon_detail_clips(plots)
    create_trajectory_ensemble_animation(plots / 'dragon_trajectory_ensemble.gif')


if __name__ == '__main__':
    main()
