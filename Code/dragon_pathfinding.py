"""Java 1.16.1 Ender Dragon pathfinding visualizations.

The source path-node geometry, adjacency masks, and holding-phase probability
rolls are exact. Continuous top-down motion between source targets is a
reduced-order interpolation for legibility.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, Wedge
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
from core.style import COLORS, STATE_COLORS, apply_style


apply_style()


def _arena_static(ax, seed=42, compact=False):
    ax.set_xlim(-76, 76)
    ax.set_ylim(-76, 76)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(COLORS['background'])

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
    ax.set_facecolor(COLORS['background'])

    positions = {
        'holding': (0.50, 0.88),
        'strafing': (0.20, 0.70),
        'charging': (0.80, 0.70),
        'landing_approach': (0.50, 0.58),
        'landing': (0.50, 0.40),
        'perching': (0.50, 0.22),
        'takeoff': (0.80, 0.38),
    }
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
            mutation_scale=8, color=COLORS['grid'], linewidth=0.9,
            connectionstyle='arc3,rad=0.08', alpha=0.85,
            shrinkA=17, shrinkB=17,
        )
        ax.add_patch(arrow)

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
    for state in STATE_ORDER:
        position = positions[state]
        node = Circle(
            position, 0.052, facecolor=COLORS['panel'],
            edgecolor=STATE_COLORS[state], linewidth=1.4, zorder=4,
        )
        ax.add_patch(node)
        ax.text(
            position[0], position[1] - 0.078, labels[state],
            ha='center', va='top', color=COLORS['muted'],
            fontsize=7.2, fontweight='normal',
        )
        nodes[state] = node

    crystal_icons = []
    for index in range(10):
        x = 0.13 + index * 0.082
        icon = Circle(
            (x, 0.055), 0.013, facecolor=COLORS['green'],
            edgecolor=COLORS['text'], linewidth=0.35,
        )
        ax.add_patch(icon)
        crystal_icons.append(icon)

    probability_background = Wedge(
        (0.50, 0.12), 0.050, 90, 450, width=0.010,
        facecolor=COLORS['grid'], edgecolor='none', alpha=0.8,
    )
    probability_wedge = Wedge(
        (0.50, 0.12), 0.050, 90, 90, width=0.010,
        facecolor=COLORS['cyan'], edgecolor='none', alpha=0.95,
    )
    ax.add_patch(probability_background)
    ax.add_patch(probability_wedge)
    probability_text = ax.text(
        0.50, 0.12, '', ha='center', va='center',
        color=COLORS['text'], fontsize=6.7, family='monospace',
    )
    return nodes, crystal_icons, probability_wedge, probability_text


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
    state_nodes, crystal_icons, probability_wedge, probability_text = (
        _draw_state_machine(machine)
    )

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
    save_path, seed=12031, trajectories=420, fps=12, frames=108,
):
    """Animate accumulated source-shaped dragon approaches and occupancy."""
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
        histogram, _, _ = np.histogram2d(path[:, 1], path[:, 0], bins=(bins, bins))
        contributions.append(histogram)
    cumulative = np.cumsum(np.asarray(contributions), axis=0)

    figure, axis = plt.subplots(figsize=(9.6, 6.4), facecolor=COLORS['background'])
    axis.set_xlim(-76, 76)
    axis.set_ylim(-76, 76)
    axis.set_aspect('equal')
    axis.set_xlabel('X (blocks)')
    axis.set_ylabel('Z (blocks)')
    axis.grid(color=COLORS['grid'], alpha=0.24, linewidth=0.45)
    for spine in axis.spines.values():
        spine.set_color(COLORS['grid'])
    for start, end in DRAGON_EDGES:
        axis.plot(
            [DRAGON_NODES[start, 0], DRAGON_NODES[end, 0]],
            [DRAGON_NODES[start, 1], DRAGON_NODES[end, 1]],
            color=COLORS['grid'], linewidth=0.45, alpha=0.24, zorder=1,
        )
    axis.add_patch(Circle(
        (0, 0), 7.5, fill=False, edgecolor=COLORS['portal'],
        linewidth=1.2, alpha=0.9, zorder=7,
    ))
    axis.add_patch(Circle(
        (0, 0), 2.0, fill=False, edgecolor=COLORS['cyan'],
        linewidth=1.0, linestyle=':', alpha=0.95, zorder=8,
    ))

    image = axis.imshow(
        np.zeros_like(cumulative[0]), origin='lower',
        extent=(bins[0], bins[-1], bins[0], bins[-1]),
        cmap='magma', vmin=0, vmax=np.sqrt(cumulative[-1]).max(),
        interpolation='bilinear', alpha=0.78, zorder=2,
    )
    lines = LineCollection([], linewidths=0.55, colors=COLORS['cyan'], alpha=0.10, zorder=4)
    axis.add_collection(lines)
    count_text = axis.text(
        0.985, 0.025, '', transform=axis.transAxes,
        ha='right', va='bottom', color=COLORS['muted'],
        fontsize=8, family='monospace',
    )

    def update(frame_index):
        shown = max(1, round((frame_index + 1) * trajectories / frames))
        image.set_data(np.sqrt(cumulative[shown - 1]))
        recent_start = max(0, shown - 55)
        lines.set_segments(paths[recent_start:shown])
        count_text.set_text(f'{shown:03d} seeded approaches')
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
