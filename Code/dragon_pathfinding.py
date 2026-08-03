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
from matplotlib.colors import to_rgba
from matplotlib.patches import (
    FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, RegularPolygon,
)
import numpy as np

from core.dragon import (
    DRAGON_EDGES,
    DRAGON_NODES,
    STATE_ORDER,
    perch_probability,
    scripted_showcase,
    simulate_perch_trajectory,
)
from core.end_visuals import (
    draw_central_island,
    draw_end_fountain,
    draw_end_spikes,
    set_crystals_alive,
)
from core.rendering import optimize_gif
from core.style import COLORS, STATE_COLORS, apply_style


apply_style()


def _arena_static(
    ax, seed=42, compact=False, axis_off=True, limits=88,
    island_alpha=0.52,
):
    ax.set_xlim(-limits, limits)
    ax.set_ylim(-limits, limits)
    ax.set_aspect('equal')
    ax.set_facecolor(COLORS['background'])
    if axis_off:
        ax.axis('off')
    else:
        ax.set_xlabel('X (blocks)')
        ax.set_ylabel('Z (blocks)')
        ax.grid(color=COLORS['grid'], alpha=0.22, linewidth=0.45)
        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
        ax.tick_params(colors=COLORS['muted'], labelsize=8)

    draw_central_island(
        ax, seed=seed, extent=limits, alpha=island_alpha, zorder=0,
    )

    for start, end in DRAGON_EDGES:
        ax.plot(
            [DRAGON_NODES[start, 0], DRAGON_NODES[end, 0]],
            [DRAGON_NODES[start, 1], DRAGON_NODES[end, 1]],
            color=COLORS['magenta'], linewidth=0.48 if compact else 0.72,
            alpha=0.25 if compact else 0.34, zorder=1,
        )

    ax.scatter(
        DRAGON_NODES[:, 0], DRAGON_NODES[:, 1],
        s=8 if compact else 13, c=COLORS['end_stone'], alpha=0.58,
        edgecolors=COLORS['end_shadow'], linewidths=0.22, zorder=2,
    )
    spikes = draw_end_spikes(ax, seed=seed, crystals_alive=10, zorder=4)
    draw_end_fountain(ax, active=False, zorder=7)
    ax.set_xlim(-limits, limits)
    ax.set_ylim(-limits, limits)
    return spikes


def _state_island_vertices(center, width, height, seed):
    random = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, 14, endpoint=False)
    jitter = random.uniform(0.88, 1.12, len(angles))
    return np.column_stack((
        center[0] + width * 0.5 * np.cos(angles) * jitter,
        center[1] + height * 0.5 * np.sin(angles) * jitter,
    ))


def _draw_state_machine(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_facecolor(COLORS['background'])

    positions = {
        'holding': (0.50, 0.90),
        'strafing': (0.20, 0.74),
        'charging': (0.80, 0.74),
        'landing_approach': (0.50, 0.64),
        'landing': (0.50, 0.48),
        'perching': (0.50, 0.32),
        'takeoff': (0.80, 0.48),
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
            mutation_scale=8, color=COLORS['magenta'], linewidth=0.85,
            connectionstyle='arc3,rad=0.08', alpha=0.48,
            shrinkA=19, shrinkB=19,
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
    for state_index, state in enumerate(STATE_ORDER):
        position = positions[state]
        vertices = _state_island_vertices(
            position, 0.235 if state != 'landing_approach' else 0.275,
            0.104, 8100 + state_index,
        )
        shadow = Polygon(
            vertices + np.array([0.0, -0.012]), closed=True,
            facecolor=COLORS['end_shadow'], edgecolor='none',
            alpha=0.34, zorder=3,
        )
        node = Polygon(
            vertices, closed=True,
            facecolor=to_rgba(COLORS['purpur'], 0.42),
            edgecolor=STATE_COLORS[state], linewidth=1.2, zorder=4,
        )
        ax.add_patch(shadow)
        ax.add_patch(node)
        ax.text(
            position[0], position[1], labels[state],
            ha='center', va='center', color=COLORS['text'],
            fontsize=9.3, fontweight='black', family='monospace',
            zorder=5,
        )
        nodes[state] = node

    hud = FancyBboxPatch(
        (0.09, 0.025), 0.82, 0.165,
        boxstyle='round,pad=0.012,rounding_size=0.025',
        facecolor=to_rgba(COLORS['panel'], 0.94),
        edgecolor=COLORS['purpur'], linewidth=0.9, zorder=4,
    )
    ax.add_patch(hud)
    ax.text(
        0.13, 0.148, 'PERCH CHANCE', ha='left', va='center',
        color=COLORS['muted'], fontsize=6.2, family='monospace', zorder=5,
    )
    probability_background = Rectangle(
        (0.34, 0.135), 0.40, 0.026,
        facecolor=COLORS['grid'], edgecolor=COLORS['muted'],
        linewidth=0.35, alpha=0.82, zorder=5,
    )
    probability_fill = Rectangle(
        (0.34, 0.135), 0.0, 0.026,
        facecolor=COLORS['magenta'], edgecolor='none', zorder=6,
    )
    ax.add_patch(probability_background)
    ax.add_patch(probability_fill)
    probability_text = ax.text(
        0.87, 0.148, '', ha='right', va='center',
        color=COLORS['text'], fontsize=6.3, family='monospace', zorder=6,
    )
    ax.text(
        0.13, 0.075, 'END CRYSTALS', ha='left', va='center',
        color=COLORS['muted'], fontsize=6.2, family='monospace', zorder=5,
    )
    crystal_icons = []
    for index in range(10):
        x = 0.36 + index * 0.051
        icon = RegularPolygon(
            (x, 0.075), numVertices=4, radius=0.012,
            orientation=np.pi / 4.0, facecolor=COLORS['magenta'],
            edgecolor=COLORS['text'], linewidth=0.32, zorder=6,
        )
        ax.add_patch(icon)
        crystal_icons.append(icon)
    return nodes, crystal_icons, probability_fill, probability_text


def create_dragon_pathfinding_animation(
    save_path, fps=12, dpi=125, colors=128,
):
    """Create the shorter README hero animation."""
    frames = scripted_showcase()
    figure = plt.figure(figsize=(12.8, 7.2), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[1.68, 0.82],
        left=0.025, right=0.985, top=0.975, bottom=0.025, wspace=0.025,
    )
    arena = figure.add_subplot(grid[0, 0])
    machine = figure.add_subplot(grid[0, 1])
    spike_artists = _arena_static(arena)
    state_nodes, crystal_icons, probability_fill, probability_text = (
        _draw_state_machine(machine)
    )

    trail = LineCollection([], linewidths=2.8, capstyle='round', zorder=9)
    arena.add_collection(trail)
    active = arena.scatter(
        [], [], s=112, marker='D', c=COLORS['blue'],
        edgecolors=COLORS['text'], linewidths=1.1, zorder=11,
    )
    direction = arena.scatter(
        [], [], s=45, marker='>', c=COLORS['text'],
        edgecolors='none', zorder=12,
    )

    history = []

    def update(frame_index):
        frame = frames[frame_index]
        if frame_index == 0:
            history.clear()
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
            node.set_facecolor(
                to_rgba(STATE_COLORS[state], 0.82)
                if is_active else to_rgba(COLORS['purpur'], 0.42)
            )
            node.set_edgecolor(COLORS['text'] if is_active else STATE_COLORS[state])
            node.set_linewidth(2.2 if is_active else 1.3)

        probability = perch_probability(frame.crystals_alive)
        probability_fill.set_width(0.40 * probability / (1.0 / 3.0))
        probability_text.set_text(
            f'{probability * 100:4.1f}%  1/(3+{frame.crystals_alive})'
        )
        for index, icon in enumerate(crystal_icons):
            alive = index < frame.crystals_alive
            color = COLORS['magenta'] if alive else COLORS['coral']
            alpha = 0.95 if alive else 0.18
            icon.set_facecolor(color)
            icon.set_alpha(alpha)
        set_crystals_alive(spike_artists, frame.crystals_alive)
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
            if frame_index == 0:
                history.clear()
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
    save_path, seed=12031, trajectories=420, fps=3, frames=180,
):
    """Animate a slower accumulation of dragon approaches and occupancy."""
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
    _arena_static(
        axis, seed=42, compact=True, axis_off=False,
        limits=88, island_alpha=0.42,
    )

    image = axis.imshow(
        np.zeros_like(cumulative[0]), origin='lower',
        extent=(bins[0], bins[-1], bins[0], bins[-1]),
        cmap='magma', vmin=0, vmax=np.sqrt(cumulative[-1]).max(),
        interpolation='bilinear', alpha=0.74, zorder=2.8,
    )
    lines = LineCollection(
        [], linewidths=0.62, colors=COLORS['cyan'],
        alpha=0.12, zorder=8,
    )
    axis.add_collection(lines)
    count_text = axis.text(
        0.985, 0.025, '', transform=axis.transAxes,
        ha='right', va='bottom', color=COLORS['muted'],
        fontsize=8, family='monospace',
        bbox=dict(
            boxstyle='round,pad=0.32', facecolor=COLORS['panel'],
            edgecolor=COLORS['purpur'], alpha=0.90,
        ),
    )

    active_frames = max(1, round(frames * 0.82))

    def update(frame_index):
        progress = min((frame_index + 1) / active_frames, 1.0)
        shown = max(1, round(progress * trajectories))
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
    create_dragon_pathfinding_animation(plots / 'dragon_pathfinding_hero.gif')
    create_dragon_detail_clips(plots)
    create_trajectory_ensemble_animation(plots / 'dragon_trajectory_ensemble.gif')


if __name__ == '__main__':
    main()
