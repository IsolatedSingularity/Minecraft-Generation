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
from matplotlib.path import Path as MarkerPath
from matplotlib.patches import (
    Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle,
)
import numpy as np
from scipy.ndimage import maximum_filter

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
    set_crystal_states,
)
from core.rendering import optimize_gif
from core.style import COLORS, STATE_COLORS, apply_style


apply_style()


DRAGON_MARKER = MarkerPath(
    np.array([
        [1.00, 0.00], [0.55, 0.18], [0.24, 0.14], [0.03, 0.72],
        [-0.22, 0.78], [-0.08, 0.18], [-0.78, 0.35], [-1.00, 0.00],
        [-0.78, -0.35], [-0.08, -0.18], [-0.22, -0.78], [0.03, -0.72],
        [0.24, -0.14], [0.55, -0.18], [1.00, 0.00],
    ]),
    [MarkerPath.MOVETO] + [MarkerPath.LINETO] * 13 + [MarkerPath.CLOSEPOLY],
)


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
            color='#7C8391', linewidth=0.50 if compact else 0.76,
            alpha=0.24 if compact else 0.38, zorder=1,
        )

    ax.scatter(
        DRAGON_NODES[:, 0], DRAGON_NODES[:, 1],
        s=8 if compact else 13, c=COLORS['end_stone'], alpha=0.58,
        edgecolors=COLORS['end_shadow'], linewidths=0.22, zorder=2,
    )
    spikes = draw_end_spikes(
        ax, seed=seed, crystals_alive=10, zorder=4,
        tower_edgecolor='#3D334A', cage_linewidth=1.55, cage_extent=3.25,
    )
    draw_end_fountain(ax, active=False, zorder=7)
    ax.set_xlim(-limits, limits)
    ax.set_ylim(-limits, limits)
    return spikes


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
        ('holding', 'strafing', 0.12),
        ('strafing', 'holding', 0.12),
        ('holding', 'charging', -0.12),
        ('charging', 'holding', -0.12),
        ('holding', 'landing_approach', 0.0),
        ('landing_approach', 'landing', 0.0),
        ('landing', 'perching', 0.0),
        ('perching', 'takeoff', 0.0),
        ('takeoff', 'holding', 0.06),
    ]
    for start, end, curvature in transitions:
        connection = FancyArrowPatch(
            positions[start], positions[end], arrowstyle='-|>',
            mutation_scale=12.0, color='#65458C', linewidth=1.65,
            connectionstyle=f'arc3,rad={curvature}', alpha=0.82,
            shrinkA=30, shrinkB=30, zorder=2,
        )
        ax.add_patch(connection)

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
    node_width = 0.245
    node_height = 0.095
    for state in STATE_ORDER:
        position = positions[state]
        shadow = Ellipse(
            (position[0], position[1] - 0.012),
            node_width, node_height,
            facecolor=COLORS['end_shadow'], edgecolor='none',
            alpha=0.34, zorder=3,
        )
        node = Ellipse(
            position, node_width, node_height,
            facecolor=to_rgba(STATE_COLORS[state], 0.62),
            edgecolor=STATE_COLORS[state], linewidth=1.25, zorder=4,
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
        (0.055, 0.018), 0.89, 0.190,
        boxstyle='round,pad=0.012,rounding_size=0.025',
        facecolor=to_rgba(COLORS['panel'], 0.94),
        edgecolor='#65458C', linewidth=1.15, zorder=4,
    )
    ax.add_patch(hud)
    ax.text(
        0.09, 0.160, 'PERCH CHANCE', ha='left', va='center',
        color=COLORS['muted'], fontsize=7.1, family='monospace', zorder=5,
    )
    probability_background = FancyBboxPatch(
        (0.34, 0.145), 0.40, 0.030,
        boxstyle='round,pad=0.002,rounding_size=0.014',
        facecolor=COLORS['grid'], edgecolor=COLORS['muted'],
        linewidth=0.35, alpha=0.82, zorder=5,
    )
    probability_fill = FancyBboxPatch(
        (0.34, 0.145), 0.002, 0.030,
        boxstyle='round,pad=0.002,rounding_size=0.014',
        facecolor='#65458C', edgecolor='none', zorder=6,
    )
    ax.add_patch(probability_background)
    ax.add_patch(probability_fill)
    probability_text = ax.text(
        0.91, 0.160, '', ha='right', va='center',
        color=COLORS['text'], fontsize=7.0, family='monospace', zorder=6,
    )
    ax.text(
        0.09, 0.076, 'END CRYSTALS', ha='left', va='center',
        color=COLORS['muted'], fontsize=7.1, family='monospace', zorder=5,
    )
    crystal_icons = []
    for index in range(10):
        x = 0.36 + index * 0.052
        icon = Circle(
            (x, 0.076), radius=0.014,
            facecolor='#9B5DE5',
            edgecolor=COLORS['text'], linewidth=0.32, zorder=6,
        )
        ax.add_patch(icon)
        crystal_icons.append(icon)
    return nodes, crystal_icons, probability_fill, probability_text


def create_dragon_pathfinding_animation(
    save_path, fps=10, dpi=100, colors=96,
):
    """Create the shorter README hero animation."""
    frames = scripted_showcase()
    if len(frames) > 270:
        indices = np.linspace(0, len(frames) - 1, 270).round().astype(int)
        frames = [frames[index] for index in indices]
    figure = plt.figure(figsize=(12.8, 7.2), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[1.76, 0.94],
        left=0.018, right=0.988, top=0.982, bottom=0.018, wspace=0.0,
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
        [], [], s=235, marker=DRAGON_MARKER, c=COLORS['blue'],
        edgecolors=COLORS['text'], linewidths=1.1, zorder=11,
    )
    fireball_glow = arena.scatter(
        [], [], s=160, marker='o', c=COLORS['orange'],
        edgecolors='none', alpha=0.22, visible=False, zorder=12,
    )
    fireball_core = arena.scatter(
        [], [], s=38, marker='o', c=COLORS['gold'],
        edgecolors=COLORS['text'], linewidths=0.45,
        visible=False, zorder=13,
    )
    explosion = Circle(
        (0, 0), 0.1, fill=False, edgecolor=COLORS['gold'],
        linewidth=2.2, alpha=0.0, zorder=14,
    )
    arena.add_patch(explosion)

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
            active.set_paths([
                DRAGON_MARKER.transformed(
                    plt.matplotlib.transforms.Affine2D().rotate_deg(angle)
                )
            ])

        if frame.fireball_position is not None:
            offset = np.asarray(frame.fireball_position).reshape(1, 2)
            fireball_glow.set_offsets(offset)
            fireball_core.set_offsets(offset)
            fireball_glow.set_visible(True)
            fireball_core.set_visible(True)
        else:
            fireball_glow.set_visible(False)
            fireball_core.set_visible(False)

        if frame.explosion_index is not None:
            spike = spike_artists[frame.explosion_index]
            explosion.center = (spike['x'], spike['z'])
            explosion.set_radius(2.0 + 8.5 * frame.explosion_phase)
            explosion.set_alpha(0.95 * (1.0 - frame.explosion_phase) + 0.12)
        else:
            explosion.set_alpha(0.0)

        for state, node in state_nodes.items():
            is_active = state == frame.state
            node.set_facecolor(to_rgba(STATE_COLORS[state], 0.84 if is_active else 0.60))
            node.set_edgecolor(COLORS['text'] if is_active else STATE_COLORS[state])
            node.set_linewidth(3.0 if is_active else 1.25)

        probability = perch_probability(frame.crystals_alive)
        probability_fill.set_width(max(0.004, 0.40 * probability / (1.0 / 3.0)))
        probability_text.set_text(
            f'{probability * 100:4.1f}%  1/(3+{frame.crystals_alive})'
        )
        alive_indices = (
            set(frame.alive_crystals)
            if frame.alive_crystals is not None
            else set(range(frame.crystals_alive))
        )
        for index, icon in enumerate(crystal_icons):
            alive = index in alive_indices
            color = '#9B5DE5' if alive else COLORS['coral']
            alpha = 0.95 if alive else 0.18
            icon.set_facecolor(color)
            icon.set_alpha(alpha)
        set_crystal_states(spike_artists, alive_indices)
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
        if len(clip) > 72:
            indices = np.linspace(0, len(clip) - 1, 72).round().astype(int)
            clip = [clip[index] for index in indices]
        figure, arena = plt.subplots(figsize=(9.6, 5.4), facecolor=COLORS['background'])
        _arena_static(arena, compact=True)
        trail = LineCollection([], linewidths=3.2, capstyle='round', zorder=9)
        arena.add_collection(trail)
        active = arena.scatter(
            [], [], s=210, marker=DRAGON_MARKER,
            c=COLORS['blue'], edgecolors=COLORS['text'],
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
                vector = points[-1] - points[-2]
                angle = np.degrees(np.arctan2(vector[1], vector[0]))
                active.set_paths([
                    DRAGON_MARKER.transformed(
                        plt.matplotlib.transforms.Affine2D().rotate_deg(angle)
                    )
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
    save_path, seed=12031, trajectories=240, fps=8, frames=144,
):
    """Animate dragon approaches with distinct-trajectory intersection counts."""
    paths = [
        simulate_perch_trajectory(
            seed + index * 7919,
            crystals_alive=10 - (index % 6),
            player_position=(34.0, -18.0),
        )[0]
        for index in range(trajectories)
    ]
    bins = np.linspace(-76, 76, 77)
    contributions = []
    for path in paths:
        histogram, _, _ = np.histogram2d(path[:, 1], path[:, 0], bins=(bins, bins))
        contributions.append(histogram > 0)
    cumulative = np.cumsum(np.asarray(contributions), axis=0)

    final_frequency = cumulative[-1]
    local_maxima = final_frequency == maximum_filter(final_frequency, size=5, mode='nearest')
    local_maxima &= final_frequency >= 4
    hotspot_rows, hotspot_columns = np.nonzero(local_maxima)
    ranking = np.argsort(final_frequency[hotspot_rows, hotspot_columns])[::-1]
    selected = []
    for candidate in ranking:
        row = hotspot_rows[candidate]
        column = hotspot_columns[candidate]
        center_x = (bins[column] + bins[column + 1]) / 2.0
        center_z = (bins[row] + bins[row + 1]) / 2.0
        if np.hypot(center_x, center_z) <= 8.0:
            continue
        if any(
            (row - hotspot_rows[other]) ** 2
            + (column - hotspot_columns[other]) ** 2 < 49
            for other in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == 10:
            break
    selected = np.asarray(selected, dtype=int)
    hotspot_rows = hotspot_rows[selected]
    hotspot_columns = hotspot_columns[selected]
    hotspot_values = final_frequency[hotspot_rows, hotspot_columns]
    hotspot_x = (bins[hotspot_columns] + bins[hotspot_columns + 1]) / 2.0
    hotspot_z = (bins[hotspot_rows] + bins[hotspot_rows + 1]) / 2.0

    figure = plt.figure(figsize=(13.4, 7.2), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[1.58, 1.0],
        left=0.065, right=0.98, top=0.90, bottom=0.17, wspace=0.24,
    )
    axis = figure.add_subplot(grid[0, 0])
    frequency_axis = figure.add_subplot(grid[0, 1])
    _arena_static(
        axis, seed=42, compact=True, axis_off=False,
        limits=88, island_alpha=0.42,
    )

    image = axis.imshow(
        np.zeros_like(cumulative[0]), origin='lower',
        extent=(bins[0], bins[-1], bins[0], bins[-1]),
        cmap='magma', vmin=0, vmax=np.sqrt(final_frequency).max(),
        interpolation='bilinear', alpha=0.74, zorder=2.8,
    )
    lines = LineCollection(
        [], linewidths=0.78, colors=COLORS['cyan'], zorder=8,
    )
    axis.add_collection(lines)
    local_trail = LineCollection([], linewidths=3.1, capstyle='round', zorder=10)
    axis.add_collection(local_trail)
    dragon = axis.scatter(
        [], [], s=245, marker=DRAGON_MARKER,
        c='#7E57C2', edgecolors=COLORS['text'], linewidths=1.15, zorder=12,
    )
    count_text = axis.text(
        0.985, 0.025, '', transform=axis.transAxes,
        ha='right', va='bottom', color=COLORS['muted'],
        fontsize=8, family='monospace',
        bbox=dict(
            boxstyle='round,pad=0.32', facecolor=COLORS['panel'],
            edgecolor=COLORS['purpur'], alpha=0.90,
        ),
    )
    bars = frequency_axis.barh(
        np.arange(len(hotspot_values)), np.zeros(len(hotspot_values)),
        color=plt.get_cmap('plasma')(np.linspace(0.28, 0.88, len(hotspot_values))),
        edgecolor=COLORS['text'], linewidth=0.45, alpha=0.92,
    )
    frequency_axis.set_yticks(
        np.arange(len(hotspot_values)),
        [f'({x:+.0f}, {z:+.0f})' for x, z in zip(hotspot_x, hotspot_z)],
    )
    frequency_axis.invert_yaxis()
    frequency_axis.set_xlim(0, max(hotspot_values) * 1.12)
    frequency_axis.set_xlabel('Distinct trajectories entering the cell')
    frequency_axis.set_ylabel('Critical X, Z cell (blocks)')
    frequency_axis.set_title('Intersection frequency at critical flight cells', fontsize=11, pad=8)
    frequency_axis.grid(axis='x', color=COLORS['grid'], alpha=0.35, linewidth=0.55)
    for spine in frequency_axis.spines.values():
        spine.set_color(COLORS['grid'])
    frequency_axis.tick_params(colors=COLORS['muted'], labelsize=7.7)
    figure.text(
        0.665, 0.055,
        'Each trajectory contributes at most once per cell.\nHigh bars mark repeatable approach corridors.',
        color=COLORS['muted'],
        fontsize=7.4, ha='left', va='top', linespacing=1.35,
    )

    figure.suptitle(
        'TRAJECTORY DISTRIBUTION AND DEGENERACY',
        color=COLORS['text'], fontsize=17, fontweight='black', y=0.97,
    )

    active_frames = max(1, round(frames * 0.88))

    def update(frame_index):
        progress = min((frame_index + 1) / active_frames, 1.0)
        shown = max(1, round(progress * trajectories))
        current_frequency = cumulative[shown - 1]
        image.set_data(np.sqrt(current_frequency))
        recent_start = max(0, shown - 28)
        recent_paths = paths[recent_start:shown]
        lines.set_segments(recent_paths)
        age = np.linspace(0.08, 1.0, len(recent_paths))
        line_colors = [
            to_rgba(plt.get_cmap('plasma')(0.14 + 0.72 * value),
                    0.05 + 0.36 * value)
            for value in age
        ]
        lines.set_colors(line_colors)

        feature_frames = 16
        feature_count = max(1, int(np.ceil(frames / feature_frames)))
        feature_slot = min(frame_index // feature_frames, feature_count - 1)
        featured_index = round(
            feature_slot * (trajectories - 1) / max(feature_count - 1, 1)
        )
        featured = paths[featured_index]
        local_phase = (frame_index % feature_frames) / max(feature_frames - 1, 1)
        point_index = min(len(featured) - 1, round(local_phase * (len(featured) - 1)))
        point = featured[point_index]
        dragon.set_offsets(point.reshape(1, 2))
        trail_start = max(0, point_index - 22)
        local_points = featured[trail_start:point_index + 1]
        if len(local_points) > 1:
            segments = np.stack([local_points[:-1], local_points[1:]], axis=1)
            local_trail.set_segments(segments)
            local_trail.set_colors([
                to_rgba(plt.get_cmap('plasma')(0.20 + 0.68 * value), 0.20 + 0.78 * value)
                for value in np.linspace(0.0, 1.0, len(segments))
            ])
            vector = local_points[-1] - local_points[-2]
            angle = np.degrees(np.arctan2(vector[1], vector[0]))
            dragon.set_paths([
                DRAGON_MARKER.transformed(
                    plt.matplotlib.transforms.Affine2D().rotate_deg(angle)
                )
            ])
        else:
            local_trail.set_segments([])

        for bar, row, column in zip(bars, hotspot_rows, hotspot_columns):
            bar.set_width(current_frequency[row, column])
        count_text.set_text(f'{shown:03d} seeded approaches')
        return []

    animation = FuncAnimation(
        figure, update, frames=frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=92)
    plt.close(figure)
    optimize_gif(save_path, colors=112)
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
