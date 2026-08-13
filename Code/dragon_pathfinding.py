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
from matplotlib.colors import PowerNorm, to_rgba
from matplotlib.path import Path as MarkerPath
from matplotlib.patches import (
    Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, RegularPolygon,
)
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import maximum_filter

from core.dragon import (
    DRAGON_EDGES,
    DRAGON_NODES,
    EXCEPTION_PHASE_TRANSITIONS,
    SOURCE_PHASE_TRANSITIONS,
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


DRAGON_SPRITE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'pngfind.com-ender-dragon-png-6585528.png'
)


def _prepare_dragon_sprite():
    """Return a compact, tinted square sprite plus a soft alpha glow."""
    source = Image.open(DRAGON_SPRITE_PATH).convert('RGBA')
    alpha_box = source.getchannel('A').getbbox()
    if alpha_box is None:
        raise ValueError(f'dragon sprite has no visible pixels: {DRAGON_SPRITE_PATH}')
    source = source.crop(alpha_box)
    source.thumbnail((88, 108), Image.Resampling.LANCZOS)
    canvas = Image.new('RGBA', (120, 120), (0, 0, 0, 0))
    canvas.alpha_composite(
        source, ((canvas.width - source.width) // 2, (canvas.height - source.height) // 2),
    )
    rgba = np.asarray(canvas).astype(float) / 255.0
    luminance = np.mean(rgba[..., :3], axis=2, keepdims=True)
    violet = np.array([0.58, 0.27, 0.82])[None, None, :]
    cyan = np.array([0.30, 0.78, 0.94])[None, None, :]
    colour = (
        0.50 * rgba[..., :3]
        + 0.34 * violet
        + 0.16 * cyan * np.clip(luminance * 1.8, 0.0, 1.0)
    )
    rgba[..., :3] = np.clip(colour, 0.0, 1.0)
    tinted = Image.fromarray(np.uint8(np.clip(rgba, 0.0, 1.0) * 255), 'RGBA')
    glow_alpha = tinted.getchannel('A').filter(
        ImageFilter.MaxFilter(7)
    ).filter(ImageFilter.GaussianBlur(2.2))
    glow = Image.new('RGBA', tinted.size, (142, 77, 204, 0))
    glow.putalpha(glow_alpha.point(lambda value: int(value * 0.32)))
    return tinted, glow


class DragonSpriteArtist:
    """Rotating raster sprite with cached state tint and violet glow."""

    def __init__(self, axis, size_blocks=12.0, zorder=12):
        self.base, self.glow = _prepare_dragon_sprite()
        self.cache = {}
        blank = np.zeros((120, 120, 4), dtype=np.uint8)
        self.half_size = float(size_blocks) / 2.0
        self.glow_artist = axis.imshow(
            blank, extent=(-1, 1, -1, 1), origin='upper',
            interpolation='bilinear', zorder=zorder - 1,
        )
        self.artist = axis.imshow(
            blank, extent=(-1, 1, -1, 1), origin='upper',
            interpolation='bilinear', zorder=zorder,
        )

    def update(self, position, angle, state):
        quantized = int(round(float(angle) / 5.0) * 5) % 360
        key = (quantized, state)
        if key not in self.cache:
            rotation = quantized - 90
            sprite = self.base.rotate(
                rotation, resample=Image.Resampling.BICUBIC, expand=False,
            )
            glow = self.glow.rotate(
                rotation, resample=Image.Resampling.BICUBIC, expand=False,
            )
            rgba = np.asarray(sprite).astype(float) / 255.0
            state_rgb = np.asarray(to_rgba(STATE_COLORS[state])[:3])
            visible = rgba[..., 3:4]
            rgba[..., :3] = np.clip(
                rgba[..., :3] * 0.84 + state_rgb[None, None, :] * 0.16 * visible,
                0.0, 1.0,
            )
            self.cache[key] = (
                np.uint8(rgba * 255), np.asarray(glow),
            )
        sprite, glow = self.cache[key]
        x, z = np.asarray(position, dtype=float)
        extent = (
            x - self.half_size, x + self.half_size,
            z - self.half_size, z + self.half_size,
        )
        self.artist.set_data(sprite)
        self.artist.set_extent(extent)
        self.glow_artist.set_data(glow)
        self.glow_artist.set_extent(extent)

    def set_alpha(self, alpha):
        self.artist.set_alpha(alpha)
        self.glow_artist.set_alpha(alpha)


def _create_breath_artists(axis):
    cloud = Circle(
        (0, 0), 0.1, facecolor=to_rgba('#7428A8', 0.0),
        edgecolor='#D794FF', linewidth=1.2, linestyle='--', zorder=10.5,
    )
    axis.add_patch(cloud)
    particles = axis.scatter(
        [], [], s=14, c='#C875FF', edgecolors='#F1D7FF',
        linewidths=0.25, alpha=0.0, zorder=11,
    )
    stream = LineCollection([], linewidths=1.25, capstyle='round', zorder=10.8)
    axis.add_collection(stream)
    return cloud, particles, stream


def _update_breath_artists(frame, cloud, particles, stream):
    if frame.breath_center is None:
        cloud.set_alpha(0.0)
        particles.set_offsets(np.empty((0, 2)))
        stream.set_segments([])
        return
    center = np.asarray(frame.breath_center, dtype=float)
    radius = float(frame.breath_radius)
    alpha = float(frame.breath_alpha)
    cloud.center = center
    cloud.set_radius(radius)
    cloud.set_facecolor(to_rgba('#7428A8', alpha * 0.48))
    cloud.set_edgecolor(to_rgba('#D794FF', min(0.92, alpha + 0.24)))
    cloud.set_alpha(1.0)
    angles = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False)
    radial = radius * (0.28 + 0.62 * ((np.arange(18) * 7) % 17) / 16.0)
    offsets = center + np.column_stack((np.cos(angles), np.sin(angles))) * radial[:, None]
    particles.set_offsets(offsets)
    particles.set_alpha(min(0.90, alpha + 0.18))
    if frame.breath_kind == 'sitting_flame':
        origin = np.asarray(frame.position, dtype=float)
        perpendicular = np.array([0.0, 1.0])
        segments = []
        for offset in np.linspace(-1.8, 1.8, 7):
            segments.append([origin + perpendicular * offset * 0.25, center + perpendicular * offset])
        stream.set_segments(segments)
        stream.set_colors([
            to_rgba('#A84DE0', 0.16 + 0.08 * index)
            for index in range(len(segments))
        ])
    else:
        stream.set_segments([])


DRAGON_MARKER = MarkerPath(
    np.array([
        [1.30, 0.00], [1.08, 0.12], [1.18, 0.31], [0.95, 0.23],
        [0.67, 0.16], [0.39, 0.30], [0.02, 0.94], [-0.18, 1.04],
        [-0.43, 0.91], [-0.34, 0.55], [-0.19, 0.24], [-0.53, 0.12],
        [-1.82, 0.00], [-0.53, -0.12], [-0.19, -0.24], [-0.34, -0.55],
        [-0.43, -0.91], [-0.18, -1.04], [0.02, -0.94], [0.39, -0.30],
        [0.67, -0.16], [0.95, -0.23], [1.18, -0.31], [1.08, -0.12],
        [1.30, 0.00],
    ]),
    [MarkerPath.MOVETO] + [MarkerPath.LINETO] * 23 + [MarkerPath.CLOSEPOLY],
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
        ax.set_axisbelow(False)
        ax.grid(
            color='#59677E', alpha=0.44, linewidth=0.68,
            zorder=3.1,
        )
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
            color='#A4AEC0', linewidth=0.72 if compact else 1.02,
            alpha=0.42 if compact else 0.54, zorder=1,
        )

    ax.scatter(
        DRAGON_NODES[:, 0], DRAGON_NODES[:, 1],
        s=11 if compact else 17, c=COLORS['end_stone'], alpha=0.76,
        edgecolors=COLORS['end_shadow'], linewidths=0.22, zorder=2,
    )
    spikes = draw_end_spikes(
        ax, seed=seed, crystals_alive=10, zorder=4,
        tower_edgecolor='#786A8B', tower_linewidth=1.35,
        cage_linewidth=1.75, cage_extent=3.25,
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
        'holding': (0.50, 0.95),
        'strafing': (0.17, 0.85),
        'charging_player': (0.83, 0.85),
        'landing_approach': (0.50, 0.81),
        'landing': (0.50, 0.69),
        'sitting_scanning': (0.27, 0.58),
        'sitting_attacking': (0.27, 0.47),
        'sitting_flaming': (0.27, 0.36),
        'takeoff': (0.68, 0.47),
        'hover': (0.74, 0.60),
        'dying': (0.74, 0.34),
    }
    curvature = {
        ('holding', 'strafing'): 0.10,
        ('strafing', 'holding'): 0.10,
        ('holding', 'landing_approach'): 0.0,
        ('landing_approach', 'landing'): 0.0,
        ('landing', 'sitting_scanning'): 0.05,
        ('sitting_scanning', 'sitting_attacking'): 0.0,
        ('sitting_scanning', 'takeoff'): -0.08,
        ('sitting_scanning', 'charging_player'): -0.22,
        ('sitting_attacking', 'sitting_flaming'): 0.0,
        ('sitting_flaming', 'sitting_scanning'): 0.20,
        ('sitting_flaming', 'takeoff'): 0.04,
        ('takeoff', 'holding'): -0.32,
        ('charging_player', 'holding'): -0.10,
        ('hover', 'holding'): 0.30,
        ('holding', 'dying'): -0.36,
    }
    transitions = [
        (start, end, curvature[(start, end)], False)
        for start, end in SOURCE_PHASE_TRANSITIONS
    ] + [
        (start, end, curvature[(start, end)], True)
        for start, end, _ in EXCEPTION_PHASE_TRANSITIONS
    ]
    for start, end, curvature, exceptional in transitions:
        connection = FancyArrowPatch(
            positions[start], positions[end], arrowstyle='-|>',
            mutation_scale=10.5,
            color='#566785' if exceptional else '#7455A3',
            linewidth=1.0 if exceptional else 1.35,
            linestyle='--' if exceptional else '-',
            connectionstyle=f'arc3,rad={curvature}',
            alpha=0.48 if exceptional else 0.80,
            shrinkA=12, shrinkB=12, zorder=2,
        )
        ax.add_patch(connection)

    labels = {
        'holding': 'HOLDING',
        'strafing': 'STRAFE',
        'charging_player': 'CHARGE',
        'landing_approach': 'APPROACH',
        'landing': 'LAND',
        'takeoff': 'TAKEOFF',
        'sitting_flaming': 'BREATH',
        'sitting_scanning': 'SEARCH',
        'sitting_attacking': 'ROAR',
        'dying': 'DYING',
        'hover': 'HOVER',
    }
    nodes = {}
    exceptional_states = {'dying', 'hover'}
    node_width = 0.215
    node_height = 0.066
    for state in STATE_ORDER:
        position = positions[state]
        shadow = Ellipse(
            (position[0], position[1] - 0.008),
            node_width, node_height,
            facecolor=COLORS['end_shadow'], edgecolor='none',
            alpha=0.18 if state in exceptional_states else 0.30, zorder=3,
        )
        node = Ellipse(
            position, node_width, node_height,
            facecolor=to_rgba(
                STATE_COLORS[state], 0.24 if state in exceptional_states else 0.58,
            ),
            edgecolor=STATE_COLORS[state],
            linewidth=0.85 if state in exceptional_states else 1.15,
            zorder=4,
        )
        ax.add_patch(shadow)
        ax.add_patch(node)
        ax.text(
            position[0], position[1], labels[state],
            ha='center', va='center', color=COLORS['text'],
            fontsize=7.2, fontweight='black', family='monospace',
            alpha=0.66 if state in exceptional_states else 1.0,
            zorder=5,
        )
        nodes[state] = node

    hud = FancyBboxPatch(
        (0.035, 0.012), 0.93, 0.202,
        boxstyle='round,pad=0.012,rounding_size=0.022',
        facecolor=to_rgba('#171D2A', 0.99),
        edgecolor='#9B6FD1', linewidth=1.45, zorder=4,
    )
    ax.add_patch(hud)
    ax.add_patch(FancyBboxPatch(
        (0.051, 0.119), 0.898, 0.078,
        boxstyle='round,pad=0.006,rounding_size=0.014',
        facecolor=to_rgba('#20293A', 0.94), edgecolor='#3B475E',
        linewidth=0.65, zorder=4.5,
    ))
    ax.add_patch(FancyBboxPatch(
        (0.051, 0.030), 0.898, 0.071,
        boxstyle='round,pad=0.006,rounding_size=0.014',
        facecolor=to_rgba('#20293A', 0.94), edgecolor='#3B475E',
        linewidth=0.65, zorder=4.5,
    ))
    ax.text(
        0.071, 0.178, 'NEXT HOLDING-PATH LANDING ROLL',
        ha='left', va='center', color='#D7DDEA',
        fontsize=7.3, fontweight='black', family='DejaVu Sans', zorder=5,
    )
    ax.text(
        0.071, 0.145, 'evaluated once at path completion',
        ha='left', va='center', color='#8492AA',
        fontsize=6.1, fontweight='bold', family='DejaVu Sans', zorder=5,
    )
    probability_background = FancyBboxPatch(
        (0.385, 0.133), 0.445, 0.032,
        boxstyle='round,pad=0.002,rounding_size=0.010',
        facecolor='#0B0F17', edgecolor='#59677E',
        linewidth=0.60, alpha=1.0, zorder=5,
    )
    probability_fill = FancyBboxPatch(
        (0.385, 0.133), 0.002, 0.032,
        boxstyle='round,pad=0.002,rounding_size=0.010',
        facecolor='#A75DE1', edgecolor='none', zorder=6,
    )
    ax.add_patch(probability_background)
    ax.add_patch(probability_fill)
    probability_text = ax.text(
        0.925, 0.159, '', ha='right', va='center',
        color='#F4ECFF', fontsize=13.0, fontweight='black',
        family='DejaVu Sans', zorder=6,
    )
    probability_formula = ax.text(
        0.925, 0.132, '', ha='right', va='center',
        color='#8F9CB1', fontsize=6.3, family='DejaVu Sans', zorder=6,
    )
    crystal_label = ax.text(
        0.071, 0.066, 'ACTIVE END CRYSTALS', ha='left', va='center',
        color='#D7DDEA', fontsize=7.1, fontweight='black',
        family='DejaVu Sans', zorder=5,
    )
    crystal_icons = []
    for index in range(10):
        x = 0.405 + index * 0.050
        glow = Circle(
            (x, 0.066), radius=0.0195,
            facecolor='#A86BE0', edgecolor='none', alpha=0.24, zorder=5.5,
        )
        cage = RegularPolygon(
            (x, 0.066), numVertices=4, radius=0.0148,
            orientation=np.pi / 4.0, facecolor=to_rgba('#121824', 0.88),
            edgecolor='#8390A4', linewidth=0.48, zorder=6,
        )
        icon = RegularPolygon(
            (x, 0.066), numVertices=4, radius=0.0088,
            orientation=0.0, facecolor='#C875FF',
            edgecolor='#F4ECFF', linewidth=0.38, zorder=6.2,
        )
        ax.add_patch(glow)
        ax.add_patch(cage)
        ax.add_patch(icon)
        crystal_icons.append({'glow': glow, 'cage': cage, 'core': icon})
    return (
        nodes, crystal_icons, probability_fill, probability_text,
        probability_formula, crystal_label,
    )


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
    (
        state_nodes, crystal_icons, probability_fill, probability_text,
        probability_formula, crystal_label,
    ) = _draw_state_machine(machine)

    trail = LineCollection([], linewidths=2.8, capstyle='round', zorder=9)
    arena.add_collection(trail)
    active = DragonSpriteArtist(arena, size_blocks=11.0, zorder=13)
    fireball_glow = arena.scatter(
        [], [], s=185, marker='o', c='#8B36C6',
        edgecolors='none', alpha=0.32, visible=False, zorder=13.2,
    )
    fireball_core = arena.scatter(
        [], [], s=42, marker='o', c='#D68BFF',
        edgecolors='#F4ECFF', linewidths=0.55,
        visible=False, zorder=13.4,
    )
    breath_cloud, breath_particles, breath_stream = _create_breath_artists(arena)
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
        angle = 0.0
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
        active.update(frame.position, angle, frame.state)

        if frame.fireball_position is not None:
            offset = np.asarray(frame.fireball_position).reshape(1, 2)
            fireball_glow.set_offsets(offset)
            fireball_core.set_offsets(offset)
            fireball_glow.set_visible(True)
            fireball_core.set_visible(True)
        else:
            fireball_glow.set_visible(False)
            fireball_core.set_visible(False)
        _update_breath_artists(
            frame, breath_cloud, breath_particles, breath_stream,
        )

        if frame.explosion_index is not None:
            spike = spike_artists[frame.explosion_index]
            explosion.center = (spike['x'], spike['z'])
            explosion.set_radius(2.0 + 8.5 * frame.explosion_phase)
            explosion.set_alpha(0.95 * (1.0 - frame.explosion_phase) + 0.12)
        else:
            explosion.set_alpha(0.0)

        for state, node in state_nodes.items():
            is_active = state == frame.state
            inactive_alpha = 0.24 if state in {'dying', 'hover'} else 0.56
            node.set_facecolor(to_rgba(
                STATE_COLORS[state], 0.86 if is_active else inactive_alpha,
            ))
            node.set_edgecolor(COLORS['text'] if is_active else STATE_COLORS[state])
            node.set_linewidth(
                2.7 if is_active else (0.85 if state in {'dying', 'hover'} else 1.15)
            )

        probability = perch_probability(frame.crystals_alive)
        probability_fill.set_width(max(0.004, 0.445 * probability / (1.0 / 3.0)))
        probability_text.set_text(f'{probability * 100:4.1f}%')
        probability_formula.set_text(f'1 / (3 + {frame.crystals_alive})')
        crystal_label.set_text(f'CRYSTALS ALIVE  {frame.crystals_alive}/10')
        alive_indices = (
            set(frame.alive_crystals)
            if frame.alive_crystals is not None
            else set(range(frame.crystals_alive))
        )
        for index, icon in enumerate(crystal_icons):
            alive = index in alive_indices
            color = '#9B5DE5' if alive else COLORS['coral']
            icon['core'].set_facecolor(color)
            icon['core'].set_alpha(0.98 if alive else 0.16)
            icon['glow'].set_facecolor(color)
            icon['glow'].set_alpha(0.28 if alive else 0.04)
            icon['cage'].set_edgecolor('#A8B2C3' if alive else '#4B5361')
            icon['cage'].set_alpha(0.95 if alive else 0.32)
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
        _arena_static(arena, compact=True, limits=70)
        figure.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
        trail = LineCollection([], linewidths=3.2, capstyle='round', zorder=9)
        arena.add_collection(trail)
        active = DragonSpriteArtist(arena, size_blocks=10.5, zorder=13)
        breath_cloud, breath_particles, breath_stream = _create_breath_artists(arena)
        permanent_titles = {
            'dragon_holding_strafe.gif': 'HOLDING, STRAFING, AND CHARGING',
            'dragon_landing_perch.gif': 'LANDING APPROACH AND PERCHED PHASES',
            'dragon_takeoff.gif': 'TAKEOFF, RETURN, AND EXCEPTIONAL END STATE',
        }
        arena.set_title(
            permanent_titles[name], color=COLORS['text'], fontsize=13.5,
            fontweight='black', pad=7,
        )
        history = []

        def update(frame_index):
            frame = clip[frame_index]
            if frame_index == 0:
                history.clear()
            history.append(frame.position.copy())
            if len(history) > 28:
                history.pop(0)
            angle = 0.0
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
            active.update(frame.position, angle, frame.state)
            _update_breath_artists(
                frame, breath_cloud, breath_particles, breath_stream,
            )
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


def trajectory_animation_state(
    frame_index, trajectories=240, fps=8, frames=152,
    final_hold=3.0, batch_size=15,
):
    """Return synchronized route-count and representative-path state.

    Each active batch accumulates genuine routes while one fixed member of
    that batch is traversed as a visual representative. The final state is
    then held exactly for ``final_hold`` seconds.
    """
    hold_frames = int(round(float(fps) * float(final_hold)))
    active_frames = int(frames) - hold_frames
    if active_frames < 1:
        raise ValueError('frames must leave at least one active frame')
    if not 0 <= int(frame_index) < int(frames):
        raise IndexError('frame_index is outside the animation')
    if int(frame_index) >= active_frames:
        return int(trajectories), int(trajectories) - 1, 1.0

    batch_count = int(np.ceil(int(trajectories) / int(batch_size)))
    scaled = int(frame_index) * batch_count / active_frames
    batch_index = min(int(scaled), batch_count - 1)
    batch_start = batch_index * int(batch_size)
    batch_end = min(batch_start + int(batch_size), int(trajectories))
    batch_start_frame = int(batch_index * active_frames / batch_count)
    batch_end_frame = int((batch_index + 1) * active_frames / batch_count)
    batch_active_frames = max(1, batch_end_frame - batch_start_frame)
    local_step = int(frame_index) - batch_start_frame + 1
    route_count = batch_end - batch_start
    shown = batch_start + max(
        1, int(np.ceil(route_count * local_step / batch_active_frames)),
    )
    traversal = min(local_step / batch_active_frames, 1.0)
    return min(shown, int(trajectories)), batch_end - 1, traversal


def create_trajectory_ensemble_animation(
    save_path, seed=12031, trajectories=240, fps=8, frames=152,
    final_hold=3.0, batch_size=15,
):
    """Animate dragon approaches with distinct-trajectory intersection counts."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    player_positions = [
        (
            (24.0 + 24.0 * ((index * 37) % trajectories) / max(trajectories - 1, 1))
            * np.cos(index * golden_angle),
            (24.0 + 24.0 * ((index * 37) % trajectories) / max(trajectories - 1, 1))
            * np.sin(index * golden_angle),
        )
        for index in range(trajectories)
    ]
    paths = [
        simulate_perch_trajectory(
            seed + index * 7919,
            crystals_alive=10 - (index % 6),
            player_position=player_positions[index],
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
        if np.hypot(center_x, center_z) <= 24.0:
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

    figure = plt.figure(figsize=(14.4, 8.1), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[1.54, 1.0],
        left=0.055, right=0.985, top=0.90, bottom=0.17, wspace=0.055,
    )
    axis = figure.add_subplot(grid[0, 0])
    frequency_axis = figure.add_subplot(grid[0, 1])
    _arena_static(
        axis, seed=42, compact=True, axis_off=False,
        limits=88, island_alpha=0.48,
    )

    image = axis.imshow(
        np.zeros_like(cumulative[0]), origin='lower',
        extent=(bins[0], bins[-1], bins[0], bins[-1]),
        cmap='viridis',
        norm=PowerNorm(
            gamma=0.5, vmin=0.0, vmax=max(float(final_frequency.max()), 1.0),
        ),
        interpolation='bilinear', alpha=0.74, zorder=2.8,
    )
    lines = LineCollection(
        [], linewidths=0.78, colors=COLORS['cyan'], zorder=8,
    )
    axis.add_collection(lines)
    local_trail = LineCollection([], linewidths=3.1, capstyle='round', zorder=10)
    axis.add_collection(local_trail)
    dragon = DragonSpriteArtist(axis, size_blocks=10.2, zorder=12)
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
        color=plt.get_cmap('viridis')(0.0),
        edgecolor=COLORS['text'], linewidth=0.45, alpha=0.92,
    )
    frequency_axis.set_yticks(
        np.arange(len(hotspot_values)),
        [f'({x:+.0f}, {z:+.0f})' for x, z in zip(hotspot_x, hotspot_z)],
    )
    frequency_axis.invert_yaxis()
    maximum_hotspot = max(float(np.max(hotspot_values)), 1.0)
    frequency_axis.set_xlim(0, maximum_hotspot * 1.18)
    frequency_axis.set_xlabel('Distinct legal routes entering the cell')
    frequency_axis.set_ylabel('Final hotspot center X, Z (blocks)')
    frequency_axis.set_title(
        'Final repeatability hotspots (counts accumulating)', fontsize=10.8, pad=8,
    )
    frequency_axis.grid(axis='x', color=COLORS['grid'], alpha=0.35, linewidth=0.55)
    for spine in frequency_axis.spines.values():
        spine.set_color(COLORS['grid'])
    frequency_axis.tick_params(colors=COLORS['muted'], labelsize=7.7)
    bar_value_texts = [
        frequency_axis.text(
            0.0, index, '', ha='left', va='center',
            color=COLORS['text'], fontsize=7.2, family='monospace',
        )
        for index in range(len(hotspot_values))
    ]
    hotspot_markers = axis.scatter(
        hotspot_x, hotspot_z, s=34, facecolors='none',
        edgecolors=plt.get_cmap('viridis')(0.98), linewidths=0.8,
        alpha=0.0, zorder=11,
    )
    figure.text(
        0.645, 0.055,
        'Final cells are fixed for comparison; each legal route contributes once per cell.\n'
        f'Active dragon represents the current {batch_size}-route batch; final hold = {final_hold:.1f} s.',
        color=COLORS['muted'],
        fontsize=7.4, ha='left', va='top', linespacing=1.35,
    )

    figure.suptitle(
        'TRAJECTORY DISTRIBUTION AND DEGENERACY',
        color=COLORS['text'], fontsize=17, fontweight='black', y=0.97,
    )

    def update(frame_index):
        shown, featured_index, local_phase = trajectory_animation_state(
            frame_index, trajectories=trajectories, fps=fps, frames=frames,
            final_hold=final_hold, batch_size=batch_size,
        )
        current_frequency = cumulative[shown - 1]
        image.set_data(current_frequency)
        recent_start = max(0, shown - 28)
        recent_paths = paths[recent_start:shown]
        lines.set_segments(recent_paths)
        age = np.linspace(0.08, 1.0, len(recent_paths))
        line_colors = [
            to_rgba(plt.get_cmap('viridis')(0.10 + 0.76 * value),
                    0.05 + 0.36 * value)
            for value in age
        ]
        lines.set_colors(line_colors)

        featured = paths[featured_index]
        point_index = min(len(featured) - 1, round(local_phase * (len(featured) - 1)))
        point = featured[point_index]
        edge_fade = min(local_phase / 0.12, 1.0)
        dragon.set_alpha(0.35 + 0.65 * max(edge_fade, 0.0))
        trail_start = max(0, point_index - 22)
        local_points = featured[trail_start:point_index + 1]
        if len(local_points) > 1:
            segments = np.stack([local_points[:-1], local_points[1:]], axis=1)
            local_trail.set_segments(segments)
            local_trail.set_colors([
                to_rgba(plt.get_cmap('viridis')(0.30 + 0.68 * value), 0.20 + 0.78 * value)
                for value in np.linspace(0.0, 1.0, len(segments))
            ])
            vector = local_points[-1] - local_points[-2]
            angle = np.degrees(np.arctan2(vector[1], vector[0]))
        else:
            local_trail.set_segments([])
            angle = 0.0
        dragon.update(point, angle, 'landing_approach')

        for bar, label, row, column in zip(
            bars, bar_value_texts, hotspot_rows, hotspot_columns,
        ):
            value = float(current_frequency[row, column])
            bar.set_width(value)
            color_value = value / maximum_hotspot
            bar.set_facecolor(plt.get_cmap('viridis')(color_value))
            label.set_x(value + maximum_hotspot * 0.015)
            label.set_text(str(int(value)) if value > 0 else '')
        count_text.set_text(f'{shown:03d} seeded approaches')
        hotspot_markers.set_alpha(0.30 + 0.70 * shown / trajectories)
        return []

    animation = FuncAnimation(
        figure, update, frames=frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
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
