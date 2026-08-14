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
from matplotlib.colors import Normalize, PowerNorm, to_rgba
from matplotlib.path import Path as MarkerPath
from matplotlib.patches import (
    Circle, Ellipse, FancyArrowPatch, FancyBboxPatch,
)
import numpy as np
from PIL import Image, ImageFilter

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
HUD_PROBABILITY_X = 0.335
HUD_PROBABILITY_WIDTH = 0.390


def _prepare_dragon_sprite():
    """Return a compact tinted sprite whose wings can be articulated."""
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
    return tinted


def _articulate_dragon_wings(sprite, extension):
    """Foreshorten the two wing layers while leaving the body rigid."""
    extension = float(np.clip(extension, 0.76, 1.0))
    output = Image.new('RGBA', sprite.size, (0, 0, 0, 0))
    left_pivot, right_pivot = 50, 70
    left = sprite.crop((0, 0, left_pivot, sprite.height))
    right = sprite.crop((right_pivot, 0, sprite.width, sprite.height))
    left_width = max(1, int(round(left.width * extension)))
    right_width = max(1, int(round(right.width * extension)))
    left = left.resize((left_width, sprite.height), Image.Resampling.BICUBIC)
    right = right.resize((right_width, sprite.height), Image.Resampling.BICUBIC)
    output.alpha_composite(left, (left_pivot - left_width, 0))
    output.alpha_composite(sprite.crop((left_pivot, 0, right_pivot, sprite.height)),
                           (left_pivot, 0))
    output.alpha_composite(right, (right_pivot, 0))
    return output


def _dragon_glow(sprite):
    alpha = sprite.getchannel('A').filter(
        ImageFilter.MaxFilter(7)
    ).filter(ImageFilter.GaussianBlur(2.2))
    glow = Image.new('RGBA', sprite.size, (142, 77, 204, 0))
    glow.putalpha(alpha.point(lambda value: int(value * 0.32)))
    return glow


class DragonSpriteArtist:
    """Rotating raster sprite with cached state tint and violet glow."""

    def __init__(self, axis, size_blocks=12.0, zorder=12):
        self.base = _prepare_dragon_sprite()
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

    def update(self, position, angle, state, scale=1.0, wing_phase=0.0):
        quantized = int(round(float(angle) / 2.0) * 2) % 360
        flap_index = int(round(6.0 * (0.5 + 0.5 * np.sin(float(wing_phase)))))
        key = (quantized, state, flap_index)
        if key not in self.cache:
            rotation = quantized - 90
            articulated = _articulate_dragon_wings(
                self.base, 0.76 + 0.24 * flap_index / 6.0,
            )
            sprite = articulated.rotate(
                rotation, resample=Image.Resampling.BICUBIC, expand=False,
            )
            glow = _dragon_glow(articulated).rotate(
                rotation, resample=Image.Resampling.BICUBIC, expand=False,
            )
            rgba = np.asarray(sprite).astype(float) / 255.0
            state_rgb = np.asarray(to_rgba(STATE_COLORS[state])[:3])
            visible = rgba[..., 3:4]
            rgba[..., :3] = np.clip(
                rgba[..., :3] * 0.52 + state_rgb[None, None, :] * 0.48 * visible,
                0.0, 1.0,
            )
            glow_rgba = np.asarray(glow).copy()
            glow_rgba[..., :3] = np.uint8(state_rgb * 255.0)
            self.cache[key] = (
                np.uint8(rgba * 255), glow_rgba,
            )
        sprite, glow = self.cache[key]
        x, z = np.asarray(position, dtype=float)
        half_size = self.half_size * float(scale)
        extent = (
            x - half_size, x + half_size,
            z - half_size, z + half_size,
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


def _update_breath_with_linger(
    frame, cloud, particles, stream, linger_state,
    projectile_linger_frames=34,
):
    """Keep projectile breath visible as a translucent source-valid cloud.

    A dragon fireball creates a 600-tick area-effect cloud that grows from
    radius three toward seven. The showcase compresses that lifetime, while
    perched breath is still removed immediately when its phase ends.
    """
    if frame.breath_center is not None:
        _update_breath_artists(frame, cloud, particles, stream)
        if frame.breath_kind == 'projectile_impact':
            linger_state.update({
                'center': np.asarray(frame.breath_center, dtype=float).copy(),
                'radius': float(frame.breath_radius),
                'alpha': float(frame.breath_alpha),
                'remaining': int(projectile_linger_frames),
            })
        return
    if linger_state.get('remaining', 0) <= 0:
        _update_breath_artists(frame, cloud, particles, stream)
        return

    remaining = int(linger_state['remaining'])
    fraction = remaining / max(int(projectile_linger_frames), 1)
    elapsed = 1.0 - fraction
    linger_state['remaining'] = remaining - 1
    proxy = frame.__class__(
        position=frame.position,
        state=frame.state,
        crystals_alive=frame.crystals_alive,
        current_node=frame.current_node,
        target_node=frame.target_node,
        alive_crystals=frame.alive_crystals,
        breath_center=linger_state['center'],
        breath_radius=min(7.0, linger_state['radius'] + 0.9 * elapsed),
        breath_alpha=linger_state['alpha'] * fraction * 0.72,
        breath_kind='projectile_impact',
    )
    _update_breath_artists(proxy, cloud, particles, stream)


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
        cage_linewidth=1.75, cage_extent=3.75, radius_scale=1.28,
        crystal_shape='diamond',
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
        'strafing': (0.17, 0.86),
        'charging_player': (0.83, 0.86),
        'landing_approach': (0.50, 0.82),
        'landing': (0.50, 0.71),
        'sitting_scanning': (0.26, 0.59),
        'sitting_attacking': (0.26, 0.48),
        'sitting_flaming': (0.26, 0.37),
        'takeoff': (0.68, 0.48),
        'hover': (0.75, 0.61),
        'dying': (0.75, 0.36),
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
    node_width = 0.205
    node_height = 0.061
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

    transitions = [
        (start, end, curvature[(start, end)], False)
        for start, end in SOURCE_PHASE_TRANSITIONS
    ] + [
        (start, end, curvature[(start, end)], True)
        for start, end, _ in EXCEPTION_PHASE_TRANSITIONS
    ]
    for start, end, curve, exceptional in transitions:
        connection = FancyArrowPatch(
            posA=positions[start], posB=positions[end],
            patchA=nodes[start], patchB=nodes[end], arrowstyle='-|>',
            mutation_scale=10.5,
            color='#566785' if exceptional else '#7455A3',
            linewidth=1.0 if exceptional else 1.35,
            linestyle='--' if exceptional else '-',
            connectionstyle=f'arc3,rad={curve}',
            alpha=0.48 if exceptional else 0.80,
            shrinkA=1.5, shrinkB=1.5, zorder=2,
        )
        ax.add_patch(connection)

    hud = FancyBboxPatch(
        (0.035, 0.012), 0.93, 0.216,
        boxstyle='round,pad=0.012,rounding_size=0.022',
        facecolor=to_rgba('#171D2A', 0.99),
        edgecolor='#9B6FD1', linewidth=1.45, zorder=4,
    )
    ax.add_patch(hud)
    ax.add_patch(FancyBboxPatch(
        (0.051, 0.126), 0.898, 0.084,
        boxstyle='round,pad=0.006,rounding_size=0.014',
        facecolor=to_rgba('#20293A', 0.94), edgecolor='#3B475E',
        linewidth=0.65, zorder=4.5,
    ))
    ax.add_patch(FancyBboxPatch(
        (0.051, 0.031), 0.898, 0.078,
        boxstyle='round,pad=0.006,rounding_size=0.014',
        facecolor=to_rgba('#20293A', 0.94), edgecolor='#3B475E',
        linewidth=0.65, zorder=4.5,
    ))
    ax.text(
        0.071, 0.187, 'NEXT HOLDING-PATH LANDING ROLL',
        ha='left', va='center', color='#D7DDEA',
        fontsize=7.7, fontweight='black', family='DejaVu Sans', zorder=5,
    )
    probability_background = FancyBboxPatch(
        (HUD_PROBABILITY_X, 0.139), HUD_PROBABILITY_WIDTH, 0.036,
        boxstyle='round,pad=0.002,rounding_size=0.010',
        facecolor='#0B0F17', edgecolor='#59677E',
        linewidth=0.60, alpha=1.0, zorder=5,
    )
    probability_fill = FancyBboxPatch(
        (HUD_PROBABILITY_X, 0.139), 0.002, 0.036,
        boxstyle='round,pad=0.002,rounding_size=0.010',
        facecolor='#A75DE1', edgecolor='none', zorder=6,
    )
    ax.add_patch(probability_background)
    ax.add_patch(probability_fill)
    probability_text = ax.text(
        0.932, 0.170, '', ha='right', va='center',
        color='#F4ECFF', fontsize=13.4, fontweight='black',
        family='DejaVu Sans', zorder=6,
    )
    probability_formula = ax.text(
        0.932, 0.140, '', ha='right', va='center',
        color='#8F9CB1', fontsize=7.0, family='DejaVu Sans', zorder=6,
    )
    crystal_label = ax.text(
        0.071, 0.070, 'ACTIVE END CRYSTALS', ha='left', va='center',
        color='#D7DDEA', fontsize=7.6, fontweight='black',
        family='DejaVu Sans', zorder=5,
    )
    crystal_icons = []
    for index in range(10):
        x = 0.425 + index * 0.0525
        glow = ax.scatter(
            [x], [0.070], s=125, marker='D', c='#A86BE0',
            edgecolors='none', alpha=0.24, zorder=5.5,
        )
        cage = ax.scatter(
            [x], [0.070], s=76, marker='D', c='#121824',
            edgecolors='#8390A4', linewidths=0.55, alpha=0.95, zorder=6,
        )
        icon = ax.scatter(
            [x], [0.070], s=28, marker='D', c='#C875FF',
            edgecolors='#F4ECFF', linewidths=0.42, zorder=6.2,
        )
        crystal_icons.append({'glow': glow, 'cage': cage, 'core': icon})
    return (
        nodes, crystal_icons, probability_fill, probability_text,
        probability_formula, crystal_label,
    )


def _set_active_graph_edge(artist, edge, color):
    """Show one decoded graph edge without replacing the flight trajectory."""
    if edge is None:
        artist.set_segments([])
        return
    left, right = edge
    artist.set_segments([[DRAGON_NODES[left], DRAGON_NODES[right]]])
    artist.set_color(color)


def _nearest_route_edge(position, node_path):
    """Return the route edge nearest a projected continuous flight position."""
    if len(node_path) < 2:
        return None
    point = np.asarray(position, dtype=float)
    best_edge = None
    best_distance = float('inf')
    for left, right in zip(node_path, node_path[1:]):
        start = DRAGON_NODES[left]
        end = DRAGON_NODES[right]
        delta = end - start
        fraction = np.clip(
            np.dot(point - start, delta) / max(np.dot(delta, delta), 1.0),
            0.0, 1.0,
        )
        distance = float(np.linalg.norm(point - (start + fraction * delta)))
        if distance < best_distance:
            best_distance = distance
            best_edge = tuple(sorted((int(left), int(right))))
    return best_edge


def create_dragon_pathfinding_animation(
    save_path, fps=10, dpi=100, colors=96,
):
    """Create the shorter README hero animation."""
    frames = scripted_showcase()
    if len(frames) > 448:
        indices = np.linspace(0, len(frames) - 1, 448).round().astype(int)
        frames = [frames[index] for index in indices]
    figure = plt.figure(figsize=(12.8, 7.2), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[1.62, 1.08],
        left=0.018, right=0.988, top=0.982, bottom=0.018, wspace=0.005,
    )
    arena = figure.add_subplot(grid[0, 0])
    machine = figure.add_subplot(grid[0, 1])
    spike_artists = _arena_static(arena)
    (
        state_nodes, crystal_icons, probability_fill, probability_text,
        probability_formula, crystal_label,
    ) = _draw_state_machine(machine)

    active_edge = LineCollection(
        [], linewidths=3.6, capstyle='round', joinstyle='round',
        alpha=0.82, zorder=8.4,
    )
    arena.add_collection(active_edge)
    trail = LineCollection(
        [], linewidths=2.8, capstyle='round', joinstyle='round', zorder=9,
    )
    arena.add_collection(trail)
    active = DragonSpriteArtist(arena, size_blocks=19.0, zorder=13)
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
    damage_flash = Circle(
        (0, 0), 0.1, fill=False, edgecolor=COLORS['coral'],
        linewidth=2.5, alpha=0.0, zorder=14.2,
    )
    arena.add_patch(damage_flash)

    history = []
    breath_linger = {}

    def update(frame_index):
        frame = frames[frame_index]
        if frame_index == 0:
            history.clear()
            breath_linger.clear()
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
        sitting = frame.state in {
            'hover', 'sitting_scanning', 'sitting_attacking', 'sitting_flaming',
        }
        wing_phase = (-0.5 * np.pi if sitting
                      else 2.0 * np.pi * frame_index / 10.0)
        active.update(
            frame.position, angle, frame.state,
            scale=1.0, wing_phase=wing_phase,
        )
        _set_active_graph_edge(
            active_edge, frame.active_edge,
            to_rgba(STATE_COLORS[frame.state], 0.86),
        )

        if frame.fireball_position is not None:
            offset = np.asarray(frame.fireball_position).reshape(1, 2)
            fireball_glow.set_offsets(offset)
            fireball_core.set_offsets(offset)
            fireball_glow.set_visible(True)
            fireball_core.set_visible(True)
        else:
            fireball_glow.set_visible(False)
            fireball_core.set_visible(False)
        _update_breath_with_linger(
            frame, breath_cloud, breath_particles, breath_stream, breath_linger,
        )

        if frame.explosion_index is not None:
            spike = spike_artists[frame.explosion_index]
            explosion.center = (spike['x'], spike['z'])
            explosion.set_radius(2.0 + 8.5 * frame.explosion_phase)
            explosion.set_alpha(0.95 * (1.0 - frame.explosion_phase) + 0.12)
        else:
            explosion.set_alpha(0.0)

        if frame.damage_pulse > 0.0:
            damage_flash.center = tuple(frame.position)
            damage_flash.set_radius(2.5 + 8.0 * frame.damage_pulse)
            damage_flash.set_alpha(0.92 * (1.0 - frame.damage_pulse) + 0.10)
        else:
            damage_flash.set_alpha(0.0)

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
        probability_fill.set_width(max(
            0.004, HUD_PROBABILITY_WIDTH * probability / (1.0 / 3.0),
        ))
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
        if len(clip) > 96:
            indices = np.linspace(0, len(clip) - 1, 96).round().astype(int)
            clip = [clip[index] for index in indices]
        figure, arena = plt.subplots(figsize=(9.6, 5.4), facecolor=COLORS['background'])
        _arena_static(arena, compact=True, limits=70)
        figure.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02)
        active_edge = LineCollection(
            [], linewidths=3.8, capstyle='round', joinstyle='round',
            alpha=0.82, zorder=8.4,
        )
        arena.add_collection(active_edge)
        trail = LineCollection(
            [], linewidths=3.2, capstyle='round', joinstyle='round', zorder=9,
        )
        arena.add_collection(trail)
        active = DragonSpriteArtist(arena, size_blocks=19.0, zorder=13)
        breath_cloud, breath_particles, breath_stream = _create_breath_artists(arena)
        permanent_titles = {
            'dragon_holding_strafe.gif': 'HOLDING AND STRAFING',
            'dragon_landing_perch.gif': 'LANDING, PERCHED DECISIONS, AND CHARGING',
            'dragon_takeoff.gif': 'TAKEOFF, RETURN, AND EXCEPTIONAL END STATE',
        }
        arena.set_title(
            permanent_titles[name], color=COLORS['text'], fontsize=13.5,
            fontweight='black', pad=7,
        )
        history = []
        breath_linger = {}

        def update(frame_index):
            frame = clip[frame_index]
            if frame_index == 0:
                history.clear()
                breath_linger.clear()
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
            sitting = frame.state in {
                'hover', 'sitting_scanning', 'sitting_attacking', 'sitting_flaming',
            }
            active.update(
                frame.position, angle, frame.state, scale=1.0,
                wing_phase=(-0.5 * np.pi if sitting
                            else 2.0 * np.pi * frame_index / 9.0),
            )
            _set_active_graph_edge(
                active_edge, frame.active_edge,
                to_rgba(STATE_COLORS[frame.state], 0.86),
            )
            _update_breath_with_linger(
                frame, breath_cloud, breath_particles, breath_stream,
                breath_linger,
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
    frame_index, trajectories=480, fps=10, frames=320,
    final_hold=3.2, batch_size=12,
):
    """Return synchronized route-count and representative-path state.

    Each active batch accumulates genuine routes while its first, already
    counted member is traversed as the visual representative. The final state
    is then held exactly for ``final_hold`` seconds.
    """
    hold_frames = int(round(float(fps) * float(final_hold)))
    active_frames = int(frames) - hold_frames
    if active_frames < 1:
        raise ValueError('frames must leave at least one active frame')
    if not 0 <= int(frame_index) < int(frames):
        raise IndexError('frame_index is outside the animation')
    if int(frame_index) >= active_frames:
        last_batch_start = (
            (int(np.ceil(int(trajectories) / int(batch_size))) - 1)
            * int(batch_size)
        )
        return int(trajectories), last_batch_start, 1.0

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
    return min(shown, int(trajectories)), batch_start, traversal


def _point_on_polyline(points, phase):
    """Return an arc-length position, heading, and visible prefix."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points[0], 0.0, points.copy()
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    target = float(np.clip(phase, 0.0, 1.0)) * cumulative[-1]
    segment = min(int(np.searchsorted(cumulative, target, side='right') - 1), len(points) - 2)
    segment_length = max(lengths[segment], 1e-9)
    fraction = (target - cumulative[segment]) / segment_length
    point = points[segment] + fraction * (points[segment + 1] - points[segment])
    prefix = np.vstack((points[:segment + 1], point))
    vector = points[segment + 1] - points[segment]
    angle = np.degrees(np.arctan2(vector[1], vector[0]))
    return point, angle, prefix


def create_trajectory_ensemble_animation(
    save_path, seed=12031, trajectories=480, fps=10, frames=320,
    final_hold=3.2, batch_size=12,
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
    simulations = [
        simulate_perch_trajectory(
            seed + index * 7919,
            crystals_alive=10 - (index % 6),
            player_position=player_positions[index],
        )
        for index in range(trajectories)
    ]
    paths = [simulation[0] for simulation in simulations]
    node_paths = [simulation[1] for simulation in simulations]
    bins = np.linspace(-76, 76, 121)
    contributions = []
    for path in paths:
        histogram, _, _ = np.histogram2d(path[:, 1], path[:, 0], bins=(bins, bins))
        contributions.append(histogram > 0)
    contributions = np.asarray(contributions)
    cumulative = np.cumsum(contributions, axis=0)
    final_frequency = cumulative[-1]
    occupied_frequency = final_frequency[final_frequency > 0]
    color_ceiling = max(
        float(np.percentile(occupied_frequency, 98.5))
        if occupied_frequency.size else 1.0,
        1.0,
    )

    edge_index = {tuple(sorted(edge)): index for index, edge in enumerate(DRAGON_EDGES)}
    route_edge_contributions = np.zeros((trajectories, len(DRAGON_EDGES)), dtype=int)
    for route_index, nodes in enumerate(node_paths):
        for left, right in zip(nodes, nodes[1:]):
            route_edge_contributions[route_index, edge_index[tuple(sorted((left, right)))]] = 1
    cumulative_edge_counts = np.cumsum(route_edge_contributions, axis=0)
    final_edge_counts = cumulative_edge_counts[-1]
    ranked_edges = np.argsort(final_edge_counts)[::-1]
    selected_edge_indices = ranked_edges[:10]
    selected_edges = [DRAGON_EDGES[index] for index in selected_edge_indices]
    selected_edge_counts = final_edge_counts[selected_edge_indices]

    figure = plt.figure(figsize=(14.4, 8.1), facecolor=COLORS['background'])
    grid = figure.add_gridspec(
        1, 2, width_ratios=[1.54, 1.0],
        left=0.055, right=0.985, top=0.90, bottom=0.115, wspace=0.055,
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
            gamma=0.64, vmin=0.0, vmax=color_ceiling, clip=True,
        ),
        interpolation='bilinear', alpha=0.74, zorder=2.8,
    )
    lines = LineCollection(
        [], linewidths=0.78, colors=COLORS['cyan'],
        capstyle='round', joinstyle='round', zorder=8,
    )
    axis.add_collection(lines)
    active_edge = LineCollection(
        [], linewidths=3.6, capstyle='round', joinstyle='round',
        alpha=0.86, zorder=9.2,
    )
    axis.add_collection(active_edge)
    local_trail = LineCollection(
        [], linewidths=3.1, capstyle='round', joinstyle='round', zorder=10,
    )
    axis.add_collection(local_trail)
    dragon = DragonSpriteArtist(axis, size_blocks=19.0, zorder=12)
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
        np.arange(len(selected_edges)), np.zeros(len(selected_edges)),
        color=plt.get_cmap('viridis')(0.0),
        edgecolor=COLORS['text'], linewidth=0.45, alpha=0.92,
    )
    frequency_axis.set_yticks(
        np.arange(len(selected_edges)),
        [f'NODE {left:02d}  TO  {right:02d}' for left, right in selected_edges],
    )
    frequency_axis.invert_yaxis()
    maximum_edge_count = max(float(np.max(selected_edge_counts)), 1.0)
    minimum_edge_count = float(np.min(selected_edge_counts))
    bar_norm = Normalize(
        vmin=minimum_edge_count,
        vmax=max(maximum_edge_count, minimum_edge_count + 1.0),
        clip=True,
    )
    maximum_final_percentage = 100.0 * maximum_edge_count / trajectories
    bar_axis_limit = min(
        100.0,
        max(35.0, 5.0 * np.ceil((maximum_final_percentage + 5.0) / 5.0)),
    )
    frequency_axis.set_xlim(0, bar_axis_limit)
    frequency_axis.set_xlabel(
        f'Share of the final {trajectories}-route ensemble (%)'
    )
    frequency_axis.set_ylabel('Decoded path-node edge')
    frequency_axis.set_title(
        'Most-used legal navigation edges', fontsize=10.8, pad=8,
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
        for index in range(len(selected_edges))
    ]
    figure.suptitle(
        'TRAJECTORY DISTRIBUTION AND DEGENERACY',
        color=COLORS['text'], fontsize=17, fontweight='black', y=0.97,
    )

    def update(frame_index):
        shown, featured_index, local_phase = trajectory_animation_state(
            frame_index, trajectories=trajectories, fps=fps, frames=frames,
            final_hold=final_hold, batch_size=batch_size,
        )
        featured = paths[featured_index]
        featured_nodes = node_paths[featured_index]
        point, angle, visible_prefix = _point_on_polyline(featured, local_phase)
        current_frequency = cumulative[shown - 1].astype(float).copy()
        if featured_index < shown:
            current_frequency -= contributions[featured_index]
            partial, _, _ = np.histogram2d(
                visible_prefix[:, 1], visible_prefix[:, 0], bins=(bins, bins),
            )
            current_frequency += partial > 0
        image.set_data(current_frequency)
        recent_start = max(0, shown - 28)
        recent_paths = [
            paths[index] for index in range(recent_start, shown)
            if index != featured_index
        ]
        lines.set_segments(recent_paths)
        age = np.linspace(0.08, 1.0, len(recent_paths))
        line_colors = [
            to_rgba(plt.get_cmap('viridis')(0.10 + 0.76 * value),
                    0.05 + 0.36 * value)
            for value in age
        ]
        lines.set_colors(line_colors)

        edge_fade = min(local_phase / 0.12, 1.0)
        dragon.set_alpha(0.35 + 0.65 * max(edge_fade, 0.0))
        local_points = visible_prefix[max(0, len(visible_prefix) - 23):]
        if len(local_points) > 1:
            segments = np.stack([local_points[:-1], local_points[1:]], axis=1)
            local_trail.set_segments(segments)
            local_trail.set_colors([
                to_rgba(plt.get_cmap('viridis')(0.30 + 0.68 * value), 0.20 + 0.78 * value)
                for value in np.linspace(0.0, 1.0, len(segments))
            ])
        else:
            local_trail.set_segments([])
        active_frames = frames - int(round(final_hold * fps))
        wing_phase = (
            2.0 * np.pi * frame_index / 9.0
            if frame_index < active_frames else -0.5 * np.pi
        )
        dragon.update(
            point, angle, 'landing_approach', scale=1.0,
            wing_phase=wing_phase,
        )
        _set_active_graph_edge(
            active_edge,
            _nearest_route_edge(point, featured_nodes),
            to_rgba(COLORS['cyan'], 0.92),
        )

        current_edge_counts = cumulative_edge_counts[shown - 1, selected_edge_indices]
        for bar, label, value in zip(bars, bar_value_texts, current_edge_counts):
            percentage = 100.0 * float(value) / trajectories
            bar.set_width(percentage)
            bar.set_facecolor(plt.get_cmap('viridis')(bar_norm(float(value))))
            label.set_x(min(percentage + 0.6, bar_axis_limit - 3.5))
            label.set_text(f'{int(value):3d} routes  {percentage:4.1f}%')
        count_text.set_text(
            f'{shown:03d}/{trajectories} exact routes  |  density capped at P98.5'
        )
        return []

    animation = FuncAnimation(
        figure, update, frames=frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=105)
    plt.close(figure)
    optimize_gif(save_path, colors=96)
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
