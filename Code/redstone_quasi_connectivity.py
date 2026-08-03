"""Minecraft-style Java 1.16.1 quasi-connectivity visualizations."""

from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np

from core.rendering import optimize_gif
from core.style import COLORS, apply_style


apply_style()


BLOCKS = {
    'platform': ('#565E68', '#414852', '#303640'),
    'piston': ('#B7B1A0', '#A97845', '#765334'),
    'piston_head': ('#C2BCA9', '#94683E', '#6E4C31'),
    'arm': ('#B9B0A0', '#77706A', '#5B5550'),
    'load': ('#F6C85F', '#CC9840', '#9B6E2F'),
    'source_off': ('#8E949D', '#6F7680', '#505761'),
    'source_on': ('#FF5A55', '#C82E3B', '#8E1C2B'),
    'note': ('#A5673F', '#754329', '#512E20'),
}


@dataclass(frozen=True)
class BudVisualState:
    source_on: bool
    extension: float
    update_pulse: float
    phase: str
    detail: str


def _ease(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def bud_animation_state(frame_index, total_frames=140):
    """Return the deterministic visual state for one BUD demonstration frame."""
    frame = int(frame_index) % int(total_frames)
    if frame < 12:
        return BudVisualState(False, 0.0, 0.0, 'READY', 'No elevated power')
    if frame < 35:
        return BudVisualState(
            True, 0.0, 0.0, 'QUASI-POWERED', 'Signal present, update missing',
        )
    if frame < 45:
        pulse = math.sin(math.pi * (frame - 35) / 10.0)
        return BudVisualState(
            True, 0.0, pulse, 'BLOCK UPDATE', 'The note block wakes the piston',
        )
    if frame < 59:
        extension = _ease((frame - 45) / 13.0)
        return BudVisualState(
            True, extension, 0.0, 'EXTENDING', 'The piston re-checks current power',
        )
    if frame < 77:
        return BudVisualState(True, 1.0, 0.0, 'EXTENDED', 'Piston state is current')
    if frame < 99:
        return BudVisualState(
            False, 1.0, 0.0, 'POWER REMOVED', 'Retraction also waits for an update',
        )
    if frame < 109:
        pulse = math.sin(math.pi * (frame - 99) / 10.0)
        return BudVisualState(
            False, 1.0, pulse, 'BLOCK UPDATE', 'A second update wakes the piston',
        )
    if frame < 123:
        extension = 1.0 - _ease((frame - 109) / 13.0)
        return BudVisualState(
            False, extension, 0.0, 'RETRACTING', 'The unpowered piston retracts',
        )
    return BudVisualState(False, 0.0, 0.0, 'READY', 'Cycle complete')


def _iso_point(x, y, z):
    return x - z, y + 0.48 * (x + z)


def _cuboid_faces(x, y, z, width=1.0, height=1.0, depth=1.0):
    p000 = _iso_point(x, y, z)
    p100 = _iso_point(x + width, y, z)
    p110 = _iso_point(x + width, y + height, z)
    p010 = _iso_point(x, y + height, z)
    p001 = _iso_point(x, y, z + depth)
    p101 = _iso_point(x + width, y, z + depth)
    p111 = _iso_point(x + width, y + height, z + depth)
    p011 = _iso_point(x, y + height, z + depth)
    return {
        'side': [p100, p101, p111, p110],
        'front': [p000, p100, p110, p010],
        'top': [p010, p110, p111, p011],
    }


def _draw_cuboid(
    ax, x, y, z, kind, width=1.0, height=1.0, depth=1.0,
    alpha=1.0, zorder=3,
):
    top, front, side = BLOCKS[kind]
    faces = _cuboid_faces(x, y, z, width, height, depth)
    for offset, (name, color) in enumerate((
        ('side', side), ('front', front), ('top', top),
    )):
        patch = Polygon(
            faces[name], closed=True, facecolor=color,
            edgecolor='#171A20', linewidth=0.72,
            alpha=alpha, zorder=zorder + offset * 0.02,
        )
        ax.add_patch(patch)

    if kind == 'note':
        for height_offset in (0.28, 0.50, 0.72):
            left = _iso_point(x + 0.18, y + height_offset, z - 0.002)
            right = _iso_point(x + 0.82, y + height_offset, z - 0.002)
            ax.plot(
                [left[0], right[0]], [left[1], right[1]],
                color='#3A2018', linewidth=1.0, alpha=0.75,
                zorder=zorder + 0.08,
            )
    if kind == 'piston':
        left = _iso_point(x + 0.08, y + 0.27, z - 0.002)
        right = _iso_point(x + 0.92, y + 0.27, z - 0.002)
        ax.plot(
            [left[0], right[0]], [left[1], right[1]],
            color='#E0BD78', linewidth=1.4, zorder=zorder + 0.08,
        )


def _draw_lever(ax, source_on):
    center = _iso_point(1.50, 2.03, 0.50)
    direction = -1.0 if source_on else 1.0
    tip = (center[0] + 0.22 * direction, center[1] + 0.34)
    ax.scatter(
        [center[0]], [center[1]], s=42,
        c='#4A4650', edgecolors='#D1D5DB', linewidths=0.55, zorder=12,
    )
    ax.plot(
        [center[0], tip[0]], [center[1], tip[1]],
        color='#C6B9A3', linewidth=3.0, solid_capstyle='round', zorder=13,
    )
    ax.scatter(
        [tip[0]], [tip[1]], s=16,
        c=COLORS['coral'] if source_on else COLORS['muted'],
        edgecolors=COLORS['text'], linewidths=0.4, zorder=14,
    )


def _draw_scene(ax, state, labels=False, panel_label=None, heading=None):
    ax.set_facecolor(COLORS['background'])
    ax.set_xlim(-3.35, 3.55)
    ax.set_ylim(-0.75, 4.85)
    ax.set_aspect('equal')
    ax.axis('off')

    blocks = []
    for block_x in range(-2, 3):
        for block_z in range(-1, 2):
            blocks.append((block_x, -0.32, block_z, 'platform', 1.0, 0.32, 1.0))
    blocks.extend([
        (-1.0, 0.0, 0.0, 'note', 1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0, 'piston', 1.0, 1.0, 1.0),
        (1.0, 1.0, 0.0, 'source_on' if state.source_on else 'source_off',
         1.0, 1.0, 1.0),
    ])
    blocks.sort(key=lambda value: (-(value[0] + value[2]), value[1]))
    for index, (x, y, z, kind, width, height, depth) in enumerate(blocks):
        _draw_cuboid(
            ax, x, y, z, kind, width, height, depth,
            zorder=2 + index * 0.06,
        )

    extension = float(state.extension)
    if extension > 0.015:
        _draw_cuboid(
            ax, 0.35, 1.0, 0.35, 'arm',
            width=0.30, height=extension, depth=0.30, zorder=8,
        )
        _draw_cuboid(
            ax, 0.0, 1.0 + extension - 0.18, 0.0, 'piston_head',
            width=1.0, height=0.18, depth=1.0, zorder=9,
        )
    _draw_cuboid(
        ax, 0.0, 1.0 + extension, 0.0, 'load',
        width=1.0, height=1.0, depth=1.0, zorder=10,
    )
    _draw_lever(ax, state.source_on)

    if state.source_on:
        source = _iso_point(1.05, 1.62, 0.05)
        virtual = _iso_point(0.50, 1.50, 0.50)
        piston = _iso_point(0.50, 0.88, 0.50)
        ax.plot(
            [source[0], virtual[0], piston[0]],
            [source[1], virtual[1], piston[1]],
            color=COLORS['magenta'], linewidth=2.0,
            linestyle=(0, (2.0, 2.0)), alpha=0.88, zorder=15,
        )
        ax.scatter(
            [virtual[0]], [virtual[1]], s=96,
            facecolors='none', edgecolors=COLORS['magenta'],
            linewidths=1.0, alpha=0.75, zorder=15,
        )

    if state.update_pulse > 0.0:
        updater = _iso_point(-0.50, 1.08, 0.50)
        ax.scatter(
            [updater[0]], [updater[1]],
            s=260 + 520 * state.update_pulse,
            facecolors='none', edgecolors=COLORS['cyan'],
            linewidths=1.6, alpha=0.85 * state.update_pulse, zorder=16,
        )
        ax.text(
            updater[0] - 0.08, updater[1] + 0.45,
            '♪', color=COLORS['cyan'], fontsize=16,
            ha='center', va='center', alpha=state.update_pulse, zorder=16,
        )

    status = FancyBboxPatch(
        (0.06, 0.035), 0.88, 0.13,
        transform=ax.transAxes,
        boxstyle='round,pad=0.012,rounding_size=0.02',
        facecolor=COLORS['panel'], edgecolor=COLORS['purpur'],
        linewidth=0.85, alpha=0.94, zorder=20,
    )
    ax.add_patch(status)
    ax.text(
        0.09, 0.112, state.phase, transform=ax.transAxes,
        color=COLORS['text'], fontsize=8.2, fontweight='bold',
        family='monospace', ha='left', va='center', zorder=21,
    )
    ax.text(
        0.09, 0.066, state.detail, transform=ax.transAxes,
        color=COLORS['muted'], fontsize=7.0,
        ha='left', va='center', zorder=21,
    )

    if panel_label:
        ax.text(
            0.035, 0.965, panel_label, transform=ax.transAxes,
            color=COLORS['text'], fontsize=11, fontweight='bold',
            ha='left', va='top', zorder=21,
        )
    if heading:
        ax.text(
            0.50, 0.965, heading, transform=ax.transAxes,
            color=COLORS['text'], fontsize=9.5, fontweight='bold',
            family='monospace', ha='center', va='top', zorder=21,
        )
    if labels:
        annotations = [
            ('elevated source', _iso_point(1.65, 2.55, 0.35), (0.88, 0.79)),
            ('update block', _iso_point(-0.60, 1.20, 0.45), (0.16, 0.72)),
            ('piston', _iso_point(0.55, 0.68, 0.15), (0.69, 0.49)),
        ]
        for text, point, text_position in annotations:
            ax.annotate(
                text, xy=point, xycoords='data',
                xytext=text_position, textcoords=ax.transAxes,
                color=COLORS['muted'], fontsize=6.6, family='monospace',
                ha='center', va='center',
                arrowprops=dict(
                    arrowstyle='-', color=COLORS['grid'], linewidth=0.7,
                ),
                zorder=22,
            )


def create_quasi_connectivity_diagram(save_path, dpi=200):
    """Create a three-stage isometric quasi-connectivity diagram."""
    figure, axes = plt.subplots(
        1, 3, figsize=(15.2, 5.5), facecolor=COLORS['background'],
    )
    states = [
        BudVisualState(
            True, 0.0, 0.0, 'QUASI-POWERED', 'Signal present, update missing',
        ),
        BudVisualState(
            True, 0.0, 1.0, 'BLOCK UPDATE', 'The adjacent note block changes',
        ),
        BudVisualState(
            True, 1.0, 0.0, 'PISTON RESPONSE', 'The powered piston extends',
        ),
    ]
    headings = ('1  QUASI POWER', '2  NEIGHBOR UPDATE', '3  STATE CHANGE')
    for index, (axis, state, heading) in enumerate(zip(axes, states, headings)):
        _draw_scene(
            axis, state, labels=index == 0,
            panel_label=f'({chr(97 + index)})', heading=heading,
        )
    figure.suptitle(
        'Quasi-connectivity in Java 1.16.1',
        color=COLORS['text'], fontsize=15, fontweight='bold', y=0.985,
    )
    figure.subplots_adjust(left=0.025, right=0.985, top=0.92, bottom=0.025, wspace=0.025)
    figure.savefig(
        save_path, dpi=dpi, facecolor=COLORS['background'],
        edgecolor='none', bbox_inches='tight',
    )
    plt.close(figure)
    return str(save_path)


def create_quasi_connectivity_animation(
    save_path, fps=10, frames=140, dpi=100, colors=96,
):
    """Animate one complete quasi-connectivity BUD piston cycle."""
    figure, axis = plt.subplots(
        figsize=(9.6, 6.0), facecolor=COLORS['background'],
    )

    def update(frame_index):
        axis.clear()
        _draw_scene(axis, bud_animation_state(frame_index, frames))
        return []

    animation = FuncAnimation(
        figure, update, frames=frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)
    optimize_gif(save_path, colors=colors)
    return str(save_path)


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)
    create_quasi_connectivity_diagram(plots / 'redstone_quasi_connectivity.png')
    create_quasi_connectivity_animation(plots / 'redstone_quasi_connectivity.gif')


if __name__ == '__main__':
    main()
