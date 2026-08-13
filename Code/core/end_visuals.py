"""Shared top-down End arena rendering primitives."""

import math

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Rectangle
import numpy as np

from .end_generation import central_island_projection, spike_layout
from .style import COLORS


ISLAND_CMAP = LinearSegmentedColormap.from_list(
    'end_stone_terraces',
    ['#55583F', '#777A56', '#A8AA78', COLORS['end_stone']],
)
BEDROCK = '#565461'
BEDROCK_DARK = '#302E39'
CRYSTAL = '#F59CFF'
CRYSTAL_CORE = '#F7F4FF'
IRON = '#AAB2C0'


def draw_central_island(
    ax, seed=42, extent=88.0, resolution=241, alpha=0.52, zorder=0,
):
    """Draw a translucent, block-stepped central End island projection."""
    x, z, terraces = central_island_projection(
        seed, extent_blocks=extent, resolution=resolution,
    )
    values = terraces.filled(0.0)
    rgba = ISLAND_CMAP(values)
    visible = ~np.ma.getmaskarray(terraces)
    rgba[..., 3] = np.where(
        visible, alpha * (0.45 + 0.55 * values), 0.0,
    )
    return ax.imshow(
        rgba,
        extent=(x[0], x[-1], z[0], z[-1]),
        origin='lower',
        interpolation='nearest',
        zorder=zorder,
    )


def draw_end_fountain(ax, active=False, zorder=7):
    """Draw the seven-block-wide End podium as a top-down block plan."""
    artists = []
    for block_z in range(-4, 5):
        for block_x in range(-4, 5):
            distance = math.hypot(block_x, block_z)
            if distance > 3.5:
                continue
            if distance > 2.5:
                color = BEDROCK
            elif active:
                color = COLORS['obsidian']
            else:
                continue
            block = Rectangle(
                (block_x - 0.5, block_z - 0.5), 1.0, 1.0,
                facecolor=color, edgecolor=BEDROCK_DARK,
                linewidth=0.68, zorder=zorder,
            )
            ax.add_patch(block)
            artists.append(block)

    pillar = Rectangle(
        (-0.5, -0.5), 1.0, 1.0,
        facecolor='#74717E', edgecolor=COLORS['text'],
        linewidth=0.92, zorder=zorder + 2,
    )
    ax.add_patch(pillar)
    artists.append(pillar)
    rim = Circle(
        (0.0, 0.0), 3.62, fill=False,
        edgecolor='#8D8998', linewidth=1.25, alpha=0.88,
        zorder=zorder + 1.5,
    )
    ax.add_patch(rim)
    artists.append(rim)
    if active:
        portal_points = np.array([
            [-1.7, -0.7], [-0.9, 1.3], [0.8, -1.5], [1.6, 0.8],
        ])
        portal = ax.scatter(
            portal_points[:, 0], portal_points[:, 1],
            s=5, c=COLORS['magenta'], alpha=0.75,
            linewidths=0, zorder=zorder + 1,
        )
        artists.append(portal)
    for torch_x, torch_z in ((0.0, 0.78), (0.78, 0.0), (0.0, -0.78), (-0.78, 0.0)):
        torch = Circle(
            (torch_x, torch_z), 0.12,
            facecolor=COLORS['gold'], edgecolor=COLORS['orange'],
            linewidth=0.3, zorder=zorder + 3,
        )
        ax.add_patch(torch)
        artists.append(torch)
    return artists


def draw_end_spikes(
    ax, seed=42, crystals_alive=10, alpha=1.0, zorder=5,
    radius_override=None, tower_edgecolor='#6E4B86',
    tower_linewidth=0.7, cage_linewidth=0.75, cage_extent=2.7,
    radius_scale=1.0,
):
    """Draw spike footprints with top-down crystals and cage marks.

    ``radius_override`` is a visual-only option for panels that need uniform
    footprints. The source radius remains available in the returned metadata.
    """
    artists = []
    for index, spike in enumerate(spike_layout(seed)):
        x = spike['x']
        z = spike['z']
        source_radius = float(spike['radius'])
        radius = float(radius_scale) * (
            source_radius if radius_override is None
            else float(radius_override)
        )
        tower = Circle(
            (x, z), radius,
            facecolor=COLORS['obsidian'], edgecolor=tower_edgecolor,
            linewidth=tower_linewidth, alpha=0.96 * alpha, zorder=zorder,
        )
        core = Circle(
            (x, z), radius * 0.68,
            facecolor='#130A20', edgecolor='none',
            alpha=0.88 * alpha, zorder=zorder + 0.1,
        )
        ax.add_patch(tower)
        ax.add_patch(core)

        alive = index < int(crystals_alive)
        glow = Circle(
            (x, z), 2.15,
            facecolor=COLORS['magenta'] if alive else COLORS['coral'],
            edgecolor='none', alpha=(0.24 if alive else 0.06) * alpha,
            zorder=zorder + 1,
        )
        crystal = Circle(
            (x, z), radius=1.25,
            facecolor=CRYSTAL if alive else COLORS['coral'],
            edgecolor=CRYSTAL_CORE, linewidth=0.55,
            alpha=(0.98 if alive else 0.14) * alpha,
            zorder=zorder + 2,
        )
        ax.add_patch(glow)
        ax.add_patch(crystal)

        cage = None
        if spike['caged']:
            cage = Rectangle(
                (x - cage_extent, z - cage_extent),
                2.0 * cage_extent, 2.0 * cage_extent,
                fill=False, edgecolor=IRON, linewidth=cage_linewidth,
                alpha=0.9 * alpha, zorder=zorder + 2.5,
            )
            ax.add_patch(cage)
            ax.plot(
                [x - cage_extent, x + cage_extent], [z, z],
                color=IRON, linewidth=cage_linewidth * 0.56,
                alpha=0.82 * alpha,
                zorder=zorder + 2.5,
            )
            ax.plot(
                [x, x], [z - cage_extent, z + cage_extent],
                color=IRON, linewidth=cage_linewidth * 0.56,
                alpha=0.82 * alpha,
                zorder=zorder + 2.5,
            )
        artists.append({
            'x': x,
            'z': z,
            'tower': tower,
            'core': core,
            'glow': glow,
            'crystal': crystal,
            'cage': cage,
            'height': spike['height'],
            'radius': radius,
            'source_radius': source_radius,
        })
    return artists


def set_crystals_alive(spike_artists, crystals_alive):
    """Update crystal visibility without changing the permanent towers."""
    set_crystal_states(spike_artists, range(int(crystals_alive)))


def set_crystal_states(spike_artists, alive_indices):
    """Update an arbitrary, possibly non-circular set of living crystals."""
    alive_indices = set(int(index) for index in alive_indices)
    for index, artists in enumerate(spike_artists):
        alive = index in alive_indices
        artists['glow'].set_facecolor(
            COLORS['magenta'] if alive else COLORS['coral']
        )
        artists['glow'].set_alpha(0.24 if alive else 0.06)
        artists['crystal'].set_facecolor(CRYSTAL if alive else COLORS['coral'])
        artists['crystal'].set_alpha(0.98 if alive else 0.14)
