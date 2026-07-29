"""Shared light iOS 10 styling for active repository visualizations."""

import matplotlib as mpl
import matplotlib.patheffects as pathEffects


COLORS = {
    'background': '#F2F2F7',
    'panel': '#FFFFFF',
    'panel_alt': '#F8F8FB',
    'grid': '#D6D6DC',
    'shadow': '#A7A7AE',
    'text': '#1C1C1E',
    'muted': '#6D6D72',
    'blue': '#4A90E2',
    'cyan': '#5AC8FA',
    'violet': '#9C8CF2',
    'magenta': '#D780D6',
    'coral': '#FF6B6B',
    'orange': '#FF9F43',
    'gold': '#F4C542',
    'green': '#5CCB73',
    'end_stone': '#D6D7A8',
    'end_shadow': '#AAA982',
    'obsidian': '#352B45',
    'purpur': '#C58AC1',
    'portal': '#30B993',
    'fortress': '#D75A5E',
    'bastion': '#C98A3E',
    'ruined_portal': '#9167D8',
    'stronghold': '#3F9BC6',
}

STATE_COLORS = {
    'holding': COLORS['blue'],
    'strafing': COLORS['coral'],
    'charging': COLORS['orange'],
    'landing_approach': COLORS['violet'],
    'landing': COLORS['magenta'],
    'perching': COLORS['green'],
    'takeoff': COLORS['gold'],
}


def apply_style():
    """Apply a polished light visual system inspired by iOS 10."""
    mpl.rcParams.update({
        'figure.facecolor': COLORS['background'],
        'savefig.facecolor': COLORS['background'],
        'axes.facecolor': COLORS['panel'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['muted'],
        'axes.titlecolor': COLORS['text'],
        'xtick.color': COLORS['muted'],
        'ytick.color': COLORS['muted'],
        'text.color': COLORS['text'],
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'font.weight': 'regular',
        'axes.linewidth': 0.7,
        'axes.titleweight': 'bold',
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.48,
        'grid.linewidth': 0.55,
        'legend.facecolor': COLORS['panel'],
        'legend.edgecolor': COLORS['grid'],
        'legend.labelcolor': COLORS['text'],
        'legend.fancybox': True,
        'legend.framealpha': 0.96,
    })


def addSoftShadow(artist, offset=(1.5, -1.5), alpha=0.18):
    """Give a patch or panel the subtle raised depth used by iOS controls."""
    artist.set_path_effects([
        pathEffects.SimplePatchShadow(
            offset=offset,
            shadow_rgbFace=COLORS['shadow'],
            alpha=alpha,
            rho=0.98,
        ),
        pathEffects.Normal(),
    ])
    return artist


def style_axis(ax, equal=False, grid=True):
    """Apply the shared white-card axis styling."""
    ax.set_facecolor(COLORS['panel'])
    addSoftShadow(ax.patch, offset=(2.0, -2.0), alpha=0.14)
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['muted'], labelsize=8)
    if grid:
        ax.grid(True, color=COLORS['grid'], alpha=0.42, linewidth=0.55)
    if equal:
        ax.set_aspect('equal', adjustable='box')
    return ax
