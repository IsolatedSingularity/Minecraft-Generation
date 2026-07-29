"""Shared scientific styling for active repository visualizations."""

import matplotlib as mpl


COLORS = {
    'background': '#090B12',
    'panel': '#111622',
    'panel_alt': '#171E2C',
    'grid': '#273043',
    'text': '#F1F5F9',
    'muted': '#9AA8BC',
    'blue': '#65C7F7',
    'cyan': '#43D9C2',
    'violet': '#A78BFA',
    'magenta': '#E879F9',
    'coral': '#FB7185',
    'orange': '#FB923C',
    'gold': '#F6C85F',
    'green': '#73D49B',
    'end_stone': '#D6D7A8',
    'end_shadow': '#777954',
    'obsidian': '#211532',
    'purpur': '#B879B3',
    'portal': '#65D6AD',
    'fortress': '#E06C75',
    'bastion': '#D6A35A',
    'ruined_portal': '#B88AE8',
    'stronghold': '#58B9D9',
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
    """Apply consistent Matplotlib defaults without a game UI motif."""
    mpl.rcParams.update({
        'figure.facecolor': COLORS['background'],
        'savefig.facecolor': COLORS['background'],
        'axes.facecolor': COLORS['background'],
        'axes.edgecolor': COLORS['grid'],
        'axes.labelcolor': COLORS['muted'],
        'axes.titlecolor': COLORS['text'],
        'xtick.color': COLORS['muted'],
        'ytick.color': COLORS['muted'],
        'text.color': COLORS['text'],
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.linewidth': 0.8,
        'grid.color': COLORS['grid'],
        'grid.alpha': 0.38,
        'grid.linewidth': 0.6,
        'legend.facecolor': COLORS['panel'],
        'legend.edgecolor': COLORS['grid'],
        'legend.labelcolor': COLORS['text'],
    })


def style_axis(ax, equal=False, grid=True):
    """Apply restrained scientific axis styling."""
    ax.set_facecolor(COLORS['background'])
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
    ax.tick_params(colors=COLORS['muted'], labelsize=8)
    if grid:
        ax.grid(True, color=COLORS['grid'], alpha=0.32, linewidth=0.55)
    if equal:
        ax.set_aspect('equal', adjustable='box')
    return ax
