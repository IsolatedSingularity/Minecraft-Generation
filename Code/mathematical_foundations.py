"""Static figures for the README's mathematical foundations."""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from core.end_generation import SimplexNoise2D
from core.lcg import MinecraftLCG
from core.style import COLORS, apply_style, style_axis


apply_style()


def create_lcg_bit_figure(save_path, dpi=190, seed=42, calls=64):
    """Show exact 48-bit state updates and the high bits returned to callers."""
    random = MinecraftLCG(seed)
    states = []
    for _ in range(int(calls)):
        random.next_bits(16)
        states.append(random.seed)

    states = np.asarray(states, dtype=np.uint64)
    bit_rows = np.array([
        [(int(state) >> shift) & 1 for shift in range(47, -1, -1)]
        for state in states[:24]
    ])

    figure, (state_axis, bit_axis) = plt.subplots(
        1, 2, figsize=(12.8, 5.7), facecolor=COLORS['background'],
        gridspec_kw={'width_ratios': [1.0, 1.42]},
    )
    calls_axis = np.arange(1, len(states) + 1)
    normalized = states.astype(np.float64) / float(1 << 48)
    state_axis.plot(
        calls_axis, normalized, color=COLORS['cyan'], linewidth=1.1,
        alpha=0.58, zorder=2,
    )
    state_axis.scatter(
        calls_axis, normalized, c=COLORS['gold'], s=23,
        edgecolors=COLORS['background'], linewidths=0.28, zorder=3,
    )
    state_axis.set_xlim(1, len(states))
    state_axis.set_ylim(-0.03, 1.03)
    state_axis.set_xlabel('Call number')
    state_axis.set_ylabel(r'Internal state $X_n / 2^{48}$')
    state_axis.set_title('The first 64 state updates', fontsize=11.5, pad=9)
    style_axis(state_axis, grid=True)

    bit_axis.imshow(
        bit_rows, aspect='auto', interpolation='nearest', origin='upper',
        cmap=plt.matplotlib.colors.ListedColormap(['#171D2A', '#B96BE3']),
        vmin=0, vmax=1,
    )
    bit_axis.axvspan(-0.5, 15.5, facecolor=COLORS['cyan'], alpha=0.13)
    bit_axis.axvline(15.5, color=COLORS['cyan'], linewidth=1.5, alpha=0.9)
    bit_axis.text(
        0.16, 0.975, 'returned by next_bits(16)',
        transform=bit_axis.transAxes, ha='center', va='top',
        color=COLORS['cyan'], fontsize=8.7, fontweight='bold',
        bbox=dict(facecolor=COLORS['background'], edgecolor='none', alpha=0.78, pad=1.5),
    )
    bit_axis.text(
        0.67, 0.975, 'retained inside the generator',
        transform=bit_axis.transAxes, ha='center', va='top',
        color=COLORS['muted'], fontsize=8.7,
        bbox=dict(facecolor=COLORS['background'], edgecolor='none', alpha=0.78, pad=1.5),
    )
    bit_axis.set_xticks([0, 7, 15, 23, 31, 39, 47])
    bit_axis.set_xticklabels(['47', '40', '32', '24', '16', '8', '0'])
    bit_axis.set_yticks([0, 5, 11, 17, 23])
    bit_axis.set_yticklabels(['1', '6', '12', '18', '24'])
    bit_axis.set_xlabel('Bit position in the 48-bit state')
    bit_axis.set_ylabel('Call number')
    bit_axis.set_title('The high bits become the answer', fontsize=11.5, pad=9)
    for spine in bit_axis.spines.values():
        spine.set_color(COLORS['grid'])
    bit_axis.tick_params(colors=COLORS['muted'])

    figure.suptitle(
        "JAVA'S 48-BIT CLOCKWORK", color=COLORS['text'],
        fontsize=17.5, fontweight='black', y=0.98,
    )
    figure.text(
        0.5, 0.925, 'The machine never improvises. It only advances.',
        ha='center', va='center', color=COLORS['muted'], fontsize=9.5,
        style='italic',
    )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.12, wspace=0.22)
    figure.savefig(save_path, dpi=dpi, facecolor=COLORS['background'])
    plt.close(figure)
    return str(save_path)


def create_brownian_composition_figure(save_path, dpi=190, seed=42):
    """Show four weighted spatial octaves and their deterministic sum."""
    coordinates = np.linspace(-3.2, 3.2, 241)
    grid_x, grid_z = np.meshgrid(coordinates, coordinates)
    sampler = SimplexNoise2D(seed)
    octaves = [
        (0.5 ** index) * sampler.sample_grid(
            grid_x * (2.0 ** index),
            grid_z * (2.0 ** index),
        )
        for index in range(4)
    ]
    combined = np.sum(octaves, axis=0)
    fields = [*octaves, combined]
    maximum = max(float(np.max(np.abs(field))) for field in fields)

    figure, axes = plt.subplots(
        2, 3, figsize=(13.4, 7.7), facecolor=COLORS['background'],
    )
    image_axes = list(axes.flat[:5])
    titles = (
        r'Octave 0: $\eta(x,z)$',
        r'Octave 1: $\frac{1}{2}\eta(2x,2z)$',
        r'Octave 2: $\frac{1}{4}\eta(4x,4z)$',
        r'Octave 3: $\frac{1}{8}\eta(8x,8z)$',
        'Weighted sum',
    )
    image = None
    for axis, field, title in zip(image_axes, fields, titles):
        image = axis.imshow(
            field, extent=(-3.2, 3.2, -3.2, 3.2), origin='lower',
            cmap='coolwarm', vmin=-maximum, vmax=maximum,
            interpolation='bilinear',
        )
        axis.set_title(title, fontsize=10.7, pad=7)
        axis.set_xlabel('x')
        axis.set_ylabel('z')
        style_axis(axis, equal=True, grid=False)

    profile_axis = axes.flat[5]
    center = len(coordinates) // 2
    for index, field in enumerate(octaves):
        profile_axis.plot(
            coordinates, field[center], linewidth=1.0,
            alpha=0.54 + 0.10 * index, label=f'octave {index}',
        )
    profile_axis.plot(
        coordinates, combined[center], color=COLORS['text'],
        linewidth=2.2, label='sum', zorder=5,
    )
    profile_axis.set_title('One slice through every scale', fontsize=10.7, pad=7)
    profile_axis.set_xlabel('x at z = 0')
    profile_axis.set_ylabel('Weighted noise value')
    style_axis(profile_axis, grid=True)
    profile_axis.legend(
        loc='upper right', ncol=2, fontsize=7.4,
        facecolor=COLORS['panel'], edgecolor=COLORS['grid'], framealpha=0.92,
    )

    figure.subplots_adjust(
        left=0.06, right=0.985, top=0.875, bottom=0.17,
        wspace=0.18, hspace=0.33,
    )
    colorbar_axis = figure.add_axes([0.20, 0.065, 0.60, 0.025])
    colorbar = figure.colorbar(image, cax=colorbar_axis, orientation='horizontal')
    colorbar.set_label('Weighted contribution')
    colorbar.outline.set_edgecolor(COLORS['grid'])
    figure.suptitle(
        'ONE LANDSCAPE, SEVERAL TEMPOS', color=COLORS['text'],
        fontsize=17.5, fontweight='black', y=0.985,
    )
    figure.text(
        0.5, 0.94, 'Broad shapes carry the weight. Fine scales roughen the edges.',
        ha='center', va='center', color=COLORS['muted'], fontsize=9.5,
        style='italic',
    )
    figure.savefig(save_path, dpi=dpi, facecolor=COLORS['background'])
    plt.close(figure)
    return str(save_path)


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)
    create_lcg_bit_figure(plots / 'lcg_bit_extraction.png')
    create_brownian_composition_figure(plots / 'brownian_noise_composition.png')


if __name__ == '__main__':
    main()
