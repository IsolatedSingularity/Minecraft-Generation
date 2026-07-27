"""Seed loading and Java 1.16.1 world-generation animation.

The animation is intentionally explicit about scope. The seed and Java LCG
states are exact; the biome panel is a compact, deterministic 1.16.1-style
layer preview rather than a claim to replace the full vanilla biome engine.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, Rectangle

from core.constants import MINECRAFT_VERSION
from core.lcg import MinecraftLCG


plt.style.use('dark_background')

COLORS = {
    'background': '#0B1020',
    'panel': '#111A2E',
    'panel_alt': '#17233B',
    'grid': '#263653',
    'text': '#E8EEF9',
    'muted': '#93A4C3',
    'accent': '#7DD3FC',
    'accent_2': '#A78BFA',
    'good': '#86EFAC',
    'warning': '#FCD34D',
    'bad': '#FB7185',
    'unloaded': '#18243A',
}

BIOME_COLORS = {
    'ocean': '#215A88',
    'plains': '#79B86A',
    'forest': '#2E7D5B',
    'desert': '#D9B35C',
    'taiga': '#416B68',
    'snowy_tundra': '#B9D5DF',
    'swamp': '#657B4B',
}


def rgb(color):
    color = color.lstrip('#')
    return np.array([int(color[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def generate_chunk_spiral(radius):
    """Return chunk coordinates in a deterministic outward square spiral."""
    chunks = [(0, 0)]
    x = z = 0
    step = 1
    while len(chunks) < (2 * radius + 1) ** 2:
        for _ in range(step):
            x += 1
            if abs(x) <= radius and abs(z) <= radius:
                chunks.append((x, z))
        for _ in range(step):
            z += 1
            if abs(x) <= radius and abs(z) <= radius:
                chunks.append((x, z))
        step += 1
        for _ in range(step):
            x -= 1
            if abs(x) <= radius and abs(z) <= radius:
                chunks.append((x, z))
        for _ in range(step):
            z -= 1
            if abs(x) <= radius and abs(z) <= radius:
                chunks.append((x, z))
        step += 1
    return chunks[:(2 * radius + 1) ** 2]


class BiomeLayerPreview:
    """Small deterministic biome layer preview for the animation."""

    def __init__(self, seed):
        self.seed = int(seed)

    def biome_at(self, chunk_x, chunk_z):
        phase = self.seed % 100000 / 100000.0
        temperature = (
            np.sin(chunk_x * 0.33 + phase * 8.0)
            + 0.45 * np.cos(chunk_z * 0.18 - phase * 5.0)
        )
        rainfall = (
            np.cos(chunk_z * 0.27 + phase * 4.0)
            + 0.35 * np.sin((chunk_x + chunk_z) * 0.14)
        )
        if temperature < -0.9:
            return 'snowy_tundra'
        if temperature > 0.95 and rainfall < -0.2:
            return 'desert'
        if rainfall > 0.85:
            return 'swamp'
        if temperature > 0.45 and rainfall > 0.15:
            return 'forest'
        if temperature < -0.15 and rainfall > 0.15:
            return 'taiga'
        if abs(chunk_x * 3 + chunk_z * 5) % 17 == 0:
            return 'ocean'
        return 'plains'


def _phase(progress):
    if progress < 0.08:
        return 'seed', progress / 0.08
    if progress < 0.17:
        return 'rng', (progress - 0.08) / 0.09
    if progress < 0.94:
        return 'chunks', (progress - 0.17) / 0.77
    return 'complete', (progress - 0.94) / 0.06


def create_seed_loading_animation(save_path, seed=-4172144997902289642,
                                  fps=20, duration=15):
    """Render the seed-loading animation to a GIF."""
    total_frames = max(2, int(fps * duration))
    chunk_radius = 10
    chunk_spiral = generate_chunk_spiral(chunk_radius)
    total_chunks = len(chunk_spiral)
    biome_preview = BiomeLayerPreview(seed)

    fig = plt.figure(figsize=(18, 10), facecolor=COLORS['background'])
    gs = fig.add_gridspec(
        12, 12, left=0.04, right=0.97, top=0.91, bottom=0.06,
        hspace=0.65, wspace=0.65,
    )
    ax_map = fig.add_subplot(gs[:, :7])
    ax_pipeline = fig.add_subplot(gs[:4, 7:])
    ax_rng = fig.add_subplot(gs[4:8, 7:])
    ax_stats = fig.add_subplot(gs[8:, 7:])

    for ax in (ax_map, ax_pipeline, ax_rng, ax_stats):
        ax.set_facecolor(COLORS['panel'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
            spine.set_linewidth(1.0)

    fig.suptitle(
        f'WORLD GENERATION | {MINECRAFT_VERSION}',
        color=COLORS['text'], fontsize=20, fontweight='bold', x=0.04, ha='left',
    )
    fig.text(
        0.97, 0.925,
        'A readable visual trace of seed -> state -> chunk layers',
        color=COLORS['muted'], fontsize=10, ha='right',
    )

    size = 2 * chunk_radius + 1
    map_rgb = np.empty((size, size, 3), dtype=float)
    map_rgb[:] = rgb(COLORS['unloaded'])
    map_image = ax_map.imshow(
        map_rgb, origin='lower', interpolation='nearest',
        extent=(-chunk_radius - 0.5, chunk_radius + 0.5,
                -chunk_radius - 0.5, chunk_radius + 0.5),
    )
    ax_map.set_title(
        'CHUNK LAYER  /  OUTWARD SPIRAL', color=COLORS['text'],
        fontsize=13, fontweight='bold', loc='left', pad=12,
    )
    ax_map.set_xlabel('Chunk X', color=COLORS['muted'])
    ax_map.set_ylabel('Chunk Z', color=COLORS['muted'])
    ax_map.tick_params(colors=COLORS['muted'], labelsize=8)
    ax_map.grid(color=COLORS['grid'], linewidth=0.5, alpha=0.45)
    ax_map.scatter(
        [0], [0], marker='*', s=180, color=COLORS['warning'],
        edgecolors=COLORS['text'], linewidths=1.2, zorder=5,
    )
    ax_map.text(
        0.02, 0.03, 'spawn', transform=ax_map.transAxes,
        color=COLORS['warning'], fontsize=9, fontweight='bold',
    )
    path_line, = ax_map.plot(
        [], [], color=COLORS['accent'], linewidth=1.0, alpha=0.85, zorder=4,
    )
    current_chunk = Rectangle(
        (-0.5, -0.5), 1, 1, fill=False, edgecolor=COLORS['warning'],
        linewidth=2.2, zorder=6,
    )
    ax_map.add_patch(current_chunk)
    map_progress_bg = FancyBboxPatch(
        (0.04, 0.94), 0.92, 0.025, transform=ax_map.transAxes,
        boxstyle='round,pad=0.01', facecolor=COLORS['grid'],
        edgecolor='none', zorder=8,
    )
    map_progress_fg = FancyBboxPatch(
        (0.04, 0.94), 0.01, 0.025, transform=ax_map.transAxes,
        boxstyle='round,pad=0.01', facecolor=COLORS['accent'],
        edgecolor='none', zorder=9,
    )
    ax_map.add_patch(map_progress_bg)
    ax_map.add_patch(map_progress_fg)
    progress_label = ax_map.text(
        0.96, 0.90, '', transform=ax_map.transAxes, ha='right',
        color=COLORS['muted'], fontsize=9,
    )

    ax_pipeline.axis('off')
    ax_pipeline.text(
        0.04, 0.92, 'GENERATION PIPELINE', transform=ax_pipeline.transAxes,
        color=COLORS['text'], fontsize=12, fontweight='bold',
    )
    pipeline_specs = [
        ('01', 'WORLD SEED', '64-bit input'),
        ('02', 'JAVA LCG', '48-bit internal state'),
        ('03', 'BIOME LAYER', '1.16.1-style preview'),
        ('04', 'CHUNK SAMPLE', '16 x 16 blocks'),
    ]
    pipeline_patches = []
    pipeline_labels = []
    for i, (number, title, subtitle) in enumerate(pipeline_specs):
        y = 0.72 - i * 0.17
        patch = FancyBboxPatch(
            (0.04, y - 0.055), 0.92, 0.12, transform=ax_pipeline.transAxes,
            boxstyle='round,pad=0.012', facecolor=COLORS['panel_alt'],
            edgecolor=COLORS['grid'], linewidth=1.0,
        )
        ax_pipeline.add_patch(patch)
        pipeline_patches.append(patch)
        label = ax_pipeline.text(
            0.08, y + 0.015, f'{number}  {title}', transform=ax_pipeline.transAxes,
            color=COLORS['text'], fontsize=9, fontweight='bold', va='center',
        )
        ax_pipeline.text(
            0.08, y - 0.025, subtitle, transform=ax_pipeline.transAxes,
            color=COLORS['muted'], fontsize=8, va='center',
        )
        pipeline_labels.append(label)

    ax_rng.axis('off')
    ax_rng.text(
        0.04, 0.92, 'JAVA RANDOM STATE', transform=ax_rng.transAxes,
        color=COLORS['text'], fontsize=12, fontweight='bold',
    )
    seed_text = ax_rng.text(
        0.04, 0.71, '', transform=ax_rng.transAxes, color=COLORS['accent'],
        fontsize=10, family='monospace',
    )
    state_text = ax_rng.text(
        0.04, 0.54, '', transform=ax_rng.transAxes, color=COLORS['good'],
        fontsize=10, family='monospace',
    )
    formula_text = ax_rng.text(
        0.04, 0.24,
        'next = (state * 0x5DEECE66D + 0xB) mod 2^48',
        transform=ax_rng.transAxes, color=COLORS['muted'], fontsize=8,
        family='monospace',
    )
    ax_rng.text(
        0.04, 0.39, 'Visual trace advances once per revealed chunk.',
        transform=ax_rng.transAxes, color=COLORS['muted'], fontsize=8,
    )

    ax_stats.axis('off')
    ax_stats.text(
        0.04, 0.92, 'RUN STATUS', transform=ax_stats.transAxes,
        color=COLORS['text'], fontsize=12, fontweight='bold',
    )
    phase_text = ax_stats.text(
        0.04, 0.72, '', transform=ax_stats.transAxes,
        color=COLORS['accent'], fontsize=12, fontweight='bold',
    )
    stats_text = ax_stats.text(
        0.04, 0.52, '', transform=ax_stats.transAxes,
        color=COLORS['text'], fontsize=9, linespacing=1.6,
    )
    legend_y = 0.08
    for i, (name, color) in enumerate(BIOME_COLORS.items()):
        x = 0.04 + (i % 4) * 0.24
        y = legend_y + (i // 4) * 0.10
        ax_stats.add_patch(Rectangle(
            (x, y), 0.025, 0.045, transform=ax_stats.transAxes,
            facecolor=color, edgecolor='none',
        ))
        ax_stats.text(
            x + 0.035, y + 0.022, name.replace('_', ' ').title(),
            transform=ax_stats.transAxes, color=COLORS['muted'],
            fontsize=7, va='center',
        )

    def update(frame):
        progress = frame / (total_frames - 1)
        phase, phase_progress = _phase(progress)
        if phase == 'chunks':
            chunks_loaded = int(phase_progress * total_chunks)
        elif phase == 'complete':
            chunks_loaded = total_chunks
        else:
            chunks_loaded = 0
        chunks_loaded = min(total_chunks, max(0, chunks_loaded))

        map_rgb[:] = rgb(COLORS['unloaded'])
        loaded = chunk_spiral[:chunks_loaded]
        for chunk_x, chunk_z in loaded:
            biome = biome_preview.biome_at(chunk_x, chunk_z)
            map_rgb[chunk_z + chunk_radius, chunk_x + chunk_radius] = rgb(
                BIOME_COLORS[biome]
            )
        map_image.set_data(map_rgb)

        if loaded:
            path_line.set_data(
                [point[0] for point in loaded],
                [point[1] for point in loaded],
            )
            chunk_x, chunk_z = loaded[-1]
            current_chunk.set_xy((chunk_x - 0.5, chunk_z - 0.5))
        else:
            path_line.set_data([], [])
            current_chunk.set_xy((-0.5, -0.5))

        map_progress_fg.set_width(max(0.01, 0.92 * progress))
        progress_label.set_text(f'{progress * 100:5.1f}%  |  {chunks_loaded}/{total_chunks}')

        active_index = {'seed': 0, 'rng': 1, 'chunks': 3, 'complete': 3}[phase]
        for i, patch in enumerate(pipeline_patches):
            active = i == active_index
            patch.set_edgecolor(COLORS['accent'] if active else COLORS['grid'])
            patch.set_linewidth(1.8 if active else 1.0)
            patch.set_facecolor(COLORS['panel_alt'] if not active else '#203454')
            pipeline_labels[i].set_color(COLORS['accent'] if active else COLORS['text'])

        trace = MinecraftLCG(seed)
        for _ in range(min(chunks_loaded, 32)):
            trace.next_bits(32)
        seed_text.set_text(f'world seed  {int(seed):d}  /  0x{int(seed) & ((1 << 64) - 1):016X}')
        state_text.set_text(f'LCG state   0x{trace.seed:012X}')
        phase_labels = {
            'seed': 'Initializing seed',
            'rng': 'Advancing Java LCG',
            'chunks': 'Revealing chunk layers',
            'complete': 'World sample ready',
        }
        phase_text.set_text(phase_labels[phase])

        counts = {}
        for chunk_x, chunk_z in loaded:
            biome = biome_preview.biome_at(chunk_x, chunk_z)
            counts[biome] = counts.get(biome, 0) + 1
        top_counts = sorted(counts.items(), key=lambda item: -item[1])[:3]
        summary = [
            f'Chunks revealed   {chunks_loaded:>3}/{total_chunks}',
            f'Biome categories  {len(counts):>3}',
        ]
        summary.extend(
            f'{name.replace("_", " ").title():<16} {count:>3}'
            for name, count in top_counts
        )
        stats_text.set_text('\\n'.join(summary))

        return []

    animation = FuncAnimation(
        fig, update, frames=total_frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)
    return save_path


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(os.path.dirname(script_dir), 'Plots')
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, 'seed_loading.gif')
    create_seed_loading_animation(output_path)
