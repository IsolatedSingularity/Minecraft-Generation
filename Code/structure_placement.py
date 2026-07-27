"""Java 1.16.1 structure-placement animation.

The candidate location and region seed are exact for the 1.16.1 village
structure set. The biome gate is shown as a deterministic visual model, not
as a claim that this small animation replaces the complete vanilla biome
generator.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

from core.constants import (
    CHUNK_SIZE,
    MINECRAFT_VERSION,
    VILLAGE_SALT,
    VILLAGE_SEPARATION,
    VILLAGE_SPACING,
)
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
    'village': '#F8C15C',
    'good': '#86EFAC',
    'bad': '#FB7185',
    'warning': '#FCD34D',
    'candidate': '#7990B4',
}

BIOME_COLORS = {
    'Plains': '#86B96E',
    'Desert': '#D9B35C',
    'Savanna': '#B6A45B',
    'Taiga': '#4D7770',
    'Snowy Tundra': '#B9D5DF',
    'Other': '#66738A',
}


class StructurePlacementSimulator:
    """Exact 1.16.1 candidate placement with a readable biome gate."""

    def __init__(self, world_seed=42, world_size=6144):
        self.world_seed = int(world_seed)
        self.world_size = world_size
        self.chunk_size = CHUNK_SIZE
        self.spacing = VILLAGE_SPACING
        self.separation = VILLAGE_SEPARATION
        self.salt = VILLAGE_SALT
        self.region_blocks = self.spacing * self.chunk_size
        self.region_radius = max(1, world_size // (2 * self.region_blocks))
        self.valid_biomes = {
            'Plains', 'Desert', 'Savanna', 'Taiga', 'Snowy Tundra',
        }
        self.regions = self._generate_spiral_regions(self.region_radius)

    @staticmethod
    def _generate_spiral_regions(radius):
        regions = [(0, 0)]
        x = z = 0
        step = 1
        while len(regions) < (2 * radius + 1) ** 2:
            for _ in range(step):
                x += 1
                if -radius <= x <= radius and -radius <= z <= radius:
                    regions.append((x, z))
            for _ in range(step):
                z += 1
                if -radius <= x <= radius and -radius <= z <= radius:
                    regions.append((x, z))
            step += 1
            for _ in range(step):
                x -= 1
                if -radius <= x <= radius and -radius <= z <= radius:
                    regions.append((x, z))
            for _ in range(step):
                z -= 1
                if -radius <= x <= radius and -radius <= z <= radius:
                    regions.append((x, z))
            step += 1
        return regions[:(2 * radius + 1) ** 2]

    def generate_region_seed(self, region_x, region_z):
        return (
            self.world_seed
            + region_x * 341873128712
            + region_z * 132897987541
            + self.salt
        ) & ((1 << 48) - 1)

    def biome_at(self, block_x, block_z):
        phase = (self.world_seed & 0xFFFF) / 65535.0
        temperature = (
            0.65 * np.sin(block_x / 1500.0 + phase * 4.0)
            + 0.35 * np.cos(block_z / 2200.0 - phase * 3.0)
        )
        rainfall = (
            0.70 * np.cos(block_z / 1800.0 + phase * 2.0)
            + 0.30 * np.sin((block_x + block_z) / 2600.0)
        )
        if temperature < -0.58:
            biome = 'Snowy Tundra'
        elif temperature > 0.58 and rainfall < -0.12:
            biome = 'Desert'
        elif temperature > 0.20 and rainfall > 0.28:
            biome = 'Savanna'
        elif rainfall > 0.62:
            biome = 'Taiga'
        elif temperature > -0.10:
            biome = 'Plains'
        else:
            biome = 'Other'
        return biome, BIOME_COLORS[biome]

    def evaluate_region(self, region_x, region_z, order):
        region_seed = self.generate_region_seed(region_x, region_z)
        rng = MinecraftLCG(region_seed)
        candidate_window = self.spacing - self.separation
        roll_x = rng.next_int(candidate_window)
        roll_z = rng.next_int(candidate_window)
        chunk_x = region_x * self.spacing + roll_x
        chunk_z = region_z * self.spacing + roll_z
        block_x = chunk_x * self.chunk_size + self.chunk_size // 2
        block_z = chunk_z * self.chunk_size + self.chunk_size // 2
        biome, biome_color = self.biome_at(block_x, block_z)
        valid = biome in self.valid_biomes
        return {
            'order': order,
            'region_x': region_x,
            'region_z': region_z,
            'region_seed': region_seed,
            'roll_x': roll_x,
            'roll_z': roll_z,
            'chunk_x': chunk_x,
            'chunk_z': chunk_z,
            'block_x': block_x,
            'block_z': block_z,
            'biome': biome,
            'biome_color': biome_color,
            'valid': valid,
        }

    def evaluate_all(self):
        return [
            self.evaluate_region(region_x, region_z, order)
            for order, (region_x, region_z) in enumerate(self.regions, start=1)
        ]


def _card(ax, y, number, title, body):
    patch = FancyBboxPatch(
        (0.04, y), 0.92, 0.145, transform=ax.transAxes,
        boxstyle='round,pad=0.012', facecolor=COLORS['panel_alt'],
        edgecolor=COLORS['grid'], linewidth=1.0,
    )
    ax.add_patch(patch)
    ax.text(
        0.08, y + 0.097, f'{number}  {title}', transform=ax.transAxes,
        color=COLORS['text'], fontsize=9, fontweight='bold',
    )
    ax.text(
        0.08, y + 0.045, body, transform=ax.transAxes,
        color=COLORS['muted'], fontsize=8, family='monospace',
    )
    return patch


def create_structure_placement_animation(save_path, frames=220, dpi=160, fps=15):
    """Render the exact candidate-placement story as a GIF."""
    simulator = StructurePlacementSimulator(world_seed=42)
    results = simulator.evaluate_all()

    fig = plt.figure(figsize=(18, 10), facecolor=COLORS['background'])
    gs = fig.add_gridspec(
        12, 12, left=0.04, right=0.97, top=0.90, bottom=0.06,
        hspace=0.62, wspace=0.62,
    )
    ax_map = fig.add_subplot(gs[:, :7])
    ax_algorithm = fig.add_subplot(gs[:7, 7:])
    ax_stats = fig.add_subplot(gs[7:, 7:])

    for ax in (ax_map, ax_algorithm, ax_stats):
        ax.set_facecolor(COLORS['panel'])
        for spine in ax.spines.values():
            spine.set_color(COLORS['grid'])
            spine.set_linewidth(1.0)

    fig.suptitle(
        f'VILLAGE CANDIDATE PLACEMENT  |  {MINECRAFT_VERSION}',
        color=COLORS['text'], fontsize=20, fontweight='bold', x=0.04, ha='left',
    )
    fig.text(
        0.97, 0.915,
        'One attempt per 32 x 32 chunk region, then a biome gate',
        color=COLORS['muted'], fontsize=10, ha='right',
    )

    world_half = simulator.world_size // 2
    ax_map.set_xlim(-world_half, world_half)
    ax_map.set_ylim(-world_half, world_half)
    ax_map.set_aspect('equal')
    ax_map.set_title(
        'CANDIDATE MAP  /  SPIRAL REVEAL', color=COLORS['text'],
        fontsize=13, fontweight='bold', loc='left', pad=12,
    )
    ax_map.set_xlabel('Block X', color=COLORS['muted'])
    ax_map.set_ylabel('Block Z', color=COLORS['muted'])
    ax_map.tick_params(colors=COLORS['muted'], labelsize=8)
    ax_map.grid(color=COLORS['grid'], linewidth=0.5, alpha=0.42)

    for coordinate in range(-world_half, world_half + 1, simulator.region_blocks):
        ax_map.axvline(coordinate, color=COLORS['grid'], linewidth=0.7, alpha=0.5)
        ax_map.axhline(coordinate, color=COLORS['grid'], linewidth=0.7, alpha=0.5)

    ax_map.scatter(
        [0], [0], marker='*', s=200, color=COLORS['warning'],
        edgecolors=COLORS['text'], linewidths=1.2, zorder=8,
    )
    ax_map.text(
        0.02, 0.03, 'world origin', transform=ax_map.transAxes,
        color=COLORS['warning'], fontsize=9, fontweight='bold',
    )
    candidate_scatter = ax_map.scatter(
        [], [], s=20, color=COLORS['candidate'], alpha=0.65,
        linewidths=0, zorder=3,
    )
    valid_scatter = ax_map.scatter(
        [], [], s=70, color=COLORS['village'], edgecolors=COLORS['text'],
        linewidths=0.7, zorder=5,
    )
    rejected_scatter = ax_map.scatter(
        [], [], s=36, color=COLORS['bad'], marker='x',
        linewidths=1.2, zorder=4,
    )
    current_region = Rectangle(
        (0, 0), simulator.region_blocks, simulator.region_blocks,
        fill=False, edgecolor=COLORS['accent'], linewidth=2.0, alpha=0.0,
        zorder=7,
    )
    ax_map.add_patch(current_region)
    map_note = ax_map.text(
        0.98, 0.03,
        'candidate = exact grid stage   |   gold = biome-pass preview',
        transform=ax_map.transAxes, ha='right', color=COLORS['muted'],
        fontsize=8,
    )

    ax_algorithm.axis('off')
    ax_algorithm.text(
        0.04, 0.94, 'ALGORITHM TRACE', transform=ax_algorithm.transAxes,
        color=COLORS['text'], fontsize=12, fontweight='bold',
    )
    cards = [
        _card(
            ax_algorithm, 0.75, '01', 'REGION SEED',
            'seed + rx*K1 + rz*K2 + salt',
        ),
        _card(
            ax_algorithm, 0.57, '02', 'JAVA RANDOM',
            'setSeed(regionSeed), nextInt(24)',
        ),
        _card(
            ax_algorithm, 0.39, '03', 'CANDIDATE CHUNK',
            'region * 32 + roll  [0, 23]',
        ),
        _card(
            ax_algorithm, 0.21, '04', 'BIOME GATE',
            'valid biome -> candidate can generate',
        ),
    ]
    region_text = ax_algorithm.text(
        0.06, 0.14, '', transform=ax_algorithm.transAxes,
        color=COLORS['accent'], fontsize=9, family='monospace',
    )
    seed_text = ax_algorithm.text(
        0.06, 0.09, '', transform=ax_algorithm.transAxes,
        color=COLORS['good'], fontsize=9, family='monospace',
    )
    outcome_text = ax_algorithm.text(
        0.60, 0.14, '', transform=ax_algorithm.transAxes,
        color=COLORS['warning'], fontsize=10, fontweight='bold',
        ha='right',
    )

    ax_stats.axis('off')
    ax_stats.text(
        0.04, 0.92, 'RUN STATUS', transform=ax_stats.transAxes,
        color=COLORS['text'], fontsize=12, fontweight='bold',
    )
    progress_text = ax_stats.text(
        0.04, 0.72, '', transform=ax_stats.transAxes,
        color=COLORS['accent'], fontsize=11, fontweight='bold',
    )
    stats_text = ax_stats.text(
        0.04, 0.53, '', transform=ax_stats.transAxes,
        color=COLORS['text'], fontsize=9, linespacing=1.6,
    )
    bar_labels = [
        ('Biome-pass previews', COLORS['good']),
        ('Biome rejects', COLORS['bad']),
    ]
    bars = []
    for i, (label, color) in enumerate(bar_labels):
        y = 0.24 - i * 0.10
        ax_stats.text(
            0.04, y + 0.02, label, transform=ax_stats.transAxes,
            color=COLORS['muted'], fontsize=8,
        )
        ax_stats.add_patch(Rectangle(
            (0.40, y), 0.52, 0.045, transform=ax_stats.transAxes,
            facecolor=COLORS['grid'], edgecolor='none',
        ))
        bar = Rectangle(
            (0.40, y), 0.01, 0.045, transform=ax_stats.transAxes,
            facecolor=color, edgecolor='none',
        )
        ax_stats.add_patch(bar)
        bars.append(bar)

    def update(frame):
        progress = frame / max(1, frames - 1)
        scanned_count = min(len(results), int(progress * len(results)))
        scanned = results[:scanned_count]
        valid = [item for item in scanned if item['valid']]
        rejected = [item for item in scanned if not item['valid']]

        candidate_scatter.set_offsets(
            np.array([[item['block_x'], item['block_z']] for item in scanned])
            if scanned else np.empty((0, 2))
        )
        candidate_scatter.set_facecolors(
            [item['biome_color'] for item in scanned]
            if scanned else []
        )
        valid_scatter.set_offsets(
            np.array([[item['block_x'], item['block_z']] for item in valid])
            if valid else np.empty((0, 2))
        )
        rejected_scatter.set_offsets(
            np.array([[item['block_x'], item['block_z']] for item in rejected])
            if rejected else np.empty((0, 2))
        )

        if scanned:
            item = scanned[-1]
            current_region.set_xy((
                item['region_x'] * simulator.region_blocks,
                item['region_z'] * simulator.region_blocks,
            ))
            current_region.set_alpha(0.95)
            region_text.set_text(
                f"region ({item['region_x']:>2}, {item['region_z']:>2})"
                f"  /  attempt {item['order']:03d}"
            )
            seed_text.set_text(
                f"0x{item['region_seed']:012X}  |  rolls {item['roll_x']:02d}, {item['roll_z']:02d}"
            )
            outcome_text.set_text(
                'BIOME PASS' if item['valid'] else 'BIOME REJECT'
            )
            outcome_text.set_color(COLORS['good'] if item['valid'] else COLORS['bad'])
            map_note.set_text(
                f"candidate chunk ({item['chunk_x']}, {item['chunk_z']})"
                f"  |  {item['biome']}"
            )
        else:
            current_region.set_alpha(0.0)
            region_text.set_text('waiting for first region')
            seed_text.set_text('')
            outcome_text.set_text('')
            map_note.set_text(
                'candidate = exact grid stage   |   gold = biome-pass preview'
            )

        viable_count = len(valid)
        rejected_count = len(rejected)
        progress_text.set_text(
            f'{scanned_count:03d}/{len(results)} regions scanned'
            f'  |  {viable_count:03d} biome-pass previews'
        )
        stats_text.set_text(
            f'World seed             42\n'
            f'Candidate window       24 x 24 chunks\n'
            f'One candidate / region  {scanned_count:03d}\n'
            f'Biome-pass share       '
            f'{100 * viable_count / max(1, scanned_count):5.1f}%'
        )
        denominator = max(1, scanned_count)
        bars[0].set_width(0.52 * viable_count / denominator)
        bars[1].set_width(0.52 * rejected_count / denominator)
        return []

    animation = FuncAnimation(
        fig, update, frames=frames, interval=1000 / fps, blit=False,
    )
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return save_path


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(os.path.dirname(script_dir), 'Plots')
    os.makedirs(plots_dir, exist_ok=True)
    output_path = os.path.join(plots_dir, 'structure_placement.gif')
    create_structure_placement_animation(output_path)
