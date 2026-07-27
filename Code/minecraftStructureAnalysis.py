"""Comprehensive Java 1.16.1 structure analysis.

The figure separates exact candidate placement from the biome-layer preview.
Village candidates use the vanilla 1.16.1 region formula and Java LCG. The
small biome panel is a deterministic visual model, while stronghold points are
the exact seeded ring candidates before the vanilla biome search.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge, FancyBboxPatch
import numpy as np

from core.constants import (
    CHUNK_SIZE,
    MINECRAFT_VERSION,
    STRONGHOLD_RINGS,
    TOTAL_STRONGHOLDS,
    VILLAGE_SALT,
    VILLAGE_SEPARATION,
    VILLAGE_SPACING,
)
from core.lcg import MinecraftLCG
from core.strongholds import generate_stronghold_candidates


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
}

BIOME_COLORS = {
    'Ocean': '#215A88',
    'Plains': '#86B96E',
    'Forest': '#2E7D5B',
    'Desert': '#D9B35C',
    'Savanna': '#B6A45B',
    'Taiga': '#4D7770',
    'Snowy Tundra': '#B9D5DF',
    'Other': '#66738A',
}


class StructureConfig:
    """Version-specific structure-set values used in the analysis."""

    CONFIGS = {
        'village': {
            'spacing': VILLAGE_SPACING,
            'separation': VILLAGE_SEPARATION,
            'salt': VILLAGE_SALT,
            'color': COLORS['village'],
        },
        'stronghold': {
            'rings': STRONGHOLD_RINGS,
            'color': '#A78BFA',
        },
    }


class StructureGenerator:
    """Generate Java 1.16.1 structure candidates."""

    def __init__(self, world_seed):
        self.world_seed = int(world_seed)

    def get_region_seed(self, region_x, region_z, salt):
        return (
            self.world_seed
            + region_x * 341873128712
            + region_z * 132897987541
            + salt
        ) & ((1 << 48) - 1)

    def get_structure_chunk(
        self, region_x, region_z, spacing=VILLAGE_SPACING,
        separation=VILLAGE_SEPARATION, salt=VILLAGE_SALT,
    ):
        region_seed = self.get_region_seed(region_x, region_z, salt)
        rng = MinecraftLCG(region_seed)
        window = spacing - separation
        offset_x = rng.next_int(window)
        offset_z = rng.next_int(window)
        return (
            region_x * spacing + offset_x,
            region_z * spacing + offset_z,
        )

    def generate_structures(self, structure_type, region_range=8):
        if structure_type == 'stronghold':
            return [
                {
                    **candidate,
                    'block_x': candidate['x'],
                    'block_z': candidate['z'],
                }
                for candidate in generate_stronghold_candidates(self.world_seed)
            ]

        if structure_type != 'village':
            return []

        config = StructureConfig.CONFIGS['village']
        positions = []
        for region_x in range(-region_range, region_range + 1):
            for region_z in range(-region_range, region_range + 1):
                chunk_x, chunk_z = self.get_structure_chunk(
                    region_x, region_z,
                    config['spacing'], config['separation'], config['salt'],
                )
                positions.append({
                    'region_x': region_x,
                    'region_z': region_z,
                    'chunk_x': chunk_x,
                    'chunk_z': chunk_z,
                    'block_x': chunk_x * CHUNK_SIZE + CHUNK_SIZE // 2,
                    'block_z': chunk_z * CHUNK_SIZE + CHUNK_SIZE // 2,
                })
        return positions


class BiomeAnalyzer:
    """Deterministic 1.16.1-style layer preview for readable map context."""

    def __init__(self, seed, world_size=16000, resolution=180):
        self.seed = int(seed)
        self.world_size = world_size
        self.resolution = resolution
        self.valid_village_biomes = {
            'Plains', 'Desert', 'Savanna', 'Taiga', 'Snowy Tundra',
        }

    def biome_at(self, block_x, block_z):
        phase = (self.seed & 0xFFFF) / 65535.0
        temperature = (
            0.65 * np.sin(block_x / 1500.0 + phase * 4.0)
            + 0.35 * np.cos(block_z / 2200.0 - phase * 3.0)
        )
        rainfall = (
            0.70 * np.cos(block_z / 1800.0 + phase * 2.0)
            + 0.30 * np.sin((block_x + block_z) / 2600.0)
        )
        if abs(block_x * 3 + block_z * 5) % 257 < 3:
            return 'Ocean'
        if temperature < -0.58:
            return 'Snowy Tundra'
        if temperature > 0.58 and rainfall < -0.12:
            return 'Desert'
        if temperature > 0.20 and rainfall > 0.28:
            return 'Savanna'
        if rainfall > 0.62:
            return 'Taiga'
        if temperature > -0.10:
            return 'Plains'
        if rainfall > 0.05:
            return 'Forest'
        return 'Other'

    def generate_biome_map(self):
        half = self.world_size // 2
        coordinates = np.linspace(-half, half, self.resolution)
        x_grid, z_grid = np.meshgrid(coordinates, coordinates)
        phase = (self.seed & 0xFFFF) / 65535.0
        temperature = (
            0.65 * np.sin(x_grid / 1500.0 + phase * 4.0)
            + 0.35 * np.cos(z_grid / 2200.0 - phase * 3.0)
        )
        rainfall = (
            0.70 * np.cos(z_grid / 1800.0 + phase * 2.0)
            + 0.30 * np.sin((x_grid + z_grid) / 2600.0)
        )
        labels = np.full(x_grid.shape, 'Other', dtype=object)
        labels[temperature > -0.10] = 'Plains'
        labels[rainfall > 0.62] = 'Taiga'
        labels[(temperature > 0.20) & (rainfall > 0.28)] = 'Savanna'
        labels[(temperature > 0.58) & (rainfall < -0.12)] = 'Desert'
        labels[temperature < -0.58] = 'Snowy Tundra'
        labels[(temperature > -0.10) & (rainfall < -0.15)] = 'Forest'
        labels[
            (np.abs(x_grid * 3 + z_grid * 5) % 257 < 3)
        ] = 'Ocean'
        order = list(BIOME_COLORS)
        values = np.zeros(labels.shape, dtype=int)
        for index, label in enumerate(order):
            values[labels == label] = index
        return coordinates, values, labels


def add_ring_band(ax, ring, alpha=0.10):
    ax.add_patch(Wedge(
        (0, 0), ring['max_radius'], 0, 360,
        width=ring['max_radius'] - ring['min_radius'],
        facecolor=ring['color'], edgecolor='none', alpha=alpha,
    ))
    ax.add_patch(Circle(
        (0, 0), ring['min_radius'], fill=False, color=ring['color'],
        alpha=0.50, linewidth=0.8, linestyle='--',
    ))
    ax.add_patch(Circle(
        (0, 0), ring['max_radius'], fill=False, color=ring['color'],
        alpha=0.75, linewidth=0.8,
    ))


def _style_axis(ax, title):
    ax.set_facecolor(COLORS['panel'])
    ax.set_title(
        title, color=COLORS['text'], fontsize=11, fontweight='bold',
        loc='left', pad=10,
    )
    ax.tick_params(colors=COLORS['muted'], labelsize=7)
    ax.grid(color=COLORS['grid'], linewidth=0.5, alpha=0.38)
    for spine in ax.spines.values():
        spine.set_color(COLORS['grid'])
        spine.set_linewidth(1.0)


class MinecraftStructureAnalyzer:
    """Create the six-panel structure analysis figure."""

    def __init__(self, seed=42, world_size=16000):
        self.seed = int(seed)
        self.world_size = world_size
        self.struct_gen = StructureGenerator(self.seed)
        self.biome_analyzer = BiomeAnalyzer(
            self.seed, world_size=world_size, resolution=180,
        )
        self.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'Plots',
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_biome_noise_fields(self):
        """Compatibility wrapper for the visual biome-layer preview."""
        return self.biome_analyzer.generate_biome_map()

    def visualize_comprehensive_analysis(self):
        print(f'Creating {MINECRAFT_VERSION} structure analysis for seed {self.seed}')
        coordinates, biome_values, biome_labels = (
            self.biome_analyzer.generate_biome_map()
        )
        villages = self.struct_gen.generate_structures('village', region_range=8)
        for village in villages:
            village['biome'] = self.biome_analyzer.biome_at(
                village['block_x'], village['block_z']
            )
            village['valid'] = (
                village['biome'] in self.biome_analyzer.valid_village_biomes
            )
        strongholds = self.struct_gen.generate_structures('stronghold')

        figure, axes = plt.subplots(
            2, 3, figsize=(18, 12), facecolor=COLORS['background'],
            gridspec_kw={'wspace': 0.28, 'hspace': 0.38},
        )
        ax_biome, ax_village, ax_stronghold = axes[0]
        ax_distance, ax_rings, ax_notes = axes[1]

        figure.suptitle(
            f'MINECRAFT STRUCTURE ANALYSIS  |  {MINECRAFT_VERSION}'
            f'  |  SEED {self.seed}',
            color=COLORS['text'], fontsize=18, fontweight='bold', y=0.98,
        )
        figure.text(
            0.5, 0.955,
            'Exact grid candidates are separated from the compact biome-layer preview.',
            ha='center', color=COLORS['muted'], fontsize=9,
        )

        half = self.world_size // 2
        cmap = ListedColormap([BIOME_COLORS[name] for name in BIOME_COLORS])
        _style_axis(ax_biome, '01  BIOME LAYER PREVIEW')
        ax_biome.imshow(
            biome_values, origin='lower', interpolation='nearest',
            extent=(-half, half, -half, half), cmap=cmap, alpha=0.93,
        )
        valid_villages = [v for v in villages if v['valid']]
        ax_biome.scatter(
            [v['block_x'] for v in valid_villages],
            [v['block_z'] for v in valid_villages],
            s=18, color=COLORS['village'], edgecolors=COLORS['text'],
            linewidths=0.25, alpha=0.85, label='biome-pass candidates',
        )
        ax_biome.set_xlim(-half, half)
        ax_biome.set_ylim(-half, half)
        ax_biome.set_xlabel('Block X', color=COLORS['muted'], fontsize=8)
        ax_biome.set_ylabel('Block Z', color=COLORS['muted'], fontsize=8)
        ax_biome.legend(
            loc='lower right', fontsize=7, framealpha=0.85,
            facecolor=COLORS['panel_alt'],
        )

        _style_axis(ax_village, '02  VILLAGE CANDIDATE GRID')
        for coordinate in range(-half, half + 1, VILLAGE_SPACING * CHUNK_SIZE):
            ax_village.axvline(coordinate, color=COLORS['grid'], linewidth=0.6)
            ax_village.axhline(coordinate, color=COLORS['grid'], linewidth=0.6)
        for valid in (True, False):
            subset = [v for v in villages if v['valid'] == valid]
            if subset:
                ax_village.scatter(
                    [v['block_x'] for v in subset],
                    [v['block_z'] for v in subset],
                    s=35 if valid else 28,
                    marker='o' if valid else 'x',
                    color=COLORS['village'] if valid else COLORS['bad'],
                    edgecolors=COLORS['text'] if valid else None,
                    linewidths=0.45,
                    alpha=0.88,
                )
        ax_village.scatter(
            [0], [0], marker='*', s=110, color=COLORS['warning'],
            edgecolors=COLORS['text'], linewidths=0.8,
        )
        ax_village.set_xlim(-half, half)
        ax_village.set_ylim(-half, half)
        ax_village.set_xlabel('Block X', color=COLORS['muted'], fontsize=8)
        ax_village.set_ylabel('Block Z', color=COLORS['muted'], fontsize=8)
        ax_village.text(
            0.03, 0.04,
            f'{len(valid_villages)} pass  /  {len(villages) - len(valid_villages)} reject',
            transform=ax_village.transAxes, color=COLORS['muted'], fontsize=8,
        )

        _style_axis(ax_stronghold, '03  STRONGHOLD RING CANDIDATES')
        max_ring = STRONGHOLD_RINGS[-1]['max_radius'] + 1800
        ax_stronghold.set_xlim(-max_ring, max_ring)
        ax_stronghold.set_ylim(-max_ring, max_ring)
        ax_stronghold.set_aspect('equal')
        for ring in STRONGHOLD_RINGS:
            add_ring_band(ax_stronghold, ring, alpha=0.08)
        for ring_index, ring in enumerate(STRONGHOLD_RINGS, start=1):
            subset = [s for s in strongholds if s['ring'] == ring_index]
            ax_stronghold.scatter(
                [s['block_x'] for s in subset],
                [s['block_z'] for s in subset],
                s=32 if ring_index > 1 else 58,
                color=ring['color'], edgecolors='white' if ring_index == 1 else 'none',
                linewidths=0.5,
            )
        ax_stronghold.scatter(
            [0], [0], marker='*', s=105, color=COLORS['warning'],
            edgecolors=COLORS['text'], linewidths=0.8,
        )
        ax_stronghold.set_xlabel('Block X', color=COLORS['muted'], fontsize=8)
        ax_stronghold.set_ylabel('Block Z', color=COLORS['muted'], fontsize=8)

        _style_axis(ax_distance, '04  VILLAGE CANDIDATE DISTANCES')
        bins = np.linspace(0, max(1, max(
            np.hypot(v['block_x'], v['block_z']) for v in villages
        )), 16)
        all_distances = [
            np.hypot(v['block_x'], v['block_z']) for v in villages
        ]
        valid_distances = [
            np.hypot(v['block_x'], v['block_z']) for v in valid_villages
        ]
        ax_distance.hist(
            all_distances, bins=bins, color=COLORS['grid'],
            edgecolor=COLORS['muted'], linewidth=0.5, label='all candidates',
        )
        ax_distance.hist(
            valid_distances, bins=bins, color=COLORS['village'],
            alpha=0.88, edgecolor=COLORS['text'], linewidth=0.4,
            label='biome-pass preview',
        )
        ax_distance.set_xlabel('Distance from origin (blocks)', color=COLORS['muted'], fontsize=8)
        ax_distance.set_ylabel('Candidate count', color=COLORS['muted'], fontsize=8)
        ax_distance.legend(
            loc='upper right', fontsize=7, framealpha=0.85,
            facecolor=COLORS['panel_alt'],
        )

        _style_axis(ax_rings, '05  STRONGHOLD RING POPULATION')
        ring_numbers = np.arange(1, len(STRONGHOLD_RINGS) + 1)
        ring_counts = [ring['count'] for ring in STRONGHOLD_RINGS]
        bars = ax_rings.bar(
            ring_numbers, ring_counts,
            color=[ring['color'] for ring in STRONGHOLD_RINGS],
            edgecolor=COLORS['text'], linewidth=0.4,
        )
        for bar, count in zip(bars, ring_counts):
            ax_rings.text(
                bar.get_x() + bar.get_width() / 2, count + 0.8,
                str(count), ha='center', color=COLORS['text'], fontsize=8,
            )
        ax_rings.set_xticks(ring_numbers)
        ax_rings.set_xlabel('Ring number', color=COLORS['muted'], fontsize=8)
        ax_rings.set_ylabel('Stronghold candidates', color=COLORS['muted'], fontsize=8)
        ax_rings.set_ylim(0, 42)
        ax_rings.text(
            0.04, 0.90, '3 + 6 + 10 + 15 + 21 + 28 + 36 + 9 = 128',
            transform=ax_rings.transAxes, color=COLORS['muted'],
            fontsize=8, family='monospace',
        )

        ax_notes.set_facecolor(COLORS['panel'])
        ax_notes.axis('off')
        for spine in ax_notes.spines.values():
            spine.set_color(COLORS['grid'])
        ax_notes.text(
            0.04, 0.92, '06  GENERATION NOTES', transform=ax_notes.transAxes,
            color=COLORS['text'], fontsize=11, fontweight='bold',
        )
        notes = [
            ('VILLAGE SEED', 'world + rx*K1 + rz*K2 + salt'),
            ('JAVA RANDOM', 'setSeed(seed), nextInt(24) twice'),
            ('CANDIDATE', 'region * 32 + roll [0, 23]'),
            ('STRONGHOLDS', 'seeded ring iterator, 128 candidates'),
            ('SCOPE', 'candidate geometry before biome search'),
        ]
        for index, (label, value) in enumerate(notes):
            y = 0.77 - index * 0.105
            ax_notes.add_patch(FancyBboxPatch(
                (0.04, y - 0.045), 0.92, 0.075,
                transform=ax_notes.transAxes,
                boxstyle='round,pad=0.008', facecolor=COLORS['panel_alt'],
                edgecolor=COLORS['grid'], linewidth=0.8,
            ))
            ax_notes.text(
                0.08, y, label, transform=ax_notes.transAxes,
                color=COLORS['accent'], fontsize=8, fontweight='bold',
            )
            ax_notes.text(
                0.36, y, value, transform=ax_notes.transAxes,
                color=COLORS['text'], fontsize=8, family='monospace',
            )
        ax_notes.text(
            0.04, 0.12,
            f'Village attempts     {len(villages)}\n'
            f'Biome-pass preview   {len(valid_villages)}\n'
            f'Stronghold candidates {TOTAL_STRONGHOLDS}',
            transform=ax_notes.transAxes, color=COLORS['muted'],
            fontsize=9, family='monospace', linespacing=1.6,
        )

        figure.savefig(
            os.path.join(self.output_dir, 'structure_analysis.png'),
            dpi=220, facecolor=COLORS['background'],
            edgecolor='none', bbox_inches='tight',
        )
        plt.close(figure)
        return os.path.join(self.output_dir, 'structure_analysis.png')


def main():
    analyzer = MinecraftStructureAnalyzer(seed=42, world_size=16000)
    output_path = analyzer.visualize_comprehensive_analysis()
    print(f'Saved {output_path}')


if __name__ == '__main__':
    main()
