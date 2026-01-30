"""
Seed Loading & World Generation Animation

Visualizes the complete world generation pipeline:
1. Seed → LCG random state initialization
2. Chunk generation spiral outward from spawn
3. Multi-layer biome noise calculation (Temperature, Humidity, Continentalness, etc.)
4. Terrain height generation
5. Biome assignment per chunk

Based on Minecraft's actual generation algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import os

# ============================================================================
# VISUAL CONFIGURATION
# ============================================================================

plt.style.use('dark_background')

COLORS = {
    'background': '#0D1117',
    'text': '#E6EDF3',
    'accent': '#58A6FF',
    'grid': '#21262D',
    'chunk_loading': '#3d5a80',
    'chunk_loaded': '#1a1a2e',
    'spawn': '#ff6b6b',
    
    # Biomes
    'ocean': '#1a5276',
    'plains': '#7dcea0',
    'forest': '#1e8449',
    'desert': '#f4d03f',
    'mountains': '#7f8c8d',
    'swamp': '#6c7a28',
    'taiga': '#2e4a3f',
    'jungle': '#27ae60',
    'snowy': '#ecf0f1',
    'badlands': '#d35400',
    
    # Noise layers
    'temperature': '#e74c3c',
    'humidity': '#3498db',
    'continentalness': '#2ecc71',
    'erosion': '#9b59b6',
    'weirdness': '#f39c12',
    'depth': '#1abc9c',
}

BIOME_COLORS = {
    'ocean': '#1a5276',
    'deep_ocean': '#0e3450',
    'plains': '#7dcea0',
    'forest': '#1e8449',
    'birch_forest': '#91c276',
    'dark_forest': '#0d4520',
    'desert': '#f4d03f',
    'mountains': '#7f8c8d',
    'snowy_plains': '#ecf0f1',
    'taiga': '#2e4a3f',
    'snowy_taiga': '#4a6670',
    'jungle': '#27ae60',
    'swamp': '#6c7a28',
    'badlands': '#d35400',
    'beach': '#f9e79f',
    'river': '#5dade2',
}

# ============================================================================
# WORLD GENERATION ALGORITHMS
# ============================================================================

class JavaLCG:
    """
    Java's Linear Congruential Generator (LCG) for random numbers.
    
    Formula: next = (seed * 0x5DEECE66D + 0xB) mod 2^48
    This is the exact RNG used by Minecraft Java Edition.
    """
    
    MULTIPLIER = 0x5DEECE66D
    ADDEND = 0xB
    MASK = (1 << 48) - 1
    
    def __init__(self, seed):
        self.seed = (seed ^ self.MULTIPLIER) & self.MASK
        self.initial_seed = seed
    
    def next(self, bits=32):
        """Generate next random bits."""
        self.seed = (self.seed * self.MULTIPLIER + self.ADDEND) & self.MASK
        return self.seed >> (48 - bits)
    
    def next_int(self, bound=None):
        """Generate random integer."""
        if bound is None:
            return self.next(32)
        
        if (bound & -bound) == bound:  # Power of 2
            return (bound * self.next(31)) >> 31
        
        bits = self.next(31)
        val = bits % bound
        while bits - val + (bound - 1) < 0:
            bits = self.next(31)
            val = bits % bound
        return val
    
    def next_float(self):
        """Generate random float [0, 1)."""
        return self.next(24) / (1 << 24)


class NoiseGenerator:
    """
    Simplified Perlin/Simplex noise for biome generation.
    
    Minecraft uses multi-octave noise with different parameters
    for each biome factor.
    """
    
    def __init__(self, seed, octaves=4, scale=1.0):
        self.rng = JavaLCG(seed)
        self.octaves = octaves
        self.scale = scale
        
        # Generate permutation table
        self.perm = list(range(256))
        for i in range(255, 0, -1):
            j = self.rng.next_int(i + 1)
            self.perm[i], self.perm[j] = self.perm[j], self.perm[i]
        self.perm = self.perm * 2
    
    def _fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _lerp(self, t, a, b):
        return a + t * (b - a)
    
    def _grad(self, hash_val, x, y):
        h = hash_val & 3
        u = x if h < 2 else y
        v = y if h < 2 else x
        return (u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v)
    
    def noise_2d(self, x, y):
        """Single octave 2D Perlin noise."""
        X = int(np.floor(x)) & 255
        Y = int(np.floor(y)) & 255
        
        x -= np.floor(x)
        y -= np.floor(y)
        
        u = self._fade(x)
        v = self._fade(y)
        
        A = self.perm[X] + Y
        B = self.perm[X + 1] + Y
        
        return self._lerp(v,
            self._lerp(u, self._grad(self.perm[A], x, y),
                         self._grad(self.perm[B], x - 1, y)),
            self._lerp(u, self._grad(self.perm[A + 1], x, y - 1),
                         self._grad(self.perm[B + 1], x - 1, y - 1)))
    
    def sample(self, x, y):
        """Multi-octave noise sample."""
        value = 0
        amplitude = 1
        frequency = self.scale
        max_value = 0
        
        for _ in range(self.octaves):
            value += self.noise_2d(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= 0.5
            frequency *= 2
        
        return value / max_value


class BiomeGenerator:
    """
    Multi-parameter biome generation system.
    
    Minecraft uses 6 noise parameters:
    - Temperature: Hot ↔ Cold
    - Humidity: Dry ↔ Wet
    - Continentalness: Ocean ↔ Inland
    - Erosion: Flat ↔ Mountainous
    - Weirdness: Normal ↔ Weird biome variants
    - Depth: Surface ↔ Underground
    """
    
    NOISE_PARAMS = {
        'temperature': {'seed_offset': 0, 'octaves': 4, 'scale': 0.025},
        'humidity': {'seed_offset': 1, 'octaves': 4, 'scale': 0.025},
        'continentalness': {'seed_offset': 2, 'octaves': 6, 'scale': 0.01},
        'erosion': {'seed_offset': 3, 'octaves': 5, 'scale': 0.02},
        'weirdness': {'seed_offset': 4, 'octaves': 4, 'scale': 0.03},
    }
    
    def __init__(self, seed):
        self.seed = seed
        self.noise_generators = {}
        
        for name, params in self.NOISE_PARAMS.items():
            self.noise_generators[name] = NoiseGenerator(
                seed + params['seed_offset'],
                params['octaves'],
                params['scale']
            )
    
    def get_noise_values(self, x, z):
        """Get all noise values for a position."""
        return {
            name: gen.sample(x, z)
            for name, gen in self.noise_generators.items()
        }
    
    def get_biome(self, x, z):
        """Determine biome from noise values."""
        noise = self.get_noise_values(x, z)
        
        temp = noise['temperature']
        humid = noise['humidity']
        cont = noise['continentalness']
        erosion = noise['erosion']
        
        # Simplified biome selection logic
        if cont < -0.4:
            return 'deep_ocean' if cont < -0.6 else 'ocean'
        elif cont < -0.1:
            return 'beach'
        elif cont < 0.1:
            if humid > 0.3:
                return 'swamp'
            return 'plains'
        else:
            # Inland biomes
            if temp > 0.5:
                if humid < -0.3:
                    return 'desert'
                elif humid > 0.3:
                    return 'jungle'
                else:
                    return 'badlands' if erosion < -0.2 else 'plains'
            elif temp > 0:
                if humid > 0.3:
                    return 'dark_forest' if erosion > 0.2 else 'forest'
                else:
                    return 'birch_forest' if humid > 0 else 'plains'
            elif temp > -0.4:
                return 'taiga' if humid > 0 else 'mountains'
            else:
                if humid > 0.2:
                    return 'snowy_taiga'
                else:
                    return 'snowy_plains'


def generate_chunk_spiral(radius):
    """
    Generate chunk coordinates in spiral order from spawn.
    
    Minecraft loads chunks in a roughly spiral pattern outward
    from the player's position.
    """
    chunks = [(0, 0)]  # Start at origin
    
    for r in range(1, radius + 1):
        # Top edge
        for x in range(-r, r + 1):
            chunks.append((x, -r))
        # Right edge
        for z in range(-r + 1, r + 1):
            chunks.append((r, z))
        # Bottom edge
        for x in range(r - 1, -r - 1, -1):
            chunks.append((x, r))
        # Left edge
        for z in range(r - 1, -r, -1):
            chunks.append((-r, z))
    
    return chunks


# ============================================================================
# ANIMATION
# ============================================================================

def create_seed_loading_animation(save_path, seed=42, fps=24, duration=12):
    """
    Create comprehensive seed loading animation showing:
    1. Seed initialization and LCG state
    2. Chunk spiral loading
    3. Noise layer computation
    4. Biome assignment
    """
    print("=" * 60)
    print("SEED LOADING & WORLD GENERATION ANIMATION")
    print(f"Seed: {seed}")
    print("=" * 60)
    
    total_frames = fps * duration
    
    # Initialize generators
    biome_gen = BiomeGenerator(seed)
    lcg = JavaLCG(seed)
    
    # Pre-generate chunk spiral
    chunk_radius = 12
    chunk_spiral = generate_chunk_spiral(chunk_radius)
    total_chunks = len(chunk_spiral)
    
    # Pre-compute biome data
    print("Pre-computing biome data...")
    biome_data = {}
    noise_data = {name: {} for name in BiomeGenerator.NOISE_PARAMS.keys()}
    
    for cx, cz in chunk_spiral:
        biome_data[(cx, cz)] = biome_gen.get_biome(cx * 16, cz * 16)
        noise_vals = biome_gen.get_noise_values(cx * 16, cz * 16)
        for name, val in noise_vals.items():
            noise_data[name][(cx, cz)] = val
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Layout
    gs = fig.add_gridspec(3, 4, width_ratios=[1.3, 0.7, 0.7, 0.7], 
                         height_ratios=[0.2, 1, 0.4],
                         hspace=0.2, wspace=0.15,
                         left=0.04, right=0.96, top=0.93, bottom=0.06)
    
    # Title panel
    ax_title = fig.add_subplot(gs[0, :])
    
    # Main chunk view
    ax_chunks = fig.add_subplot(gs[1, 0])
    
    # Noise layer panels (3 columns)
    ax_temp = fig.add_subplot(gs[1, 1])
    ax_humid = fig.add_subplot(gs[1, 2])
    ax_cont = fig.add_subplot(gs[1, 3])
    
    # Bottom info panels
    ax_seed = fig.add_subplot(gs[2, 0])
    ax_formula = fig.add_subplot(gs[2, 1:3])
    ax_stats = fig.add_subplot(gs[2, 3])
    
    for ax in [ax_title, ax_chunks, ax_temp, ax_humid, ax_cont, 
               ax_seed, ax_formula, ax_stats]:
        ax.set_facecolor(COLORS['background'])
    
    def init():
        return []
    
    def update(frame):
        # Clear all axes
        for ax in [ax_title, ax_chunks, ax_temp, ax_humid, ax_cont,
                   ax_seed, ax_formula, ax_stats]:
            ax.cla()
            ax.set_facecolor(COLORS['background'])
        
        progress = frame / total_frames
        
        # Calculate phase
        if progress < 0.1:
            phase = 'seed_init'
            phase_progress = progress / 0.1
        elif progress < 0.15:
            phase = 'lcg_scramble'
            phase_progress = (progress - 0.1) / 0.05
        elif progress < 0.85:
            phase = 'chunk_loading'
            phase_progress = (progress - 0.15) / 0.7
        else:
            phase = 'complete'
            phase_progress = (progress - 0.85) / 0.15
        
        # Calculate chunks loaded
        if phase in ['chunk_loading', 'complete']:
            if phase == 'complete':
                chunks_loaded = total_chunks
            else:
                chunks_loaded = int(phase_progress * total_chunks)
        else:
            chunks_loaded = 0
        
        # ====================================================================
        # TITLE PANEL
        # ====================================================================
        
        ax_title.set_xlim(0, 10)
        ax_title.set_ylim(0, 1)
        ax_title.axis('off')
        
        phase_titles = {
            'seed_init': 'Initializing World Seed...',
            'lcg_scramble': 'Scrambling Random State (LCG)...',
            'chunk_loading': 'Generating Chunks...',
            'complete': 'World Generation Complete!'
        }
        
        ax_title.text(5, 0.5, phase_titles[phase], ha='center', va='center',
                     color=COLORS['accent'], fontsize=18, fontweight='bold')
        
        # Progress bar
        ax_title.axhline(0.15, xmin=0.1, xmax=0.9, color=COLORS['grid'], 
                        linewidth=8, alpha=0.3)
        ax_title.axhline(0.15, xmin=0.1, xmax=0.1 + 0.8 * progress, 
                        color=COLORS['accent'], linewidth=8)
        
        ax_title.text(9.5, 0.15, f'{progress*100:.0f}%', ha='right', va='center',
                     color=COLORS['text'], fontsize=10)
        
        # ====================================================================
        # CHUNK VIEW
        # ====================================================================
        
        ax_chunks.set_xlim(-chunk_radius - 1, chunk_radius + 1)
        ax_chunks.set_ylim(-chunk_radius - 1, chunk_radius + 1)
        ax_chunks.set_aspect('equal')
        ax_chunks.set_title('Chunk Generation (Spiral Pattern)', 
                           color=COLORS['text'], fontsize=11)
        ax_chunks.axis('off')
        
        # Draw all chunks
        for i, (cx, cz) in enumerate(chunk_spiral):
            if i < chunks_loaded:
                biome = biome_data[(cx, cz)]
                color = BIOME_COLORS.get(biome, COLORS['chunk_loaded'])
                alpha = 0.8
            else:
                color = COLORS['grid']
                alpha = 0.2
            
            rect = Rectangle((cx - 0.45, cz - 0.45), 0.9, 0.9,
                            facecolor=color, edgecolor=COLORS['grid'],
                            alpha=alpha, linewidth=0.3)
            ax_chunks.add_patch(rect)
        
        # Highlight currently loading chunk
        if phase == 'chunk_loading' and chunks_loaded < total_chunks:
            cx, cz = chunk_spiral[chunks_loaded]
            highlight = Rectangle((cx - 0.5, cz - 0.5), 1, 1,
                                 facecolor='none', edgecolor=COLORS['spawn'],
                                 linewidth=2, alpha=0.9)
            ax_chunks.add_patch(highlight)
        
        # Spawn marker
        ax_chunks.scatter([0], [0], c=COLORS['spawn'], s=100, marker='*', 
                         zorder=10, edgecolors='white', linewidth=1)
        ax_chunks.text(0, -1.5, 'Spawn', ha='center', color=COLORS['text'],
                      fontsize=8)
        
        # ====================================================================
        # NOISE LAYER PANELS
        # ====================================================================
        
        noise_axes = [
            (ax_temp, 'temperature', 'Temperature', COLORS['temperature']),
            (ax_humid, 'humidity', 'Humidity', COLORS['humidity']),
            (ax_cont, 'continentalness', 'Continentalness', COLORS['continentalness']),
        ]
        
        for ax, noise_name, title, cmap_color in noise_axes:
            ax.set_xlim(-chunk_radius - 1, chunk_radius + 1)
            ax.set_ylim(-chunk_radius - 1, chunk_radius + 1)
            ax.set_aspect('equal')
            ax.set_title(title, color=cmap_color, fontsize=10, fontweight='bold')
            ax.axis('off')
            
            # Create colormap
            cmap = mcolors.LinearSegmentedColormap.from_list(
                noise_name, ['#0a0512', cmap_color, '#ffffff'])
            
            # Draw noise values for loaded chunks
            for i, (cx, cz) in enumerate(chunk_spiral[:chunks_loaded]):
                val = noise_data[noise_name][(cx, cz)]
                normalized = (val + 1) / 2  # Normalize to [0, 1]
                color = cmap(normalized)
                
                rect = Rectangle((cx - 0.45, cz - 0.45), 0.9, 0.9,
                                facecolor=color, edgecolor='none', alpha=0.8)
                ax.add_patch(rect)
            
            # Colorbar indicator
            ax.text(chunk_radius, -chunk_radius - 0.5, 'Cold' if noise_name == 'temperature' 
                   else ('Dry' if noise_name == 'humidity' else 'Ocean'),
                   ha='right', color=COLORS['text'], fontsize=7)
            ax.text(-chunk_radius, -chunk_radius - 0.5, 'Hot' if noise_name == 'temperature'
                   else ('Wet' if noise_name == 'humidity' else 'Inland'),
                   ha='left', color=COLORS['text'], fontsize=7)
        
        # ====================================================================
        # SEED INFO PANEL
        # ====================================================================
        
        ax_seed.set_xlim(0, 10)
        ax_seed.set_ylim(0, 4)
        ax_seed.axis('off')
        ax_seed.set_title('Seed State', color=COLORS['text'], fontsize=10)
        
        # Show seed value with animation effect
        if phase == 'seed_init':
            display_seed = str(seed)[:int(len(str(seed)) * phase_progress) + 1]
            ax_seed.text(5, 3, f'Seed: {display_seed}{"_" if phase_progress < 1 else ""}',
                        ha='center', color=COLORS['accent'], fontsize=12,
                        fontweight='bold', family='monospace')
        else:
            ax_seed.text(5, 3, f'Seed: {seed}', ha='center', color=COLORS['accent'],
                        fontsize=12, fontweight='bold', family='monospace')
        
        # LCG state
        if phase in ['lcg_scramble', 'chunk_loading', 'complete']:
            lcg_visual = JavaLCG(seed)
            for _ in range(chunks_loaded):
                lcg_visual.next()
            
            ax_seed.text(5, 2, f'LCG State:', ha='center', color=COLORS['text'], fontsize=9)
            ax_seed.text(5, 1.3, f'{lcg_visual.seed:015d}', ha='center',
                        color=COLORS['text'], fontsize=9, family='monospace',
                        bbox=dict(boxstyle='round', facecolor=COLORS['grid'], alpha=0.5))
        
        # Chunks counter
        ax_seed.text(5, 0.3, f'Chunks: {chunks_loaded}/{total_chunks}',
                    ha='center', color=COLORS['text'], fontsize=10)
        
        # ====================================================================
        # FORMULA PANEL
        # ====================================================================
        
        ax_formula.set_xlim(0, 10)
        ax_formula.set_ylim(0, 4)
        ax_formula.axis('off')
        ax_formula.set_title('Generation Formulas', color=COLORS['text'], fontsize=10)
        
        formulas = [
            ('LCG:', 'next = (seed × 0x5DEECE66D + 0xB) mod 2⁴⁸'),
            ('Noise:', 'N(x,z) = Σᵢ (amplitude^i × perlin(x×freq^i, z×freq^i))'),
            ('Biome:', 'B = f(temperature, humidity, continentalness, ...)'),
        ]
        
        for i, (name, formula) in enumerate(formulas):
            y = 3 - i * 1.1
            ax_formula.text(0.5, y, name, color=COLORS['accent'], fontsize=9,
                           fontweight='bold')
            ax_formula.text(2, y, formula, color=COLORS['text'], fontsize=8,
                           family='monospace')
        
        # ====================================================================
        # STATS PANEL
        # ====================================================================
        
        ax_stats.set_xlim(0, 10)
        ax_stats.set_ylim(0, 4)
        ax_stats.axis('off')
        ax_stats.set_title('Generation Stats', color=COLORS['text'], fontsize=10)
        
        # Count biomes
        biome_counts = {}
        for i in range(chunks_loaded):
            cx, cz = chunk_spiral[i]
            biome = biome_data[(cx, cz)]
            biome_counts[biome] = biome_counts.get(biome, 0) + 1
        
        # Show top biomes
        sorted_biomes = sorted(biome_counts.items(), key=lambda x: -x[1])[:4]
        
        ax_stats.text(5, 3.5, 'Biome Distribution:', ha='center', 
                     color=COLORS['text'], fontsize=9)
        
        for i, (biome, count) in enumerate(sorted_biomes):
            y = 2.7 - i * 0.6
            color = BIOME_COLORS.get(biome, COLORS['text'])
            ax_stats.add_patch(Rectangle((0.5, y - 0.15), 0.3, 0.3, 
                                        facecolor=color, alpha=0.8))
            ax_stats.text(1.2, y, f'{biome.replace("_", " ").title()}: {count}',
                         color=COLORS['text'], fontsize=8, va='center')
        
        return []
    
    # Create animation
    print(f"Generating {total_frames} frames at {fps} FPS...")
    anim = FuncAnimation(fig, update, frames=total_frames, init_func=init,
                        interval=1000/fps, blit=False)
    
    # Save
    print(f"Saving animation to: {save_path}")
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=100)
    print("✓ Seed loading animation saved!")
    
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(os.path.dirname(script_dir), "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Use a famous seed
    SEED = -4172144997902289642  # "dream" speedrun seed (example)
    
    output_path = os.path.join(plots_dir, "seed_loading.gif")
    create_seed_loading_animation(output_path, seed=SEED, fps=20, duration=15)
