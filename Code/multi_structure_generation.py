"""
Multi-Structure Animated Generation Sequence

Visualizes the complete structure generation pipeline for multiple structure types:
- Villages, Temples, Strongholds, Ocean Monuments, Nether Fortresses
- Shows spacing, separation, and salt-based seeding
- Animated sequence of each structure type being placed

Demonstrates the elegant mathematics behind structure placement.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, RegularPolygon, Circle, FancyBboxPatch, Wedge
import matplotlib.patheffects as pe
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
    'grid_major': '#30363D',
    'highlight': '#ff6b6b',
    
    # Structure colors
    'village': '#7dcea0',
    'desert_temple': '#f4d03f',
    'jungle_temple': '#27ae60',
    'stronghold': '#9b59b6',
    'ocean_monument': '#3498db',
    'nether_fortress': '#e74c3c',
    'woodland_mansion': '#8b4513',
    'pillager_outpost': '#5d6d7e',
    'ancient_city': '#1abc9c',
    'trail_ruins': '#d35400',
    
    # Region colors
    'region_valid': '#2ecc71',
    'region_invalid': '#c0392b',
    'region_checking': '#f39c12',
}

# ============================================================================
# STRUCTURE DEFINITIONS
# ============================================================================

STRUCTURES = {
    'village': {
        'name': 'Village',
        'spacing': 34,
        'separation': 8,
        'salt': 10387312,
        'color': COLORS['village'],
        'biomes': ['plains', 'desert', 'savanna', 'taiga', 'snowy_plains'],
        'symbol': 's',  # square
        'description': 'Generated in habitable biomes\nwith villager spawns',
    },
    'desert_temple': {
        'name': 'Desert Temple',
        'spacing': 32,
        'separation': 8,
        'salt': 14357617,
        'color': COLORS['desert_temple'],
        'biomes': ['desert'],
        'symbol': '^',  # triangle
        'description': 'Hidden treasure chambers\nwith TNT trap below',
    },
    'jungle_temple': {
        'name': 'Jungle Temple',
        'spacing': 32,
        'separation': 8,
        'salt': 14357619,
        'color': COLORS['jungle_temple'],
        'biomes': ['jungle'],
        'symbol': '^',
        'description': 'Puzzle mechanisms with\narrow and lever traps',
    },
    'ocean_monument': {
        'name': 'Ocean Monument',
        'spacing': 32,
        'separation': 5,
        'salt': 10387313,
        'color': COLORS['ocean_monument'],
        'biomes': ['deep_ocean'],
        'symbol': 'D',  # diamond
        'description': 'Guardian-defended prismarine\nstructures with elder guardians',
    },
    'stronghold': {
        'name': 'Stronghold',
        'spacing': 0,  # Special ring-based
        'separation': 0,
        'salt': 0,
        'color': COLORS['stronghold'],
        'biomes': ['any'],
        'symbol': 'p',  # pentagon
        'description': '128 total in concentric rings\ncontaining End Portal',
        'rings': [(1408, 2688, 3), (4480, 5760, 6), (7552, 8832, 10), (10624, 11904, 15)],
    },
    'nether_fortress': {
        'name': 'Nether Fortress',
        'spacing': 27,
        'separation': 4,
        'salt': 30084232,
        'color': COLORS['nether_fortress'],
        'biomes': ['nether_wastes', 'soul_sand_valley', 'crimson_forest', 'warped_forest', 'basalt_deltas'],
        'symbol': 'h',  # hexagon
        'description': 'Blaze spawners and wither\nskeletons in the Nether',
    },
    'woodland_mansion': {
        'name': 'Woodland Mansion',
        'spacing': 80,
        'separation': 20,
        'salt': 10387319,
        'color': COLORS['woodland_mansion'],
        'biomes': ['dark_forest'],
        'symbol': 'H',  # big house
        'description': 'Rare structure with\nillagers and unique loot',
    },
    'ancient_city': {
        'name': 'Ancient City',
        'spacing': 24,
        'separation': 8,
        'salt': 20083232,
        'color': COLORS['ancient_city'],
        'biomes': ['deep_dark'],
        'symbol': 'o',  # circle
        'description': 'Deep underground with\nWarden and sculk sensors',
    },
}

GENERATION_ORDER = [
    'stronghold',
    'village',
    'desert_temple', 
    'jungle_temple',
    'ocean_monument',
    'woodland_mansion',
    'nether_fortress',
    'ancient_city',
]


# ============================================================================
# STRUCTURE GENERATION ALGORITHMS
# ============================================================================

class StructureGenerator:
    """
    Generates structure positions using Minecraft's algorithm.
    """
    
    def __init__(self, seed):
        self.seed = seed
    
    def java_lcg_next(self, seed):
        """Single LCG step."""
        return (seed * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
    
    def get_region_seed(self, region_x, region_z, salt):
        """
        Calculate seed for a specific region.
        
        Formula: regionSeed = worldSeed + regionX * K1 + regionZ * K2 + salt
        """
        K1 = 341873128712
        K2 = 132897987541
        
        region_seed = self.seed + region_x * K1 + region_z * K2 + salt
        return region_seed & ((1 << 48) - 1)
    
    def get_structure_position(self, region_x, region_z, spacing, separation, salt):
        """
        Calculate structure position within a region.
        
        The structure generates at a random position within the valid
        area of the region (spacing - separation).
        """
        region_seed = self.get_region_seed(region_x, region_z, salt)
        
        # Initialize RNG
        rng_seed = (region_seed ^ 0x5DEECE66D) & ((1 << 48) - 1)
        
        # Get random offsets
        rng_seed = self.java_lcg_next(rng_seed)
        rand1 = (rng_seed >> 17) % (spacing - separation)
        
        rng_seed = self.java_lcg_next(rng_seed)
        rand2 = (rng_seed >> 17) % (spacing - separation)
        
        # Calculate chunk position
        chunk_x = region_x * spacing + rand1
        chunk_z = region_z * spacing + rand2
        
        return chunk_x, chunk_z
    
    def generate_structure_positions(self, structure_key, region_range=5):
        """Generate all structure positions in a region range."""
        struct = STRUCTURES[structure_key]
        positions = []
        
        if structure_key == 'stronghold':
            # Special case: ring-based generation
            for ring_min, ring_max, count in struct['rings'][:2]:  # First 2 rings for viz
                for i in range(count):
                    angle = 2 * np.pi * i / count + np.random.random() * 0.3
                    radius = (ring_min + ring_max) / 2 / 16  # Convert to chunks
                    chunk_x = int(radius * np.cos(angle))
                    chunk_z = int(radius * np.sin(angle))
                    positions.append((chunk_x, chunk_z))
        else:
            spacing = struct['spacing']
            separation = struct['separation']
            salt = struct['salt']
            
            for rx in range(-region_range, region_range + 1):
                for rz in range(-region_range, region_range + 1):
                    chunk_x, chunk_z = self.get_structure_position(
                        rx, rz, spacing, separation, salt
                    )
                    positions.append((chunk_x, chunk_z))
        
        return positions


# ============================================================================
# ANIMATION
# ============================================================================

def create_multi_structure_animation(save_path, seed=42, fps=20, duration=20):
    """
    Create animated sequence showing multiple structure types generating.
    """
    print("=" * 60)
    print("MULTI-STRUCTURE GENERATION ANIMATION")
    print(f"Seed: {seed}")
    print("=" * 60)
    
    total_frames = fps * duration
    frames_per_structure = total_frames // len(GENERATION_ORDER)
    
    # Initialize generator
    gen = StructureGenerator(seed)
    
    # Pre-compute all structure positions
    print("Pre-computing structure positions...")
    all_positions = {}
    for key in GENERATION_ORDER:
        all_positions[key] = gen.generate_structure_positions(key)
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Layout
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 0.8, 0.7], 
                         height_ratios=[1, 0.35],
                         hspace=0.15, wspace=0.12,
                         left=0.04, right=0.96, top=0.93, bottom=0.06)
    
    # Main map
    ax_map = fig.add_subplot(gs[0, 0])
    
    # Current structure detail
    ax_detail = fig.add_subplot(gs[0, 1])
    
    # Algorithm panel
    ax_algo = fig.add_subplot(gs[0, 2])
    
    # Structure legend
    ax_legend = fig.add_subplot(gs[1, :])
    
    for ax in [ax_map, ax_detail, ax_algo, ax_legend]:
        ax.set_facecolor(COLORS['background'])
    
    def draw_structure_marker(ax, x, z, symbol, color, size=80, alpha=1.0, animated=False, anim_frame=0):
        """Draw a structure marker at the given position."""
        if animated:
            # Pulsing effect
            size *= 1 + 0.2 * np.sin(anim_frame * 0.3)
        
        if symbol == 's':
            ax.scatter([x], [z], s=size, c=color, marker='s', alpha=alpha,
                      edgecolors='white', linewidth=0.5)
        elif symbol == '^':
            ax.scatter([x], [z], s=size, c=color, marker='^', alpha=alpha,
                      edgecolors='white', linewidth=0.5)
        elif symbol == 'D':
            ax.scatter([x], [z], s=size, c=color, marker='D', alpha=alpha,
                      edgecolors='white', linewidth=0.5)
        elif symbol == 'p':
            ax.scatter([x], [z], s=size, c=color, marker='p', alpha=alpha,
                      edgecolors='white', linewidth=0.5)
        elif symbol == 'h':
            ax.scatter([x], [z], s=size, c=color, marker='H', alpha=alpha,
                      edgecolors='white', linewidth=0.5)
        elif symbol == 'H':
            ax.scatter([x], [z], s=size * 1.5, c=color, marker='8', alpha=alpha,
                      edgecolors='white', linewidth=0.5)
        elif symbol == 'o':
            ax.scatter([x], [z], s=size, c=color, marker='o', alpha=alpha,
                      edgecolors='white', linewidth=0.5)
        else:
            ax.scatter([x], [z], s=size, c=color, marker='o', alpha=alpha,
                      edgecolors='white', linewidth=0.5)
    
    def init():
        return []
    
    def update(frame):
        # Clear axes
        for ax in [ax_map, ax_detail, ax_algo, ax_legend]:
            ax.cla()
            ax.set_facecolor(COLORS['background'])
        
        # Determine current structure being generated
        current_struct_idx = min(frame // frames_per_structure, len(GENERATION_ORDER) - 1)
        current_struct_key = GENERATION_ORDER[current_struct_idx]
        current_struct = STRUCTURES[current_struct_key]
        
        # Progress within current structure
        struct_frame = frame % frames_per_structure
        struct_progress = struct_frame / frames_per_structure
        
        # ====================================================================
        # MAIN MAP
        # ====================================================================
        
        map_range = 200  # chunks
        ax_map.set_xlim(-map_range, map_range)
        ax_map.set_ylim(-map_range, map_range)
        ax_map.set_aspect('equal')
        ax_map.set_title('World Structure Map', color=COLORS['text'],
                        fontsize=14, fontweight='bold')
        
        # Draw grid
        for i in range(-map_range, map_range + 1, 50):
            ax_map.axhline(i, color=COLORS['grid'], linewidth=0.3, alpha=0.3)
            ax_map.axvline(i, color=COLORS['grid'], linewidth=0.3, alpha=0.3)
        
        # Draw spawn
        ax_map.scatter([0], [0], c=COLORS['highlight'], s=150, marker='*',
                      zorder=100, edgecolors='white', linewidth=1)
        
        # Draw previously generated structures (faded)
        for i, key in enumerate(GENERATION_ORDER[:current_struct_idx]):
            struct = STRUCTURES[key]
            positions = all_positions[key]
            
            for cx, cz in positions:
                if abs(cx) < map_range and abs(cz) < map_range:
                    draw_structure_marker(ax_map, cx, cz, struct['symbol'],
                                        struct['color'], size=60, alpha=0.6)
        
        # Draw current structure positions with animation
        positions = all_positions[current_struct_key]
        num_to_show = int(len(positions) * struct_progress)
        
        for i, (cx, cz) in enumerate(positions[:num_to_show]):
            if abs(cx) < map_range and abs(cz) < map_range:
                is_new = i == num_to_show - 1
                draw_structure_marker(ax_map, cx, cz, current_struct['symbol'],
                                    current_struct['color'], size=80 if is_new else 60,
                                    alpha=1.0, animated=is_new, anim_frame=frame)
        
        # Region grid for current structure (if applicable)
        if current_struct['spacing'] > 0:
            spacing = current_struct['spacing']
            for rx in range(-4, 5):
                ax_map.axvline(rx * spacing, color=current_struct['color'],
                             linewidth=1, alpha=0.2, linestyle='--')
                ax_map.axhline(rx * spacing, color=current_struct['color'],
                             linewidth=1, alpha=0.2, linestyle='--')
        
        ax_map.set_xlabel('Chunk X', color=COLORS['text'])
        ax_map.set_ylabel('Chunk Z', color=COLORS['text'])
        
        # ====================================================================
        # STRUCTURE DETAIL PANEL
        # ====================================================================
        
        ax_detail.set_xlim(0, 10)
        ax_detail.set_ylim(0, 10)
        ax_detail.axis('off')
        ax_detail.set_title(f'Currently Generating: {current_struct["name"]}',
                           color=current_struct['color'], fontsize=12, fontweight='bold')
        
        # Structure icon (large)
        draw_structure_marker(ax_detail, 5, 7.5, current_struct['symbol'],
                            current_struct['color'], size=500, alpha=0.9)
        
        # Parameters
        params = [
            ('Spacing:', f'{current_struct["spacing"]} chunks' if current_struct['spacing'] > 0 else 'Ring-based'),
            ('Separation:', f'{current_struct["separation"]} chunks' if current_struct['separation'] > 0 else 'N/A'),
            ('Salt:', f'{current_struct["salt"]}' if current_struct['salt'] > 0 else 'Special'),
            ('Biomes:', ', '.join(current_struct['biomes'][:2]) + ('...' if len(current_struct['biomes']) > 2 else '')),
        ]
        
        y_pos = 5.5
        for label, value in params:
            ax_detail.text(1, y_pos, label, color=COLORS['text'], fontsize=9)
            ax_detail.text(4, y_pos, value, color=COLORS['accent'], fontsize=9)
            y_pos -= 0.7
        
        # Description
        ax_detail.text(5, 1.5, current_struct['description'], ha='center',
                      color=COLORS['text'], fontsize=8, style='italic',
                      bbox=dict(boxstyle='round', facecolor=COLORS['grid'], alpha=0.5))
        
        # Progress bar
        ax_detail.axhline(0.3, xmin=0.1, xmax=0.9, color=COLORS['grid'], linewidth=6)
        ax_detail.axhline(0.3, xmin=0.1, xmax=0.1 + 0.8 * struct_progress,
                         color=current_struct['color'], linewidth=6)
        
        # ====================================================================
        # ALGORITHM PANEL
        # ====================================================================
        
        ax_algo.set_xlim(0, 10)
        ax_algo.set_ylim(0, 10)
        ax_algo.axis('off')
        ax_algo.set_title('Generation Algorithm', color=COLORS['text'], fontsize=11)
        
        # Show algorithm steps
        if current_struct_key == 'stronghold':
            steps = [
                '1. Calculate ring boundaries',
                '2. Distribute evenly in ring',
                '3. Add random angular offset',
                '4. Validate Y < 0 (underground)',
            ]
        else:
            steps = [
                '1. Divide world into regions',
                '2. Hash: seed + regionX×K1 + regionZ×K2 + salt',
                '3. Random offset within (spacing - separation)',
                '4. Check biome suitability',
            ]
        
        y_pos = 8.5
        for i, step in enumerate(steps):
            highlight = i <= int(struct_progress * len(steps))
            color = COLORS['accent'] if highlight else COLORS['grid']
            ax_algo.text(0.5, y_pos, step, color=color, fontsize=9)
            y_pos -= 1.2
        
        # Current calculation (animated)
        if num_to_show > 0 and current_struct['spacing'] > 0:
            cx, cz = positions[num_to_show - 1]
            rx = cx // current_struct['spacing']
            rz = cz // current_struct['spacing']
            
            ax_algo.text(5, 3, f'Region: ({rx}, {rz})', ha='center',
                        color=COLORS['text'], fontsize=9, family='monospace')
            ax_algo.text(5, 2.2, f'Position: ({cx}, {cz})', ha='center',
                        color=current_struct['color'], fontsize=10, family='monospace',
                        fontweight='bold')
        
        # ====================================================================
        # LEGEND
        # ====================================================================
        
        ax_legend.set_xlim(0, 18)
        ax_legend.set_ylim(0, 2)
        ax_legend.axis('off')
        
        # Draw all structure types
        x_pos = 1
        for key in GENERATION_ORDER:
            struct = STRUCTURES[key]
            is_current = key == current_struct_key
            is_done = GENERATION_ORDER.index(key) < current_struct_idx
            
            alpha = 1.0 if (is_current or is_done) else 0.3
            
            draw_structure_marker(ax_legend, x_pos, 1, struct['symbol'],
                                struct['color'], size=100 if is_current else 70, alpha=alpha)
            
            name_parts = struct['name'].split()
            ax_legend.text(x_pos, 0.3, '\n'.join(name_parts), ha='center',
                          color=COLORS['text'] if alpha > 0.5 else COLORS['grid'],
                          fontsize=7)
            
            x_pos += 2.2
        
        return []
    
    # Create animation
    print(f"Generating {total_frames} frames at {fps} FPS...")
    anim = FuncAnimation(fig, update, frames=total_frames, init_func=init,
                        interval=1000/fps, blit=False)
    
    # Save
    print(f"Saving animation to: {save_path}")
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=100)
    print("✓ Multi-structure animation saved!")
    
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(os.path.dirname(script_dir), "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    output_path = os.path.join(plots_dir, "multi_structure_generation.gif")
    create_multi_structure_animation(output_path, seed=42, fps=15, duration=24)
