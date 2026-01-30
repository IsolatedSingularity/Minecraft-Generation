"""
Enhanced Structure Placement Animation

Visualizes Minecraft's village placement algorithm with spiral revelation,
salt-based seed calculation display, and biome suitability coloring.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import os

# ============================================================================
# VISUAL CONFIGURATION
# ============================================================================

plt.style.use('dark_background')

COLORS = {
    'background': '#0D1117',
    'grid': '#21262D',
    'text': '#E6EDF3',
    'accent': '#58A6FF',
    'village': '#FFD700',
    'spawn': '#FF4444',
    'region_highlight': '#58A6FF',
    'biome_plains': '#7CFC00',
    'biome_desert': '#F4A460',
    'biome_savanna': '#BDB76B',
    'biome_taiga': '#228B22',
    'biome_snowy': '#E0E0E0',
}

# ============================================================================
# STRUCTURE PLACEMENT ALGORITHM
# ============================================================================

class StructurePlacementSimulator:
    """
    Simulates Minecraft's grid-based structure placement algorithm.
    """
    
    def __init__(self, world_seed=42, world_size=8000):
        self.world_seed = world_seed
        self.world_size = world_size
        self.chunk_size = 16
        self.village_spacing = 32  # chunks
        self.village_separation = 8  # chunks
        self.village_salt = 10387312
        
        # Pre-calculate all regions in spiral order
        self.regions = self._generate_spiral_regions()
        self.villages = []
        self.current_idx = 0
        
    def _generate_spiral_regions(self):
        """Generate region coordinates in spiral order from center."""
        spacing = self.village_spacing * self.chunk_size
        max_regions = self.world_size // spacing
        
        regions = []
        x, z = 0, 0
        dx, dz = 0, -1
        
        for _ in range((max_regions * 2) ** 2):
            if (-max_regions <= x <= max_regions) and (-max_regions <= z <= max_regions):
                center_x = x * spacing
                center_z = z * spacing
                if abs(center_x) <= self.world_size//2 and abs(center_z) <= self.world_size//2:
                    regions.append({
                        'region_x': x,
                        'region_z': z,
                        'center_x': center_x,
                        'center_z': center_z,
                    })
            
            # Spiral movement
            if x == z or (x < 0 and x == -z) or (x > 0 and x == 1-z):
                dx, dz = -dz, dx
            x, z = x + dx, z + dz
        
        return regions
    
    def generate_region_seed(self, region_x, region_z):
        """Calculate region seed using Minecraft's algorithm."""
        return (self.world_seed + 
                region_x * region_x * 4987142 + 
                region_x * 5947611 + 
                region_z * region_z * 4392871 + 
                region_z * 389711 + 
                self.village_salt) & 0xFFFFFFFF
    
    def get_biome_suitability(self, x, z):
        """Calculate biome suitability (simplified noise-based)."""
        # Multi-scale noise for varied biome distribution
        scale1, scale2 = 3000, 1500
        temp = (np.sin(x / scale1) + 0.5 * np.cos(z / scale2) + 
                0.3 * np.sin((x + z) / 1000)) / 1.8
        humid = (np.cos(x / 3500) + 0.4 * np.sin(z / 2500)) / 1.4
        
        # Classify biome and return suitability
        if temp > 0.3 and -0.3 < humid < 0.3:
            return 0.95, 'Plains'
        elif temp > 0.5 and humid < -0.2:
            return 0.85, 'Desert'
        elif temp > 0.2 and humid > 0.2:
            return 0.75, 'Savanna'
        elif temp < -0.2:
            return 0.65, 'Taiga'
        elif temp < -0.4:
            return 0.55, 'Snowy'
        else:
            return 0.3, 'Other'
    
    def process_next_region(self):
        """Process the next region in spiral order."""
        if self.current_idx >= len(self.regions):
            return None
        
        region = self.regions[self.current_idx]
        self.current_idx += 1
        
        # Calculate region seed
        seed = self.generate_region_seed(region['region_x'], region['region_z'])
        np.random.seed(seed & 0x7FFFFFFF)
        
        # Check spawn probability (similar to Minecraft's check)
        spawn_check = np.random.randint(0, self.village_spacing)
        spawns = spawn_check < self.village_separation
        
        # Get biome suitability
        suitability, biome = self.get_biome_suitability(
            region['center_x'], region['center_z']
        )
        
        # Village only spawns if conditions met
        village = None
        if spawns and suitability > 0.5:
            # Add position offset
            offset_x = np.random.uniform(-self.village_spacing * self.chunk_size // 4,
                                        self.village_spacing * self.chunk_size // 4)
            offset_z = np.random.uniform(-self.village_spacing * self.chunk_size // 4,
                                        self.village_spacing * self.chunk_size // 4)
            
            village = {
                'x': region['center_x'] + offset_x,
                'z': region['center_z'] + offset_z,
                'biome': biome,
                'suitability': suitability,
            }
            self.villages.append(village)
        
        return {
            'region': region,
            'seed': seed,
            'spawns': spawns,
            'suitability': suitability,
            'biome': biome,
            'village': village,
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_structure_placement_animation(save_path, frames=200, dpi=200, fps=15):
    """
    Create high-quality structure placement animation.
    """
    print("=" * 60)
    print("STRUCTURE PLACEMENT ALGORITHM VISUALIZATION")
    print("=" * 60)
    
    # Initialize
    sim = StructurePlacementSimulator(world_seed=42, world_size=8000)
    
    # Create figure - wider aspect ratio for better spacing
    fig = plt.figure(figsize=(20, 9))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Layout - significantly wider right column to prevent text overlap
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], height_ratios=[1.2, 1],
                         hspace=0.35, wspace=0.25,
                         left=0.04, right=0.97, top=0.92, bottom=0.08)
    
    ax_map = fig.add_subplot(gs[:, 0])
    ax_algo = fig.add_subplot(gs[0, 1])
    ax_stats = fig.add_subplot(gs[1, 1])
    
    for ax in [ax_map, ax_algo, ax_stats]:
        ax.set_facecolor(COLORS['background'])
        ax.tick_params(colors=COLORS['text'])
    
    # Map setup
    world_size = sim.world_size
    ax_map.set_xlim(-world_size//2, world_size//2)
    ax_map.set_ylim(-world_size//2, world_size//2)
    ax_map.set_aspect('equal')
    ax_map.set_title('Overworld Structure Distribution', color=COLORS['text'],
                    fontsize=16, fontweight='bold', pad=15)
    ax_map.set_xlabel('X Coordinate', color=COLORS['text'])
    ax_map.set_ylabel('Z Coordinate', color=COLORS['text'])
    
    # Draw background grid
    spacing = sim.village_spacing * sim.chunk_size
    for x in range(-world_size//2, world_size//2 + 1, spacing):
        ax_map.axvline(x, color=COLORS['grid'], alpha=0.3, linewidth=0.5)
    for z in range(-world_size//2, world_size//2 + 1, spacing):
        ax_map.axhline(z, color=COLORS['grid'], alpha=0.3, linewidth=0.5)
    
    # Spawn point
    ax_map.scatter(0, 0, c=COLORS['spawn'], s=200, marker='*', 
                  zorder=10, edgecolors='white', linewidth=2, label='Spawn')
    
    # Dynamic elements
    region_highlight = Rectangle((0, 0), spacing, spacing, fill=False,
                                 edgecolor=COLORS['region_highlight'],
                                 linewidth=3, alpha=0, zorder=5)
    ax_map.add_patch(region_highlight)
    
    village_scatter = ax_map.scatter([], [], c=[], s=80, marker='s',
                                     edgecolors='white', linewidth=1,
                                     cmap='YlOrBr', zorder=6, alpha=0.9)
    
    # Algorithm panel - with proper margins and expanded bounds
    ax_algo.set_xlim(-0.5, 12.5)
    ax_algo.set_ylim(-0.5, 10.5)
    ax_algo.set_title('Placement Algorithm', color=COLORS['text'],
                     fontsize=14, fontweight='bold', pad=10)
    ax_algo.axis('off')
    
    # Algorithm steps with better spacing
    algo_steps = [
        "1. Calculate Region Seed",
        "2. Check Spawn Probability",
        "3. Evaluate Biome Suitability",
        "4. Place Structure (if valid)",
    ]
    
    step_texts = []
    for i, step in enumerate(algo_steps):
        txt = ax_algo.text(0.5, 9.5 - i*1.3, step, color=COLORS['text'],
                          fontsize=10, alpha=0.5)
        step_texts.append(txt)
    
    # Seed calculation display - properly positioned with more room
    seed_box = FancyBboxPatch((0.3, 3.5), 11.5, 2.5, boxstyle="round,pad=0.1",
                             facecolor=COLORS['grid'], alpha=0.5)
    ax_algo.add_patch(seed_box)
    
    seed_formula = ax_algo.text(6.0, 5.3, 
        "S = (seed + x²·4987142 + x·5947611\n     + z²·4392871 + z·389711 + salt) mod 2³²",
        ha='center', va='center', color=COLORS['accent'], fontsize=9,
        family='monospace')
    seed_result = ax_algo.text(6.0, 4.0, '', ha='center', va='center',
                              color=COLORS['village'], fontsize=10,
                              family='monospace', fontweight='bold')
    
    # Result indicator - properly positioned
    result_text = ax_algo.text(6.0, 1.2, '', ha='center', va='center',
                              fontsize=13, fontweight='bold')
    
    # Stats panel - with proper margins and expanded bounds
    ax_stats.set_xlim(-0.5, 12.5)
    ax_stats.set_ylim(-0.5, 6.5)
    ax_stats.set_title('Generation Statistics', color=COLORS['text'],
                      fontsize=14, fontweight='bold', pad=10)
    ax_stats.axis('off')
    
    stats_text = ax_stats.text(0.3, 5.8, '', color=COLORS['text'], fontsize=10,
                              verticalalignment='top', family='monospace')
    
    # Biome distribution bars
    biome_colors = {
        'Plains': '#7CFC00',
        'Desert': '#F4A460',
        'Savanna': '#BDB76B',
        'Taiga': '#228B22',
    }
    biome_counts = {b: 0 for b in biome_colors}
    biome_bars = {}
    
    for i, (biome, color) in enumerate(biome_colors.items()):
        ax_stats.text(0.3, 2.8 - i*0.6, biome, color=COLORS['text'], fontsize=9)
        bar = FancyBboxPatch((2.3, 2.6 - i*0.6), 0.1, 0.4,
                            boxstyle="round,pad=0.02",
                            facecolor=color, alpha=0.8)
        ax_stats.add_patch(bar)
        biome_bars[biome] = bar
    
    # Animation data
    all_villages = []
    all_suitabilities = []
    regions_processed = 0
    
    def animate(frame):
        nonlocal regions_processed, all_villages, all_suitabilities
        
        # Process multiple regions per frame for smoother animation
        regions_per_frame = 3
        
        for _ in range(regions_per_frame):
            result = sim.process_next_region()
            if result is None:
                break
            
            regions_processed += 1
            region = result['region']
            
            # Update region highlight
            region_highlight.set_xy((region['center_x'] - spacing//2,
                                    region['center_z'] - spacing//2))
            region_highlight.set_alpha(0.8)
            
            # Highlight current algorithm step
            for i, txt in enumerate(step_texts):
                if i == 0:  # Seed calculation
                    txt.set_alpha(1.0 if frame % 4 == 0 else 0.5)
                elif i == 1:  # Probability check
                    txt.set_alpha(1.0 if frame % 4 == 1 else 0.5)
                elif i == 2:  # Biome check
                    txt.set_alpha(1.0 if frame % 4 == 2 else 0.5)
                else:  # Placement
                    txt.set_alpha(1.0 if frame % 4 == 3 else 0.5)
            
            # Update seed display
            seed_result.set_text(f"Region ({region['region_x']}, {region['region_z']})\n"
                               f"Seed: 0x{result['seed']:08X}")
            
            # Update result
            if result['village']:
                result_text.set_text('✓ VILLAGE PLACED')
                result_text.set_color('#00FF00')
                
                all_villages.append([result['village']['x'], result['village']['z']])
                all_suitabilities.append(result['suitability'])
                
                # Update biome counts
                biome = result['biome']
                if biome in biome_counts:
                    biome_counts[biome] += 1
            else:
                if not result['spawns']:
                    result_text.set_text('✗ PROBABILITY FAILED')
                    result_text.set_color('#FF6B6B')
                else:
                    result_text.set_text('✗ BIOME UNSUITABLE')
                    result_text.set_color('#FFA500')
        
        # Update village scatter
        if all_villages:
            village_scatter.set_offsets(all_villages)
            village_scatter.set_array(np.array(all_suitabilities))
        
        # Update stats
        stats_text.set_text(
            f"Regions Scanned: {regions_processed}\n"
            f"Villages Found:  {len(all_villages)}\n"
            f"Spawn Rate:      {100*len(all_villages)/max(1,regions_processed):.1f}%\n"
            f"World Seed:      {sim.world_seed}"
        )
        
        # Update biome bars
        max_count = max(biome_counts.values()) if biome_counts.values() else 1
        for biome, bar in biome_bars.items():
            width = 6 * biome_counts[biome] / max(max_count, 1)
            bar.set_width(max(0.1, width))
        
        return [region_highlight, village_scatter, seed_result, result_text, stats_text]
    
    print(f"Generating {frames} frames at {dpi} DPI...")
    anim = animation.FuncAnimation(fig, animate, frames=frames,
                                  interval=1000/fps, blit=False)
    
    try:
        print(f"Saving animation to: {save_path}")
        anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
        print("✓ Animation saved successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("Trying with reduced settings...")
        anim.save(save_path, writer='pillow', fps=10, dpi=150)
    
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(os.path.dirname(script_dir), "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    output_path = os.path.join(plots_dir, "structure_placement.gif")
    create_structure_placement_animation(output_path, frames=200, dpi=200, fps=15)
