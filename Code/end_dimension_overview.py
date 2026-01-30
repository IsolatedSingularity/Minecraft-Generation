"""
End Dimension Overview Visualization

Comprehensive visualization of the End dimension structure:
- Central island with Exit Portal and Obsidian Pillars
- 20 End Gateway positions (radius 96 blocks)
- Outer island generation rings (overflow-based concentric bands)
- Mathematical formulas governing generation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyBboxPatch, RegularPolygon
from matplotlib.collections import PathCollection
import matplotlib.colors as mcolors
import os

# ============================================================================
# VISUAL CONFIGURATION
# ============================================================================

plt.style.use('dark_background')

COLORS = {
    'background': '#0D1117',
    'void': '#0a0512',
    'text': '#E6EDF3',
    'accent': '#58A6FF',
    'central_island': '#3d3a50',
    'end_stone': '#d4d8a8',
    'obsidian': '#1a0a2e',
    'portal': '#2ecc71',
    'gateway': '#9b59b6',
    'gateway_beam': '#e056fd',
    'crystal': '#00ff88',
    'outer_island': '#2d2845',
    'ring_1': '#9b59b6',
    'ring_2': '#8e44ad',
    'ring_3': '#6c3483',
    'grid': '#21262D',
}

# ============================================================================
# END DIMENSION MATHEMATICS
# ============================================================================

def calculate_end_gateway_positions():
    """
    Calculate positions of all 20 End Gateway portals.
    
    Formula: x = floor(96 * cos(π * k / 10))
             z = floor(96 * sin(π * k / 10))
    for k = 0, 1, 2, ..., 19
    
    Gateways spawn at Y=75, radius 96 blocks from center.
    """
    gateways = []
    radius = 96
    for k in range(20):
        angle = np.pi * k / 10
        x = int(np.floor(radius * np.cos(angle)))
        z = int(np.floor(radius * np.sin(angle)))
        gateways.append({
            'x': x,
            'z': z,
            'angle': angle,
            'k': k
        })
    return gateways


def calculate_outer_island_rings(max_radius=500000):
    """
    Calculate outer island generation rings.
    
    Outer islands generate where:
    1. sin(X² + Z² / 43748131634) > 0 
    2. X² + Z² - 1,000,000 > 0 (outside 1000 block radius)
    
    Due to arithmetic overflow, rings appear at:
    - First ring ends at ±370,720 blocks
    - Second ring starts at ±524,288 blocks
    - Pattern repeats with increasing gaps
    """
    rings = []
    
    # Inner void (no islands < 1000 blocks)
    rings.append({
        'name': 'Central Void',
        'inner': 0,
        'outer': 1000,
        'color': COLORS['void'],
        'islands': False
    })
    
    # Main outer islands region
    rings.append({
        'name': 'Outer Islands',
        'inner': 1000,
        'outer': 370720,
        'color': COLORS['outer_island'],
        'islands': True
    })
    
    # First gap (overflow)
    rings.append({
        'name': 'Void Gap 1',
        'inner': 370720,
        'outer': 524288,
        'color': COLORS['void'],
        'islands': False
    })
    
    # Second ring
    rings.append({
        'name': 'Ring 2',
        'inner': 524288,
        'outer': 1048576,
        'color': COLORS['outer_island'],
        'islands': True
    })
    
    return rings


def generate_outer_island_samples(n_samples=2000, seed=42):
    """
    Generate sample positions for outer island visualization.
    Uses simplified noise to represent island distribution.
    """
    np.random.seed(seed)
    islands = []
    
    # Generate islands in valid regions
    for _ in range(n_samples):
        # Random angle and radius in outer region
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(1200, 15000)  # Scaled for visualization
        
        # Add some clustering
        if np.random.random() < 0.3:
            # Cluster around end cities (simplified)
            radius += np.random.uniform(-500, 500)
        
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        
        # Size variation
        size = np.random.uniform(50, 300)
        
        islands.append({
            'x': x,
            'z': z,
            'radius': radius,
            'size': size
        })
    
    return islands


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_end_dimension_overview(save_path, dpi=300):
    """
    Create comprehensive End dimension overview visualization.
    """
    print("=" * 60)
    print("END DIMENSION OVERVIEW VISUALIZATION")
    print("=" * 60)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Main layout
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 0.8], height_ratios=[1.2, 1],
                         hspace=0.2, wspace=0.15,
                         left=0.05, right=0.95, top=0.92, bottom=0.08)
    
    # Main overview (top-left, spans 2 rows)
    ax_main = fig.add_subplot(gs[:, 0])
    
    # Central island detail (top-middle)
    ax_central = fig.add_subplot(gs[0, 1])
    
    # Gateway positions (top-right)
    ax_gateway = fig.add_subplot(gs[0, 2])
    
    # Ring structure (bottom-middle, spans 2 cols)
    ax_rings = fig.add_subplot(gs[1, 1:])
    
    for ax in [ax_main, ax_central, ax_gateway, ax_rings]:
        ax.set_facecolor(COLORS['void'])
        ax.tick_params(colors=COLORS['text'])
    
    # ========================================================================
    # MAIN OVERVIEW
    # ========================================================================
    
    ax_main.set_xlim(-18000, 18000)
    ax_main.set_ylim(-18000, 18000)
    ax_main.set_aspect('equal')
    ax_main.set_title('The End — Dimension Overview', color=COLORS['text'],
                     fontsize=16, fontweight='bold', pad=15)
    
    # Draw outer island samples
    islands = generate_outer_island_samples(n_samples=3000)
    island_x = [i['x'] for i in islands]
    island_z = [i['z'] for i in islands]
    island_sizes = [i['size'] / 10 for i in islands]
    
    ax_main.scatter(island_x, island_z, s=island_sizes, c=COLORS['end_stone'],
                   alpha=0.3, marker='o', edgecolors='none')
    
    # Draw ring boundaries
    inner_void = Circle((0, 0), 1000, fill=True, color=COLORS['void'], 
                        alpha=1, zorder=5)
    ax_main.add_patch(inner_void)
    
    # Central island
    central = Circle((0, 0), 150, fill=True, color=COLORS['central_island'],
                    alpha=0.9, zorder=10)
    ax_main.add_patch(central)
    
    # Gateway ring indicator
    gateway_ring = Circle((0, 0), 96 * 50, fill=False, 
                         edgecolor=COLORS['gateway'], linewidth=1, 
                         linestyle='--', alpha=0.5, zorder=11)
    ax_main.add_patch(gateway_ring)
    
    # Distance markers
    for dist in [5000, 10000, 15000]:
        ring = Circle((0, 0), dist, fill=False, edgecolor=COLORS['grid'],
                     linewidth=0.5, linestyle=':', alpha=0.3)
        ax_main.add_patch(ring)
        ax_main.text(dist * 0.707, dist * 0.707, f'{dist:,}', 
                    color=COLORS['text'], fontsize=8, alpha=0.5)
    
    # Axis labels
    ax_main.set_xlabel('X Coordinate (blocks)', color=COLORS['text'])
    ax_main.set_ylabel('Z Coordinate (blocks)', color=COLORS['text'])
    
    # Legend annotation
    ax_main.annotate('Central\nIsland', xy=(0, 0), xytext=(3000, 3000),
                    color=COLORS['text'], fontsize=9,
                    arrowprops=dict(arrowstyle='->', color=COLORS['accent'], alpha=0.6))
    
    ax_main.annotate('Outer Islands\n(1,000+ blocks)', xy=(8000, 0), xytext=(12000, 5000),
                    color=COLORS['text'], fontsize=9,
                    arrowprops=dict(arrowstyle='->', color=COLORS['accent'], alpha=0.6))
    
    # ========================================================================
    # CENTRAL ISLAND DETAIL
    # ========================================================================
    
    ax_central.set_xlim(-120, 120)
    ax_central.set_ylim(-120, 120)
    ax_central.set_aspect('equal')
    ax_central.set_title('Central Island Structure', color=COLORS['text'],
                        fontsize=12, fontweight='bold')
    
    # Main island
    island_main = Circle((0, 0), 100, fill=True, color=COLORS['central_island'],
                        alpha=0.8)
    ax_central.add_patch(island_main)
    
    # Exit portal (center)
    portal = Circle((0, 0), 8, fill=True, color=COLORS['portal'], alpha=0.9, zorder=5)
    ax_central.add_patch(portal)
    ax_central.text(0, -15, 'Exit Portal', ha='center', color=COLORS['text'], fontsize=8)
    
    # Obsidian pillars (10 total, various heights)
    pillar_radius = 76
    pillar_heights = [76, 79, 82, 85, 88, 91, 94, 97, 100, 103]  # Y heights
    
    for i in range(10):
        angle = 2 * np.pi * i / 10
        x = pillar_radius * np.cos(angle)
        z = pillar_radius * np.sin(angle)
        
        # Pillar base
        pillar = Circle((x, z), 5, fill=True, color=COLORS['obsidian'], 
                       alpha=0.9, zorder=4)
        ax_central.add_patch(pillar)
        
        # Crystal on top
        crystal = Circle((x, z), 2.5, fill=True, color=COLORS['crystal'],
                        alpha=0.9, zorder=5)
        ax_central.add_patch(crystal)
    
    # Gateway positions around island
    gateways = calculate_end_gateway_positions()
    gateway_x = [g['x'] for g in gateways]
    gateway_z = [g['z'] for g in gateways]
    
    ax_central.scatter(gateway_x, gateway_z, c=COLORS['gateway'], s=30,
                      marker='s', alpha=0.9, zorder=6, edgecolors='white', linewidth=0.5)
    
    # Gateway ring
    gateway_circle = Circle((0, 0), 96, fill=False, edgecolor=COLORS['gateway_beam'],
                           linewidth=1.5, linestyle='--', alpha=0.7)
    ax_central.add_patch(gateway_circle)
    
    ax_central.axis('off')
    
    # ========================================================================
    # GATEWAY POSITIONS DETAIL
    # ========================================================================
    
    ax_gateway.set_xlim(-130, 130)
    ax_gateway.set_ylim(-170, 130)
    ax_gateway.set_aspect('equal')
    ax_gateway.set_title('End Gateway Positions', color=COLORS['text'],
                        fontsize=12, fontweight='bold')
    
    # Draw gateway positions with indices
    for g in gateways:
        ax_gateway.scatter(g['x'], g['z'], c=COLORS['gateway'], s=60,
                          marker='s', alpha=0.9, edgecolors='white', linewidth=1)
        ax_gateway.text(g['x'] + 8, g['z'], str(g['k']), color=COLORS['text'],
                       fontsize=7, alpha=0.7)
    
    # Gateway ring
    theta = np.linspace(0, 2*np.pi, 100)
    ax_gateway.plot(96 * np.cos(theta), 96 * np.sin(theta), 
                   color=COLORS['gateway_beam'], linewidth=1.5, linestyle='--', alpha=0.5)
    
    # Formula
    formula_text = (
        "Gateway Position Formula:\n"
        "x = ⌊96·cos(πk/10)⌋\n"
        "z = ⌊96·sin(πk/10)⌋\n"
        "k = 0, 1, 2, ..., 19\n"
        "Y = 75 (constant)"
    )
    ax_gateway.text(0, -145, formula_text, ha='center', va='top',
                   color=COLORS['accent'], fontsize=8, family='monospace',
                   bbox=dict(boxstyle='round', facecolor=COLORS['background'], 
                            alpha=0.8, edgecolor=COLORS['grid']))
    
    ax_gateway.axis('off')
    
    # ========================================================================
    # RING STRUCTURE DIAGRAM
    # ========================================================================
    
    ax_rings.set_xlim(0, 10)
    ax_rings.set_ylim(0, 6)
    ax_rings.set_title('End Dimension Ring Structure (Overflow Pattern)', 
                      color=COLORS['text'], fontsize=12, fontweight='bold')
    ax_rings.axis('off')
    
    # Ring data
    ring_data = [
        ('Central Island', '0', '~150', COLORS['central_island'], True),
        ('Void (No Islands)', '150', '1,000', COLORS['void'], False),
        ('Outer Islands', '1,000', '370,720', COLORS['outer_island'], True),
        ('Void Gap (Overflow)', '370,720', '524,288', COLORS['void'], False),
        ('Ring 2', '524,288', '1,048,576+', COLORS['outer_island'], True),
    ]
    
    # Draw table
    headers = ['Region', 'Inner (blocks)', 'Outer (blocks)', 'Islands?']
    for i, h in enumerate(headers):
        ax_rings.text(0.5 + i * 2.3, 5.3, h, color=COLORS['accent'], 
                     fontsize=9, fontweight='bold')
    
    ax_rings.axhline(5.1, color=COLORS['grid'], linewidth=1, alpha=0.5)
    
    for j, (name, inner, outer, color, has_islands) in enumerate(ring_data):
        y = 4.5 - j * 0.8
        
        # Color indicator
        rect = FancyBboxPatch((0.1, y - 0.25), 0.3, 0.5, boxstyle="round,pad=0.02",
                             facecolor=color, alpha=0.8)
        ax_rings.add_patch(rect)
        
        ax_rings.text(0.5, y, name, color=COLORS['text'], fontsize=9, va='center')
        ax_rings.text(2.8, y, inner, color=COLORS['text'], fontsize=9, va='center')
        ax_rings.text(5.1, y, outer, color=COLORS['text'], fontsize=9, va='center')
        ax_rings.text(7.4, y, '✓' if has_islands else '✗', 
                     color='#00ff88' if has_islands else '#ff4444',
                     fontsize=12, va='center', fontweight='bold')
    
    # Explanation
    explanation = (
        "The End dimension's outer islands generate in concentric rings due to\n"
        "an arithmetic overflow bug in the island generation formula:\n\n"
        "    sin(X² + Z² / 43748131634) > 0 AND X² + Z² > 1,000,000\n\n"
        "This creates periodic void gaps at specific distances from center."
    )
    ax_rings.text(5, 0.6, explanation, ha='center', va='center',
                 color=COLORS['text'], fontsize=8, family='monospace',
                 bbox=dict(boxstyle='round', facecolor=COLORS['background'],
                          alpha=0.7, edgecolor=COLORS['grid']))
    
    # Title
    fig.suptitle('', y=0.98)
    
    # Save
    print(f"Saving to: {save_path}")
    plt.savefig(save_path, dpi=dpi, facecolor=COLORS['background'],
               edgecolor='none', bbox_inches='tight')
    print("✓ End dimension overview saved!")
    plt.close(fig)
    
    return save_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(os.path.dirname(script_dir), "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    output_path = os.path.join(plots_dir, "end_dimension_overview.png")
    create_end_dimension_overview(output_path, dpi=300)
