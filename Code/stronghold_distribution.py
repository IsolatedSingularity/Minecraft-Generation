"""
Stronghold Ring Distribution Visualization

Creates a publication-quality visualization of Minecraft's stronghold
placement algorithm with concentric rings and polar coordinate mathematics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyArrowPatch
from matplotlib.lines import Line2D
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
    'spawn': '#FF4444',
}

RING_COLORS = [
    '#FF6B6B',  # Ring 1 - Coral
    '#4ECDC4',  # Ring 2 - Teal
    '#45B7D1',  # Ring 3 - Sky Blue
    '#96CEB4',  # Ring 4 - Sage
    '#FFEAA7',  # Ring 5 - Yellow
    '#DDA0DD',  # Ring 6 - Plum
    '#87CEEB',  # Ring 7 - Light Blue
    '#F0E68C',  # Ring 8 - Khaki
]

# ============================================================================
# STRONGHOLD GENERATION
# ============================================================================

STRONGHOLD_RINGS = [
    {'count': 3, 'min_radius': 1280, 'max_radius': 2816},
    {'count': 6, 'min_radius': 4352, 'max_radius': 5888},
    {'count': 10, 'min_radius': 7424, 'max_radius': 8960},
    {'count': 15, 'min_radius': 10496, 'max_radius': 12032},
    {'count': 21, 'min_radius': 13568, 'max_radius': 15104},
    {'count': 28, 'min_radius': 16640, 'max_radius': 18176},
    {'count': 36, 'min_radius': 19712, 'max_radius': 21248},
    {'count': 9, 'min_radius': 22784, 'max_radius': 24320},
]


def generate_strongholds(seed=42):
    """Generate all 128 stronghold positions."""
    np.random.seed(seed)
    strongholds = []
    
    for ring_idx, ring in enumerate(STRONGHOLD_RINGS):
        count = ring['count']
        min_r = ring['min_radius']
        max_r = ring['max_radius']
        color = RING_COLORS[ring_idx]
        
        # Base angle with slight randomization
        base_angle = np.random.uniform(0, 2*np.pi / count)
        
        for i in range(count):
            # Angular distribution with jitter
            angle = base_angle + i * 2*np.pi / count
            angle += np.random.normal(0, np.pi / (count * 4))
            
            # Radius within ring bounds
            radius = np.random.uniform(min_r, max_r)
            
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            
            strongholds.append({
                'x': x,
                'z': z,
                'radius': radius,
                'angle': angle,
                'ring': ring_idx + 1,
                'color': color,
            })
    
    return strongholds


def create_stronghold_distribution(save_path, dpi=300):
    """Create publication-quality stronghold distribution plot."""
    print("=" * 60)
    print("STRONGHOLD RING DISTRIBUTION")
    print("=" * 60)
    
    # Generate data
    strongholds = generate_strongholds(seed=42)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    # Set up axes
    max_radius = STRONGHOLD_RINGS[-1]['max_radius'] + 2000
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect('equal')
    
    # Title
    ax.set_title('Stronghold Distribution in the Overworld\n128 Strongholds Across 8 Concentric Rings',
                color=COLORS['text'], fontsize=18, fontweight='bold', pad=20)
    
    # Draw rings
    for ring_idx, ring in enumerate(STRONGHOLD_RINGS):
        color = RING_COLORS[ring_idx]
        
        # Ring band (wedge)
        wedge = Wedge((0, 0), ring['max_radius'], 0, 360,
                     width=ring['max_radius'] - ring['min_radius'],
                     color=color, alpha=0.15)
        ax.add_patch(wedge)
        
        # Ring boundaries
        inner_circle = Circle((0, 0), ring['min_radius'], fill=False,
                             color=color, alpha=0.5, linewidth=1.5,
                             linestyle='--')
        outer_circle = Circle((0, 0), ring['max_radius'], fill=False,
                             color=color, alpha=0.5, linewidth=1.5)
        ax.add_patch(inner_circle)
        ax.add_patch(outer_circle)
        
        # Ring label
        label_angle = np.pi / 4 + ring_idx * 0.15
        label_r = (ring['min_radius'] + ring['max_radius']) / 2
        label_x = label_r * np.cos(label_angle)
        label_z = label_r * np.sin(label_angle)
        ax.annotate(f'Ring {ring_idx + 1}\n({ring["count"]})',
                   xy=(label_x, label_z), ha='center', va='center',
                   color=color, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['background'],
                            edgecolor=color, alpha=0.9))
    
    # Draw strongholds
    for sh in strongholds:
        ax.scatter(sh['x'], sh['z'], c=sh['color'], s=120, alpha=0.9,
                  edgecolors='white', linewidth=1.5, zorder=5)
    
    # Spawn point
    ax.scatter(0, 0, c=COLORS['spawn'], s=400, marker='*',
              edgecolors='white', linewidth=2, zorder=10, label='World Spawn')
    
    # Cardinal directions
    for angle, label in [(0, 'E (+X)'), (np.pi/2, 'N (-Z)'), 
                         (np.pi, 'W (-X)'), (3*np.pi/2, 'S (+Z)')]:
        r = max_radius - 1000
        x = r * np.cos(angle)
        z = r * np.sin(angle)
        ax.annotate(label, xy=(x, z), ha='center', va='center',
                   color=COLORS['text'], fontsize=12, fontweight='bold',
                   alpha=0.7)
    
    # Distance scale
    scale_y = -max_radius + 1500
    for dist in [5000, 10000, 15000, 20000]:
        ax.axhline(scale_y, xmin=0.5, xmax=0.5+dist/(2*max_radius),
                  color=COLORS['text'], alpha=0.5, linewidth=2)
        ax.text(dist, scale_y - 800, f'{dist:,}', ha='center',
               color=COLORS['text'], fontsize=9)
    ax.text(10000, scale_y - 1800, 'Distance (blocks)', ha='center',
           color=COLORS['text'], fontsize=10)
    
    # Statistics box
    total_strongholds = sum(ring['count'] for ring in STRONGHOLD_RINGS)
    stats_text = (
        f"Total Strongholds: {total_strongholds}\n"
        f"First Ring Distance: 1,280 - 2,816 blocks\n"
        f"Last Ring Distance: 22,784 - 24,320 blocks\n"
        f"Angular Distribution: θ = 2π·i/n + random"
    )
    ax.text(-max_radius + 1000, max_radius - 1000, stats_text,
           color=COLORS['text'], fontsize=11, verticalalignment='top',
           family='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['background'],
                    edgecolor=COLORS['accent'], alpha=0.9))
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['spawn'],
               markersize=15, label='World Spawn', linestyle='None'),
    ]
    for ring_idx, ring in enumerate(STRONGHOLD_RINGS):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=RING_COLORS[ring_idx],
                   markersize=10, 
                   label=f'Ring {ring_idx+1} ({ring["count"]} strongholds)',
                   linestyle='None')
        )
    
    ax.legend(handles=legend_elements, loc='lower right',
             facecolor=COLORS['background'], edgecolor=COLORS['grid'],
             labelcolor=COLORS['text'], fontsize=9)
    
    # Grid
    ax.grid(True, color=COLORS['grid'], alpha=0.3, linestyle='--')
    ax.set_xlabel('X Coordinate (blocks)', color=COLORS['text'], fontsize=12)
    ax.set_ylabel('Z Coordinate (blocks)', color=COLORS['text'], fontsize=12)
    ax.tick_params(colors=COLORS['text'])
    
    # Save
    plt.tight_layout()
    print(f"Saving to: {save_path}")
    plt.savefig(save_path, dpi=dpi, facecolor=COLORS['background'],
               bbox_inches='tight')
    print("✓ Saved successfully!")
    plt.close(fig)
    
    return save_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(os.path.dirname(script_dir), "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    output_path = os.path.join(plots_dir, "stronghold_rings.png")
    create_stronghold_distribution(output_path, dpi=300)
