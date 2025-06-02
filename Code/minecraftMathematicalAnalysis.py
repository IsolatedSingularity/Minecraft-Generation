"""
Minecraft Mathematical Foundation Analysis

This script creates specialized visualizations focusing on:
1. Linear Congruential Generator patterns and seed analysis
2. Mathematical precision of structure placement algorithms
3. Speedrunning optimization visualizations
4. Probabilistic analysis of generation parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import seaborn as sns
import os

# Configure global styling
plt.style.use('dark_background')
backgroundColor = '#0D1117'
gridLineColor = '#21262D'
textFontColor = '#E6EDF3'
accentColor = '#58A6FF'
plotsPath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots")

class MinecraftMathAnalyzer:
    def __init__(self, world_seed=42):
        """Initialize the mathematical analyzer."""
        self.world_seed = world_seed
        
        # LCG constants (Java Random implementation)
        self.lcg_multiplier = 0x5DEECE66D
        self.lcg_addend = 0xB
        self.lcg_modulus = 2**48
        
        # Structure generation constants
        self.village_salt = 10387312
        self.fortress_salt = 30084232
        
        # Stronghold ring data
        self.stronghold_rings = [
            {'count': 3, 'min_radius': 1280, 'max_radius': 2816},
            {'count': 6, 'min_radius': 4352, 'max_radius': 5888},
            {'count': 10, 'min_radius': 7424, 'max_radius': 8960},
            {'count': 15, 'min_radius': 10496, 'max_radius': 12032},
            {'count': 21, 'min_radius': 13568, 'max_radius': 15104},
            {'count': 28, 'min_radius': 16640, 'max_radius': 18176}
        ]
    
    def lcg_next(self, seed):
        """Generate next value using Linear Congruential Generator."""
        return (self.lcg_multiplier * seed + self.lcg_addend) % self.lcg_modulus
    
    def generate_region_seed(self, region_x, region_z, salt):
        """Generate region seed using Minecraft's exact algorithm."""
        return (self.world_seed + 
                region_x * region_x * 4987142 + 
                region_x * 5947611 + 
                region_z * region_z * 4392871 + 
                region_z * 389711 + 
                salt) & 0xFFFFFFFF
    
    def visualize_speedrunning_optimization(self):
        """Create visualizations for speedrunning optimization strategies."""
        print("Creating speedrunning optimization analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor(backgroundColor)
        
        # 1. Stronghold triangulation visualization
        ax1.set_facecolor(backgroundColor)
        ax1.set_aspect('equal')
        
        # Generate first ring strongholds
        np.random.seed(self.world_seed)
        angles = np.linspace(0, 2*np.pi, 3, endpoint=False)
        angles += np.random.normal(0, np.pi/12, 3)  # Small randomization
        
        strongholds = []
        for angle in angles:
            radius = np.random.uniform(1280, 2816)
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            strongholds.append((x, z))
        
        # Plot strongholds
        for i, (x, z) in enumerate(strongholds):
            ax1.scatter(x, z, c='#FF6B6B', s=200, alpha=0.8, 
                       edgecolors='white', linewidth=2, zorder=5)
            ax1.text(x*1.1, z*1.1, f'SH{i+1}', color=textFontColor, 
                    fontsize=12, ha='center', weight='bold')
        
        # Draw spawn point
        ax1.scatter(0, 0, c='red', s=300, marker='*', zorder=10, 
                   edgecolors='white', linewidth=2)
        ax1.text(0, -200, 'SPAWN', color=textFontColor, fontsize=12, 
                ha='center', weight='bold')
        
        # Draw ring boundaries
        min_circle = Circle((0, 0), 1280, fill=False, color='white', alpha=0.5, linewidth=2)
        max_circle = Circle((0, 0), 2816, fill=False, color='white', alpha=0.5, linewidth=2)
        ax1.add_patch(min_circle)
        ax1.add_patch(max_circle)
        
        # Simulate triangulation
        # Show eye of ender throws
        throw_point = (500, 300)  # Player position
        ax1.scatter(throw_point[0], throw_point[1], c='#00FF00', s=100, 
                   marker='^', zorder=8, label='Player Position')
        
        # Draw triangulation lines to nearest stronghold
        nearest_sh = min(strongholds, key=lambda sh: np.sqrt((sh[0]-throw_point[0])**2 + (sh[1]-throw_point[1])**2))
        ax1.plot([throw_point[0], nearest_sh[0]], [throw_point[1], nearest_sh[1]], 
                'g--', linewidth=2, alpha=0.7, label='Eye Trajectory')
        
        # Calculate and show angle
        angle_to_sh = np.degrees(np.arctan2(nearest_sh[1]-throw_point[1], 
                                          nearest_sh[0]-throw_point[0]))
        ax1.text(throw_point[0]+200, throw_point[1]+200, f'Angle: {angle_to_sh:.1f}°', 
                color='#00FF00', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round', facecolor=backgroundColor, alpha=0.8))
        
        ax1.set_title('Stronghold Triangulation Strategy', color=textFontColor, fontsize=14)
        ax1.set_xlabel('X Coordinate', color=textFontColor)
        ax1.set_ylabel('Z Coordinate', color=textFontColor)
        ax1.set_xlim(-3500, 3500)
        ax1.set_ylim(-3500, 3500)
        ax1.legend()
        ax1.tick_params(colors=textFontColor)
        ax1.grid(True, color=gridLineColor, alpha=0.3)
        
        # 2. Route optimization
        ax2.set_facecolor(backgroundColor)
        
        # Generate multiple resource locations
        np.random.seed(self.world_seed)
        
        # Generate villages (early game resources)
        villages = []
        for _ in range(8):
            x = np.random.uniform(-1000, 1000)
            z = np.random.uniform(-1000, 1000)
            villages.append((x, z))
        
        # Lava pools (nether portal)
        lava_pools = []
        for _ in range(3):
            x = np.random.uniform(-800, 800)
            z = np.random.uniform(-800, 800)
            lava_pools.append((x, z))
        
        # Calculate optimal route using simple nearest neighbor
        start = (0, 0)
        current_pos = start
        route = [start]
        remaining_villages = villages.copy()
        
        while remaining_villages:
            distances = [np.sqrt((v[0]-current_pos[0])**2 + (v[1]-current_pos[1])**2) 
                        for v in remaining_villages]
            nearest_idx = np.argmin(distances)
            next_village = remaining_villages.pop(nearest_idx)
            route.append(next_village)
            current_pos = next_village
        
        # Add nearest lava pool
        lava_distances = [np.sqrt((l[0]-current_pos[0])**2 + (l[1]-current_pos[1])**2) 
                         for l in lava_pools]
        nearest_lava = lava_pools[np.argmin(lava_distances)]
        route.append(nearest_lava)
        
        # Add nearest stronghold
        route.append(nearest_sh)
        
        # Plot route
        route_x = [p[0] for p in route]
        route_z = [p[1] for p in route]
        ax2.plot(route_x, route_z, 'b-', linewidth=3, alpha=0.7, label='Optimal Route')
        
        # Plot locations
        if villages:
            villages_arr = np.array(villages)
            ax2.scatter(villages_arr[:, 0], villages_arr[:, 1], c='#FFD700', s=100, 
                       alpha=0.8, label='Villages', marker='s')
        
        lava_arr = np.array(lava_pools)
        ax2.scatter(lava_arr[:, 0], lava_arr[:, 1], c='#FF4500', s=150, 
                   alpha=0.8, label='Lava Pools', marker='D')
        
        ax2.scatter(nearest_sh[0], nearest_sh[1], c='#FF6B6B', s=200, 
                   alpha=0.8, label='Target Stronghold')
        
        ax2.scatter(0, 0, c='red', s=200, marker='*', label='Spawn')
        
        # Calculate total distance
        total_distance = sum(np.sqrt((route[i+1][0]-route[i][0])**2 + 
                                   (route[i+1][1]-route[i][1])**2) 
                          for i in range(len(route)-1))
        
        ax2.text(0.02, 0.98, f'Total Distance: {total_distance:.0f} blocks\\n'
                            f'Estimated Time: {total_distance/4.3/60:.1f} minutes',
                transform=ax2.transAxes, color=textFontColor, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=backgroundColor, alpha=0.9))
        
        ax2.set_title('Speedrun Route Optimization', color=textFontColor, fontsize=14)
        ax2.set_xlabel('X Coordinate', color=textFontColor)
        ax2.set_ylabel('Z Coordinate', color=textFontColor)
        ax2.legend()
        ax2.tick_params(colors=textFontColor)
        ax2.grid(True, color=gridLineColor, alpha=0.3)
        
        # 3. Probability analysis for speedrunning
        ax3.set_facecolor(backgroundColor)
        
        # Analyze stronghold distance probabilities
        distances = []
        for _ in range(10000):
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.uniform(1280, 2816)
            distances.append(radius)
        
        # Create probability distribution
        hist, bins = np.histogram(distances, bins=50, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        ax3.fill_between(bin_centers, hist, alpha=0.7, color=accentColor, label='Distance Distribution')
        ax3.axvline(np.mean(distances), color='#FF6B6B', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(distances):.0f}')
        ax3.axvline(np.percentile(distances, 10), color='#FFD700', linestyle='--', linewidth=2, 
                   label=f'10th percentile: {np.percentile(distances, 10):.0f}')
        
        ax3.set_title('Stronghold Distance Probability', color=textFontColor, fontsize=14)
        ax3.set_xlabel('Distance from Spawn', color=textFontColor)
        ax3.set_ylabel('Probability Density', color=textFontColor)
        ax3.legend()
        ax3.tick_params(colors=textFontColor)
        ax3.grid(True, color=gridLineColor, alpha=0.3)
        
        # 4. Seed comparison for speedrunning viability
        ax4.set_facecolor(backgroundColor)
        
        # Test multiple seeds
        test_seeds = [42, 12345, 98765, 314159, 271828, 161803, 577215, 123456]
        seed_metrics = []
        
        for seed in test_seeds:
            temp_analyzer = MinecraftMathAnalyzer(seed)
            
            # Calculate first stronghold distance
            np.random.seed(seed)
            angle = np.random.uniform(0, 2*np.pi/3)
            radius = np.random.uniform(1280, 2816)
            sh_distance = radius
            
            # Count nearby villages
            village_count = 0
            for region_x in range(-3, 4):
                for region_z in range(-3, 4):
                    region_seed = temp_analyzer.generate_region_seed(region_x, region_z, 
                                                                   temp_analyzer.village_salt)
                    np.random.seed(region_seed & 0xFFFFFFFF)
                    if np.random.randint(0, 32) < 8:
                        village_count += 1
            
            # Calculate speedrun score (lower stronghold distance + more villages = better)
            score = 3000 - sh_distance + village_count * 50
            seed_metrics.append((seed, sh_distance, village_count, score))
        
        # Sort by score
        seed_metrics.sort(key=lambda x: x[3], reverse=True)
        
        # Create bar chart
        seeds = [str(s[0]) for s in seed_metrics]
        scores = [s[3] for s in seed_metrics]
        colors = ['#00FF00' if i < 3 else '#FFD700' if i < 6 else '#FF6B6B' 
                 for i in range(len(scores))]
        
        bars = ax4.bar(range(len(seeds)), scores, color=colors, alpha=0.8)
        ax4.set_title('Seed Viability for Speedrunning', color=textFontColor, fontsize=14)
        ax4.set_xlabel('Seed', color=textFontColor)
        ax4.set_ylabel('Speedrun Score', color=textFontColor)
        ax4.set_xticks(range(len(seeds)))
        ax4.set_xticklabels(seeds, rotation=45)
        ax4.tick_params(colors=textFontColor)
        ax4.grid(True, color=gridLineColor, alpha=0.3)
        
        # Add legend for colors
        legend_elements = [
            mpatches.Patch(color='#00FF00', label='Excellent'),
            mpatches.Patch(color='#FFD700', label='Good'),
            mpatches.Patch(color='#FF6B6B', label='Average')        ]
        ax4.legend(handles=legend_elements, loc='upper right')
        
        plt.suptitle('Minecraft Speedrunning Optimization Analysis', 
                    color=textFontColor, fontsize=16)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(plotsPath, "minecraft_speedrunning_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=backgroundColor)
        print(f"Saved speedrunning analysis to {save_path}")
        plt.show()

def main():
    """Main function to run all mathematical analyses."""
    print("Starting Minecraft Mathematical Analysis...")
    
    # Initialize analyzer
    analyzer = MinecraftMathAnalyzer(world_seed=42)
    
    # Generate all visualizations
    analyzer.visualize_speedrunning_optimization()
    
    print("Minecraft mathematical analysis completed!")

if __name__ == "__main__":
    main()
