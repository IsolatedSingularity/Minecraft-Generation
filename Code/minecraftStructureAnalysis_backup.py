"""
Advanced Minecraft Structure Analysis and Visualization

This script creates comprehensive visualizations of Minecraft's underlying mathematical structures:
1. Multi-layered biome generation with noise field analysis
2. Structure placement algorithms with grid visualization
3. Seed-based deterministic pattern analysis
4. Cross-dimensional structure correlation mapping
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Wedge
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import seaborn as sns
from scipy.spatial.distance import cdist
import os

# Simple noise implementation to replace the noise library
def simple_perlin_noise(x, y, seed=0):
    """Simple 2D noise function as a replacement for pnoise2"""
    import math
    np.random.seed(seed)
    # Simple fractal noise using sine waves
    freq1, freq2, freq3 = 0.01, 0.05, 0.1
    amp1, amp2, amp3 = 1.0, 0.5, 0.25
    
    noise = (amp1 * np.sin(freq1 * x) * np.cos(freq1 * y) +
             amp2 * np.sin(freq2 * x) * np.cos(freq2 * y) +
             amp3 * np.sin(freq3 * x) * np.cos(freq3 * y))
    
    # Add some randomness based on position
    hash_val = hash((int(x/10), int(y/10), seed)) % 1000000
    np.random.seed(hash_val)
    noise += 0.3 * (np.random.random() - 0.5)
    
    return noise / 2.0  # Normalize to roughly [-1, 1]

# Configure global styling
plt.style.use('dark_background')
backgroundColor = '#0D1117'
gridLineColor = '#21262D'
textFontColor = '#E6EDF3'
accentColor = '#58A6FF'
plotsPath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Plots")

class MinecraftStructureAnalyzer:
    def __init__(self, world_seed=42, world_size=20000):
        """Initialize the Minecraft structure analyzer with configurable parameters."""
        self.world_seed = world_seed
        self.world_size = world_size
        self.chunk_size = 16
        np.random.seed(world_seed)
        
        # Minecraft generation constants
        self.village_spacing = 32
        self.village_separation = 8
        self.village_salt = 10387312
        
        self.fortress_spacing = 27
        self.fortress_separation = 4
        self.fortress_salt = 30084232
        
        # Stronghold ring definitions
        self.stronghold_rings = [
            {'count': 3, 'min_radius': 1280, 'max_radius': 2816, 'color': '#FF6B6B'},
            {'count': 6, 'min_radius': 4352, 'max_radius': 5888, 'color': '#4ECDC4'},
            {'count': 10, 'min_radius': 7424, 'max_radius': 8960, 'color': '#45B7D1'},
            {'count': 15, 'min_radius': 10496, 'max_radius': 12032, 'color': '#96CEB4'},
            {'count': 21, 'min_radius': 13568, 'max_radius': 15104, 'color': '#FFEAA7'},
            {'count': 28, 'min_radius': 16640, 'max_radius': 18176, 'color': '#DDA0DD'}
        ]
    
    def generate_biome_noise_fields(self, resolution=512):
        """Generate temperature and humidity noise fields for biome determination."""
        print("Generating biome noise fields...")
        
        # Create coordinate grids
        x = np.linspace(-self.world_size//2, self.world_size//2, resolution)
        z = np.linspace(-self.world_size//2, self.world_size//2, resolution)
        X, Z = np.meshgrid(x, z)
        
        # Generate temperature field with multiple octaves
        temperature = np.zeros_like(X)
        humidity = np.zeros_like(X)
        continentalness = np.zeros_like(X)
        erosion = np.zeros_like(X)
        
        # Multi-octave noise generation
        for octave in range(6):
            frequency = 2 ** octave / 1000.0
            amplitude = 1.0 / (2 ** octave)            
            temp_octave = np.array([[simple_perlin_noise(i * frequency, j * frequency, 
                                           self.world_seed) 
                                   for i in x] for j in z])
            
            humid_octave = np.array([[simple_perlin_noise(i * frequency, j * frequency, 
                                            self.world_seed + 1000) 
                                    for i in x] for j in z])
            
            cont_octave = np.array([[simple_perlin_noise(i * frequency * 0.5, j * frequency * 0.5, 
                                           self.world_seed + 2000) 
                                   for i in x] for j in z])
            
            erosion_octave = np.array([[simple_perlin_noise(i * frequency * 2.0, j * frequency * 2.0, 
                                              self.world_seed + 3000) 
                                      for i in x] for j in z])
            
            temperature += temp_octave * amplitude
            humidity += humid_octave * amplitude
            continentalness += cont_octave * amplitude
            erosion += erosion_octave * amplitude
        
        # Normalize to [-1, 1] range
        temperature = np.clip(temperature, -1, 1)
        humidity = np.clip(humidity, -1, 1)
        continentalness = np.clip(continentalness, -1, 1)
        erosion = np.clip(erosion, -1, 1)
        
        return X, Z, temperature, humidity, continentalness, erosion
    
    def classify_biomes(self, temperature, humidity, continentalness, erosion):
        """Classify biomes based on noise parameters."""
        biomes = np.zeros_like(temperature, dtype=int)
        biome_names = ['Ocean', 'Plains', 'Desert', 'Forest', 'Taiga', 'Mountains', 'Swamp']
        
        # Simplified biome classification logic
        ocean_mask = continentalness < -0.3
        desert_mask = (temperature > 0.5) & (humidity < -0.2) & ~ocean_mask
        mountain_mask = (erosion > 0.4) & ~ocean_mask
        taiga_mask = (temperature < -0.3) & ~ocean_mask & ~mountain_mask
        swamp_mask = (humidity > 0.5) & (temperature > -0.2) & ~ocean_mask & ~mountain_mask
        forest_mask = (temperature > -0.2) & (temperature < 0.5) & ~ocean_mask & ~desert_mask & ~mountain_mask & ~swamp_mask
        plains_mask = ~ocean_mask & ~desert_mask & ~mountain_mask & ~taiga_mask & ~swamp_mask & ~forest_mask
        
        biomes[ocean_mask] = 0      # Ocean
        biomes[plains_mask] = 1     # Plains
        biomes[desert_mask] = 2     # Desert
        biomes[forest_mask] = 3     # Forest
        biomes[taiga_mask] = 4      # Taiga
        biomes[mountain_mask] = 5   # Mountains
        biomes[swamp_mask] = 6      # Swamp
        
        return biomes, biome_names
    
    def generate_region_seed(self, region_x, region_z, salt):
        """Generate region seed using Minecraft's algorithm."""
        return (self.world_seed + 
                region_x * region_x * 4987142 + 
                region_x * 5947611 + 
                region_z * region_z * 4392871 + 
                region_z * 389711 + 
                salt) & 0xFFFFFFFF
    
    def generate_villages(self):
        """Generate village positions using Minecraft's grid-based algorithm."""
        print("Generating village positions...")
        villages = []
        
        # Calculate number of regions
        regions_per_axis = self.world_size // (self.village_spacing * self.chunk_size)
        
        for region_x in range(-regions_per_axis//2, regions_per_axis//2):
            for region_z in range(-regions_per_axis//2, regions_per_axis//2):
                # Generate region seed
                region_seed = self.generate_region_seed(region_x, region_z, self.village_salt)
                np.random.seed(region_seed & 0xFFFFFFFF)
                
                # Check if village spawns in this region
                if np.random.randint(0, self.village_spacing) < self.village_separation:
                    # Calculate village position within region
                    offset_x = np.random.randint(0, self.village_spacing - self.village_separation)
                    offset_z = np.random.randint(0, self.village_spacing - self.village_separation)
                    
                    chunk_x = region_x * self.village_spacing + offset_x
                    chunk_z = region_z * self.village_spacing + offset_z
                    
                    # Convert to block coordinates
                    block_x = chunk_x * self.chunk_size + np.random.randint(0, self.chunk_size)
                    block_z = chunk_z * self.chunk_size + np.random.randint(0, self.chunk_size)
                    
                    if abs(block_x) <= self.world_size//2 and abs(block_z) <= self.world_size//2:
                        villages.append((block_x, block_z))
        
        return np.array(villages)
    
    def generate_strongholds(self):
        """Generate stronghold positions using concentric ring algorithm."""
        print("Generating stronghold positions...")
        strongholds = []
        
        for ring in self.stronghold_rings:
            count = ring['count']
            min_radius = ring['min_radius']
            max_radius = ring['max_radius']
            color = ring['color']
            
            # Generate angles with slight randomization
            base_angle = np.random.uniform(0, 2*np.pi/count)
            angles = [base_angle + i * 2*np.pi/count + np.random.normal(0, np.pi/(count*4)) 
                     for i in range(count)]
            
            # Generate radii
            radii = [np.random.uniform(min_radius, max_radius) for _ in range(count)]
            
            # Convert to cartesian coordinates
            for angle, radius in zip(angles, radii):
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)
                strongholds.append((x, z, color))
        
        return strongholds
    
    def visualize_comprehensive_structure_analysis(self):
        """Create a comprehensive visualization of all Minecraft structures."""
        print("Creating comprehensive structure analysis...")
        
        # Generate all data
        X, Z, temperature, humidity, continentalness, erosion = self.generate_biome_noise_fields()
        biomes, biome_names = self.classify_biomes(temperature, humidity, continentalness, erosion)
        villages = self.generate_villages()
        strongholds = self.generate_strongholds()
        
        # Create figure with 6 subplots
        fig = plt.figure(figsize=(20, 16))
        fig.patch.set_facecolor(backgroundColor)
        
        # Define custom colormaps
        biome_colors = ['#1E3A8A', '#22C55E', '#EAB308', '#16A34A', '#0EA5E9', '#6B7280', '#84CC16']
        biome_cmap = LinearSegmentedColormap.from_list('biomes', biome_colors, N=len(biome_colors))
        
        # 1. Temperature field
        ax1 = plt.subplot(2, 3, 1)
        temp_plot = ax1.contourf(X, Z, temperature, levels=50, cmap='coolwarm', alpha=0.8)
        ax1.set_title('Temperature Field', color=textFontColor, fontsize=14)
        ax1.set_xlabel('X Coordinate', color=textFontColor)
        ax1.set_ylabel('Z Coordinate', color=textFontColor)
        plt.colorbar(temp_plot, ax=ax1, label='Temperature')
        ax1.set_facecolor(backgroundColor)
        
        # 2. Humidity field
        ax2 = plt.subplot(2, 3, 2)
        humid_plot = ax2.contourf(X, Z, humidity, levels=50, cmap='Blues', alpha=0.8)
        ax2.set_title('Humidity Field', color=textFontColor, fontsize=14)
        ax2.set_xlabel('X Coordinate', color=textFontColor)
        ax2.set_ylabel('Z Coordinate', color=textFontColor)
        plt.colorbar(humid_plot, ax=ax2, label='Humidity')
        ax2.set_facecolor(backgroundColor)
        
        # 3. Biome classification
        ax3 = plt.subplot(2, 3, 3)
        biome_plot = ax3.contourf(X, Z, biomes, levels=len(biome_names), cmap=biome_cmap, alpha=0.8)
        ax3.set_title('Biome Classification', color=textFontColor, fontsize=14)
        ax3.set_xlabel('X Coordinate', color=textFontColor)
        ax3.set_ylabel('Z Coordinate', color=textFontColor)
        cbar = plt.colorbar(biome_plot, ax=ax3, ticks=range(len(biome_names)))
        cbar.set_ticklabels(biome_names)
        ax3.set_facecolor(backgroundColor)
        
        # 4. Village distribution with grid
        ax4 = plt.subplot(2, 3, 4)
        ax4.set_facecolor(backgroundColor)
        
        # Draw grid lines for village regions
        grid_spacing = self.village_spacing * self.chunk_size
        grid_range = self.world_size // 2
        for x in range(-grid_range, grid_range + 1, grid_spacing):
            ax4.axvline(x, color=gridLineColor, alpha=0.3, linewidth=0.5)
        for z in range(-grid_range, grid_range + 1, grid_spacing):
            ax4.axhline(z, color=gridLineColor, alpha=0.3, linewidth=0.5)
        
        # Plot villages
        if len(villages) > 0:
            ax4.scatter(villages[:, 0], villages[:, 1], c='#FFD700', s=30, alpha=0.8, label=f'Villages ({len(villages)})')
        
        ax4.set_title('Village Distribution with Grid', color=textFontColor, fontsize=14)
        ax4.set_xlabel('X Coordinate', color=textFontColor)
        ax4.set_ylabel('Z Coordinate', color=textFontColor)
        ax4.set_xlim(-self.world_size//2, self.world_size//2)
        ax4.set_ylim(-self.world_size//2, self.world_size//2)
        ax4.legend()
        
        # 5. Stronghold rings
        ax5 = plt.subplot(2, 3, 5)
        ax5.set_facecolor(backgroundColor)
        ax5.set_aspect('equal')
        
        # Draw rings
        for ring in self.stronghold_rings:
            min_circle = Circle((0, 0), ring['min_radius'], fill=False, 
                              color=ring['color'], alpha=0.3, linewidth=2)
            max_circle = Circle((0, 0), ring['max_radius'], fill=False, 
                              color=ring['color'], alpha=0.5, linewidth=2)
            ax5.add_patch(min_circle)
            ax5.add_patch(max_circle)
        
        # Plot strongholds
        for x, z, color in strongholds:
            ax5.scatter(x, z, c=color, s=100, alpha=0.8, edgecolors='white', linewidth=1)
        
        ax5.scatter(0, 0, c='red', s=200, marker='*', label='World Spawn')
        ax5.set_title('Stronghold Ring Distribution', color=textFontColor, fontsize=14)
        ax5.set_xlabel('X Coordinate', color=textFontColor)
        ax5.set_ylabel('Z Coordinate', color=textFontColor)
        max_radius = self.stronghold_rings[-1]['max_radius'] + 1000
        ax5.set_xlim(-max_radius, max_radius)
        ax5.set_ylim(-max_radius, max_radius)
        ax5.legend()
        
        # 6. Combined structure map
        ax6 = plt.subplot(2, 3, 6)
        ax6.set_facecolor(backgroundColor)
        
        # Background biome map
        biome_bg = ax6.contourf(X, Z, biomes, levels=len(biome_names), cmap=biome_cmap, alpha=0.3)
        
        # Plot all structures
        if len(villages) > 0:
            ax6.scatter(villages[:, 0], villages[:, 1], c='#FFD700', s=20, alpha=0.8, 
                       label=f'Villages ({len(villages)})')
        
        for x, z, color in strongholds:
            ax6.scatter(x, z, c=color, s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
        
        ax6.scatter(0, 0, c='red', s=100, marker='*', label='World Spawn')
        ax6.set_title('Integrated Structure Map', color=textFontColor, fontsize=14)
        ax6.set_xlabel('X Coordinate', color=textFontColor)
        ax6.set_ylabel('Z Coordinate', color=textFontColor)
        ax6.set_xlim(-self.world_size//2, self.world_size//2)
        ax6.set_ylim(-self.world_size//2, self.world_size//2)
        ax6.legend()
        
        # Set colors for all axes
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.tick_params(colors=textFontColor)
            ax.grid(True, color=gridLineColor, alpha=0.3)
        
        plt.suptitle(f'Minecraft World Structure Analysis (Seed: {self.world_seed})', 
                    color=textFontColor, fontsize=18, y=0.95)
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(plotsPath, "minecraft_comprehensive_analysis.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor=backgroundColor)
        print(f"Saved comprehensive analysis to {save_path}")
        plt.show()

def main():
    """Main function to run all visualizations."""
    print("Starting Minecraft Structure Analysis...")
    
    # Initialize analyzer
    analyzer = MinecraftStructureAnalyzer(world_seed=42, world_size=20000)
    
    # Generate comprehensive analysis
    analyzer.visualize_comprehensive_structure_analysis()
    
    print("Minecraft structure analysis completed!")

if __name__ == "__main__":
    main()
