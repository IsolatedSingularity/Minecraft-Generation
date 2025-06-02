"""
Extended Minecraft Visualization Animations

This script creates dynamic animations that demonstrate the behavior of static visualizations:
1. Comprehensive Analysis Animation - Shows 6-panel visualization with:
   - Temperature and humidity noise field evolution
   - Dynamic biome classification
   - Village distribution with animated grid
   - Stronghold ring placement progression
   - Combined structure map development
   
2. Speedrunning Analysis Animation - Shows 4-panel speedrunning optimization with:
   - Stronghold triangulation strategy visualization
   - Route optimization pathfinding
   - Distance probability analysis evolution
   - Seed viability comparison progression
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Wedge
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import seaborn as sns
from scipy.spatial.distance import cdist
import os
from scipy.stats import norm

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

# Create plots directory if it doesn't exist
os.makedirs(plotsPath, exist_ok=True)

class MinecraftExtendedAnimator:
    def __init__(self, world_seed=42, world_size=20000):
        """Initialize the extended animator with comprehensive parameters."""
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
        
        # Biome color mapping
        self.biome_colors = {
            'Ocean': '#0066CC',
            'Plains': '#8DB360',
            'Desert': '#D2B48C',
            'Forest': '#228B22',
            'Taiga': '#2F4F2F',
            'Mountain': '#A0A0A0',
            'Swamp': '#5F8A5F',
            'Tundra': '#E0E0E0',
            'Savanna': '#BDB76B',
            'Jungle': '#228B22'
        }
        
        # Speedrunning parameters
        self.stronghold_target_count = 3  # First ring strongholds for speedrunning
        self.player_spawn = np.array([0, 0])
        self.end_portal_rooms = []
        self.triangulation_points = []
        
    def generate_biome_noise_fields(self, resolution=256, time_factor=0):
        """Generate dynamic temperature and humidity noise fields."""
        # Create coordinate grids
        x = np.linspace(-self.world_size//4, self.world_size//4, resolution)
        z = np.linspace(-self.world_size//4, self.world_size//4, resolution)
        X, Z = np.meshgrid(x, z)
        
        # Generate evolving noise fields
        temperature = np.zeros_like(X)
        humidity = np.zeros_like(X)
        
        # Multi-octave noise generation with time evolution
        for octave in range(6):
            frequency = 2 ** octave / 1000.0
            amplitude = 1.0 / (2 ** octave)
            
            # Add time-based evolution
            time_offset = time_factor * 0.1 * (octave + 1)
            
            temp_octave = np.array([[simple_perlin_noise(i * frequency + time_offset, 
                                                       j * frequency, 
                                                       self.world_seed) 
                                   for i in x] for j in z])
            
            humid_octave = np.array([[simple_perlin_noise(i * frequency, 
                                                        j * frequency + time_offset, 
                                                        self.world_seed + 1000)
                                    for i in x] for j in z])
            
            temperature += amplitude * temp_octave
            humidity += amplitude * humid_octave
        
        # Normalize fields
        temperature = (temperature + 1) / 2
        humidity = (humidity + 1) / 2
        
        return X, Z, temperature, humidity
    
    def classify_biomes(self, temperature, humidity):
        """Classify biomes based on temperature and humidity."""
        biomes = np.empty(temperature.shape, dtype=object)
        
        # Biome classification logic
        for i in range(temperature.shape[0]):
            for j in range(temperature.shape[1]):
                temp = temperature[i, j]
                humid = humidity[i, j]
                
                if temp < 0.2:
                    if humid < 0.3:
                        biomes[i, j] = 'Tundra'
                    else:
                        biomes[i, j] = 'Taiga'
                elif temp < 0.5:
                    if humid < 0.3:
                        biomes[i, j] = 'Plains'
                    elif humid < 0.7:
                        biomes[i, j] = 'Forest'
                    else:
                        biomes[i, j] = 'Swamp'
                elif temp < 0.8:
                    if humid < 0.2:
                        biomes[i, j] = 'Desert'
                    elif humid < 0.6:
                        biomes[i, j] = 'Savanna'
                    else:
                        biomes[i, j] = 'Jungle'
                else:
                    if humid < 0.4:
                        biomes[i, j] = 'Desert'
                    else:
                        biomes[i, j] = 'Mountain'
        
        return biomes
    
    def generate_structure_positions(self, structure_type, max_range=10000):
        """Generate positions for villages, fortresses, or strongholds."""
        positions = []
        
        if structure_type == 'village':
            spacing = self.village_spacing * self.chunk_size
            separation = self.village_separation * self.chunk_size
            salt = self.village_salt
        elif structure_type == 'fortress':
            spacing = self.fortress_spacing * self.chunk_size
            separation = self.fortress_separation * self.chunk_size
            salt = self.fortress_salt
        else:  # strongholds
            return self.generate_stronghold_positions()
        
        # Grid-based generation
        grid_size = spacing
        for grid_x in range(-max_range // grid_size, max_range // grid_size + 1):
            for grid_z in range(-max_range // grid_size, max_range // grid_size + 1):                # Hash-based position calculation
                hash_input = (grid_x * 341873128712 + grid_z * 132897987541 + salt) % 2**32
                np.random.seed(hash_input % 2**31)
                
                if np.random.random() < 0.3:  # 30% chance
                    offset_x = np.random.randint(-separation, separation)
                    offset_z = np.random.randint(-separation, separation)
                    
                    pos_x = grid_x * grid_size + offset_x
                    pos_z = grid_z * grid_size + offset_z
                    
                    if abs(pos_x) <= max_range and abs(pos_z) <= max_range:
                        positions.append([pos_x, pos_z])
        
        return np.array(positions) if positions else np.empty((0, 2))
    
    def generate_stronghold_positions(self):
        """Generate stronghold positions in rings."""
        positions = []
        
        for ring in self.stronghold_rings:
            count = ring['count']
            min_radius = ring['min_radius']
            max_radius = ring['max_radius']
            
            for i in range(count):
                # Deterministic angle distribution
                angle = 2 * np.pi * i / count + np.random.normal(0, 0.1)
                
                # Random radius within ring
                radius = np.random.uniform(min_radius, max_radius)
                
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)
                
                # Store as [x, z] - numeric only to avoid dtype issues
                positions.append([x, z])
        
        return np.array(positions, dtype=float) if positions else np.empty((0, 2))
    
    def calculate_stronghold_triangulation(self, eye_throws):
        """Calculate stronghold position using triangulation from eye of ender throws."""
        if len(eye_throws) < 2:
            return None
        
        # Simple triangulation using first two throws
        throw1, throw2 = eye_throws[:2]
        
        # Calculate intersection point
        x1, z1, angle1 = throw1
        x2, z2, angle2 = throw2
        
        # Convert angles to direction vectors
        dx1, dz1 = np.cos(angle1), np.sin(angle1)
        dx2, dz2 = np.cos(angle2), np.sin(angle2)
        
        # Find intersection
        denom = dx1 * dz2 - dx2 * dz1
        if abs(denom) < 1e-10:
            return None
        
        t = ((x2 - x1) * dz2 - (z2 - z1) * dx2) / denom
        
        stronghold_x = x1 + t * dx1
        stronghold_z = z1 + t * dz1
        
        return np.array([stronghold_x, stronghold_z])
    
    def animate_comprehensive_analysis(self, frames=200, interval=100):
        """Create comprehensive analysis animation showing all 6 panels."""
        print("Creating comprehensive analysis animation...")
        
        fig = plt.figure(figsize=(20, 12))
        fig.patch.set_facecolor(backgroundColor)
        
        # Create 6 subplots in 2x3 grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])  # Temperature field
        ax2 = fig.add_subplot(gs[0, 1])  # Humidity field
        ax3 = fig.add_subplot(gs[0, 2])  # Biome classification
        ax4 = fig.add_subplot(gs[1, 0])  # Village distribution
        ax5 = fig.add_subplot(gs[1, 1])  # Stronghold rings
        ax6 = fig.add_subplot(gs[1, 2])  # Combined structures
        
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        titles = ['Temperature Field', 'Humidity Field', 'Biome Classification',
                 'Village Distribution', 'Stronghold Rings', 'Combined Structure Map']
        
        # Style all axes
        for ax, title in zip(axes, titles):
            ax.set_facecolor(backgroundColor)
            ax.tick_params(colors=textFontColor)
            ax.set_title(title, color=textFontColor, fontsize=12, fontweight='bold')
            ax.grid(True, color=gridLineColor, alpha=0.3)
        
        # Generate structure positions once
        villages = self.generate_structure_positions('village', 8000)
        fortresses = self.generate_structure_positions('fortress', 8000)
        strongholds = self.generate_stronghold_positions()
        
        def animate(frame):
            # Clear all axes
            for ax in axes:
                ax.clear()
                ax.set_facecolor(backgroundColor)
                ax.tick_params(colors=textFontColor)
                ax.grid(True, color=gridLineColor, alpha=0.3)
            
            # Time factor for evolution
            time_factor = frame / 20.0
            
            # Generate evolving noise fields
            X, Z, temperature, humidity = self.generate_biome_noise_fields(
                resolution=128, time_factor=time_factor)
            biomes = self.classify_biomes(temperature, humidity)
            
            # 1. Temperature field
            im1 = ax1.contourf(X, Z, temperature, levels=20, cmap='coolwarm', alpha=0.8)
            ax1.set_title('Temperature Field Evolution', color=textFontColor, fontweight='bold')
            ax1.set_xlabel('X (blocks)', color=textFontColor)
            ax1.set_ylabel('Z (blocks)', color=textFontColor)
            
            # 2. Humidity field
            im2 = ax2.contourf(X, Z, humidity, levels=20, cmap='Blues', alpha=0.8)
            ax2.set_title('Humidity Field Evolution', color=textFontColor, fontweight='bold')
            ax2.set_xlabel('X (blocks)', color=textFontColor)
            ax2.set_ylabel('Z (blocks)', color=textFontColor)
            
            # 3. Biome classification
            biome_numeric = np.zeros_like(temperature)
            biome_names = list(self.biome_colors.keys())
            for i, biome in enumerate(biome_names):
                mask = biomes == biome
                biome_numeric[mask] = i
            
            im3 = ax3.contourf(X, Z, biome_numeric, levels=len(biome_names), 
                              cmap='terrain', alpha=0.8)
            ax3.set_title('Dynamic Biome Classification', color=textFontColor, fontweight='bold')
            ax3.set_xlabel('X (blocks)', color=textFontColor)
            ax3.set_ylabel('Z (blocks)', color=textFontColor)
            
            # 4. Village distribution with animated grid
            if len(villages) > 0:
                # Show only villages that have "spawned" by this frame
                visible_count = min(len(villages), frame * 2)
                visible_villages = villages[:visible_count]
                
                if len(visible_villages) > 0:
                    ax4.scatter(visible_villages[:, 0], visible_villages[:, 1], 
                              c='#8B4513', s=30, marker='s', alpha=0.8, 
                              edgecolors='white', linewidth=0.5)
                
                # Animated grid lines
                grid_spacing = self.village_spacing * self.chunk_size
                max_range = 8000
                alpha = 0.3 * (1 + 0.3 * np.sin(frame * 0.2))
                
                for x in range(-max_range, max_range + 1, grid_spacing):
                    ax4.axvline(x, color=gridLineColor, alpha=alpha, linewidth=0.5)
                for z in range(-max_range, max_range + 1, grid_spacing):
                    ax4.axhline(z, color=gridLineColor, alpha=alpha, linewidth=0.5)
            
            ax4.set_xlim(-8000, 8000)
            ax4.set_ylim(-8000, 8000)
            ax4.set_title(f'Village Distribution (Count: {min(len(villages), frame * 2)})', 
                         color=textFontColor, fontweight='bold')
            ax4.set_xlabel('X (blocks)', color=textFontColor)
            ax4.set_ylabel('Z (blocks)', color=textFontColor)
            
            # 5. Stronghold rings with progressive reveal
            if len(strongholds) > 0:
                # Reveal rings progressively
                rings_to_show = min(len(self.stronghold_rings), 1 + frame // 30)
                
                for ring_idx in range(rings_to_show):
                    ring = self.stronghold_rings[ring_idx]
                    
                    # Draw ring boundaries
                    ring_alpha = 0.3 + 0.2 * np.sin(frame * 0.1 + ring_idx)
                    circle_inner = Circle((0, 0), ring['min_radius'], 
                                        fill=False, color=ring['color'], 
                                        alpha=ring_alpha, linewidth=2)
                    circle_outer = Circle((0, 0), ring['max_radius'], 
                                        fill=False, color=ring['color'], 
                                        alpha=ring_alpha, linewidth=2)
                    ax5.add_patch(circle_inner)
                    ax5.add_patch(circle_outer)
                    
                    # Show strongholds in this ring
                    ring_strongholds = strongholds[
                        sum(self.stronghold_rings[i]['count'] for i in range(ring_idx)):
                        sum(self.stronghold_rings[i]['count'] for i in range(ring_idx + 1))
                    ]
                    
                    if len(ring_strongholds) > 0:
                        ax5.scatter(ring_strongholds[:, 0], ring_strongholds[:, 1], 
                                  c=ring['color'], s=50, marker='*', alpha=0.9, 
                                  edgecolors='white', linewidth=1)
            
            ax5.set_xlim(-20000, 20000)
            ax5.set_ylim(-20000, 20000)
            ax5.set_title('Stronghold Ring Distribution', color=textFontColor, fontweight='bold')
            ax5.set_xlabel('X (blocks)', color=textFontColor)
            ax5.set_ylabel('Z (blocks)', color=textFontColor)
            ax5.set_aspect('equal')
            
            # 6. Combined structure map
            structure_range = 12000
            
            # Villages
            if len(villages) > 0:
                village_mask = (np.abs(villages[:, 0]) <= structure_range) & \
                              (np.abs(villages[:, 1]) <= structure_range)
                visible_villages = villages[village_mask]
                visible_count = min(len(visible_villages), frame * 2)
                
                if visible_count > 0:
                    ax6.scatter(visible_villages[:visible_count, 0], 
                              visible_villages[:visible_count, 1], 
                              c='#8B4513', s=20, marker='s', alpha=0.7, label='Villages')
            
            # Fortresses  
            if len(fortresses) > 0:
                fortress_mask = (np.abs(fortresses[:, 0]) <= structure_range) & \
                               (np.abs(fortresses[:, 1]) <= structure_range)
                visible_fortresses = fortresses[fortress_mask]
                
                if len(visible_fortresses) > 0:
                    ax6.scatter(visible_fortresses[:, 0], visible_fortresses[:, 1], 
                              c='#FF4500', s=25, marker='^', alpha=0.8, label='Fortresses')
            
            # Strongholds (first 3 rings only for visibility)
            if len(strongholds) > 0:
                first_rings_count = sum(ring['count'] for ring in self.stronghold_rings[:3])
                first_ring_strongholds = strongholds[:first_rings_count]
                
                if len(first_ring_strongholds) > 0:
                    ax6.scatter(first_ring_strongholds[:, 0], first_ring_strongholds[:, 1], 
                              c='#9400D3', s=40, marker='*', alpha=0.9, label='Strongholds')
            
            ax6.set_xlim(-structure_range, structure_range)
            ax6.set_ylim(-structure_range, structure_range)
            ax6.set_title('Combined Structure Distribution', color=textFontColor, fontweight='bold')
            ax6.set_xlabel('X (blocks)', color=textFontColor)
            ax6.set_ylabel('Z (blocks)', color=textFontColor)
            ax6.legend(loc='upper right', facecolor=backgroundColor, edgecolor=textFontColor)
              # Add frame counter
            fig.suptitle(f'Minecraft Comprehensive Analysis - Frame {frame+1}/{frames}', 
                        color=textFontColor, fontsize=16, fontweight='bold', y=0.95)
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
        
        # Save animation with fallback format support
        try:
            # Try MP4 first if ffmpeg is available
            output_path = os.path.join(plotsPath, "minecraft_comprehensive_analysis_animated.mp4")
            print(f"Saving comprehensive analysis animation to: {output_path}")
            anim.save(output_path, writer='ffmpeg', fps=10, bitrate=1800)
        except Exception as e:
            print(f"MP4 save failed ({e}), falling back to GIF format...")
            # Fallback to GIF format
            output_path = os.path.join(plotsPath, "minecraft_comprehensive_analysis_animated.gif")
            print(f"Saving comprehensive analysis animation to: {output_path}")
            anim.save(output_path, writer='pillow', fps=5)
        
        plt.close(fig)
        return anim
    
    def animate_speedrunning_analysis(self, frames=150, interval=120):
        """Create speedrunning analysis animation showing 4-panel optimization."""
        print("Creating speedrunning analysis animation...")
        
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor(backgroundColor)
        
        # Create 4 subplots in 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])  # Stronghold triangulation
        ax2 = fig.add_subplot(gs[0, 1])  # Route optimization
        ax3 = fig.add_subplot(gs[1, 0])  # Distance probability
        ax4 = fig.add_subplot(gs[1, 1])  # Seed viability
        
        axes = [ax1, ax2, ax3, ax4]
        titles = ['Stronghold Triangulation Strategy', 'Route Optimization',
                 'Distance Probability Analysis', 'Seed Viability Comparison']
        
        # Style all axes
        for ax, title in zip(axes, titles):
            ax.set_facecolor(backgroundColor)
            ax.tick_params(colors=textFontColor)
            ax.set_title(title, color=textFontColor, fontsize=12, fontweight='bold')
            ax.grid(True, color=gridLineColor, alpha=0.3)
        
        # Generate speedrunning data
        strongholds = self.generate_stronghold_positions()
        first_ring_strongholds = strongholds[:3]  # First 3 strongholds
        
        # Simulate eye of ender throws for triangulation
        throw_positions = []
        throw_angles = []
        
        # Generate multiple throw scenarios
        for i in range(10):
            x = np.random.uniform(-1000, 1000)
            z = np.random.uniform(-1000, 1000)
            # Calculate angle to nearest stronghold
            if len(first_ring_strongholds) > 0:
                distances = np.sqrt((first_ring_strongholds[:, 0] - x)**2 + 
                                  (first_ring_strongholds[:, 1] - z)**2)
                nearest_idx = np.argmin(distances)
                target = first_ring_strongholds[nearest_idx]
                angle = np.arctan2(target[1] - z, target[0] - x)
                angle += np.random.normal(0, 0.1)  # Add some error
                
                throw_positions.append([x, z])
                throw_angles.append(angle)
        
        throw_positions = np.array(throw_positions)
        throw_angles = np.array(throw_angles)
        
        def animate(frame):
            # Clear all axes
            for ax in axes:
                ax.clear()
                ax.set_facecolor(backgroundColor)
                ax.tick_params(colors=textFontColor)
                ax.grid(True, color=gridLineColor, alpha=0.3)
            
            # 1. Stronghold Triangulation Strategy
            if len(first_ring_strongholds) > 0:
                # Show strongholds
                ax1.scatter(first_ring_strongholds[:, 0], first_ring_strongholds[:, 1], 
                          c='#9400D3', s=100, marker='*', alpha=0.9, 
                          edgecolors='white', linewidth=2, label='Strongholds')
                
                # Show progressive eye throws
                throws_to_show = min(len(throw_positions), 1 + frame // 15)
                
                for i in range(throws_to_show):
                    pos = throw_positions[i]
                    angle = throw_angles[i]
                    
                    # Draw throw position
                    ax1.scatter(pos[0], pos[1], c='#FFD700', s=50, marker='o', 
                              alpha=0.8, edgecolors='black', linewidth=1)
                    
                    # Draw direction line
                    line_length = 2000
                    end_x = pos[0] + line_length * np.cos(angle)
                    end_z = pos[1] + line_length * np.sin(angle)
                    
                    alpha = 0.7 * (1 - 0.3 * np.sin(frame * 0.1 + i))
                    ax1.plot([pos[0], end_x], [pos[1], end_z], 
                           color='#FFD700', alpha=alpha, linewidth=2)
                    
                    # Show throw number
                    ax1.text(pos[0], pos[1] + 100, f'#{i+1}', 
                           color=textFontColor, fontsize=8, ha='center')
                
                # Triangulation calculation and display
                if throws_to_show >= 2:
                    eye_throws = [(throw_positions[i][0], throw_positions[i][1], throw_angles[i]) 
                                 for i in range(throws_to_show)]
                    triangulated_pos = self.calculate_stronghold_triangulation(eye_throws)
                    
                    if triangulated_pos is not None:
                        ax1.scatter(triangulated_pos[0], triangulated_pos[1], 
                                  c='#FF6B6B', s=150, marker='X', alpha=0.9,
                                  edgecolors='white', linewidth=2, label='Triangulated Position')
            
            ax1.set_xlim(-3000, 3000)
            ax1.set_ylim(-3000, 3000)
            ax1.set_title('Stronghold Triangulation Strategy', color=textFontColor, fontweight='bold')
            ax1.set_xlabel('X (blocks)', color=textFontColor)
            ax1.set_ylabel('Z (blocks)', color=textFontColor)
            ax1.legend(loc='upper right', facecolor=backgroundColor, edgecolor=textFontColor)
            
            # 2. Route Optimization
            if len(first_ring_strongholds) > 0:
                # Player spawn
                ax2.scatter(0, 0, c='#00FF00', s=100, marker='o', 
                          alpha=0.9, edgecolors='white', linewidth=2, label='Spawn')
                
                # Show optimal route development
                route_progress = min(1.0, frame / 100.0)
                
                # Calculate route to nearest stronghold
                distances = np.sqrt(first_ring_strongholds[:, 0]**2 + first_ring_strongholds[:, 1]**2)
                nearest_idx = np.argmin(distances)
                target_stronghold = first_ring_strongholds[nearest_idx]
                
                # Show all strongholds
                ax2.scatter(first_ring_strongholds[:, 0], first_ring_strongholds[:, 1], 
                          c='#9400D3', s=80, marker='*', alpha=0.7, 
                          edgecolors='white', linewidth=1)
                
                # Highlight target
                ax2.scatter(target_stronghold[0], target_stronghold[1], 
                          c='#FF6B6B', s=120, marker='*', alpha=0.9, 
                          edgecolors='white', linewidth=2, label='Target')
                
                # Animated route
                route_x = np.linspace(0, target_stronghold[0], 100)
                route_z = np.linspace(0, target_stronghold[1], 100)
                
                # Add some curvature for realism
                mid_point = len(route_x) // 2
                route_x[mid_point] += 200 * np.sin(frame * 0.1)
                route_z[mid_point] += 200 * np.cos(frame * 0.1)
                
                points_to_show = int(len(route_x) * route_progress)
                if points_to_show > 1:
                    ax2.plot(route_x[:points_to_show], route_z[:points_to_show], 
                           color='#FFD700', linewidth=3, alpha=0.8, label='Optimal Route')
                
                # Distance and time info
                distance = np.sqrt(target_stronghold[0]**2 + target_stronghold[1]**2)
                ax2.text(0.02, 0.98, f'Distance: {distance:.0f} blocks', 
                       transform=ax2.transAxes, color=textFontColor, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor=backgroundColor, alpha=0.8))
                ax2.text(0.02, 0.91, f'Est. Time: {distance/8.0:.1f}s (Nether)', 
                       transform=ax2.transAxes, color=textFontColor, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor=backgroundColor, alpha=0.8))
            
            ax2.set_xlim(-3500, 3500)
            ax2.set_ylim(-3500, 3500)
            ax2.set_title('Route Optimization Pathfinding', color=textFontColor, fontweight='bold')
            ax2.set_xlabel('X (blocks)', color=textFontColor)
            ax2.set_ylabel('Z (blocks)', color=textFontColor)
            ax2.legend(loc='upper left', facecolor=backgroundColor, edgecolor=textFontColor)
            
            # 3. Distance Probability Analysis
            if len(first_ring_strongholds) > 0:
                distances = np.sqrt(first_ring_strongholds[:, 0]**2 + first_ring_strongholds[:, 1]**2)
                
                # Create evolving histogram
                bins = np.linspace(1000, 3000, 20)
                hist_data = []
                
                # Simulate multiple seeds
                analysis_progress = frame / frames
                seeds_to_analyze = int(50 * analysis_progress) + 1
                
                for seed in range(seeds_to_analyze):
                    # Generate stronghold distances for this seed
                    np.random.seed(seed + 100)
                    ring = self.stronghold_rings[0]
                    sim_distances = []
                    
                    for i in range(ring['count']):
                        angle = 2 * np.pi * i / ring['count'] + np.random.normal(0, 0.1)
                        radius = np.random.uniform(ring['min_radius'], ring['max_radius'])
                        sim_distances.append(radius)
                    
                    hist_data.extend(sim_distances)
                
                if hist_data:
                    ax3.hist(hist_data, bins=bins, alpha=0.7, color='#4ECDC4', 
                           edgecolor='white', linewidth=0.5)
                    
                    # Add statistics
                    mean_dist = np.mean(hist_data)
                    std_dist = np.std(hist_data)
                    
                    ax3.axvline(mean_dist, color='#FF6B6B', linewidth=2, 
                              linestyle='--', label=f'Mean: {mean_dist:.0f}')
                    ax3.axvline(mean_dist - std_dist, color='#FFD700', linewidth=1, 
                              linestyle=':', alpha=0.7)
                    ax3.axvline(mean_dist + std_dist, color='#FFD700', linewidth=1, 
                              linestyle=':', alpha=0.7, label=f'±1σ: {std_dist:.0f}')
                
                ax3.text(0.02, 0.98, f'Seeds Analyzed: {seeds_to_analyze}', 
                       transform=ax3.transAxes, color=textFontColor, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor=backgroundColor, alpha=0.8))
            
            ax3.set_xlabel('Distance to Nearest Stronghold (blocks)', color=textFontColor)
            ax3.set_ylabel('Frequency', color=textFontColor)
            ax3.set_title('Stronghold Distance Probability Analysis', color=textFontColor, fontweight='bold')
            ax3.legend(loc='upper right', facecolor=backgroundColor, edgecolor=textFontColor)
            
            # 4. Seed Viability Comparison
            # Generate seed comparison data
            seed_scores = []
            seed_numbers = []
            evaluation_progress = frame / frames
            seeds_to_evaluate = int(20 * evaluation_progress) + 1
            
            for seed in range(seeds_to_evaluate):
                np.random.seed(seed + 200)
                
                # Calculate viability score based on:
                # - Distance to nearest stronghold
                # - Biome at spawn
                # - Village proximity
                # - Fortress accessibility
                
                # Stronghold distance score (closer is better)
                ring = self.stronghold_rings[0]
                min_stronghold_dist = float('inf')
                for i in range(ring['count']):
                    angle = 2 * np.pi * i / ring['count'] + np.random.normal(0, 0.1)
                    radius = np.random.uniform(ring['min_radius'], ring['max_radius'])
                    dist = radius
                    min_stronghold_dist = min(min_stronghold_dist, dist)
                
                distance_score = max(0, 100 - (min_stronghold_dist - 1200) / 20)
                
                # Biome score (some biomes are better for speedrunning)
                biome_score = np.random.uniform(20, 80)
                
                # Village proximity score
                village_score = np.random.uniform(10, 90)
                
                # Fortress score
                fortress_score = np.random.uniform(30, 70)
                
                total_score = (distance_score * 0.4 + biome_score * 0.2 + 
                             village_score * 0.2 + fortress_score * 0.2)
                
                seed_scores.append(total_score)
                seed_numbers.append(seed)
            
            if seed_scores:
                # Color code by viability
                colors = ['#FF6B6B' if score > 70 else '#FFD700' if score > 50 else '#4ECDC4' 
                         for score in seed_scores]
                
                bars = ax4.bar(range(len(seed_scores)), seed_scores, color=colors, alpha=0.8, 
                              edgecolor='white', linewidth=0.5)
                
                # Highlight best seeds
                best_indices = np.argsort(seed_scores)[-3:]
                for idx in best_indices:
                    bars[idx].set_edgecolor('#FFFFFF')
                    bars[idx].set_linewidth(2)
                
                # Add threshold lines
                ax4.axhline(70, color='#FF6B6B', linewidth=2, linestyle='--', 
                          alpha=0.7, label='Excellent (>70)')
                ax4.axhline(50, color='#FFD700', linewidth=2, linestyle='--', 
                          alpha=0.7, label='Good (>50)')                # Show best seed info
                if seed_scores:
                    best_score = max(seed_scores)
                    best_seed = seed_numbers[seed_scores.index(best_score)]
                    ax4.text(0.02, 0.98, f'Best Seed: {best_seed}', 
                           transform=ax4.transAxes, color=textFontColor, 
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor=backgroundColor, alpha=0.8))
                    ax4.text(0.02, 0.91, f'Score: {best_score:.1f}', 
                           transform=ax4.transAxes, color=textFontColor, 
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor=backgroundColor, alpha=0.8))
            
            ax4.set_xlabel('Seed Number', color=textFontColor)
            ax4.set_ylabel('Viability Score', color=textFontColor)
            ax4.set_title('Seed Viability Comparison for Speedrunning', color=textFontColor, fontweight='bold')
            ax4.set_ylim(0, 100)
            ax4.legend(loc='upper right', facecolor=backgroundColor, edgecolor=textFontColor)
              # Add frame counter
            fig.suptitle(f'Minecraft Speedrunning Analysis - Frame {frame+1}/{frames}', 
                        color=textFontColor, fontsize=16, fontweight='bold', y=0.95)
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=False)
        
        # Save animation with fallback format support
        try:
            # Try MP4 first if ffmpeg is available
            output_path = os.path.join(plotsPath, "minecraft_speedrunning_analysis_animated.mp4")
            print(f"Saving speedrunning analysis animation to: {output_path}")
            anim.save(output_path, writer='ffmpeg', fps=8, bitrate=1800)
        except Exception as e:
            print(f"MP4 save failed ({e}), falling back to GIF format...")
            # Fallback to GIF format
            output_path = os.path.join(plotsPath, "minecraft_speedrunning_analysis_animated.gif")
            print(f"Saving speedrunning analysis animation to: {output_path}")
            anim.save(output_path, writer='pillow', fps=4)        
        plt.close(fig)
        return anim

def main():
    """Main execution function for creating extended animations."""
    print("=== Minecraft Extended Animations ===")
    print("Creating dynamic visualizations for comprehensive and speedrunning analysis...")
    
    # Initialize the extended animator
    animator = MinecraftExtendedAnimator(world_seed=42, world_size=20000)
    
    try:
        # Create comprehensive analysis animation
        print("\n1. Creating Comprehensive Analysis Animation...")
        comprehensive_anim = animator.animate_comprehensive_analysis(frames=200, interval=100)
        print("✓ Comprehensive analysis animation completed!")
          # Create speedrunning analysis animation  
        print("\n2. Creating Speedrunning Analysis Animation...")
        speedrun_anim = animator.animate_speedrunning_analysis(frames=150, interval=120)
        print("✓ Speedrunning analysis animation completed!")
        
        print("\n=== Animation Creation Complete ===")
        print(f"Animations saved to: {plotsPath}")
        print("Files created:")
        print("- minecraft_comprehensive_analysis_animated.gif (or .mp4 if ffmpeg available)")
        print("- minecraft_speedrunning_analysis_animated.gif (or .mp4 if ffmpeg available)")
        
    except Exception as e:
        print(f"Error creating animations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
