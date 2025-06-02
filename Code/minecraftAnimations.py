"""
Advanced Minecraft Structure Animations

This script creates dynamic animations showing:
1. World generation progression through noise layer accumulation
2. Structure placement algorithm visualization
3. Dragon pathfinding with real-time AI state changes
4. Seed evolution and pattern emergence
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import networkx as nx
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

class MinecraftAnimator:
    def __init__(self, world_seed=42):
        """Initialize the Minecraft animator."""
        self.world_seed = world_seed
        self.chunk_size = 16
        np.random.seed(world_seed)
          # Dragon AI states - Updated to include all behavior types with single words
        self.dragon_states = ['HOLDING', 'STRAFING', 'APPROACH', 'LANDING', 'PERCHING', 'TAKEOFF', 'CHARGING']
        self.dragon_nodes = self.generate_dragon_nodes()
        
    def generate_dragon_nodes(self):
        """Generate Ender Dragon pathfinding nodes."""
        nodes = {}
        
        # Central fountain
        nodes['fountain'] = (0, 0)
        
        # End pillars (obsidian towers)
        pillar_radius = 76
        pillar_angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
        for i, angle in enumerate(pillar_angles):
            x = pillar_radius * np.cos(angle)
            z = pillar_radius * np.sin(angle)
            nodes[f'pillar_{i}'] = (x, z)
        
        # Flight path nodes
        outer_radius = 100
        inner_radius = 60
        center_radius = 30
        
        # Outer ring
        outer_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        for i, angle in enumerate(outer_angles):
            x = outer_radius * np.cos(angle)
            z = outer_radius * np.sin(angle)
            nodes[f'outer_{i}'] = (x, z)
        
        # Inner ring
        inner_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for i, angle in enumerate(inner_angles):
            x = inner_radius * np.cos(angle)
            z = inner_radius * np.sin(angle)
            nodes[f'inner_{i}'] = (x, z)
        
        # Center ring
        center_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
        for i, angle in enumerate(center_angles):
            x = center_radius * np.cos(angle)
            z = center_radius * np.sin(angle)
            nodes[f'center_{i}'] = (x, z)
        
        return nodes
  
    
    def animate_structure_placement(self, save_path):
        """Animate the structure placement algorithm step by step."""
        print("Creating structure placement animation...")
        
        fig, ax = plt.subplots(figsize=(12, 12))
        fig.patch.set_facecolor(backgroundColor)
        ax.set_facecolor(backgroundColor)
        
        # Parameters
        world_size = 8000
        village_spacing = 32 * 16  # 32 chunks * 16 blocks
        regions_per_axis = world_size // village_spacing
        
        # Set up plot
        ax.set_xlim(-world_size//2, world_size//2)
        ax.set_ylim(-world_size//2, world_size//2)
        ax.set_aspect('equal')
        ax.set_title('Structure Placement Algorithm', color=textFontColor, fontsize=16)
        ax.set_xlabel('X Coordinate', color=textFontColor)
        ax.set_ylabel('Z Coordinate', color=textFontColor)
        ax.tick_params(colors=textFontColor)
        ax.grid(True, color=gridLineColor, alpha=0.3)
        
        # Draw initial grid
        for x in range(-world_size//2, world_size//2 + 1, village_spacing):
            ax.axvline(x, color=gridLineColor, alpha=0.5, linewidth=0.5)
        for z in range(-world_size//2, world_size//2 + 1, village_spacing):
            ax.axhline(z, color=gridLineColor, alpha=0.5, linewidth=0.5)
        
        # Generate all regions
        regions = []
        for region_x in range(-regions_per_axis//2, regions_per_axis//2):
            for region_z in range(-regions_per_axis//2, regions_per_axis//2):
                center_x = region_x * village_spacing
                center_z = region_z * village_spacing
                regions.append((region_x, region_z, center_x, center_z))
        
        # Animation elements
        current_region_highlight = Rectangle((0, 0), village_spacing, village_spacing,
                                           fill=False, edgecolor=accentColor, linewidth=3, alpha=0)
        ax.add_patch(current_region_highlight)
        
        villages_plotted = []
        
        # Text displays
        info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, 
                           color=textFontColor, fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor=backgroundColor, alpha=0.9))
        
        algorithm_text = ax.text(0.02, 0.02, 
                               "Algorithm Steps:\\n"
                               "1. Divide world into regions\\n"
                               "2. Generate region seed\\n"
                               "3. Check spawn probability\\n"
                               "4. Place structure if conditions met",
                               transform=ax.transAxes, color=textFontColor, fontsize=10,
                               bbox=dict(boxstyle='round', facecolor=backgroundColor, alpha=0.9))
        
        def animate_placement(frame):
            if frame >= len(regions):
                return []
            
            region_x, region_z, center_x, center_z = regions[frame]
            
            # Update region highlight
            current_region_highlight.set_xy((center_x - village_spacing//2, 
                                           center_z - village_spacing//2))
            current_region_highlight.set_alpha(0.8)
            
            # Generate region seed (simplified)
            region_seed = (self.world_seed + 
                          region_x * region_x * 4987142 + 
                          region_x * 5947611 + 
                          region_z * region_z * 4392871 + 
                          region_z * 389711 + 
                          10387312) & 0xFFFFFFFF
            
            np.random.seed(region_seed)
            
            # Check if structure spawns (40% probability)
            spawns = np.random.random() < 0.4
            
            if spawns:
                # Place village with some randomness
                offset_x = np.random.uniform(-village_spacing//4, village_spacing//4)
                offset_z = np.random.uniform(-village_spacing//4, village_spacing//4)
                
                village_x = center_x + offset_x
                village_z = center_z + offset_z
                
                # Add village marker
                village_marker = ax.scatter(village_x, village_z, c='#FFD700', s=80, 
                                          alpha=0.9, edgecolors='white', linewidth=1,
                                          marker='s', zorder=5)
                villages_plotted.append(village_marker)
            
            # Update info text
            info_text.set_text(f'Region: ({region_x}, {region_z})\\n'
                              f'Seed: {region_seed & 0xFFFF}...\\n'
                              f'Probability: 40%\\n'
                              f'Result: {"SPAWN" if spawns else "NO SPAWN"}\\n'                              f'Progress: {frame+1}/{len(regions)}')
            
            return [current_region_highlight, info_text]
        
        # Create animation - Fixed to show all regions instead of stopping at 100
        anim = animation.FuncAnimation(fig, animate_placement, frames=len(regions),
                                     interval=50, blit=False, repeat=False)
        
        # Save animation
        try:
            print(f"Saving structure placement animation to {save_path}")
            anim.save(save_path, writer='pillow', fps=10, dpi=150)
            print("Structure placement animation saved successfully")
        except Exception as e:
            print(f"Error saving animation: {e}")
            # Save static image instead - Fixed to use appropriate frame
            animate_placement(min(len(regions) - 1, len(regions) // 2))
            plt.savefig(save_path.replace('.gif', '.png'), dpi=150, facecolor=backgroundColor)
        
        plt.close(fig)
    
    def animate_dragon_pathfinding(self, save_path):
        """Animate the Ender Dragon's AI pathfinding and state transitions."""
        print("Creating dragon pathfinding animation...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.patch.set_facecolor(backgroundColor)
        
        # Set up dragon arena plot
        ax1.set_facecolor(backgroundColor)
        ax1.set_xlim(-120, 120)
        ax1.set_ylim(-120, 120)
        ax1.set_aspect('equal')
        ax1.set_title('Ender Dragon Pathfinding', color=textFontColor, fontsize=14)
        ax1.set_xlabel('X Coordinate', color=textFontColor)
        ax1.set_ylabel('Z Coordinate', color=textFontColor)
        ax1.tick_params(colors=textFontColor)
        ax1.grid(True, color=gridLineColor, alpha=0.3)
        
        # Set up state diagram plot
        ax2.set_facecolor(backgroundColor)
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.set_title('AI State Machine', color=textFontColor, fontsize=14)
        ax2.axis('off')
        
        # Draw End environment
        # Central fountain
        fountain = Circle((0, 0), 8, color='#808080', alpha=0.7)
        ax1.add_patch(fountain)
        
        # End pillars
        pillar_radius = 76
        pillar_angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
        pillars = []
        for angle in pillar_angles:
            x = pillar_radius * np.cos(angle)
            z = pillar_radius * np.sin(angle)
            pillar = Circle((x, z), 6, color='#4B0082', alpha=0.7)
            ax1.add_patch(pillar)
            pillars.append((x, z))
        
        # Create navigation graph
        G = nx.Graph()
        node_positions = {}
        
        # Add nodes
        for name, pos in self.dragon_nodes.items():
            G.add_node(name)
            node_positions[name] = pos
        
        # Add edges (simplified connectivity)
        # Connect outer ring
        outer_nodes = [f'outer_{i}' for i in range(12)]
        for i in range(12):
            G.add_edge(outer_nodes[i], outer_nodes[(i+1)%12])
        
        # Connect inner ring
        inner_nodes = [f'inner_{i}' for i in range(8)]
        for i in range(8):
            G.add_edge(inner_nodes[i], inner_nodes[(i+1)%8])
        
        # Connect center ring
        center_nodes = [f'center_{i}' for i in range(4)]
        for i in range(4):
            G.add_edge(center_nodes[i], center_nodes[(i+1)%4])
            G.add_edge('fountain', center_nodes[i])
        
        # Draw graph
        nx.draw_networkx_edges(G, node_positions, edge_color='#E0E0E0', width=1, ax=ax1, alpha=0.5)
        nx.draw_networkx_nodes(G, node_positions, node_color='#31688E', node_size=30, ax=ax1, alpha=0.7)
        
        # Dragon position and path
        dragon_pos = ax1.scatter(0, 0, c='#FF1493', s=200, marker='D', 
                               zorder=10, edgecolors='white', linewidth=2)
          # Path visualization
        path_line, = ax1.plot([], [], 'r-', linewidth=3, alpha=0.7, zorder=8)
        
        # State diagram elements - Improved layout and styling
        state_boxes = {}
        state_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FF4500']
        
        # Better positioning for more states
        positions = [
            (2, 8.5), (8, 8.5),    # HOLDING, STRAFING
            (2, 6.5), (8, 6.5),    # APPROACH, LANDING  
            (2, 4.5), (8, 4.5),    # PERCHING, TAKEOFF
            (5, 2.5)               # CHARGING (centered)
        ]
        
        for i, state in enumerate(self.dragon_states):
            x, y = positions[i]
            box = FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6, boxstyle="round,pad=0.1",
                               facecolor=state_colors[i], alpha=0.3, edgecolor='white')
            ax2.add_patch(box)
            ax2.text(x, y, state, ha='center', va='center',
                    color=textFontColor, fontsize=10, weight='bold')
            state_boxes[state] = box
        
        # Animation data - Extended to show all behaviors
        simulation_steps = 140  # Increased to show all behaviors
        dragon_path = []
        states = []
        
        # Generate dragon movement simulation with all behavior types
        current_node = 'fountain'
        current_state = 'HOLDING'
        
        for step in range(simulation_steps):
            # More sophisticated state machine with all behaviors
            if step % 20 == 0:  # Change state every 20 frames
                current_state = self.dragon_states[step // 20 % len(self.dragon_states)]
            
            # Move dragon based on state - Updated for all behaviors
            if current_state == 'HOLDING':
                # Circle around outer nodes
                angle = step * 0.15
                x = 85 * np.cos(angle)
                z = 85 * np.sin(angle)
            elif current_state == 'STRAFING':
                # Move in straight lines with fireball patterns
                x = 70 * np.cos(step * 0.1)
                z = 45 * np.sin(step * 0.12)
            elif current_state == 'APPROACH':
                # Move toward fountain gradually
                target = self.dragon_nodes['fountain']
                progress = (step % 20) / 20.0
                start_radius = 80
                x = target[0] + (start_radius - start_radius * progress) * np.cos(step * 0.1)
                z = target[1] + (start_radius - start_radius * progress) * np.sin(step * 0.1)
            elif current_state == 'LANDING':
                # Spiral down toward perch
                radius = max(5, 25 - (step % 20))
                angle = step * 0.4
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)
            elif current_state == 'PERCHING':
                # Stay near fountain with minimal movement
                x = np.random.uniform(-8, 8)
                z = np.random.uniform(-8, 8)
            elif current_state == 'TAKEOFF':
                # Move away from fountain in expanding spiral
                radius = min(60, 5 + (step % 20) * 3)
                angle = step * 0.3
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)
            else:  # CHARGING
                # Fast direct movement toward player position (simulated)
                target_x, target_z = 40, 30  # Simulated player position
                progress = (step % 20) / 20.0
                x = progress * target_x + (1 - progress) * 80 * np.cos(step * 0.05)
                z = progress * target_z + (1 - progress) * 80 * np.sin(step * 0.05)
            
            dragon_path.append((x, z))
            states.append(current_state)
        
        # Text displays
        state_text = ax1.text(0.02, 0.98, "", transform=ax1.transAxes, 
                             color=textFontColor, fontsize=12, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor=backgroundColor, alpha=0.9))
        
        def animate_dragon(frame):
            if frame >= len(dragon_path):
                return []
            
            x, z = dragon_path[frame]
            current_state = states[frame]
            
            # Update dragon position
            dragon_pos.set_offsets([[x, z]])
            
            # Update path (show last 10 positions)
            path_start = max(0, frame - 10)
            path_x = [pos[0] for pos in dragon_path[path_start:frame+1]]
            path_z = [pos[1] for pos in dragon_path[path_start:frame+1]]
            path_line.set_data(path_x, path_z)
            
            # Update state highlighting
            for state, box in state_boxes.items():
                if state == current_state:
                    box.set_alpha(0.8)
                    box.set_edgecolor('#FFD700')
                    box.set_linewidth(2)
                else:
                    box.set_alpha(0.3)
                    box.set_edgecolor('white')
                    box.set_linewidth(1)
              # Update text
            state_text.set_text(f'Frame: {frame+1}/{len(dragon_path)}\\n'
                               f'State: {current_state}\\n'
                               f'Position: ({x:.1f}, {z:.1f})')
            
            return [dragon_pos, path_line, state_text]
        
        # Create animation - Adjusted timing for smoother playback
        anim = animation.FuncAnimation(fig, animate_dragon, frames=len(dragon_path),
                                     interval=150, blit=False, repeat=True)
        
        # Save animation
        try:
            print(f"Saving dragon pathfinding animation to {save_path}")
            anim.save(save_path, writer='pillow', fps=5, dpi=150)
            print("Dragon pathfinding animation saved successfully")
        except Exception as e:
            print(f"Error saving animation: {e}")
            # Save static image instead
            animate_dragon(len(dragon_path) // 2)
            plt.savefig(save_path.replace('.gif', '.png'), dpi=150, facecolor=backgroundColor)
        
        plt.close(fig)

def main():
    """Main function to run all animations."""
    print("Starting Minecraft Animation Generation...")
    
    # Initialize animator
    animator = MinecraftAnimator(world_seed=42)
    
    # Generate animations
    try:
        animator.animate_structure_placement(os.path.join(plotsPath, "minecraft_structure_placement.gif"))
    except Exception as e:
        print(f"Error in structure placement animation: {e}")
    
    try:
        animator.animate_dragon_pathfinding(os.path.join(plotsPath, "minecraft_dragon_pathfinding.gif"))
    except Exception as e:
        print(f"Error in dragon pathfinding animation: {e}")
    
    print("Minecraft animation generation completed!")

if __name__ == "__main__":
    main()
