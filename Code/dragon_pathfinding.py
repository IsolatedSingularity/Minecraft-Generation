"""
Enhanced Ender Dragon Pathfinding Visualization

Creates a publication-quality animation showing the Ender Dragon's AI
behavior with actual graph traversal, state transitions, and crystal mechanics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch, Wedge, FancyArrowPatch
from matplotlib.collections import LineCollection
import networkx as nx
import os

# ============================================================================
# VISUAL CONFIGURATION
# ============================================================================

plt.style.use('dark_background')

# Color scheme
COLORS = {
    'background': '#0D1117',
    'grid': '#21262D',
    'text': '#E6EDF3',
    'accent': '#58A6FF',
    'dragon': '#9B59B6',
    'dragon_trail': '#8E44AD',
    'crystal': '#00FF88',
    'crystal_destroyed': '#FF4444',
    'fountain': '#708090',
    'obsidian': '#1a0a2e',
    'outer_ring': '#440154',
    'inner_ring': '#31688E', 
    'center_ring': '#35B779',
    'central': '#FDE725',
    'edge': '#E0E0E0',
    'active_edge': '#FFD700',
    'fireball': '#FF6B00',
}

# State colors
STATE_COLORS = {
    'HOLDING': '#3498DB',
    'STRAFING': '#E74C3C',
    'APPROACH': '#F39C12',
    'LANDING': '#9B59B6',
    'PERCHING': '#2ECC71',
    'TAKEOFF': '#1ABC9C',
    'CHARGING': '#E91E63',
}

# ============================================================================
# DRAGON AI SIMULATION
# ============================================================================

class EnderDragonAI:
    """
    Simulates the Ender Dragon's AI behavior with authentic mechanics.
    """
    
    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(seed)
        
        # Arena setup
        self.pillar_radius = 76
        self.pillar_count = 10
        self.fountain_pos = np.array([0.0, 0.0])
        
        # Generate pillar positions
        pillar_angles = np.linspace(0, 2*np.pi, self.pillar_count, endpoint=False)
        self.pillars = [(self.pillar_radius * np.cos(a), 
                        self.pillar_radius * np.sin(a)) for a in pillar_angles]
        
        # Crystal states (True = alive)
        self.crystals = [True] * self.pillar_count
        self.crystals_alive = self.pillar_count
        
        # Build pathfinding graph
        self.graph = self._build_pathfinding_graph()
        self.node_positions = nx.get_node_attributes(self.graph, 'pos')
        
        # Dragon state
        self.current_state = 'HOLDING'
        self.current_node = 'outer_0'
        self.target_node = None
        self.position = np.array(self.node_positions['outer_0'])
        self.path_history = [self.position.copy()]
        self.state_history = ['HOLDING']
        
        # State timing
        self.state_timer = 0
        self.state_duration = {
            'HOLDING': 60,
            'STRAFING': 40,
            'APPROACH': 30,
            'LANDING': 25,
            'PERCHING': 50,
            'TAKEOFF': 20,
            'CHARGING': 35,
        }
        
        # Fireballs
        self.fireballs = []
        
    def _build_pathfinding_graph(self):
        """Build the dragon's navigation graph."""
        G = nx.Graph()
        
        # Outer ring nodes (12)
        outer_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
        for i, angle in enumerate(outer_angles):
            pos = (100 * np.cos(angle), 100 * np.sin(angle))
            G.add_node(f'outer_{i}', pos=pos, ring='outer')
        
        # Inner ring nodes (8)
        inner_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for i, angle in enumerate(inner_angles):
            pos = (60 * np.cos(angle), 60 * np.sin(angle))
            G.add_node(f'inner_{i}', pos=pos, ring='inner')
        
        # Center ring nodes (4)
        center_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
        for i, angle in enumerate(center_angles):
            pos = (30 * np.cos(angle), 30 * np.sin(angle))
            G.add_node(f'center_{i}', pos=pos, ring='center')
        
        # Fountain node
        G.add_node('fountain', pos=(0, 0), ring='fountain')
        
        # Connect outer ring
        for i in range(12):
            G.add_edge(f'outer_{i}', f'outer_{(i+1)%12}')
        
        # Connect inner ring
        for i in range(8):
            G.add_edge(f'inner_{i}', f'inner_{(i+1)%8}')
        
        # Connect center ring
        for i in range(4):
            G.add_edge(f'center_{i}', f'center_{(i+1)%4}')
            G.add_edge(f'center_{i}', 'fountain')
        
        # Outer to inner connections
        outer_inner = [(0,0), (1,1), (2,1), (3,2), (4,3), (5,3), 
                       (6,4), (7,5), (8,5), (9,6), (10,7), (11,7)]
        for o, i in outer_inner:
            G.add_edge(f'outer_{o}', f'inner_{i}')
        
        # Inner to center connections
        inner_center = [(0,0), (1,0), (2,1), (3,1), (4,2), (5,2), (6,3), (7,3)]
        for i, c in inner_center:
            G.add_edge(f'inner_{i}', f'center_{c}')
        
        return G
    
    def get_perch_probability(self):
        """Calculate probability of landing based on crystal count."""
        return 1.0 / (3.0 + self.crystals_alive)
    
    def destroy_crystal(self, idx):
        """Destroy a crystal and update AI behavior."""
        if self.crystals[idx]:
            self.crystals[idx] = False
            self.crystals_alive -= 1
            return True
        return False
    
    def choose_next_state(self):
        """Choose next AI state based on probabilities."""
        if self.current_state == 'PERCHING':
            return 'TAKEOFF'
        
        if self.current_state == 'TAKEOFF':
            return 'HOLDING'
        
        if self.current_state == 'LANDING':
            return 'PERCHING'
        
        if self.current_state == 'APPROACH':
            return 'LANDING'
        
        # Random state transitions
        roll = np.random.random()
        perch_prob = self.get_perch_probability()
        
        if self.current_state == 'HOLDING':
            if roll < perch_prob:
                return 'APPROACH'
            elif roll < 0.4:
                return 'STRAFING'
            elif roll < 0.5:
                return 'CHARGING'
            return 'HOLDING'
        
        if self.current_state in ['STRAFING', 'CHARGING']:
            if roll < 0.6:
                return 'HOLDING'
            elif roll < 0.8:
                return 'STRAFING'
            return 'CHARGING'
        
        return 'HOLDING'
    
    def get_target_node(self):
        """Get next target node based on current state."""
        if self.current_state == 'APPROACH':
            # Move toward center
            ring = self.graph.nodes[self.current_node].get('ring', 'outer')
            if ring == 'outer':
                candidates = [n for n in self.graph.neighbors(self.current_node)
                             if self.graph.nodes[n]['ring'] == 'inner']
            elif ring == 'inner':
                candidates = [n for n in self.graph.neighbors(self.current_node)
                             if self.graph.nodes[n]['ring'] == 'center']
            elif ring == 'center':
                return 'fountain'
            else:
                return self.current_node
            return np.random.choice(candidates) if candidates else self.current_node
        
        elif self.current_state == 'TAKEOFF':
            # Move away from center
            ring = self.graph.nodes[self.current_node].get('ring', 'fountain')
            if ring == 'fountain':
                candidates = [f'center_{i}' for i in range(4)]
            elif ring == 'center':
                candidates = [n for n in self.graph.neighbors(self.current_node)
                             if self.graph.nodes[n]['ring'] == 'inner']
            elif ring == 'inner':
                candidates = [n for n in self.graph.neighbors(self.current_node)
                             if self.graph.nodes[n]['ring'] == 'outer']
            else:
                candidates = [n for n in self.graph.neighbors(self.current_node)]
            return np.random.choice(candidates) if candidates else self.current_node
        
        elif self.current_state in ['HOLDING', 'STRAFING']:
            # Move along current ring or to neighbors
            candidates = list(self.graph.neighbors(self.current_node))
            return np.random.choice(candidates) if candidates else self.current_node
        
        return self.current_node
    
    def update(self):
        """Update dragon position and state for one frame."""
        self.state_timer += 1
        
        # State transition
        if self.state_timer >= self.state_duration.get(self.current_state, 40):
            self.state_timer = 0
            new_state = self.choose_next_state()
            self.current_state = new_state
            self.target_node = self.get_target_node()
        
        # Movement
        if self.target_node and self.target_node != self.current_node:
            target_pos = np.array(self.node_positions[self.target_node])
            direction = target_pos - self.position
            dist = np.linalg.norm(direction)
            
            # Movement speed varies by state
            speed = {
                'HOLDING': 2.5,
                'STRAFING': 4.0,
                'APPROACH': 2.0,
                'LANDING': 1.5,
                'PERCHING': 0.2,
                'TAKEOFF': 3.0,
                'CHARGING': 5.0,
            }.get(self.current_state, 2.0)
            
            if dist > speed:
                self.position += direction / dist * speed
            else:
                self.position = target_pos.copy()
                self.current_node = self.target_node
                self.target_node = self.get_target_node()
        
        # Add oscillation for flying
        if self.current_state not in ['PERCHING', 'LANDING']:
            self.position += np.array([
                0.5 * np.sin(self.state_timer * 0.3),
                0.5 * np.cos(self.state_timer * 0.25)
            ])
        
        # Spawn fireballs during strafing
        if self.current_state == 'STRAFING' and self.state_timer % 15 == 0:
            self.fireballs.append({
                'pos': self.position.copy(),
                'vel': np.random.randn(2) * 2,
                'life': 30
            })
        
        # Update fireballs
        for fb in self.fireballs:
            fb['pos'] += fb['vel']
            fb['life'] -= 1
        self.fireballs = [fb for fb in self.fireballs if fb['life'] > 0]
        
        # Record history
        self.path_history.append(self.position.copy())
        self.state_history.append(self.current_state)
        
        # Keep history manageable
        if len(self.path_history) > 100:
            self.path_history.pop(0)
            self.state_history.pop(0)


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_dragon_pathfinding_animation(save_path, frames=300, dpi=200, fps=20):
    """
    Create high-quality dragon pathfinding animation.
    
    Parameters
    ----------
    save_path : str
        Output file path
    frames : int
        Number of animation frames
    dpi : int
        Resolution (dots per inch)
    fps : int
        Frames per second
    """
    print("=" * 60)
    print("ENDER DRAGON PATHFINDING VISUALIZATION")
    print("=" * 60)
    
    # Initialize
    dragon = EnderDragonAI(seed=42)
    
    # Create figure
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Layout: main arena (left), state diagram (top right), stats (bottom right)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[1.2, 1],
                         hspace=0.15, wspace=0.1)
    
    ax_arena = fig.add_subplot(gs[:, 0])
    ax_states = fig.add_subplot(gs[0, 1])
    ax_stats = fig.add_subplot(gs[1, 1])
    
    for ax in [ax_arena, ax_states, ax_stats]:
        ax.set_facecolor(COLORS['background'])
        ax.tick_params(colors=COLORS['text'])
    
    # Arena setup
    ax_arena.set_xlim(-130, 130)
    ax_arena.set_ylim(-130, 130)
    ax_arena.set_aspect('equal')
    ax_arena.set_title('The End - Dragon AI Pathfinding', color=COLORS['text'], 
                      fontsize=16, fontweight='bold', pad=15)
    ax_arena.axis('off')
    
    # Draw static elements
    # End island
    island = Circle((0, 0), 120, color='#2d1f3d', alpha=0.3)
    ax_arena.add_patch(island)
    
    # Fountain
    fountain = Circle((0, 0), 8, color=COLORS['fountain'], alpha=0.8, zorder=5)
    ax_arena.add_patch(fountain)
    ax_arena.plot(0, 0, 'o', color='#4a4a6a', markersize=10, zorder=6)
    
    # Draw pathfinding graph edges
    for u, v in dragon.graph.edges():
        pos_u = dragon.node_positions[u]
        pos_v = dragon.node_positions[v]
        ax_arena.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], 
                     color=COLORS['edge'], alpha=0.15, linewidth=1, zorder=1)
    
    # Draw nodes
    for node, pos in dragon.node_positions.items():
        ring = dragon.graph.nodes[node].get('ring', 'outer')
        color = {
            'outer': COLORS['outer_ring'],
            'inner': COLORS['inner_ring'],
            'center': COLORS['center_ring'],
            'fountain': COLORS['central']
        }.get(ring, COLORS['outer_ring'])
        ax_arena.plot(pos[0], pos[1], 'o', color=color, markersize=6, alpha=0.6, zorder=2)
    
    # Pillars with crystals
    pillar_patches = []
    crystal_patches = []
    for i, (px, pz) in enumerate(dragon.pillars):
        pillar = Circle((px, pz), 6, color=COLORS['obsidian'], alpha=0.9, zorder=4)
        ax_arena.add_patch(pillar)
        pillar_patches.append(pillar)
        
        crystal = Circle((px, pz), 3, color=COLORS['crystal'], alpha=0.9, zorder=5)
        ax_arena.add_patch(crystal)
        crystal_patches.append(crystal)
    
    # Dynamic elements
    dragon_marker, = ax_arena.plot([], [], 'D', color=COLORS['dragon'], 
                                   markersize=18, zorder=10, markeredgecolor='white',
                                   markeredgewidth=2)
    trail_line, = ax_arena.plot([], [], '-', color=COLORS['dragon_trail'], 
                               alpha=0.6, linewidth=3, zorder=8)
    active_edge, = ax_arena.plot([], [], '-', color=COLORS['active_edge'],
                                linewidth=4, alpha=0.8, zorder=7)
    
    # State diagram
    ax_states.set_xlim(0, 10)
    ax_states.set_ylim(0, 8)
    ax_states.set_title('AI State Machine', color=COLORS['text'], 
                       fontsize=14, fontweight='bold')
    ax_states.axis('off')
    
    state_positions = {
        'HOLDING': (2, 6.5),
        'STRAFING': (5, 7),
        'CHARGING': (8, 6.5),
        'APPROACH': (2, 4),
        'LANDING': (5, 4),
        'PERCHING': (5, 1.5),
        'TAKEOFF': (8, 4),
    }
    
    state_boxes = {}
    for state, (x, y) in state_positions.items():
        box = FancyBboxPatch((x-0.9, y-0.4), 1.8, 0.8, boxstyle="round,pad=0.1",
                            facecolor=STATE_COLORS[state], alpha=0.3, 
                            edgecolor='white', linewidth=1)
        ax_states.add_patch(box)
        ax_states.text(x, y, state, ha='center', va='center',
                      color=COLORS['text'], fontsize=9, fontweight='bold')
        state_boxes[state] = box
    
    # Draw state transitions (arrows)
    transitions = [
        ('HOLDING', 'STRAFING'), ('HOLDING', 'APPROACH'), ('HOLDING', 'CHARGING'),
        ('STRAFING', 'HOLDING'), ('STRAFING', 'CHARGING'),
        ('CHARGING', 'HOLDING'), ('CHARGING', 'STRAFING'),
        ('APPROACH', 'LANDING'), ('LANDING', 'PERCHING'),
        ('PERCHING', 'TAKEOFF'), ('TAKEOFF', 'HOLDING'),
    ]
    for start, end in transitions:
        start_pos = state_positions[start]
        end_pos = state_positions[end]
        ax_states.annotate('', xy=end_pos, xytext=start_pos,
                          arrowprops=dict(arrowstyle='->', color=COLORS['grid'],
                                        alpha=0.4, connectionstyle='arc3,rad=0.1'))
    
    # Stats panel
    ax_stats.set_xlim(0, 10)
    ax_stats.set_ylim(0, 6)
    ax_stats.set_title('Dragon Statistics', color=COLORS['text'],
                      fontsize=14, fontweight='bold')
    ax_stats.axis('off')
    
    stats_text = ax_stats.text(0.5, 5, '', color=COLORS['text'], fontsize=11,
                              verticalalignment='top', family='monospace')
    
    # Perch probability bar
    ax_stats.text(0.5, 2.5, 'Perch Probability:', color=COLORS['text'], fontsize=10)
    prob_bar_bg = FancyBboxPatch((0.5, 1.5), 9, 0.6, boxstyle="round,pad=0.05",
                                 facecolor=COLORS['grid'], alpha=0.5)
    ax_stats.add_patch(prob_bar_bg)
    prob_bar = FancyBboxPatch((0.5, 1.5), 1, 0.6, boxstyle="round,pad=0.05",
                              facecolor=COLORS['accent'], alpha=0.8)
    ax_stats.add_patch(prob_bar)
    prob_text = ax_stats.text(5, 1.8, '', color=COLORS['text'], fontsize=10,
                             ha='center', va='center')
    
    # Crystal count
    crystal_icons = []
    for i in range(10):
        icon = Circle((0.8 + i*0.9, 0.5), 0.25, color=COLORS['crystal'], alpha=0.9)
        ax_stats.add_patch(icon)
        crystal_icons.append(icon)
    ax_stats.text(0.5, 1.0, 'End Crystals:', color=COLORS['text'], fontsize=10)
    
    # Fireball scatter
    fireball_scatter = ax_arena.scatter([], [], c=COLORS['fireball'], s=50, 
                                        alpha=0.8, marker='o', zorder=9)
    
    def animate(frame):
        # Destroy a crystal periodically
        if frame > 0 and frame % 80 == 0:
            alive_indices = [i for i, c in enumerate(dragon.crystals) if c]
            if alive_indices:
                idx = np.random.choice(alive_indices)
                dragon.destroy_crystal(idx)
                crystal_patches[idx].set_color(COLORS['crystal_destroyed'])
                crystal_patches[idx].set_alpha(0.3)
        
        # Update dragon
        dragon.update()
        
        # Update dragon marker
        dragon_marker.set_data([dragon.position[0]], [dragon.position[1]])
        
        # Update trail
        if len(dragon.path_history) > 1:
            trail_x = [p[0] for p in dragon.path_history[-30:]]
            trail_z = [p[1] for p in dragon.path_history[-30:]]
            trail_line.set_data(trail_x, trail_z)
        
        # Update active edge
        if dragon.target_node and dragon.current_node:
            pos_curr = dragon.node_positions[dragon.current_node]
            pos_targ = dragon.node_positions[dragon.target_node]
            active_edge.set_data([pos_curr[0], pos_targ[0]], 
                                [pos_curr[1], pos_targ[1]])
        
        # Update state boxes
        for state, box in state_boxes.items():
            if state == dragon.current_state:
                box.set_alpha(0.9)
                box.set_edgecolor(COLORS['active_edge'])
                box.set_linewidth(3)
            else:
                box.set_alpha(0.3)
                box.set_edgecolor('white')
                box.set_linewidth(1)
        
        # Update stats
        perch_prob = dragon.get_perch_probability()
        stats_text.set_text(
            f"Frame: {frame:03d}/{frames}\n"
            f"State: {dragon.current_state}\n"
            f"Node: {dragon.current_node}\n"
            f"Crystals: {dragon.crystals_alive}/10\n"
            f"Position: ({dragon.position[0]:.1f}, {dragon.position[1]:.1f})"
        )
        
        # Update probability bar
        prob_bar.set_width(9 * perch_prob)
        prob_text.set_text(f'{perch_prob*100:.1f}%')
        
        # Update crystal icons
        for i, (icon, alive) in enumerate(zip(crystal_icons, dragon.crystals)):
            icon.set_color(COLORS['crystal'] if alive else COLORS['crystal_destroyed'])
            icon.set_alpha(0.9 if alive else 0.2)
        
        # Update fireballs
        if dragon.fireballs:
            fb_positions = np.array([fb['pos'] for fb in dragon.fireballs])
            fireball_scatter.set_offsets(fb_positions)
        else:
            fireball_scatter.set_offsets(np.empty((0, 2)))
        
        return [dragon_marker, trail_line, active_edge, stats_text, 
                prob_bar, prob_text, fireball_scatter]
    
    print(f"Generating {frames} frames at {dpi} DPI...")
    anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                  interval=1000/fps, blit=False)
    
    try:
        print(f"Saving animation to: {save_path}")
        anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
        print("✓ Animation saved successfully!")
    except Exception as e:
        print(f"Error saving GIF: {e}")
        # Fallback to lower quality
        print("Trying with reduced settings...")
        anim.save(save_path, writer='pillow', fps=10, dpi=150)
    
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    # Get output path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(os.path.dirname(script_dir), "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    output_path = os.path.join(plots_dir, "dragon_pathfinding.gif")
    create_dragon_pathfinding_animation(output_path, frames=300, dpi=200, fps=20)
