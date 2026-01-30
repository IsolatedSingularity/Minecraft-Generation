"""
One-Shot Ender Dragon Kill — 3D Simulation

Visualizes the legendary MCSR one-shot technique:
- Arrow velocity manipulation via statistics overflow
- Dragon perching mechanics in 3D
- Projectile physics and damage calculation
- Frame-perfect timing visualization

The one-shot exploits integer overflow in arrow velocity statistics,
generating impossible speeds that instantly kill the dragon on impact.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.patches as mpatches
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
    'grid': '#21262D',
    'dragon_body': '#2d2d44',
    'dragon_eyes': '#ff00ff',
    'dragon_wing': '#1a1a2e',
    'end_stone': '#d4d8a8',
    'obsidian': '#1a0a2e',
    'arrow_normal': '#8B4513',
    'arrow_boosted': '#ff4444',
    'portal': '#2ecc71',
    'crystal': '#00ff88',
    'pillar': '#1a0a2e',
    'trajectory': '#ff6b6b',
    'damage': '#ff0000',
    'velocity_bar': '#00ff88',
    'stats_overflow': '#ff00ff',
}

# ============================================================================
# ONE-SHOT PHYSICS & MATHEMATICS
# ============================================================================

class ArrowPhysics:
    """
    Arrow velocity and damage calculations.
    
    Normal arrow: 
        - Base velocity: 3.0 blocks/tick at full charge
        - Damage: velocity * 2 (rounded up)
        - Max damage with Power V + crit: ~25
        
    Statistics Overflow Exploit:
        - Statistics stores cumulative distance traveled
        - Integer overflow at 2^31-1 (2,147,483,647)
        - Causes velocity calculation to return MAX_INT
        - Arrow "teleports" with massive kinetic energy
    """
    
    BASE_VELOCITY = 3.0  # blocks per tick
    TICKS_PER_SECOND = 20
    GRAVITY = 0.05  # blocks per tick²
    DRAG = 0.99  # velocity multiplier per tick
    
    # Overflow values
    MAX_INT = 2147483647
    OVERFLOW_THRESHOLD = 2000000000
    
    @staticmethod
    def normal_trajectory(start_pos, velocity, ticks=100):
        """Calculate normal arrow trajectory with gravity and drag."""
        positions = [start_pos.copy()]
        vel = velocity.copy()
        pos = start_pos.copy()
        
        for _ in range(ticks):
            vel[1] -= ArrowPhysics.GRAVITY  # Gravity
            vel *= ArrowPhysics.DRAG  # Air resistance
            pos = pos + vel
            positions.append(pos.copy())
            
            if pos[1] < 0:  # Hit ground
                break
        
        return np.array(positions)
    
    @staticmethod
    def calculate_damage(velocity_magnitude, is_critical=False, power_level=5):
        """
        Calculate arrow damage from velocity.
        
        Damage = ceil(velocity * 2)
        + Power enchantment: +25% per level
        + Critical: x1.5
        """
        base_damage = np.ceil(velocity_magnitude * 2)
        
        # Power enchantment bonus
        power_bonus = 1 + (power_level * 0.25) if power_level > 0 else 1
        damage = base_damage * power_bonus
        
        # Critical hit
        if is_critical:
            damage *= 1.5
        
        return int(damage)
    
    @staticmethod
    def overflow_velocity():
        """
        Simulate statistics overflow generating max velocity.
        
        The exploit works by:
        1. Accumulating arrow travel distance in statistics
        2. When stats approach MAX_INT, overflow occurs
        3. Velocity calculation wraps to negative then back positive
        4. Results in effectively infinite velocity
        """
        return ArrowPhysics.MAX_INT / 1000  # Scaled for visualization


class DragonModel:
    """
    Ender Dragon 3D model and perching mechanics.
    
    Dragon perch position: Above exit portal (0, ~4, 0)
    Perch duration: ~5 seconds
    Vulnerable hitbox: Head (1.0 block radius)
    """
    
    PERCH_HEIGHT = 4.0
    PERCH_DURATION = 100  # ticks
    HEAD_RADIUS = 1.0
    BODY_LENGTH = 8.0
    WING_SPAN = 12.0
    HEALTH = 200
    
    @staticmethod
    def get_perch_position(portal_pos=np.array([0, 0, 0])):
        """Get dragon perch position above portal."""
        return portal_pos + np.array([0, DragonModel.PERCH_HEIGHT, -2])
    
    @staticmethod
    def generate_body_mesh(center):
        """Generate simplified dragon body geometry."""
        # Body vertices (elongated ellipsoid)
        body_verts = []
        for i in np.linspace(0, 2*np.pi, 8):
            for j in np.linspace(-1, 1, 5):
                x = center[0] + 0.8 * np.cos(i) * (1 - j**2)**0.5
                y = center[1] + 0.6 * np.sin(i) * (1 - j**2)**0.5
                z = center[2] + j * 2.5
                body_verts.append([x, y, z])
        return np.array(body_verts)
    
    @staticmethod
    def generate_wing_mesh(center, time=0, wing_side=1):
        """Generate wing geometry with flapping animation."""
        wing_angle = np.sin(time * 0.1) * 0.3  # Gentle flapping
        
        # Wing vertices
        wing_base = center + np.array([wing_side * 1.0, 0, -1])
        wing_tip = center + np.array([wing_side * 6.0, np.sin(wing_angle) * 2, -0.5])
        wing_back = center + np.array([wing_side * 4.0, np.sin(wing_angle) * 1.5, 2])
        
        return np.array([wing_base, wing_tip, wing_back])
    
    @staticmethod  
    def generate_head_mesh(center):
        """Generate dragon head geometry."""
        head_offset = np.array([0, 0.3, -3.5])
        head_center = center + head_offset
        
        # Simple head as vertices
        return head_center, DragonModel.HEAD_RADIUS


# ============================================================================
# ANIMATION
# ============================================================================

def create_oneshot_animation(save_path, fps=30, duration=8):
    """
    Create 3D animation of one-shot dragon kill.
    
    Timeline:
    - 0-2s: Setup, show normal arrow stats
    - 2-4s: Statistics manipulation, overflow buildup
    - 4-5s: Arrow release with boosted velocity
    - 5-6s: Impact and damage calculation
    - 6-8s: Dragon death, stats display
    """
    print("=" * 60)
    print("ONE-SHOT ENDER DRAGON — 3D SIMULATION")
    print("=" * 60)
    
    total_frames = fps * duration
    
    # Create figure with 3D subplot and info panels
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(COLORS['background'])
    
    # Layout
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 0.7, 0.8], height_ratios=[1, 0.3],
                         hspace=0.15, wspace=0.15,
                         left=0.05, right=0.95, top=0.92, bottom=0.08)
    
    # 3D view (main)
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Velocity graph
    ax_velocity = fig.add_subplot(gs[0, 1])
    
    # Stats panel
    ax_stats = fig.add_subplot(gs[0, 2])
    
    # Timeline
    ax_timeline = fig.add_subplot(gs[1, :])
    
    for ax in [ax_velocity, ax_stats, ax_timeline]:
        ax.set_facecolor(COLORS['background'])
        ax.tick_params(colors=COLORS['text'])
    
    # Animation state
    state = {
        'phase': 'setup',  # setup, overflow, release, impact, death
        'arrow_pos': np.array([-15.0, 8.0, 10.0]),
        'arrow_vel': np.array([0.0, 0.0, 0.0]),
        'arrow_released': False,
        'statistics_value': 0,
        'overflow_triggered': False,
        'dragon_health': DragonModel.HEALTH,
        'damage_dealt': 0,
        'velocity_magnitude': 3.0,
        'impact_frame': -1,
    }
    
    # Portal and pillar positions
    portal_pos = np.array([0, 0, 0])
    dragon_perch = DragonModel.get_perch_position(portal_pos)
    player_pos = np.array([-15, 3, 10])
    
    def init():
        return []
    
    def update(frame):
        # Clear axes
        ax_3d.cla()
        ax_velocity.cla()
        ax_stats.cla()
        ax_timeline.cla()
        
        # Configure 3D axis
        ax_3d.set_facecolor(COLORS['void'])
        ax_3d.set_xlim(-20, 20)
        ax_3d.set_ylim(-5, 25)
        ax_3d.set_zlim(-20, 20)
        ax_3d.set_xlabel('X', color=COLORS['text'])
        ax_3d.set_ylabel('Y', color=COLORS['text'])
        ax_3d.set_zlabel('Z', color=COLORS['text'])
        ax_3d.set_title('The End — One-Shot Technique', color=COLORS['text'],
                       fontsize=14, fontweight='bold')
        
        # Calculate phase based on frame
        progress = frame / total_frames
        
        if progress < 0.25:
            state['phase'] = 'setup'
        elif progress < 0.5:
            state['phase'] = 'overflow'
            # Build up statistics
            overflow_progress = (progress - 0.25) / 0.25
            state['statistics_value'] = int(overflow_progress * ArrowPhysics.OVERFLOW_THRESHOLD)
        elif progress < 0.55:
            state['phase'] = 'overflow'
            state['overflow_triggered'] = True
            state['statistics_value'] = ArrowPhysics.MAX_INT
            state['velocity_magnitude'] = ArrowPhysics.overflow_velocity()
        elif progress < 0.7:
            state['phase'] = 'release'
            if not state['arrow_released']:
                state['arrow_released'] = True
                state['arrow_pos'] = player_pos.copy() + np.array([0, 1.5, 0])
                # Direction toward dragon head
                direction = dragon_perch - state['arrow_pos']
                direction = direction / np.linalg.norm(direction)
                # Boosted velocity
                state['arrow_vel'] = direction * 150  # Extreme speed
            
            # Move arrow (almost instant due to speed)
            state['arrow_pos'] += state['arrow_vel'] * 0.1
            
            # Check impact
            if np.linalg.norm(state['arrow_pos'] - dragon_perch) < 2:
                state['phase'] = 'impact'
                state['impact_frame'] = frame
                state['damage_dealt'] = int(state['velocity_magnitude'] * 0.001)
                state['dragon_health'] = 0
        elif progress < 0.85:
            state['phase'] = 'impact'
        else:
            state['phase'] = 'death'
        
        # ====================================================================
        # DRAW 3D SCENE
        # ====================================================================
        
        # Draw end platform (simplified)
        platform_size = 12
        xx, zz = np.meshgrid(np.linspace(-platform_size, platform_size, 5),
                            np.linspace(-platform_size, platform_size, 5))
        yy = np.zeros_like(xx)
        ax_3d.plot_surface(xx, yy, zz, alpha=0.3, color=COLORS['end_stone'])
        
        # Draw exit portal
        portal_theta = np.linspace(0, 2*np.pi, 20)
        portal_x = 3 * np.cos(portal_theta)
        portal_z = 3 * np.sin(portal_theta)
        portal_y = np.ones_like(portal_x) * 0.5
        ax_3d.plot(portal_x, portal_y, portal_z, color=COLORS['portal'], 
                  linewidth=3, alpha=0.8)
        ax_3d.plot_surface(np.outer(np.cos(portal_theta), np.linspace(0, 3, 10)).T,
                          np.ones((10, 20)) * 0.5,
                          np.outer(np.sin(portal_theta), np.linspace(0, 3, 10)).T,
                          alpha=0.5, color=COLORS['portal'])
        
        # Draw obsidian pillars
        pillar_angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
        pillar_radius = 10
        for i, angle in enumerate(pillar_angles):
            px = pillar_radius * np.cos(angle)
            pz = pillar_radius * np.sin(angle)
            pillar_height = 8 + i
            
            # Pillar
            ax_3d.bar3d(px-0.5, 0, pz-0.5, 1, pillar_height, 1,
                       color=COLORS['obsidian'], alpha=0.7)
            
            # Crystal on top
            ax_3d.scatter([px], [pillar_height + 0.5], [pz], 
                         c=COLORS['crystal'], s=50, marker='D')
        
        # Draw dragon (perching)
        dragon_alpha = 1.0 if state['dragon_health'] > 0 else max(0.1, 1.0 - (frame - state['impact_frame']) / 30)
        
        # Dragon body (elongated shape)
        body_center = dragon_perch
        
        # Body as ellipsoid points
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        body_x = body_center[0] + 1.5 * np.outer(np.cos(u), np.sin(v))
        body_y = body_center[1] + 1 * np.outer(np.sin(u), np.sin(v))
        body_z = body_center[2] + 3 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax_3d.plot_surface(body_x, body_y, body_z, color=COLORS['dragon_body'], 
                          alpha=0.7 * dragon_alpha)
        
        # Dragon head
        head_pos = body_center + np.array([0, 0.5, -4])
        ax_3d.scatter([head_pos[0]], [head_pos[1]], [head_pos[2]], 
                     c=COLORS['dragon_body'], s=200, marker='o', alpha=dragon_alpha)
        # Eyes
        ax_3d.scatter([head_pos[0]-0.3, head_pos[0]+0.3], 
                     [head_pos[1]+0.2, head_pos[1]+0.2],
                     [head_pos[2]-0.3, head_pos[2]-0.3],
                     c=COLORS['dragon_eyes'], s=30, marker='o', alpha=dragon_alpha)
        
        # Dragon wings
        wing_flap = np.sin(frame * 0.15) * 0.5
        for side in [-1, 1]:
            wing_base = body_center + np.array([side * 1.5, 0, 0])
            wing_mid = body_center + np.array([side * 5, 2 + wing_flap * side, -1])
            wing_tip = body_center + np.array([side * 7, 1 + wing_flap * side * 0.5, 1])
            wing_back = body_center + np.array([side * 4, 0.5, 2])
            
            # Wing membrane
            wing_verts = [
                [wing_base, wing_mid, wing_back],
                [wing_mid, wing_tip, wing_back]
            ]
            for verts in wing_verts:
                tri = Poly3DCollection([verts], alpha=0.5 * dragon_alpha)
                tri.set_facecolor(COLORS['dragon_wing'])
                tri.set_edgecolor(COLORS['dragon_body'])
                ax_3d.add_collection3d(tri)
        
        # Draw player position
        ax_3d.scatter([player_pos[0]], [player_pos[1]], [player_pos[2]],
                     c=COLORS['accent'], s=100, marker='^', label='Player')
        
        # Draw arrow
        if state['arrow_released'] and state['phase'] in ['release', 'impact']:
            arrow_color = COLORS['arrow_boosted'] if state['overflow_triggered'] else COLORS['arrow_normal']
            
            # Arrow trail
            if state['phase'] == 'release':
                trail_start = player_pos + np.array([0, 1.5, 0])
                ax_3d.plot([trail_start[0], state['arrow_pos'][0]],
                          [trail_start[1], state['arrow_pos'][1]],
                          [trail_start[2], state['arrow_pos'][2]],
                          color=COLORS['trajectory'], linewidth=2, alpha=0.5,
                          linestyle='--')
            
            # Arrow
            ax_3d.scatter([state['arrow_pos'][0]], [state['arrow_pos'][1]], 
                         [state['arrow_pos'][2]], c=arrow_color, s=80, marker='>')
        
        # Impact effect
        if state['phase'] == 'impact':
            impact_radius = (frame - state['impact_frame']) * 0.3
            impact_theta = np.linspace(0, 2*np.pi, 30)
            for r in np.linspace(0.5, impact_radius, 3):
                ix = dragon_perch[0] + r * np.cos(impact_theta)
                iy = dragon_perch[1] + np.ones_like(impact_theta) * 0.5
                iz = dragon_perch[2] + r * np.sin(impact_theta)
                ax_3d.plot(ix, iy, iz, color=COLORS['damage'], 
                          alpha=max(0, 1 - r/impact_radius), linewidth=2)
        
        # Camera angle
        ax_3d.view_init(elev=20, azim=45 + frame * 0.3)
        
        # ====================================================================
        # VELOCITY GRAPH
        # ====================================================================
        
        ax_velocity.set_facecolor(COLORS['background'])
        ax_velocity.set_title('Arrow Velocity', color=COLORS['text'], fontsize=11)
        ax_velocity.set_xlim(0, 10)
        ax_velocity.set_ylim(0, 120)
        ax_velocity.set_xlabel('Time', color=COLORS['text'])
        ax_velocity.set_ylabel('Velocity (blocks/tick)', color=COLORS['text'])
        
        # Build velocity history
        time_points = np.linspace(0, progress * 10, 50)
        velocity_values = []
        for t in time_points:
            if t < 5:
                velocity_values.append(3.0)  # Normal
            elif t < 5.5:
                velocity_values.append(3.0 + (t - 5) * 200)  # Spike
            else:
                velocity_values.append(min(100, 3.0 + (t - 5) * 200))
        
        ax_velocity.fill_between(time_points, velocity_values, alpha=0.3, 
                                color=COLORS['velocity_bar'])
        ax_velocity.plot(time_points, velocity_values, color=COLORS['velocity_bar'],
                        linewidth=2)
        
        # Overflow marker
        if state['overflow_triggered']:
            ax_velocity.axvline(5.5, color=COLORS['stats_overflow'], linestyle='--',
                               linewidth=2, label='OVERFLOW')
            ax_velocity.text(5.6, 80, 'OVERFLOW!', color=COLORS['stats_overflow'],
                           fontsize=10, fontweight='bold', rotation=90)
        
        ax_velocity.legend(loc='upper left', fontsize=8)
        ax_velocity.spines['top'].set_visible(False)
        ax_velocity.spines['right'].set_visible(False)
        ax_velocity.spines['bottom'].set_color(COLORS['text'])
        ax_velocity.spines['left'].set_color(COLORS['text'])
        
        # ====================================================================
        # STATS PANEL
        # ====================================================================
        
        ax_stats.set_facecolor(COLORS['background'])
        ax_stats.set_xlim(0, 10)
        ax_stats.set_ylim(0, 10)
        ax_stats.axis('off')
        ax_stats.set_title('Statistics', color=COLORS['text'], fontsize=11)
        
        stats_text = [
            ('Phase:', state['phase'].upper()),
            ('', ''),
            ('Arrow Stats:', ''),
            (f'  Distance:', f'{state["statistics_value"]:,}'),
            (f'  Velocity:', f'{state["velocity_magnitude"]:.1f} b/t'),
            ('', ''),
            ('Dragon:', ''),
            (f'  Health:', f'{state["dragon_health"]}/200'),
            (f'  State:', 'PERCHING'),
            ('', ''),
            ('Damage:', ''),
            (f'  Dealt:', f'{state["damage_dealt"]:,}' if state['damage_dealt'] > 0 else '—'),
        ]
        
        y_pos = 9.5
        for label, value in stats_text:
            color = COLORS['text']
            if 'OVERFLOW' in str(value) or state['statistics_value'] > ArrowPhysics.OVERFLOW_THRESHOLD:
                if 'Distance' in label:
                    color = COLORS['stats_overflow']
            if state['damage_dealt'] > 0 and 'Dealt' in label:
                color = COLORS['damage']
            
            ax_stats.text(0.5, y_pos, label, color=COLORS['text'], fontsize=9)
            ax_stats.text(5, y_pos, str(value), color=color, fontsize=9, 
                         fontweight='bold' if value else 'normal')
            y_pos -= 0.75
        
        # Overflow warning
        if state['overflow_triggered']:
            ax_stats.text(5, 0.5, '⚠ INTEGER OVERFLOW', color=COLORS['stats_overflow'],
                         fontsize=10, fontweight='bold', ha='center',
                         bbox=dict(boxstyle='round', facecolor=COLORS['background'],
                                  edgecolor=COLORS['stats_overflow']))
        
        # ====================================================================
        # TIMELINE
        # ====================================================================
        
        ax_timeline.set_facecolor(COLORS['background'])
        ax_timeline.set_xlim(0, 1)
        ax_timeline.set_ylim(0, 1)
        ax_timeline.axis('off')
        
        # Timeline bar
        ax_timeline.axhline(0.5, color=COLORS['grid'], linewidth=3, alpha=0.3)
        ax_timeline.axhline(0.5, color=COLORS['accent'], linewidth=3, 
                           xmax=progress, alpha=0.8)
        
        # Phase markers
        phases = [
            (0.0, 'Setup'),
            (0.25, 'Statistics\nManipulation'),
            (0.5, 'Overflow'),
            (0.55, 'Release'),
            (0.7, 'Impact'),
            (0.85, 'Death'),
        ]
        
        for pos, name in phases:
            ax_timeline.scatter([pos], [0.5], s=100, c=COLORS['accent'] if progress >= pos else COLORS['grid'],
                               marker='o', zorder=5)
            ax_timeline.text(pos, 0.15, name, ha='center', color=COLORS['text'],
                           fontsize=8, alpha=1.0 if progress >= pos else 0.5)
        
        # Current position marker
        ax_timeline.scatter([progress], [0.5], s=150, c=COLORS['accent'],
                           marker='v', zorder=6)
        
        # Title
        phase_titles = {
            'setup': 'Preparing shot — Normal arrow velocity',
            'overflow': 'Manipulating statistics — Building to overflow...',
            'release': 'ARROW RELEASED — Extreme velocity!',
            'impact': 'IMPACT! — Massive damage dealt',
            'death': 'Dragon defeated in ONE SHOT!'
        }
        ax_timeline.text(0.5, 0.85, phase_titles.get(state['phase'], ''),
                        ha='center', color=COLORS['accent'], fontsize=12,
                        fontweight='bold')
        
        return []
    
    # Create animation
    print(f"Generating {total_frames} frames at {fps} FPS...")
    anim = FuncAnimation(fig, update, frames=total_frames, init_func=init,
                        interval=1000/fps, blit=False)
    
    # Save
    print(f"Saving animation to: {save_path}")
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=100)
    print("✓ One-shot animation saved!")
    
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(os.path.dirname(script_dir), "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    output_path = os.path.join(plots_dir, "oneshot_dragon.gif")
    create_oneshot_animation(output_path, fps=20, duration=10)
