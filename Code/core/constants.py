"""
Minecraft Generation Constants

Authentic values from Minecraft's world generation.
"""

# ============================================================================
# VISUAL STYLING
# ============================================================================

# Dark theme colors (GitHub-inspired)
BACKGROUND_COLOR = '#0D1117'
GRID_COLOR = '#21262D'
TEXT_COLOR = '#E6EDF3'
ACCENT_COLOR = '#58A6FF'

# Viridis-inspired palette for data visualization
PALETTE = {
    'purple': '#440154',
    'blue': '#31688E',
    'teal': '#35B779',
    'yellow': '#FDE725',
    'coral': '#FF6B6B',
    'cyan': '#4ECDC4',
    'gold': '#FFD700',
    'magenta': '#FF1493',
}

# ============================================================================
# WORLD GENERATION PARAMETERS
# ============================================================================

CHUNK_SIZE = 16  # Blocks per chunk

# Village generation
VILLAGE_SPACING = 32  # Chunks between village region centers
VILLAGE_SEPARATION = 8  # Minimum chunks between villages
VILLAGE_SALT = 10387312

# Nether Fortress generation  
FORTRESS_SPACING = 27
FORTRESS_SEPARATION = 4
FORTRESS_SALT = 30084232

# Ocean Monument generation
MONUMENT_SPACING = 32
MONUMENT_SEPARATION = 5
MONUMENT_SALT = 10387313

# ============================================================================
# STRONGHOLD RING DEFINITIONS
# ============================================================================

STRONGHOLD_RINGS = [
    {'count': 3, 'min_radius': 1280, 'max_radius': 2816, 'color': '#FF6B6B'},
    {'count': 6, 'min_radius': 4352, 'max_radius': 5888, 'color': '#4ECDC4'},
    {'count': 10, 'min_radius': 7424, 'max_radius': 8960, 'color': '#45B7D1'},
    {'count': 15, 'min_radius': 10496, 'max_radius': 12032, 'color': '#96CEB4'},
    {'count': 21, 'min_radius': 13568, 'max_radius': 15104, 'color': '#FFEAA7'},
    {'count': 28, 'min_radius': 16640, 'max_radius': 18176, 'color': '#DDA0DD'},
    {'count': 36, 'min_radius': 19712, 'max_radius': 21248, 'color': '#87CEEB'},
    {'count': 9, 'min_radius': 22784, 'max_radius': 24320, 'color': '#F0E68C'},
]

TOTAL_STRONGHOLDS = sum(ring['count'] for ring in STRONGHOLD_RINGS)  # 128

# ============================================================================
# ENDER DRAGON AI PARAMETERS
# ============================================================================

# Dragon behavioral states
DRAGON_STATES = [
    'HOLDING',    # Circling the outer ring
    'STRAFING',   # Attack runs with fireballs
    'APPROACH',   # Moving toward fountain
    'LANDING',    # Descending to perch
    'PERCHING',   # Stationary on fountain (vulnerable)
    'TAKEOFF',    # Launching from fountain
    'CHARGING',   # Direct charge attack at player
]

# Dragon arena dimensions
END_PILLAR_COUNT = 10
END_PILLAR_RADIUS = 76  # Blocks from center
END_FOUNTAIN_RADIUS = 8

# Pathfinding node radii
DRAGON_OUTER_RING_RADIUS = 100
DRAGON_INNER_RING_RADIUS = 60
DRAGON_CENTER_RING_RADIUS = 30

# Node counts per ring
DRAGON_OUTER_NODE_COUNT = 12
DRAGON_INNER_NODE_COUNT = 8
DRAGON_CENTER_NODE_COUNT = 4

# AI parameters
DRAGON_BASE_PERCH_PROBABILITY = 1/3  # With 0 crystals
DRAGON_CRYSTAL_PERCH_MODIFIER = 1  # Added to denominator per crystal

# ============================================================================
# BIOME CLASSIFICATION THRESHOLDS
# ============================================================================

BIOME_COLORS = {
    'Ocean': '#1E3A8A',
    'Plains': '#22C55E',
    'Desert': '#EAB308',
    'Forest': '#16A34A',
    'Taiga': '#0EA5E9',
    'Mountains': '#6B7280',
    'Swamp': '#84CC16',
    'Tundra': '#E0E0E0',
    'Savanna': '#BDB76B',
    'Jungle': '#228B22',
}
