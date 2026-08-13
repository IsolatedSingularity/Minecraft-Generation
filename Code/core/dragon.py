"""Java 1.16.1 Ender Dragon path topology and reduced-order simulation."""

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .constants import DRAGON_NODE_CONNECTIONS
from .lcg import MinecraftLCG


STATE_ORDER = [
    'holding', 'strafing', 'landing_approach', 'landing', 'takeoff',
    'sitting_flaming', 'sitting_scanning', 'sitting_attacking',
    'charging_player', 'dying', 'hover',
]

# Direct phase changes observed in the Java 1.16.1 phase implementations.
# HOVER -> HOLDING is the fight-manager bootstrap, while the representative
# airborne -> DYING edge is handled separately by EnderDragonEntity damage.
SOURCE_PHASE_TRANSITIONS = (
    ('holding', 'strafing'),
    ('strafing', 'holding'),
    ('holding', 'landing_approach'),
    ('landing_approach', 'landing'),
    ('landing', 'sitting_scanning'),
    ('sitting_scanning', 'sitting_attacking'),
    ('sitting_scanning', 'takeoff'),
    ('sitting_scanning', 'charging_player'),
    ('sitting_attacking', 'sitting_flaming'),
    ('sitting_flaming', 'sitting_scanning'),
    ('takeoff', 'holding'),
    ('charging_player', 'holding'),
)

EXCEPTION_PHASE_TRANSITIONS = (
    ('hover', 'holding', 'fight bootstrap'),
    ('sitting_flaming', 'takeoff', 'sufficient sitting damage'),
    ('holding', 'dying', 'representative lethal airborne damage'),
)


@dataclass(frozen=True)
class DragonFrame:
    position: np.ndarray
    state: str
    crystals_alive: int
    current_node: int | None
    target_node: int | None
    alive_crystals: tuple[int, ...] | None = None
    fireball_position: np.ndarray | None = None
    breath_center: np.ndarray | None = None
    breath_radius: float = 0.0
    breath_alpha: float = 0.0
    breath_kind: str | None = None
    explosion_index: int | None = None
    explosion_phase: float = 0.0
    damage_pulse: float = 0.0
    active_edge: tuple[int, int] | None = None


def build_dragon_nodes():
    """Return the exact 24 horizontal path-node coordinates."""
    nodes = []
    for index in range(24):
        if index < 12:
            radius = 60.0
            angle = math.pi * index / 6.0
        elif index < 20:
            radius = 40.0
            angle = math.pi * (index - 12) / 4.0
        else:
            radius = 20.0
            angle = math.pi * (index - 20) / 2.0
        nodes.append(np.array([
            math.floor(radius * math.cos(angle)),
            math.floor(radius * math.sin(angle)),
        ], dtype=float))
    return np.array(nodes)


DRAGON_NODES = build_dragon_nodes()


def dragon_edges():
    """Decode the source adjacency bitmasks into unique undirected edges."""
    edges = set()
    for start, mask in enumerate(DRAGON_NODE_CONNECTIONS):
        for end in range(24):
            if mask & (1 << end):
                edges.add(tuple(sorted((start, end))))
    return sorted(edges)


DRAGON_EDGES = dragon_edges()


def dragon_adjacency():
    """Return the decoded source graph as a node-to-neighbours mapping."""
    adjacency = {index: [] for index in range(24)}
    for left, right in DRAGON_EDGES:
        adjacency[left].append(right)
        adjacency[right].append(left)
    for neighbors in adjacency.values():
        neighbors.sort()
    return adjacency


DRAGON_ADJACENCY = dragon_adjacency()


def shortest_path(start, finish, crystals_alive=10):
    """Run the dragon's weighted node search on the allowed node subset."""
    minimum_node = 0 if crystals_alive > 0 else 12
    allowed = set(range(minimum_node, 24))
    if start not in allowed:
        start = min(allowed, key=lambda i: np.linalg.norm(DRAGON_NODES[i] - DRAGON_NODES[start]))
    if finish not in allowed:
        finish = min(allowed, key=lambda i: np.linalg.norm(DRAGON_NODES[i] - DRAGON_NODES[finish]))

    queue = [(0.0, start)]
    distance = {start: 0.0}
    previous = {}
    while queue:
        current_distance, current = heapq.heappop(queue)
        if current == finish:
            break
        if current_distance != distance.get(current):
            continue
        for neighbor in DRAGON_ADJACENCY[current]:
            if neighbor not in allowed:
                continue
            weight = float(np.linalg.norm(
                DRAGON_NODES[current] - DRAGON_NODES[neighbor]
            ))
            proposed = current_distance + weight
            if proposed < distance.get(neighbor, float('inf')):
                distance[neighbor] = proposed
                previous[neighbor] = current
                heapq.heappush(queue, (proposed, neighbor))

    if finish not in distance:
        return [start]
    path = [finish]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return list(reversed(path))


def nearest_node(position, crystals_alive=10):
    minimum_node = 0 if crystals_alive > 0 else 12
    point = np.asarray(position, dtype=float)
    return min(
        range(minimum_node, 24),
        key=lambda index: np.linalg.norm(DRAGON_NODES[index] - point),
    )


def perch_probability(crystals_alive):
    return 1.0 / (int(crystals_alive) + 3.0)


def holding_transition(random, crystals_alive, player_distance=40.0):
    """Match the holding-phase landing and strafe decisions."""
    if random.next_int(int(crystals_alive) + 3) == 0:
        return 'landing_approach'
    distance_term = int(player_distance * player_distance / 512.0)
    if (
        random.next_int(abs(distance_term) + 2) == 0
        or random.next_int(int(crystals_alive) + 2) == 0
    ):
        return 'strafing'
    return 'holding'


def smooth_segment(start, end, samples=12, bend=0.0):
    """Interpolate a visually smooth top-down flight segment."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    length = np.linalg.norm(delta)
    perpendicular = np.array([-delta[1], delta[0]]) / max(length, 1.0)
    points = []
    for value in np.linspace(0.0, 1.0, samples, endpoint=False):
        eased = value * value * (3.0 - 2.0 * value)
        offset = math.sin(math.pi * value) * bend
        points.append(start + eased * delta + perpendicular * offset)
    return points


def catmull_rom_path(control_points, samples_per_segment=10):
    """Interpolate a fluid curve through a sequence of graph targets."""
    values = np.asarray(control_points, dtype=float)
    if len(values) < 2:
        return values.copy()
    padded = np.vstack((values[0], values, values[-1]))
    output = []
    for index in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[index - 1:index + 3]
        for t in np.linspace(0.0, 1.0, int(samples_per_segment), endpoint=False):
            t2 = t * t
            t3 = t2 * t
            point = 0.5 * (
                2.0 * p1
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            output.append(point)
    output.append(values[-1].copy())
    return np.asarray(output)


def _wrap_degrees(value):
    """Wrap an angle to the same signed interval used by MathHelper."""
    return (float(value) + 180.0) % 360.0 - 180.0


def source_steered_path(
    control_points, samples_per_target=12, arrival_radius=10.0,
    maximum_ticks_per_target=220, turn_mode='airborne',
):
    """Integrate a top-down reduction of the Java dragon steering loop.

    The target sequence can be source path nodes or phase-specific targets.
    Yaw error clamping, damped turn momentum, forward acceleration, alignment,
    and velocity retention follow ``EnderDragonEntity.tickMovement``. Vertical
    acceleration and block collision are intentionally omitted.
    """
    targets = np.asarray(control_points, dtype=float)
    if len(targets) < 2:
        return targets.copy()

    position = targets[0].copy()
    initial_delta = targets[1] - targets[0]
    initial_length = max(float(np.linalg.norm(initial_delta)), 1.0)
    initial_direction = initial_delta / initial_length
    yaw = math.degrees(math.atan2(initial_direction[1], initial_direction[0]))
    velocity = initial_direction * 0.38
    turn_momentum = 0.0
    microsteps = [position.copy()]

    for target in targets[1:]:
        for tick in range(int(maximum_ticks_per_target)):
            delta = target - position
            distance = float(np.linalg.norm(delta))
            if distance <= float(arrival_radius) and tick >= 2:
                break

            desired_yaw = math.degrees(math.atan2(delta[1], delta[0]))
            yaw_error = np.clip(_wrap_degrees(desired_yaw - yaw), -50.0, 50.0)
            horizontal_speed = float(np.linalg.norm(velocity))
            speed_term = horizontal_speed + 1.0
            capped_speed = min(speed_term, 40.0)
            if turn_mode == 'landing':
                turn_scale = capped_speed / speed_term
            else:
                turn_scale = 0.70 / capped_speed / speed_term
            turn_momentum *= 0.8
            turn_momentum += float(yaw_error) * turn_scale
            yaw = _wrap_degrees(yaw + turn_momentum * 0.1)

            radians = math.radians(yaw)
            forward = np.array([math.cos(radians), math.sin(radians)])
            target_direction = delta / max(distance, 1.0e-9)
            alignment = max((float(np.dot(forward, target_direction)) + 0.5) / 1.5, 0.0)
            distance_weight = 2.0 / (distance * distance + 1.0)
            acceleration = 0.06 * (
                alignment * distance_weight + (1.0 - distance_weight)
            )
            velocity = velocity + forward * acceleration
            position = position + velocity
            microsteps.append(position.copy())

            speed = float(np.linalg.norm(velocity))
            if speed > 1.0e-9:
                velocity_direction = velocity / speed
                retention = 0.8 + 0.15 * (
                    float(np.dot(velocity_direction, forward)) + 1.0
                ) / 2.0
                velocity *= retention

            # The Java phases ask for another path target inside a ten-block
            # radius. In this top-down reduction the same boundary prevents a
            # missed target from turning into a decorative orbit.
            if tick > 8 and distance > 150.0:
                break

    microsteps = np.asarray(microsteps)
    output_count = max(
        2, int(samples_per_target) * max(len(targets) - 1, 1) + 1,
    )
    source_index = np.linspace(0.0, len(microsteps) - 1.0, output_count)
    left = np.floor(source_index).astype(int)
    right = np.minimum(left + 1, len(microsteps) - 1)
    fraction = (source_index - left)[:, None]
    return microsteps[left] * (1.0 - fraction) + microsteps[right] * fraction


def path_coordinates(indices, samples_per_edge=10, bend=1.2):
    """Return a continuous spline through a legal sequence of path nodes."""
    if len(indices) == 1:
        return [DRAGON_NODES[indices[0]].copy()]
    del bend  # Retained for compatibility with older callers.
    return list(catmull_rom_path(
        [DRAGON_NODES[index] for index in indices],
        samples_per_segment=samples_per_edge,
    ))


def simulate_perch_trajectory(
    seed, crystals_alive=10, player_position=(34.0, -18.0),
    max_holding_segments=24,
):
    """Simulate one source-shaped landing approach in top-down projection.

    ``LandingApproachPhase`` finds the nearest current node, selects the node
    opposite the nearby player at radius 40, and appends the exit portal as
    the final path target. The seed chooses a representative current outer
    node; the graph route and portal approach then follow that source logic.
    Continuous movement remains a reduced-order top-down integration.
    """
    del max_holding_segments  # Kept for compatibility with older callers.
    random = MinecraftLCG(seed)
    player = np.asarray(player_position, dtype=float)
    current = random.next_int(12) if crystals_alive > 0 else 12 + random.next_int(8)

    direction = player / max(float(np.linalg.norm(player)), 1.0)
    opposite = -direction * 40.0
    landing_node = nearest_node(opposite, crystals_alive)
    node_path = shortest_path(current, landing_node, crystals_alive)

    control_points = [DRAGON_NODES[index] for index in node_path]
    control_points.append(np.zeros(2))
    coordinates = source_steered_path(control_points, samples_per_target=14)
    return coordinates, node_path


def _append_frames(
    frames, start, end, count, state, crystals, bend=0.0,
    alive_crystals=None,
):
    points = smooth_segment(start, end, samples=count, bend=bend)
    for index, point in enumerate(points):
        frames.append(DragonFrame(
            position=np.asarray(point),
            state=state,
            crystals_alive=crystals,
            current_node=None,
            target_node=None,
            alive_crystals=(
                tuple(range(int(crystals)))
                if alive_crystals is None else tuple(alive_crystals)
            ),
        ))


def scripted_showcase():
    """Return a fluid representative fight loop for the README hero.

    The path-node portions use legal routes. Player-targeted strafe and charge
    portions leave the graph, as their corresponding vanilla phases do.
    """
    frames = []
    alive = set(range(10))

    # PhaseManager initializes an entity in HOVER; EnderDragonFight then
    # bootstraps the live fight into HOLDING_PATTERN. Keep the exceptional
    # phase visible without pretending it belongs to the repeating fight loop.
    for _ in range(10):
        frames.append(DragonFrame(
            np.zeros(2), 'hover', len(alive), None, None,
            alive_crystals=tuple(sorted(alive)),
        ))

    def active_route_edge(point, graph_nodes):
        if graph_nodes is None or len(graph_nodes) < 2:
            return None
        position = np.asarray(point, dtype=float)
        nearest = None
        nearest_distance = float('inf')
        for left, right in zip(graph_nodes, graph_nodes[1:]):
            start = DRAGON_NODES[left]
            end = DRAGON_NODES[right]
            delta = end - start
            fraction = np.clip(
                np.dot(position - start, delta) / max(np.dot(delta, delta), 1.0),
                0.0, 1.0,
            )
            distance = float(np.linalg.norm(position - (start + fraction * delta)))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest = tuple(sorted((int(left), int(right))))
        return nearest if nearest_distance <= 11.0 else None

    def append_curve(points, state, samples=9, fireball=False, graph_nodes=None):
        curve = source_steered_path(
            points, samples_per_target=samples,
            turn_mode='landing' if state == 'landing' else 'airborne',
        )
        player = np.array([31.0, -17.0])
        for index, point in enumerate(curve):
            fireball_position = None
            breath_center = None
            breath_radius = 0.0
            breath_alpha = 0.0
            breath_kind = None
            fraction = index / max(len(curve) - 1, 1)
            if fireball and 0.32 <= fraction < 0.62:
                phase = (fraction - 0.32) / 0.30
                fireball_position = (1.0 - phase) * point + phase * player
            elif fireball and 0.62 <= fraction <= 0.96:
                phase = (fraction - 0.62) / 0.34
                breath_center = player.copy()
                breath_radius = 3.0 + 4.0 * phase
                breath_alpha = 0.48 * (1.0 - 0.48 * phase)
                breath_kind = 'projectile_impact'
            elif state == 'landing':
                breath_center = np.asarray(point) + np.array([2.8, 0.0])
                breath_radius = 1.2 + 1.4 * fraction
                breath_alpha = 0.18 + 0.12 * (1.0 - fraction)
                breath_kind = 'landing_particles'
            frames.append(DragonFrame(
                np.asarray(point), state, len(alive), None, None,
                alive_crystals=tuple(sorted(alive)),
                fireball_position=fireball_position,
                breath_center=breath_center,
                breath_radius=breath_radius,
                breath_alpha=breath_alpha,
                breath_kind=breath_kind,
                active_edge=active_route_edge(point, graph_nodes),
            ))

    first_holding = shortest_path(0, 15, crystals_alive=len(alive))
    append_curve(
        [np.zeros(2)] + [DRAGON_NODES[index] for index in first_holding],
        'holding', samples=10, graph_nodes=first_holding,
    )
    append_curve(
        [DRAGON_NODES[15], np.array([-10.0, -38.0]),
         np.array([31.0, -17.0]), DRAGON_NODES[6]],
        'strafing', samples=11, fireball=True,
    )

    # Strafe returns to Holding before any later phase decision.
    strafe_return = shortest_path(6, 18, crystals_alive=len(alive))
    append_curve(
        [DRAGON_NODES[index] for index in strafe_return],
        'holding', samples=9, graph_nodes=strafe_return,
    )

    # Crystal losses start after the flight has been established and jump
    # around the spike ring instead of walking it in angular order.
    explosion_order = (7, 2, 9, 4)
    second_holding = shortest_path(18, 3, crystals_alive=len(alive))
    holding_curve = source_steered_path(
        [DRAGON_NODES[index] for index in second_holding], samples_per_target=11,
    )
    event_frames = (8, 18, 28, 38)
    for frame_index, point in enumerate(holding_curve):
        explosion_index = None
        explosion_phase = 0.0
        for event_index, start in enumerate(event_frames):
            crystal_index = explosion_order[event_index]
            if start <= frame_index < start + 6:
                explosion_index = crystal_index
                explosion_phase = (frame_index - start + 1) / 6.0
            if frame_index >= start + 3:
                alive.discard(crystal_index)
        frames.append(DragonFrame(
            np.asarray(point), 'holding', len(alive), None, None,
            alive_crystals=tuple(sorted(alive)),
            explosion_index=explosion_index,
            explosion_phase=explosion_phase,
            active_edge=active_route_edge(point, second_holding),
        ))

    landing_route = shortest_path(3, 15, crystals_alive=len(alive))
    append_curve(
        [DRAGON_NODES[index] for index in landing_route],
        'landing_approach', samples=9, graph_nodes=landing_route,
    )
    append_curve(
        [DRAGON_NODES[landing_route[-1]], np.array([11.0, -6.0]), np.zeros(2)],
        'landing', samples=12,
    )

    sitting_index = 0

    def append_sitting(state, count):
        nonlocal sitting_index
        for local_index in range(count):
            angle = 2.0 * math.pi * sitting_index / 48.0
            point = np.array([0.9 * math.cos(angle), 0.55 * math.sin(angle)])
            sitting_index += 1
            flaming = state == 'sitting_flaming'
            flame_phase = local_index / max(count - 1, 1)
            ignition = min(1.0, flame_phase / 0.28) if flaming else 0.0
            frames.append(DragonFrame(
                point, state, len(alive), None, None,
                alive_crystals=tuple(sorted(alive)),
                breath_center=(np.array([5.0, 0.0]) if flaming else None),
                breath_radius=(5.0 * ignition if flaming else 0.0),
                breath_alpha=(
                    (0.48 - 0.16 * flame_phase)
                    * (0.35 + 0.65 * ignition)
                    if flaming else 0.0
                ),
                breath_kind=('sitting_flame' if flaming else None),
            ))

    # A distant player can be selected directly from Sitting Scanning. The
    # source sets Takeoff and then Charging Player in the same server tick, so
    # Charging is the visible phase here.
    append_sitting('sitting_scanning', 14)
    player = np.array([31.0, -17.0])
    append_curve(
        [np.zeros(2), np.array([12.0, -6.0]), player],
        'charging_player', samples=11,
    )
    charge_node = nearest_node(player, crystals_alive=len(alive))
    charge_return = shortest_path(charge_node, 8, crystals_alive=len(alive))
    append_curve(
        [player, DRAGON_NODES[charge_node]]
        + [DRAGON_NODES[index] for index in charge_return[1:]],
        'holding', samples=9, graph_nodes=charge_return,
    )

    # Return for a second, abbreviated but transition-valid perched sequence.
    player_direction = player / max(float(np.linalg.norm(player)), 1.0)
    second_landing_node = nearest_node(
        -player_direction * 40.0, crystals_alive=len(alive),
    )
    second_landing = shortest_path(8, second_landing_node, crystals_alive=len(alive))
    append_curve(
        [DRAGON_NODES[index] for index in second_landing],
        'landing_approach', samples=8, graph_nodes=second_landing,
    )
    append_curve(
        [DRAGON_NODES[second_landing[-1]], np.array([11.0, -6.0]), np.zeros(2)],
        'landing', samples=10,
    )
    append_sitting('sitting_scanning', 12)
    append_sitting('sitting_attacking', 10)
    append_sitting('sitting_flaming', 28)

    # Sufficient damage while sitting or hovering forces Takeoff. A brief
    # coral pulse makes that external trigger explicit in the animation.
    for pulse_index in range(5):
        frames.append(DragonFrame(
            np.zeros(2), 'sitting_flaming', len(alive), None, None,
            alive_crystals=tuple(sorted(alive)),
            breath_center=np.array([5.0, 0.0]),
            breath_radius=5.0,
            breath_alpha=0.24,
            breath_kind='sitting_flame',
            damage_pulse=(pulse_index + 1) / 5.0,
        ))

    takeoff_route = shortest_path(20, 8, crystals_alive=len(alive))
    append_curve(
        [np.zeros(2)] + [DRAGON_NODES[index] for index in takeoff_route],
        'takeoff', samples=9, graph_nodes=takeoff_route,
    )
    append_curve(
        [DRAGON_NODES[index] for index in (8, 9, 10, 11, 0)],
        'holding', samples=8, graph_nodes=(8, 9, 10, 11, 0),
    )
    # DyingPhase steers toward the exit portal. This is an explicit exceptional
    # showcase after the ordinary combat loop, not a transition caused by the
    # preceding scripted holding decision.
    append_curve(
        [DRAGON_NODES[0], np.array([38.0, 5.0]), np.array([17.0, -2.0]),
         np.zeros(2)],
        'dying', samples=6,
    )
    return frames
