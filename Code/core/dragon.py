"""Java 1.16.1 Ender Dragon path topology and reduced-order simulation."""

from dataclasses import dataclass
import heapq
import math

import numpy as np

from .constants import DRAGON_NODE_CONNECTIONS
from .lcg import MinecraftLCG


STATE_ORDER = [
    'holding', 'strafing', 'charging', 'landing_approach',
    'landing', 'perching', 'takeoff',
]


@dataclass(frozen=True)
class DragonFrame:
    position: np.ndarray
    state: str
    crystals_alive: int
    current_node: int | None
    target_node: int | None
    alive_crystals: tuple[int, ...] | None = None
    fireball_position: np.ndarray | None = None
    explosion_index: int | None = None
    explosion_phase: float = 0.0


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
    adjacency = {index: [] for index in range(24)}
    for left, right in DRAGON_EDGES:
        adjacency[left].append(right)
        adjacency[right].append(left)

    while queue:
        current_distance, current = heapq.heappop(queue)
        if current == finish:
            break
        if current_distance != distance.get(current):
            continue
        for neighbor in adjacency[current]:
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


def path_coordinates(indices, samples_per_edge=10, bend=1.2):
    if len(indices) == 1:
        return [DRAGON_NODES[indices[0]].copy()]
    points = []
    for edge_index, (start, end) in enumerate(zip(indices, indices[1:])):
        direction = -1.0 if edge_index % 2 else 1.0
        points.extend(smooth_segment(
            DRAGON_NODES[start], DRAGON_NODES[end],
            samples=samples_per_edge, bend=bend * direction,
        ))
    points.append(DRAGON_NODES[indices[-1]].copy())
    return points


def simulate_perch_trajectory(
    seed, crystals_alive=10, player_position=(34.0, -18.0),
    max_holding_segments=24,
):
    """Simulate holding decisions and the source-shaped landing approach.

    The node graph and phase rolls are source-faithful. Continuous movement
    between targets is a reduced-order top-down interpolation.
    """
    random = MinecraftLCG(seed)
    player = np.asarray(player_position, dtype=float)
    current = random.next_int(12) if crystals_alive > 0 else 12 + random.next_int(8)
    clockwise = random.next_int(2) == 0
    node_path = [current]

    for _ in range(max_holding_segments):
        decision = holding_transition(
            random, crystals_alive, float(np.linalg.norm(player)),
        )
        if decision == 'landing_approach':
            break
        if random.next_int(8) == 0:
            clockwise = not clockwise
            current += 6
        if crystals_alive > 0:
            current = (current + (1 if clockwise else -1)) % 12
        else:
            current = 12 + ((current - 12 + (1 if clockwise else -1)) & 7)
        node_path.append(current)
        if decision == 'strafing':
            side = 1.0 if random.next_int(2) else -1.0
            strafe_point = player + np.array([side * 8.0, -side * 5.0])
            node_path.append(nearest_node(strafe_point, crystals_alive))
            current = node_path[-1]

    direction = player / max(float(np.linalg.norm(player)), 1.0)
    opposite = -direction * 40.0
    landing_node = nearest_node(opposite, crystals_alive)
    approach = shortest_path(current, landing_node, crystals_alive)
    node_path.extend(approach[1:])

    coordinates = path_coordinates(node_path, samples_per_edge=7, bend=1.4)
    coordinates.extend(smooth_segment(
        coordinates[-1], np.zeros(2), samples=18, bend=2.0,
    ))
    coordinates.append(np.zeros(2))
    return np.array(coordinates), node_path


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
    """Return a fluid, event-rich fight loop for the README hero."""
    frames = []
    alive = set(range(10))

    def append_curve(points, state, samples=9, fireball=False):
        curve = catmull_rom_path(points, samples_per_segment=samples)
        player = np.array([31.0, -17.0])
        for index, point in enumerate(curve):
            fireball_position = None
            if fireball and 0.34 <= index / max(len(curve) - 1, 1) <= 0.82:
                phase = (index / max(len(curve) - 1, 1) - 0.34) / 0.48
                fireball_position = (1.0 - phase) * point + phase * player
            frames.append(DragonFrame(
                np.asarray(point), state, len(alive), None, None,
                alive_crystals=tuple(sorted(alive)),
                fireball_position=fireball_position,
            ))

    append_curve(
        [DRAGON_NODES[index] for index in (0, 1, 2, 3, 4, 5, 6)],
        'holding', samples=9,
    )
    append_curve(
        [DRAGON_NODES[6], np.array([-12.0, -44.0]),
         np.array([31.0, -17.0]), DRAGON_NODES[8]],
        'strafing', samples=11, fireball=True,
    )
    append_curve(
        [DRAGON_NODES[index] for index in (8, 9, 10, 11, 0, 1)],
        'charging', samples=8,
    )

    # Crystal losses start after the flight has been established and jump
    # around the spike ring instead of walking it in angular order.
    explosion_order = (7, 2, 9, 4)
    holding_curve = catmull_rom_path(
        [DRAGON_NODES[index] for index in (1, 2, 3, 4, 5, 6)],
        samples_per_segment=10,
    )
    event_frames = (10, 23, 36, 47)
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
        ))

    landing_route = shortest_path(6, 15, crystals_alive=len(alive))
    append_curve(
        [DRAGON_NODES[index] for index in landing_route],
        'landing_approach', samples=9,
    )
    append_curve(
        [DRAGON_NODES[landing_route[-1]], np.array([11.0, -6.0]), np.zeros(2)],
        'landing', samples=12,
    )

    for index in range(26):
        angle = 2.0 * math.pi * index / 26.0
        point = np.array([0.9 * math.cos(angle), 0.55 * math.sin(angle)])
        frames.append(DragonFrame(
            point, 'perching', len(alive), None, None,
            alive_crystals=tuple(sorted(alive)),
        ))

    takeoff_route = shortest_path(20, 8, crystals_alive=len(alive))
    append_curve(
        [np.zeros(2)] + [DRAGON_NODES[index] for index in takeoff_route],
        'takeoff', samples=9,
    )
    append_curve(
        [DRAGON_NODES[index] for index in (8, 9, 10, 11, 0)],
        'holding', samples=8,
    )
    return frames
