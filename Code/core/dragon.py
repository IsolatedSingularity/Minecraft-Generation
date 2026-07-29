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


def _append_frames(frames, start, end, count, state, crystals, bend=0.0):
    points = smooth_segment(start, end, samples=count, bend=bend)
    for index, point in enumerate(points):
        frames.append(DragonFrame(
            position=np.asarray(point),
            state=state,
            crystals_alive=crystals,
            current_node=None,
            target_node=None,
        ))


def scripted_showcase():
    """Return a compact deterministic loop covering the major phase groups."""
    frames = []
    route = [0, 1, 2, 3, 4]
    for index, (start, end) in enumerate(zip(route, route[1:])):
        _append_frames(
            frames, DRAGON_NODES[start], DRAGON_NODES[end], 7,
            'holding', 10 - index, bend=1.1,
        )

    _append_frames(
        frames, DRAGON_NODES[4], np.array([30.0, -18.0]), 10,
        'strafing', 6, bend=-4.0,
    )
    _append_frames(
        frames, np.array([30.0, -18.0]), DRAGON_NODES[5], 8,
        'charging', 5, bend=2.0,
    )

    landing_route = shortest_path(5, 15, crystals_alive=4)
    for start, end in zip(landing_route, landing_route[1:]):
        _append_frames(
            frames, DRAGON_NODES[start], DRAGON_NODES[end], 6,
            'landing_approach', 4, bend=0.8,
        )
    _append_frames(
        frames, DRAGON_NODES[landing_route[-1]], np.zeros(2), 12,
        'landing', 3, bend=-1.5,
    )

    for index in range(18):
        angle = 2.0 * math.pi * index / 18.0
        point = np.array([0.7 * math.cos(angle), 0.45 * math.sin(angle)])
        frames.append(DragonFrame(point, 'perching', 2, None, None))

    takeoff_route = shortest_path(20, 0, crystals_alive=1)
    previous = np.zeros(2)
    for node in takeoff_route:
        _append_frames(
            frames, previous, DRAGON_NODES[node], 6,
            'takeoff', 1, bend=1.3,
        )
        previous = DRAGON_NODES[node]
    _append_frames(
        frames, previous, DRAGON_NODES[0], 8,
        'holding', 1, bend=-1.0,
    )
    frames.append(DragonFrame(DRAGON_NODES[0].copy(), 'holding', 1, 0, 1))
    return frames
