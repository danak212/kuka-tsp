import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from itertools import product, permutations
from numpy.typing import ArrayLike
from shapely.geometry import LineString
from dataclasses import dataclass


class Map:
    def __init__(self, map_size, num_lines, num_connected, line_length, precision):
        self.size = map_size
        self.precision = precision
        self.lines = self.generate_lines(num_lines, num_connected, line_length)
        self.walls = None

    # Utility Functions
    def set_walls(self, walls):
        self.walls = walls

    # Calculates distance between two points
    def dist(self, p1: tuple, p2: tuple):
        return np.linalg.norm(np.subtract(p1, p2))

    # Return a random value in [min_val; max_val] with a certain float precision, set by map
    def random(self, min_val: float, max_val: float) -> float:
        return np.round(np.random.uniform(min_val, max_val) / self.precision) * self.precision

    # Return a random point as a tuple (x, y)
    def random_point(self) -> tuple:
        return self.random(0, self.size[0]), self.random(0, self.size[1])

    # Generate work lines, a certain number of connected lines, and within a given length range
    def generate_lines(self, num_lines: int, num_connected: int, line_length: tuple) -> ArrayLike:
        assert num_lines > num_connected * 2, "Number of lines must be greater than twice the connected lines."
        min_length, max_length = line_length

        lines = []

        # Generate connected lines
        for _ in range(num_connected):
            p1, p2 = self.random_point(), self.random_point()

            while not (min_length <= self.dist(p1, p2) <= max_length):
                p2 = self.random_point()
            lines.append((p1, p2))

            # Choose which end of the first line will be shared by second line
            shared_point = p1 if np.random.randint(0, 2) == 0 else p2
            second_point = self.random_point()

            while not (min_length <= self.dist(shared_point, second_point) <= max_length):
                second_point = self.random_point()
            lines.append((shared_point, second_point))
            num_lines -= 2

        # Generate remaining lines
        for _ in range(num_lines):
            p1, p2 = self.random_point(), self.random_point()

            while not (min_length <= self.dist(p1, p2) <= max_length):
                p2 = self.random_point()
            lines.append((p1, p2))

        return lines

    def plot(self, ax) -> None:
        colors = ['blue', 'green', 'orange', 'purple', 'brown']  # New colors for each segment
        labels = ['A', 'B', 'C', 'D', 'E']

        for i, line in enumerate(self.lines):
            p1, p2 = line
            x1, y1 = p1
            x2, y2 = p2
            label = f"{labels[i]}1 to {labels[i]}2"  # Renamed segments
            ax.plot([x1, x2], [y1, y2], color=colors[i], linewidth=2, label=label)
            ax.scatter([x1, x2], [y1, y2], c=colors[i])

        if self.walls is not None:
            for wall in self.walls:
                p1, p2 = wall
                x1, y1 = p1
                x2, y2 = p2
                ax.plot([x1, x2], [y1, y2], 'red', linewidth=3, label="Obstacle")


@dataclass
class Node:
    id: str
    start: np.ndarray
    end: np.ndarray

    def cost(self, other):
        return np.linalg.norm(np.subtract(self.end, other.start))


@dataclass
class Edge:
    node1: Node
    node2: Node

    def cost(self):
        return self.node1.cost(self.node2)


class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, id, start, end):
        if id not in self.nodes:
            self.nodes[id] = Node(id, start, end)

    def add_edge(self, id1, id2):
        if id1 in self.nodes and id2 in self.nodes:
            edge = Edge(self.nodes[id1], self.nodes[id2])
            self.edges[id1 + id2] = edge

    def get_cost(self, id1, id2):
        edge = self.edges.get(id1 + id2)
        return edge.cost() if edge else float('inf')

    def plot(self, ax, plot_nodes=True, plot_all_edges=False):
        if plot_nodes:
            for key, node in self.nodes.items():
                x, y = node.start
                ax.scatter(x, y, facecolors='none', edgecolors='red', s=200, linewidths=2, marker='o')
                ax.text(x + 0.04, y + 0.01, key, fontsize=12, color="black")

        if plot_all_edges:
            for key, edge in self.edges.items():
                x1, y1 = edge.node1.end
                x2, y2 = edge.node2.start
                ax.plot([x1, x2], [y1, y2], 'k-')
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                ax.text(mid_x, mid_y, f'{edge.cost:.2f}', fontsize=10, color='red')

    def plot_path(self, ax, path):
        if path:
            for i in range(len(path) - 1):
                node1, node2 = self.nodes[path[i]], self.nodes[path[i + 1]]
                x1, y1 = node1.end
                x2, y2 = node2.start
                ax.plot([x1, x2], [y1, y2], 'g--')

    def get_path_cost(self, path):
        return sum(self.get_cost(path[i], path[i + 1]) for i in range(len(path) - 1))

    def tsp_brute_force(self, start_id, end_id, pairs):
        all_nodes = set(self.nodes.keys()) - {start_id, end_id}
        reduced_nodes = [list(pair) for pair in pairs]
        combinations = product(*reduced_nodes)

        min_cost, best_path = float('inf'), None
        for comb in combinations:
            for perm in permutations(comb):
                path = [start_id] + list(perm) + [end_id]
                cost = self.get_path_cost(path)
                if cost < min_cost:
                    min_cost, best_path = cost, path
        return best_path, min_cost


def plot(grid_size, work_map, graph, path):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    ax.set_xlim((0., grid_size[0]))
    ax.set_ylim((0., grid_size[1]))
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.grid(which='minor', linestyle=':', linewidth=0.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.set_aspect('auto')
    plt.style.use('seaborn-v0_8-paper')

    plt.xlabel("x (m)", fontsize='large')
    plt.ylabel("y (m)", fontsize='large')
    plt.title("Movement problem on Kuka", fontsize='large')

    work_map.plot(ax)
    graph.plot(ax)
    graph.plot_path(ax, path)

    plt.savefig('a4_print.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    np.random.seed(42)  # Changed the seed
    params = {
        'map_size': [4, 2],
        'num_lines': 5,
        'num_connected': 1,
        'line_length': [0.2, 0.5],
        'precision': 0.1,
    }

    work_map = Map(**params)
    work_map.set_walls([[(0.5, 1.5), (1.5, 1.5)]])  # Changed obstacle location
    graph = Graph()

    graph.add_node('start', (0, 0), (0, 0))
    graph.add_node('end', (0, 0), (0, 0))

    for i, line in enumerate(work_map.lines):
        p1, p2 = line
        graph.add_node(f'{chr(65 + i)}1', p1, p2)
        graph.add_node(f'{chr(65 + i)}2', p2, p1)

    for id1 in graph.nodes:
        for id2 in graph.nodes:
            if id1 == id2: continue

            node1, node2 = graph.nodes[id1], graph.nodes[id2]
            edge_line = LineString([node1.end, node2.start])

            tests = []
            for wall in work_map.walls:
                p1, p2 = wall
                wall_line = LineString([p1, p2])
                tests.append(edge_line.intersects(wall_line))

            if np.any(tests):
                continue

            graph.add_edge(id1, id2)

    pairs = [(f'{chr(65 + i)}1', f'{chr(65 + i)}2') for i in range(len(work_map.lines))]
    best_path, min_cost = graph.tsp_brute_force('start', 'end', pairs)

    print(best_path, min_cost)
    plot(work_map.size, work_map, graph, best_path)


if __name__ == "__main__":
    main()
