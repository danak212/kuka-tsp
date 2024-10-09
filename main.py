import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import random

from itertools import product, permutations

class Map:
    def __init__(self, **params):

        self.size = params['map_size']
        self.precision = params['precision']

        self.lines = self.generate_lines(params['num_lines'], params['num_connected'], params['line_length'])

        self.walls = None
    ### Point, line UTILS

    def set_walls(self, walls):
        self.walls = walls
        
    def length(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def random(self, min, max):
        x = np.random.uniform(min, max)
        return np.round(x / self.precision) * self.precision
    
    def random_point(self):
        max_x, max_y = self.size
        x, y = self.random(0, max_x), self.random(0, max_y)
        return (x, y)
    
    def generate_lines(self, num_lines, num_connected, line_length):
        assert num_lines > num_connected*2
        min_length, max_length = line_length

        lines = []
        # Generate connected lines
        for i in range(num_connected):
            # Generate first line randomly
            p1, p2 = self.random_point(), self.random_point()

            # Make sure line length has correct length
            while self.length(p1, p2) > max_length or self.length(p1, p2) < min_length:
                p2 = self.random_point()
            lines.append((p1, p2))
            
            # Generate a line connected with the previous
            rand_index = np.random.randint(0, 2)
            shared_point = (p1, p2)[rand_index]
            second_point = self.random_point()
            
            # Make sure line length is correct
            while self.length(shared_point, second_point) > max_length or self.length(shared_point, second_point) < min_length:
                second_point = self.random_point()
            lines.append((shared_point, second_point))
            num_lines -= 2
        
        # Generate standard lines
        for i in range(num_lines):
            p1 = self.random_point()
            p2 = self.random_point()
            
            dist = np.inf
            while dist > max_length or dist < min_length:
                p2 = self.random_point()
                dist = self.length(p1, p2)
            lines.append((p1, p2))

        lines = np.array(lines, dtype=float)
        return  np.reshape(lines, (-1, 2, 2)) # Update lines

    def plot(self, ax):

        for i, line in enumerate(self.lines):
            if i == 0:   
                ax.plot(line[:, 0], line[:, 1], 'black', linewidth=1.5, label="working_path")
            else:
                ax.plot(line[:, 0], line[:, 1], 'black', linewidth=1.5)

            ax.scatter(line[:, 0], line[:, 1], c='black')

        if self.walls is not None:
            for wall in self.walls:
                ax.plot(wall[:, 0], wall[:, 1], 'red', linewidth=3, label="obstacle")

class Node:
    def __init__(self, id, start, end):
        self.id = id
        self.start = start
        self.end = end

    def get_start(self):
        x, y = self.start
        return (x.ravel(), y.ravel())
    
    def get_end(self):
        x, y = self.end
        return (x.ravel(), y.ravel())
    
    def __repr__(self):
        return f"Node({self.id}: ({self.start}, {self.end}))"
    
class Edge:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        self.cost = self.calculate_cost()

    def calculate_cost(self):
        cost = np.linalg.norm(np.array(self.node1.get_end()) - np.array(self.node2.get_start()))
        return cost

class Graph:
    def __init__(self):
        self.nodes= {}
        self.edges = []

    def add_node(self, id, start, end):
        if id not in self.nodes:
            node = Node(id, start, end)
            self.nodes[id] = node

    def add_edge(self, id1, id2):
        if id1 in self.nodes and id2 in self.nodes:
            node1 = self.nodes[id1]
            node2 = self.nodes[id2]
            edge = Edge(node1, node2)
            self.edges.append(edge)
    
    def get_node(self, id):
        return self.nodes[id]
    
    def plot(self, ax, plot_nodes=True, plot_all_edges=False):
        # Plot nodes
        if plot_nodes:
            for key, node in self.nodes.items():
                x, y = node.get_start()
                ax.scatter(x, y, facecolors='none', edgecolors='red', s=200, linewidths=2, marker='o')
                ax.text(x + 0.04, y + 0.01, key, fontsize=12, color="black")
        
        # Plot edges
        if plot_all_edges:
            for edge in self.edges:
                x1, y1 = edge.node1.get_end()
                x2, y2 = edge.node2.get_start()
                ax.plot([x1, x2], [y1, y2], 'k-')

                mid_x = x1 + x2 / 2
                mid_y = y1 + y2 / 2
                ax.text(mid_x, mid_y, f'{edge.cost:.2f}', fontsize=10, color='red')

    def plot_path(self, ax, path):
            for i in range(len(path) - 1):
                node1 = self.nodes[path[i]]
                node2 = self.nodes[path[i+1]]

                x1, y1 = node1.get_end()
                x2, y2 = node2.get_start()
                ax.plot([x1, x2], [y1, y2], 'g--')

    def get_cost(self, path):
        total_cost = 0
        for i in range(len(path) - 1):
            node1 = self.nodes[path[i]]
            node2 = self.nodes[path[i+1]]
            total_cost += np.linalg.norm(np.array(node1.get_end()) - np.array(node2.get_start()))
        return total_cost

    def tsp_brute_force(self, start_id, end_id, pairs):
        all_nodes = set(self.nodes.keys()) - {start_id, end_id}

        # Generate permuations while selecitng only one from each pair
        reduced_nodes = []
        for pair in pairs:
            pair_nodes = list(pair)
            reduced_nodes.append(pair_nodes)
        
        combinations = product(*reduced_nodes)

        min_cost = float('inf')
        best_path = None
        for comb in combinations:
            comb = list(comb)
            for perm in permutations(comb):
                path = [start_id] + list(perm) + [end_id]
                cost = self.get_cost(path)
                if cost < min_cost:
                    min_cost = cost
                    best_path = path
        return best_path, min_cost

    
def plot(grid_size, map, graph, path):
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Adjust the subplot parameters

    # Style
    max_x, max_y = grid_size
    ax.set_xlim((0, max_x))
    ax.set_ylim((0, max_y))

    ax.grid()   
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))  
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    ax.grid(which='minor', linestyle=':', linewidth='0.5')
    ax.set_aspect('auto')
    plt.style.use('seaborn-v0_8-paper')  # Choose a style suitable for printing


    # Labels
    plt.xlabel("x (m)", fontsize='large')
    plt.ylabel("y (m)", fontsize='large')
    plt.title("TSP Problem on Kuka", fontsize='large')

    plt.tight_layout()
    plt.legend(fontsize='large')
    

    # Plot map
    map.plot(ax)

    # Plot graph
    graph.plot(ax)
    graph.plot_path(ax, path)

    plt.savefig('a4_print.png', dpi=300, bbox_inches='tight')  # Save as PNG with high resolution
    plt.show()

# TSP




def main():
    np.random.seed(42)
    params = {
        'map_size': [4, 2],
        'num_lines': 6,
        'num_connected': 1,
        'line_length': [0.2, 0.5],
        'precision': 0.1,
    }

    work_map = Map(**params) 
    work_map.set_walls(np.reshape([[0, 1], [0.5, 1]], (-1, 2, 2)))

    lines = work_map.lines
    graph = Graph()

    # Add start and end nodes:
    graph.add_node(f'start', np.zeros((2, 1)), np.zeros((2, 1)))
    graph.add_node(f'end', np.zeros((2, 1)), np.zeros((2, 1)))

    # Add nodes
    for i, line in enumerate(lines):
        # Add node for each line in both directions
        p1, p2 = line
        graph.add_node(f'p_{i}a', p1, p2)
        graph.add_node(f'p_{i}b', p2, p1)

    # Add edges
    for id1, node1 in graph.nodes.items():
        for id2, node2 in graph.nodes.items():
            if id1 == id2:
                continue
            graph.add_edge(id1, id2)
        
    
    # TSP
    print("Starting TSP")
    pairs = [(f'p_{i}a', f'p_{i}b') for i, _ in enumerate(lines)]
    best_path, min_cost = graph.tsp_brute_force('start', 'end', pairs)
    print(best_path, min_cost)

    plot(work_map.size, work_map, graph, best_path)

if __name__ == "__main__":
    main()