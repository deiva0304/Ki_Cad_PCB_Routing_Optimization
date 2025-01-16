import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree
import plotly.graph_objects as go
import time
import sexpdata
import pandas as pd
from tabulate import tabulate
import heapq

# Parsing KiCad file
def parse_kicad_file(file_path):
    with open(file_path, 'r') as file:
        pcb_data = file.read()

    pcb_expr = sexpdata.loads(pcb_data)
    components, traces = [], []

    for element in pcb_expr:
        if isinstance(element, list):
            if element[0] == sexpdata.Symbol('module'):
                components.append(parse_component(element))
            elif element[0] in [sexpdata.Symbol('gr_line'), sexpdata.Symbol('segment')]:
                traces.append(parse_trace(element))

    return components, traces

def parse_component(component_expr):
    position = None
    for item in component_expr:
        if isinstance(item, list) and item[0] == sexpdata.Symbol('at'):
            position = (float(item[1]), float(item[2]))
    return position

def parse_trace(trace_expr):
    start, end = None, None
    for item in trace_expr:
        if isinstance(item, list):
            if item[0] == sexpdata.Symbol('start'):
                start = (float(item[1]), float(item[2]))
            elif item[0] == sexpdata.Symbol('end'):
                end = (float(item[1]), float(item[2]))
    return (start, end)

# Create grid with obstacles
def create_grid_from_kicad(components, traces, grid_size):
    grid = np.zeros(grid_size, dtype=int)
    for start, end in traces:
        if start and end:
            x1, y1 = map(int, start)
            x2, y2 = map(int, end)
            for x in range(min(x1, x2), max(x1, x2) + 1):
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    grid[x, y] = 1
    return grid

class GeneticAlgorithm:
    def __init__(self, grid, start, goal, population_size=50, generations=100, mutation_rate=0.1):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.best_path = None
        self.best_cost = float('inf')

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            path = [self.start]
            current = self.start
            while current != self.goal:
                neighbors = self.get_neighbors(current)
                if not neighbors: break
                next_node = random.choice(neighbors)
                path.append(next_node)
                current = next_node
            population.append(path)
        return population

    def fitness(self, path):
        return len(path) + self.grid_penalty(path)

    def grid_penalty(self, path):
        # Penalize paths that pass through obstacles
        penalty = 0
        for x, y in path:
            if self.grid[x, y] == 1:
                penalty += 100  # Large penalty for obstacle
        return penalty

    def get_neighbors(self, current):
        x, y = current
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid.shape[0] and 0 <= ny < self.grid.shape[1] and self.grid[nx, ny] == 0:
                neighbors.append((nx, ny))
        return neighbors

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutate(self, path):
        if random.random() < self.mutation_rate:
            idx = random.randint(0, len(path) - 1)
            neighbors = self.get_neighbors(path[idx])
            if neighbors:
                path[idx] = random.choice(neighbors)
        return path

    def optimize(self):
        population = self.initialize_population()
        for generation in range(self.generations):
            population = sorted(population, key=self.fitness)
            if self.fitness(population[0]) < self.best_cost:
                self.best_cost = self.fitness(population[0])
                self.best_path = population[0]

            # Select top 50% of the population
            selected = population[:self.population_size // 2]
            next_generation = []

            # Create offspring via crossover and mutation
            while len(next_generation) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_generation.append(child)

            population = next_generation
        return self.best_path, self.best_cost
    
# Compute MST using Genetic Algorithm
def compute_mst_with_ga(components, grid, population_size=50, generations=100):
    num_components = len(components)
    distance_matrix = np.full((num_components, num_components), np.inf)

    for i in range(num_components):
        for j in range(i + 1, num_components):
            start = tuple(map(int, components[i]))
            goal = tuple(map(int, components[j]))
            ga = GeneticAlgorithm(grid, start, goal, population_size, generations)
            path, cost = ga.optimize()
            distance_matrix[i, j] = cost
            distance_matrix[j, i] = cost

    mst = minimum_spanning_tree(distance_matrix).toarray().astype(float)
    return mst

# Lee's Algorithm for shortest path
def lee_algorithm(grid, start, goal):
    rows, cols = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    distance = np.full_like(grid, -1, dtype=int)
    distance[start] = 0
    queue = deque([start])

    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0 and distance[nx, ny] == -1:
                distance[nx, ny] = distance[x, y] + 1
                queue.append((nx, ny))
                if (nx, ny) == goal:
                    break

    if distance[goal] == -1:
        return None

    path = [goal]
    x, y = goal
    while (x, y) != start:
        for dx, dy in directions:
            nx, ny = x - dx, y - dy
            if 0 <= nx < rows and 0 <= ny < cols and distance[nx, ny] == distance[x, y] - 1:
                path.append((nx, ny))
                x, y = nx, ny
                break

    path.reverse()
    return path

def compute_mst(components, grid):
    num_components = len(components)
    distance_matrix = np.full((num_components, num_components), np.inf)

    for i in range(num_components):
        for j in range(i + 1, num_components):
            start = tuple(map(int, components[i]))
            goal = tuple(map(int, components[j]))
            path = lee_algorithm(grid, start, goal)
            if path:
                distance_matrix[i, j] = len(path)
                distance_matrix[j, i] = len(path)

    mst = minimum_spanning_tree(distance_matrix).toarray().astype(float)
    return mst

def a_star_search(grid, start, goal, heuristic_func):
    rows, cols = grid.shape
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic_func(start, goal)}

    while open_list:
        current_priority, current = heapq.heappop(open_list)

        if current == goal:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        x, y = current
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
            neighbor = (x + dx, y + dy)
            nx, ny = neighbor

            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic_func(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
                    came_from[neighbor] = current

    return None  # No path found

# Manhattan Distance Heuristic
def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Compute MST using A* for pairwise paths
def compute_mst_with_a_star(components, grid):
    num_components = len(components)
    distance_matrix = np.full((num_components, num_components), np.inf)

    for i in range(num_components):
        for j in range(i + 1, num_components):
            start = tuple(map(int, components[i]))
            goal = tuple(map(int, components[j]))
            path = a_star_search(grid, start, goal, manhattan_distance)
            if path:
                distance_matrix[i, j] = len(path)
                distance_matrix[j, i] = len(path)

    mst = minimum_spanning_tree(distance_matrix).toarray().astype(float)
    return mst

# Simulated Annealing for pathfinding
def simulated_annealing_routing(grid, start, goal, max_iterations=1000, initial_temperature=100):
    current_path = [start]
    current_cost = compute_cost(grid, current_path, goal)
    temperature = initial_temperature

    for iteration in range(max_iterations):
        if current_cost == 0:
            break  # Path found

        # Generate a neighbor (random movement)
        new_path = current_path[:]
        if new_path[-1] != goal:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            x, y = new_path[-1]
            dx, dy = random.choice(directions)
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] == 0:
                new_path.append((nx, ny))

        # Compute the cost of the new path
        new_cost = compute_cost(grid, new_path, goal)

        # Decide whether to accept the new path
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or random.uniform(0, 1) < np.exp(-delta_cost / temperature):
            current_path, current_cost = new_path, new_cost

        # Decrease the temperature
        temperature *= 0.99

    # Validate final path
    if current_path[-1] == goal:
        return current_path
    else:
        return None


# Cost function for SA
def compute_cost(grid, path, goal):
    if path[-1] == goal:
        return 0
    x, y = path[-1]
    gx, gy = goal
    return abs(x - gx) + abs(y - gy)  # Manhattan distance to goal


# Compute MST using Simulated Annealing for pairwise distances
def compute_mst_with_sa(components, grid):
    num_components = len(components)
    distance_matrix = np.full((num_components, num_components), np.inf)

    for i in range(num_components):
        for j in range(i + 1, num_components):
            start = tuple(map(int, components[i]))
            goal = tuple(map(int, components[j]))
            path = simulated_annealing_routing(grid, start, goal)
            if path:
                distance_matrix[i, j] = len(path)
                distance_matrix[j, i] = len(path)

    mst = minimum_spanning_tree(distance_matrix).toarray().astype(float)
    return mst




def compute_path_lengths(components, grid, algorithm):
    path_lengths = []
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            start = tuple(map(int, components[i]))
            goal = tuple(map(int, components[j]))
            if algorithm == "Lee":
                path = lee_algorithm(grid, start, goal)
            elif algorithm == "GA":
                ga = GeneticAlgorithm(grid, start, goal, population_size=50, generations=100)
                path, _ = ga.optimize()
            elif algorithm == "A*":
                path = a_star_search(grid, start, goal, manhattan_distance)
            elif algorithm == "SA":
                path = simulated_annealing_routing(grid, start, goal)
            else:
                raise ValueError("Unknown algorithm!")
            if path:
                path_lengths.append(len(path))
    return path_lengths

def measure_execution_time(components, grid, algorithm):
    import time
    start_time = time.time()
    compute_path_lengths(components, grid, algorithm)
    return time.time() - start_time

# MST
def compute_mst_with_algorithms(components, grid, algorithm):
    if algorithm == "Lee":
        mst = compute_mst(components, grid)
    elif algorithm == "GA":
        mst = compute_mst_with_ga(components, grid)
    elif algorithm == "A*":
        mst = compute_mst_with_a_star(components, grid)
    elif algorithm == "SA":
        mst = compute_mst_with_sa(components, grid)
    else:
        raise ValueError("Unknown algorithm!")
    return mst


def compare_mst_metrics(components, grid):
    results = []

    for algorithm in ["Lee", "GA", "A*", "SA"]:
        # Compute MST
        mst = compute_mst_with_algorithms(components, grid, algorithm)

        # Calculate path lengths
        total_path_length = 0
        execution_time = measure_execution_time(components, grid, algorithm)

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                if mst[i, j] > 0:
                    start = tuple(map(int, components[i]))
                    goal = tuple(map(int, components[j]))

                    if algorithm == "Lee":
                        path = lee_algorithm(grid, start, goal)
                    elif algorithm == "GA":
                        ga = GeneticAlgorithm(grid, start, goal, population_size=50, generations=100)
                        path, _ = ga.optimize()
                    elif algorithm == "A*":
                        path = a_star_search(grid, start, goal, manhattan_distance)
                    elif algorithm == "SA":
                        path = simulated_annealing_routing(grid, start, goal)

                    if path:
                        total_path_length += len(path)

        results.append([algorithm, total_path_length, execution_time])

    # Create DataFrame for comparison
    df = pd.DataFrame(results, columns=["Algorithm", "Total Path Length", "Execution Time (s)"])
    print(tabulate(df, headers="keys", tablefmt="grid"))
    return df


def visualize_mst_paths(grid, components, mst, algorithm):
    fig = go.Figure()

    # Add grid as background
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 1:
                fig.add_trace(go.Scatter(
                    x=[y], y=[x],
                    mode='markers',
                    marker=dict(color='black', size=3),
                    showlegend=False
                ))

    # Add components
    for i, component in enumerate(components):
        x, y = component
        fig.add_trace(go.Scatter(
            x=[y], y=[x],
            mode='markers+text',
            marker=dict(color='blue', size=10),
            text=f"C{i + 1}",
            textposition="top center",
            name=f"Component {i + 1}"
        ))

    # Add MST paths
    for i, start in enumerate(components):
        for j, weight in enumerate(mst[i]):
            if weight > 0:
                start_pos = tuple(map(int, start))
                goal_pos = tuple(map(int, components[j]))

                if algorithm == "Lee":
                    path = lee_algorithm(grid, start_pos, goal_pos)
                elif algorithm == "GA":
                    ga = GeneticAlgorithm(grid, start_pos, goal_pos, population_size=50, generations=100)
                    path, _ = ga.optimize()
                elif algorithm == "A*":
                    path = a_star_search(grid, start_pos, goal_pos, manhattan_distance)
                elif algorithm == "SA":
                    path = simulated_annealing_routing(grid, start_pos, goal_pos)

                if path:
                    path_x, path_y = zip(*path)
                    fig.add_trace(go.Scatter(
                        x=path_y, y=path_x,
                        mode='lines',
                        line=dict(width=2),
                        name=f"{algorithm} Path ({i + 1} to {j + 1})"
                    ))

    fig.update_layout(
        title=f"MST Paths Using {algorithm}",
        xaxis_title="X-axis (grid units)",
        yaxis_title="Y-axis (grid units)",
        showlegend=True
    )
    fig.show()


if __name__ == "__main__":
    kicad_file_path = "./data/Arduino_As_Uno.kicad_pcb"

    # Parse KiCad file
    components, traces = parse_kicad_file(kicad_file_path)

    # Print parsed components
    print("Parsed Components:")
    for i, component in enumerate(components):
        print(f"Component {i + 1}: Position {component}")

    # Print parsed traces
    print("\nParsed Traces:")
    for i, trace in enumerate(traces):
        print(f"Trace {i + 1}: Start {trace[0]}, End {trace[1]}")

    # Define grid size
    GRID_SIZE = (200, 200)

    # Create grid
    grid = create_grid_from_kicad(components, traces, GRID_SIZE)

    # Print grid
    print("\nGrid (with obstacles):")
    print(grid)

    # Compute and visualize MST using Lee's Algorithm
    print("\nComputing MST using Lee's Algorithm...")
    mst_lee = compute_mst_with_algorithms(components, grid, "Lee")
    visualize_mst_paths(grid, components, mst_lee, "Lee")

    # Compute and visualize MST using A*
    print("\nComputing MST using A* Algorithm...")
    mst_a_star = compute_mst_with_algorithms(components, grid, "A*")
    visualize_mst_paths(grid, components, mst_a_star, "A*")

    # Compute and visualize MST using SA
    print("\nComputing MST using SA Algorithm...")
    mst_sa = compute_mst_with_algorithms(components, grid, "SA")
    visualize_mst_paths(grid, components, mst_a_star, "SA")

    # Compute and visualize MST using Genetic Algorithm
    print("\nComputing MST using Genetic Algorithm...")
    mst_ga = compute_mst_with_algorithms(components, grid, "GA")
    visualize_mst_paths(grid, components, mst_ga, "GA")

    # Compare Metrics
    print("\nComparing Metrics...")
    compare_mst_metrics(components, grid)
