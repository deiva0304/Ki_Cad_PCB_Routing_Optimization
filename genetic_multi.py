import numpy as np
import random
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree
import sexpdata
import plotly.graph_objects as go

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

# Genetic Algorithm for Pathfinding
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

# Visualize the routing
def visualize_multi_routing_interactive_ga(grid, mst, components):
    fig = go.Figure()

    # Initialize bounds
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Add grid as background (grayscale for obstacles)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 1:  # Obstacle
                fig.add_trace(go.Scatter(
                    x=[y], y=[x],
                    mode='markers',
                    marker=dict(color='black', size=5),  # Increase obstacle size
                    showlegend=False
                ))
                # Update bounds for obstacles
                min_x, min_y = min(min_x, y), min(min_y, x)
                max_x, max_y = max(max_x, y), max(max_y, x)

    # Add components as blue points and label them
    for i, component in enumerate(components):
        x, y = component
        fig.add_trace(go.Scatter(
            x=[y], y=[x],
            mode='markers+text',
            marker=dict(color='blue', size=10),
            text=f"C{i + 1}",  # Label the component
            textposition="top center",
            name=f"Component {i + 1}"
        ))
        # Update bounds for components
        min_x, min_y = min(min_x, y), min(min_y, x)
        max_x, max_y = max(max_x, y), max(max_y, x)

    # Add MST paths with unique colors
    for i, start in enumerate(components):
        for j, weight in enumerate(mst[i]):
            if weight > 0:
                ga = GeneticAlgorithm(grid, tuple(map(int, start)), tuple(map(int, components[j])), population_size=50, generations=100)
                path, _ = ga.optimize()
                path_x, path_y = zip(*path)
                color = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"  # Random color
                fig.add_trace(go.Scatter(
                    x=path_y, y=path_x,
                    mode='lines',
                    line=dict(color=color, width=4),  # Increase path width
                    name=f"Path {i + 1} to {j + 1}"
                ))
                # Update bounds for paths
                min_x, min_y = min(min_x, min(path_y)), min(min_y, min(path_x))
                max_x, max_y = max(max_x, max(path_y)), max(max_y, max(path_x))

    # Add padding to the bounds
    padding = 10
    min_x, min_y = max(0, min_x - padding), max(0, min_y - padding)
    max_x, max_y = min(grid.shape[1], max_x + padding), min(grid.shape[0], max_y + padding)

    # Configure layout with dynamically calculated axis ranges
    fig.update_layout(
        title="Interactive Multi-Component Routing with Genetic Algorithm",
        xaxis=dict(
            title="X-axis (grid units)",
            range=[min_x, max_x],  # Focused X-axis range
            scaleanchor="y"
        ),
        yaxis=dict(
            title="Y-axis (grid units)",
            range=[min_y, max_y],  # Focused Y-axis range
            scaleanchor="x"
        ),
        showlegend=True
    )

    fig.show()


# Main section with GA
if __name__ == "__main__":
    kicad_file_path = "./data/Arduino_Nano.kicad_pcb"

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

    # Compute MST with GA
    if len(components) > 1:
        print("\nComputing MST with GA...")
        mst = compute_mst_with_ga(components, grid)

        # Print MST matrix
        print("\nMST Matrix (Distance Matrix):")
        print(mst)

        # Visualize routing
        visualize_multi_routing_interactive_ga(grid, mst, components)
    else:
        print("Not enough components to route.")
