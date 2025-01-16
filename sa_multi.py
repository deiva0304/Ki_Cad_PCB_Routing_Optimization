import numpy as np
from scipy.spatial import distance
from scipy.sparse.csgraph import minimum_spanning_tree
import sexpdata
import plotly.graph_objects as go
import random


# Parsing KiCad file (same as before)
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
def compute_mst(components, grid):
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


# Visualize the routing with Simulated Annealing paths
def visualize_multi_routing_interactive_colored(grid, mst, components):
    fig = go.Figure()

    # Add grid as background (grayscale for obstacles)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 1:  # Obstacle
                fig.add_trace(go.Scatter(
                    x=[y], y=[x],
                    mode='markers',
                    marker=dict(color='black', size=2),
                    showlegend=False
                ))

    # Add components as blue points
    for i, component in enumerate(components):
        x, y = component
        fig.add_trace(go.Scatter(
            x=[y], y=[x],
            mode='markers',
            marker=dict(color='blue', size=10),
            name=f"Component {i + 1}"
        ))

    # Add MST paths with unique colors
    for i, start in enumerate(components):
        for j, weight in enumerate(mst[i]):
            if weight > 0:
                path = simulated_annealing_routing(grid, tuple(map(int, start)), tuple(map(int, components[j])))
                if path:
                    path_x, path_y = zip(*path)
                    color = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"  # Random color
                    fig.add_trace(go.Scatter(
                        x=path_y, y=path_x,
                        mode='lines',
                        line=dict(color=color, width=2),
                        name=f"Path {i + 1} to {j + 1}"
                    ))

    # Configure layout with updated axis ranges
    fig.update_layout(
        title="Interactive Multi-Component Routing with Simulated Annealing",
        xaxis=dict(
            title="X-axis (grid units)",
            range=[0, grid.shape[1]],
            scaleanchor="y"
        ),
        yaxis=dict(
            title="Y-axis (grid units)",
            range=[0, grid.shape[0]],
            scaleanchor="x"
        ),
        showlegend=True
    )

    fig.show()


if __name__ == "__main__":
    kicad_file_path = "./data/Arduino_Nano.kicad_pcb"

    # Parse KiCad file
    components, traces = parse_kicad_file(kicad_file_path)

    # Print parsed components
    print("Parsed Components:")
    for i, component in enumerate(components):
        print(f"Component {i + 1}: Position {component}")

    # Define grid size
    GRID_SIZE = (200, 200)

    # Create grid
    grid = create_grid_from_kicad(components, traces, GRID_SIZE)

    # Compute MST
    if len(components) > 1:
        print("\nComputing MST with Simulated Annealing...")
        mst = compute_mst(components, grid)

        # Visualize the routing
        visualize_multi_routing_interactive_colored(grid, mst, components)
    else:
        print("Not enough components to route.")
