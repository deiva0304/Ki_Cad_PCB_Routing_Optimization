import heapq
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
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

# A* Search Algorithm
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

# Visualize the routing
def visualize_multi_routing_interactive_a_star(grid, mst, components):
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
                    marker=dict(color='black', size=5),  # Increase obstacle size for better visibility
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
                path = a_star_search(grid, tuple(map(int, start)), tuple(map(int, components[j])), manhattan_distance)
                if path:
                    path_x, path_y = zip(*path)
                    color = f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})"  # Random color
                    fig.add_trace(go.Scatter(
                        x=path_y, y=path_x,
                        mode='lines',
                        line=dict(color=color, width=4),  # Increase path width for better visibility
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
        title="Interactive Multi-Component Routing with A* Algorithm",
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


# Main section with A*
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

    # Compute MST with A*
    if len(components) > 1:
        print("\nComputing MST with A*...")
        mst = compute_mst_with_a_star(components, grid)

        # Print MST matrix
        print("\nMST Matrix (Distance Matrix):")
        print(mst)

        # Visualize routing
        visualize_multi_routing_interactive_a_star(grid, mst, components)
    else:
        print("Not enough components to route.")
