import numpy as np
import heapq
import sexpdata
import plotly.graph_objects as go
import random


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

# A* Search Algorithm for obstacle-aware pathfinding
def a_star_search(grid, start, goal):
    rows, cols = grid.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        x, y = current
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
                neighbor = (nx, ny)
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + abs(nx - goal[0]) + abs(ny - goal[1])
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
                    came_from[neighbor] = current

    return None  # No valid path found

# Kruskal's Algorithm with A* for MST
def compute_mst_with_kruskal(grid, components):
    num_components = len(components)
    edges = []

    # Calculate all pairwise paths and their lengths
    for i in range(num_components):
        for j in range(i + 1, num_components):
            start = tuple(map(int, components[i]))
            goal = tuple(map(int, components[j]))
            path = a_star_search(grid, start, goal)
            if path:
                edges.append((len(path), i, j, path))  # (weight, start, end, path)

    # Sort edges by weight
    edges.sort()

    # Kruskal's MST construction
    mst_edges = []
    parent = list(range(num_components))  # Union-Find parent array

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            parent[root2] = root1

    for weight, start, end, path in edges:
        if find(start) != find(end):
            union(start, end)
            mst_edges.append((start, end, path))
            if len(mst_edges) == num_components - 1:
                break

    return mst_edges

# Visualize the routing
def visualize_multi_routing_interactive_colored(grid, mst_edges, components):
    fig = go.Figure()

    # Add grid as background (grayscale for obstacles)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 1:
                fig.add_trace(go.Scatter(
                    x=[y], y=[x],
                    mode='markers',
                    marker=dict(color='black', size=5),
                    showlegend=False
                ))

    # Add components as blue points
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
    for start, end, path in mst_edges:
        path_x, path_y = zip(*path)
        fig.add_trace(go.Scatter(
            x=path_y, y=path_x,
            mode='lines',
            line=dict(color=f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})", width=4),
            name=f"Path {start + 1} to {end + 1}"
        ))

    # Configure layout
    fig.update_layout(
        title="Obstacle-Aware MST Routing with Kruskal's Algorithm and A*",
        xaxis=dict(title="X-axis"),
        yaxis=dict(title="Y-axis"),
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

    # Create grid
    GRID_SIZE = (200, 200)
    grid = create_grid_from_kicad(components, traces, GRID_SIZE)

    # Compute MST with Kruskal's Algorithm
    if len(components) > 1:
        print("\nComputing MST with Kruskal's Algorithm and A*...")
        mst_edges = compute_mst_with_kruskal(grid, components)

        # Visualize the routing
        visualize_multi_routing_interactive_colored(grid, mst_edges, components)
    else:
        print("Not enough components to route.")
