import numpy as np
import random
from collections import deque
from scipy.spatial.distance import euclidean
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


# A* Algorithm
def a_star_search(grid, start, goal, heuristic_func):
    """
    A* search algorithm to find the shortest path in a grid.
    
    Args:
        grid (np.ndarray): The grid representing the space (0: free, 1: obstacle).
        start (tuple): Starting point (x, y) as integers.
        goal (tuple): Goal point (x, y) as integers.
        heuristic_func (function): Heuristic function to estimate the cost to the goal.
        
    Returns:
        list: The path from start to goal as a list of (x, y) tuples, or None if no path exists.
    """
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    
    rows, cols = grid.shape
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic_func(start, goal)}

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
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
                neighbor = (nx, ny)
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic_func(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
                    came_from[neighbor] = current

    return None

# Manhattan Distance Heuristic
def manhattan_distance(a, b):
    """
    Calculates the Manhattan distance between two points a and b.
    
    Args:
        a (tuple): First point (x, y).
        b (tuple): Second point (x, y).
        
    Returns:
        int: Manhattan distance.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])



# Kruskal's Algorithm
def compute_mst_with_kruskal(components, distance_matrix):
    """
    Computes the Minimum Spanning Tree (MST) using Kruskal's Algorithm.
    
    Args:
        components (list): List of component positions.
        distance_matrix (np.ndarray): Distance matrix representing graph edges.
        
    Returns:
        np.ndarray: Adjacency matrix of the MST.
    """
    num_components = len(components)
    edges = []

    # Collect all edges from the distance matrix
    for i in range(num_components):
        for j in range(i + 1, num_components):
            if distance_matrix[i, j] < np.inf:
                edges.append((distance_matrix[i, j], i, j))

    # Sort edges by weight
    edges.sort(key=lambda x: x[0])

    # Initialize Union-Find structure
    parent = list(range(num_components))
    rank = [0] * num_components


    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])  # Path compression
        return parent[u]

    def union(u, v):
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            if rank[root_u] > rank[root_v]:
                parent[root_v] = root_u
            elif rank[root_u] < rank[root_v]:
                parent[root_u] = root_v
            else:
                parent[root_v] = root_u
                rank[root_u] += 1

    # Build the MST
    mst_edges = []
    for weight, u, v in edges:
        if find(u) != find(v):
            union(u, v)
            mst_edges.append((u, v, weight))

    # Construct the MST adjacency matrix
    mst = np.zeros((num_components, num_components))
    for u, v, weight in mst_edges:
        mst[u, v] = mst[v, u] = weight

    return mst



## Prim's Algorithm
def compute_mst_with_prim(components, distance_matrix):
    """
    Computes the Minimum Spanning Tree (MST) using Prim's Algorithm.
    
    Args:
        components (list): List of component positions.
        distance_matrix (np.ndarray): Distance matrix representing graph edges.
        
    Returns:
        np.ndarray: Adjacency matrix of the MST.
    """
    num_components = len(components)
    mst = np.zeros((num_components, num_components))  # Initialize MST adjacency matrix
    selected = [False] * num_components  # Track visited nodes
    edge_count = 0  # Count edges added to the MST

    # Priority queue to select the minimum edge (weight, current_node, parent_node)
    min_heap = [(0, 0, -1)]  # Start with node 0

    while edge_count < num_components:
        weight, u, prev = heapq.heappop(min_heap)

        # Skip if the node is already included in MST
        if selected[u]:
            continue

        # Mark the node as selected
        selected[u] = True
        edge_count += 1

        # Add the edge to the MST if it is not the starting node
        if prev != -1:
            mst[u, prev] = mst[prev, u] = weight

        # Add all edges from the current node to the priority queue
        for v in range(num_components):
            if not selected[v] and distance_matrix[u, v] < np.inf:
                heapq.heappush(min_heap, (distance_matrix[u, v], v, u))

    return mst


def compare_mst_direct_metrics(components, grid):
    """
    Compares MST metrics (execution time and total path length) for various algorithms directly.
    """
    results = []
    num_components = len(components)

    algorithms = ["Kruskal", "Prim"]
    for algorithm in algorithms:
        start_time = time.time()  # Start time measurement
        num_components = len(components)
        distance_matrix = np.full((num_components, num_components), np.inf)

            # Calculate distances using A*
        for i in range(num_components):
            for j in range(i + 1, num_components):
                    start = tuple(map(int, components[i]))
                    goal = tuple(map(int, components[j]))
                    path = a_star_search(grid, start, goal, manhattan_distance)
                    if path:
                        distance_matrix[i, j] = len(path)
                        distance_matrix[j, i] = len(path)
        if algorithm == "Kruskal":
            mst = compute_mst_with_kruskal(components, distance_matrix)
        elif algorithm == "Prim":
            mst = compute_mst_with_prim(components, distance_matrix)

        end_time = time.time()  # End time measurement

        # Calculate total path length
        total_path_length = np.sum(mst[mst > 0])

        # Calculate execution time
        execution_time = end_time - start_time

        # Append the results
        results.append([algorithm, total_path_length, execution_time])

    # Create DataFrame for comparison
    df = pd.DataFrame(results, columns=["Algorithm", "Total Path Length", "Execution Time (s)"])
    print(tabulate(df, headers="keys", tablefmt="grid"))
    return df


# Updated MST Visualization Function
def visualize_mst_paths(grid, components, mst, algorithm):
    """
    Visualizes the MST paths generated by different algorithms on a grid.
    
    Args:
        grid (np.ndarray): The grid representing the space (0: free, 1: obstacle).
        components (list): List of component positions as (x, y) tuples.
        mst (np.ndarray): Adjacency matrix of the MST (nonzero values indicate edges).
        algorithm (str): Name of the algorithm used to compute the MST.
    """
    fig = go.Figure()

    # Add obstacles from the grid
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 1:  # Obstacle
                fig.add_trace(go.Scatter(
                    x=[y], y=[x],
                    mode='markers',
                    marker=dict(color='black', size=3),
                    showlegend=False
                ))

    # Add components as nodes (blue points)
    for i, component in enumerate(components):
        x, y = map(int, component)
        fig.add_trace(go.Scatter(
            x=[y], y=[x],
            mode='markers+text',
            marker=dict(color='blue', size=10),
            text=f"C{i + 1}",  # Label each component
            textposition="top center",
            name=f"Component {i + 1}"
        ))

    # Add MST paths
    for i in range(len(components)):
        for j in range(len(components)):
            if mst[i, j] > 0:  # Edge exists in the MST
                start = components[i]
                goal = components[j]
                path = a_star_search(grid, start, goal, manhattan_distance)
                if path:
                    path_x, path_y = zip(*path)
                    fig.add_trace(go.Scatter(
                        x=path_y, y=path_x,
                        mode='lines',
                        line=dict(color=f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})", width=2),
                        name=f"{algorithm} Path ({i + 1} to {j + 1})"
                    ))

    # Configure plot layout
    fig.update_layout(
        title=f"MST Paths Using {algorithm}",
        xaxis_title="X-axis (grid units)",
        yaxis_title="Y-axis (grid units)",
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Equal scaling for x and y axes
        yaxis=dict(scaleanchor="x", scaleratio=1),
        showlegend=True
    )
    fig.show()

def compare_time_space_complexity():
    complexities = [
        ["Kruskal", "O(E log E + E * α(V))", "O(V + E)"],
        ["Prim", "O((V + E) log V)", "O(V + E)"]
    ]
    df = pd.DataFrame(complexities, columns=["Algorithm", "Time Complexity", "Space Complexity"])
    print("\nTime and Space Complexity Comparison:")
    print(tabulate(df, headers="keys", tablefmt="grid"))
    return df

    




if __name__ == "__main__":
    kicad_file_path = "./data/raspberrrypi_hat.kicad_pcb"

    # Parse KiCad file
    components, traces = parse_kicad_file(kicad_file_path)

    # Define grid size
    GRID_SIZE = (300, 300)
    grid = create_grid_from_kicad(components, traces, GRID_SIZE)

    num_components = len(components)
    distance_matrix = np.full((num_components, num_components), np.inf)

    # Calculate distances using A*
    for i in range(num_components):
        for j in range(i + 1, num_components):
            start = tuple(map(int, components[i]))
            goal = tuple(map(int, components[j]))
            path = a_star_search(grid, start, goal, manhattan_distance)
            if path:
                distance_matrix[i, j] = len(path)
                distance_matrix[j, i] = len(path)
    


    # Visualize MST for each algorithm
    print("\nVisualizing MST for each algorithm...")

    # Kruskal
    print("\nComputing and Visualizing MST using Kruskal...")
    mst_kruskal = compute_mst_with_kruskal(components, distance_matrix)
    visualize_mst_paths(grid, components, mst_kruskal, "Kruskal")

    # Prim
    print("\nComputing and Visualizing MST using Prim...")
    mst_prim = compute_mst_with_prim(components, distance_matrix)
    visualize_mst_paths(grid, components, mst_prim, "Prim")

    # Compare Metrics
    print("\nComparing Metrics...")
    compare_mst_direct_metrics(components, grid)

    complexity_df = compare_time_space_complexity()