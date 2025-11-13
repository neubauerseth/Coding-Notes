"""
Dijkstra's Algorithm
===================

Used for: Shortest path in weighted graphs with non-negative weights

Time Complexity: O((V + E) log V) with min-heap
Space Complexity: O(V) for distances and heap
"""

import heapq
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
import math

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()
    
    def add_edge(self, u, v, weight, directed=True):
        """Add weighted edge to graph"""
        self.graph[u].append((v, weight))
        self.vertices.add(u)
        self.vertices.add(v)
        if not directed:
            self.graph[v].append((u, weight))

# ==================== BRUTE FORCE APPROACH ====================
def dijkstra_brute_force(graph: Dict[int, List[Tuple[int, int]]], start: int) -> Dict[int, int]:
    """
    Brute Force Dijkstra: Without using heap optimization
    
    Time Complexity: O(V²) - selecting minimum distance vertex takes O(V)
    Space Complexity: O(V) - for distance array
    
    Problems:
    - O(V²) complexity even for sparse graphs
    - Inefficient for large graphs
    - Manual minimum finding
    """
    # Get all vertices
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        for neighbor, _ in neighbors:
            all_vertices.add(neighbor)
    all_vertices.add(start)
    
    # Initialize distances
    distances = {v: float('inf') for v in all_vertices}
    distances[start] = 0
    visited = set()
    
    for _ in range(len(all_vertices)):
        # Find unvisited vertex with minimum distance (O(V) operation)
        min_vertex = None
        min_distance = float('inf')
        
        for vertex in all_vertices:
            if vertex not in visited and distances[vertex] < min_distance:
                min_distance = distances[vertex]
                min_vertex = vertex
        
        if min_vertex is None:
            break
        
        visited.add(min_vertex)
        
        # Update distances to neighbors
        for neighbor, weight in graph.get(min_vertex, []):
            if neighbor not in visited:
                new_distance = distances[min_vertex] + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
    
    return distances

# ==================== OPTIMIZED APPROACH ====================
def dijkstra_optimized(graph: Dict[int, List[Tuple[int, int]]], start: int) -> Dict[int, int]:
    """
    Optimized Dijkstra using min-heap (priority queue)
    
    Time Complexity: O((V + E) log V)
    Space Complexity: O(V)
    
    Advantages:
    - Much faster for sparse graphs
    - Efficient minimum extraction with heap
    - Scalable to large graphs
    """
    # Get all vertices
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        for neighbor, _ in neighbors:
            all_vertices.add(neighbor)
    all_vertices.add(start)
    
    # Initialize distances
    distances = {v: float('inf') for v in all_vertices}
    distances[start] = 0
    
    # Priority queue: (distance, vertex)
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, vertex = heapq.heappop(pq)
        
        if vertex in visited:
            continue
        
        visited.add(vertex)
        
        # Update distances to neighbors
        for neighbor, weight in graph.get(vertex, []):
            if neighbor not in visited:
                new_distance = current_dist + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(pq, (new_distance, neighbor))
    
    return distances

def dijkstra_with_path(graph: Dict[int, List[Tuple[int, int]]], start: int, end: int) -> Tuple[int, List[int]]:
    """
    Dijkstra with path reconstruction
    
    Time Complexity: O((V + E) log V)
    Space Complexity: O(V)
    """
    # Get all vertices
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        for neighbor, _ in neighbors:
            all_vertices.add(neighbor)
    all_vertices.update([start, end])
    
    # Initialize
    distances = {v: float('inf') for v in all_vertices}
    distances[start] = 0
    previous = {v: None for v in all_vertices}
    
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, vertex = heapq.heappop(pq)
        
        if vertex == end:
            break
        
        if vertex in visited:
            continue
        
        visited.add(vertex)
        
        for neighbor, weight in graph.get(vertex, []):
            if neighbor not in visited:
                new_distance = current_dist + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = vertex
                    heapq.heappush(pq, (new_distance, neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    
    path.reverse()
    
    if distances[end] == float('inf'):
        return float('inf'), []
    
    return distances[end], path

def dijkstra_all_pairs(graph: Dict[int, List[Tuple[int, int]]]) -> Dict[Tuple[int, int], int]:
    """
    All-pairs shortest path using Dijkstra from each vertex
    
    Time Complexity: O(V * (V + E) log V)
    Space Complexity: O(V²)
    """
    # Get all vertices
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        for neighbor, _ in neighbors:
            all_vertices.add(neighbor)
    
    all_distances = {}
    
    for start in all_vertices:
        distances = dijkstra_optimized(graph, start)
        for end in all_vertices:
            all_distances[(start, end)] = distances[end]
    
    return all_distances

def dijkstra_k_shortest_paths(graph: Dict[int, List[Tuple[int, int]]], start: int, end: int, k: int) -> List[Tuple[int, List[int]]]:
    """
    Find k shortest paths using modified Dijkstra
    
    Time Complexity: O(k * (V + E) log V)
    Space Complexity: O(k * V)
    """
    import heapq
    
    # Priority queue: (distance, path)
    pq = [(0, [start])]
    paths_found = []
    visited_paths = set()
    
    while pq and len(paths_found) < k:
        current_dist, path = heapq.heappop(pq)
        current_vertex = path[-1]
        
        # Convert path to tuple for hashing
        path_tuple = tuple(path)
        if path_tuple in visited_paths:
            continue
        visited_paths.add(path_tuple)
        
        if current_vertex == end:
            paths_found.append((current_dist, path))
            continue
        
        # Explore neighbors
        for neighbor, weight in graph.get(current_vertex, []):
            if neighbor not in path:  # Avoid cycles
                new_path = path + [neighbor]
                new_dist = current_dist + weight
                heapq.heappush(pq, (new_dist, new_path))
    
    return paths_found

def dijkstra_with_constraints(graph: Dict[int, List[Tuple[int, int]]], start: int, end: int, 
                            max_distance: int) -> Optional[List[int]]:
    """
    Dijkstra with distance constraint
    
    Time Complexity: O((V + E) log V)
    Space Complexity: O(V)
    """
    distances = {start: 0}
    previous = {start: None}
    pq = [(0, start)]
    visited = set()
    
    while pq:
        current_dist, vertex = heapq.heappop(pq)
        
        if vertex == end and current_dist <= max_distance:
            # Reconstruct path
            path = []
            current = end
            while current is not None:
                path.append(current)
                current = previous[current]
            return path[::-1]
        
        if vertex in visited or current_dist > max_distance:
            continue
        
        visited.add(vertex)
        
        for neighbor, weight in graph.get(vertex, []):
            new_distance = current_dist + weight
            if (neighbor not in visited and new_distance <= max_distance and
                (neighbor not in distances or new_distance < distances[neighbor])):
                distances[neighbor] = new_distance
                previous[neighbor] = vertex
                heapq.heappush(pq, (new_distance, neighbor))
    
    return None

# ==================== GRID DIJKSTRA ====================
def dijkstra_grid(grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> int:
    """
    Dijkstra on 2D grid where each cell has a cost
    
    Time Complexity: O(V log V) where V = rows * cols
    Space Complexity: O(V)
    """
    if not grid or not grid[0]:
        return -1
    
    rows, cols = len(grid), len(grid[0])
    if not (0 <= start[0] < rows and 0 <= start[1] < cols and
            0 <= end[0] < rows and 0 <= end[1] < cols):
        return -1
    
    # Initialize distances
    distances = {}
    for r in range(rows):
        for c in range(cols):
            distances[(r, c)] = float('inf')
    
    distances[start] = grid[start[0]][start[1]]
    pq = [(grid[start[0]][start[1]], start)]
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while pq:
        current_dist, (row, col) = heapq.heappop(pq)
        
        if (row, col) == end:
            return current_dist
        
        if current_dist > distances[(row, col)]:
            continue
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if 0 <= new_row < rows and 0 <= new_col < cols:
                new_distance = current_dist + grid[new_row][new_col]
                
                if new_distance < distances[(new_row, new_col)]:
                    distances[(new_row, new_col)] = new_distance
                    heapq.heappush(pq, (new_distance, (new_row, new_col)))
    
    return -1 if distances[end] == float('inf') else distances[end]

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    # Create sample weighted graph
    graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('C', 1), ('D', 5)],
        'C': [('D', 8), ('E', 10)],
        'D': [('E', 2)],
        'E': []
    }
    
    print("Weighted Graph:")
    for vertex, edges in graph.items():
        print(f"  {vertex} -> {edges}")
    
    print("\n--- Distance Calculations ---")
    print("Brute Force:", dijkstra_brute_force(graph, 'A'))
    print("Optimized:", dijkstra_optimized(graph, 'A'))
    
    print("\n--- Path Finding ---")
    distance, path = dijkstra_with_path(graph, 'A', 'E')
    print(f"Shortest path A to E: {path} (distance: {distance})")
    
    print("\n--- K Shortest Paths ---")
    k_paths = dijkstra_k_shortest_paths(graph, 'A', 'E', 3)
    for i, (dist, path) in enumerate(k_paths, 1):
        print(f"Path {i}: {path} (distance: {dist})")
    
    print("\n--- Grid Dijkstra ---")
    # Grid where each cell has a cost
    cost_grid = [
        [1, 3, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 9, 1]
    ]
    
    print("Cost Grid:")
    for row in cost_grid:
        print(row)
    
    min_cost = dijkstra_grid(cost_grid, (0, 0), (3, 3))
    print(f"Minimum cost path from (0,0) to (3,3): {min_cost}")
    
    print("\n--- Network Routing Example ---")
    # Network with routers and latencies
    network = {
        'Router1': [('Router2', 10), ('Router3', 15)],
        'Router2': [('Router4', 12), ('Router5', 15)],
        'Router3': [('Router5', 10)],
        'Router4': [('Router6', 2)],
        'Router5': [('Router6', 5)],
        'Router6': []
    }
    
    print("Network topology:")
    for router, connections in network.items():
        print(f"  {router} -> {connections}")
    
    distances = dijkstra_optimized(network, 'Router1')
    print(f"\nLatencies from Router1:")
    for router, latency in distances.items():
        if latency != float('inf'):
            print(f"  To {router}: {latency}ms")

"""
DIJKSTRA'S ALGORITHM PATTERNS:

1. Single Source Shortest Path:
   - GPS navigation systems
   - Network routing protocols
   - Social network analysis

2. All-Pairs Shortest Path:
   - Distance matrices
   - Transportation planning
   - Game pathfinding precomputation

3. Constrained Shortest Path:
   - Budget-constrained travel
   - Resource-limited routing
   - Time-constrained delivery

WHEN TO USE DIJKSTRA:
- Weighted graphs with non-negative weights
- Need shortest path between vertices
- Network routing and navigation
- Resource allocation problems

WHEN NOT TO USE DIJKSTRA:
- Negative edge weights (use Bellman-Ford)
- Unweighted graphs (use BFS instead)
- Need all shortest paths between all pairs (use Floyd-Warshall)
- Very dense graphs (consider Floyd-Warshall)

OPTIMIZATIONS:
- Bidirectional search for single target
- A* algorithm with heuristics
- Early termination when target found
- Fibonacci heap for better complexity (rarely needed in practice)

COMPLEXITY COMPARISON:
- Brute Force: O(V²) - good for dense graphs
- Min-Heap: O((V + E) log V) - good for sparse graphs
- Fibonacci Heap: O(E + V log V) - theoretical improvement
"""