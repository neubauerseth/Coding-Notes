"""
Breadth-First Search (BFS) Algorithm
====================================

Used for: Shortest path (unweighted), level-order traversal, minimum spanning tree

Time Complexity: O(V + E) where V = vertices, E = edges
Space Complexity: O(V) for queue and visited set
"""

from collections import deque, defaultdict
from typing import List, Set, Dict, Optional, Tuple

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()
    
    def add_edge(self, u, v, directed=True):
        """Add edge to graph"""
        self.graph[u].append(v)
        self.vertices.add(u)
        self.vertices.add(v)
        if not directed:
            self.graph[v].append(u)

# ==================== BRUTE FORCE APPROACH ====================
def bfs_brute_force(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Brute Force BFS: Simple implementation without optimizations
    
    Time Complexity: O(V + E) - visits each vertex and edge once
    Space Complexity: O(V) - queue can hold all vertices in worst case
    
    Problems:
    - Uses list as queue (O(n) for pop(0))
    - No distance tracking
    - Inefficient for large graphs
    """
    if start not in graph and not any(start in neighbors for neighbors in graph.values()):
        return [start]
    
    visited = []
    queue = [start]  # Using list as queue (inefficient)
    
    while queue:
        node = queue.pop(0)  # O(n) operation!
        if node not in visited:
            visited.append(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited and neighbor not in queue:
                    queue.append(neighbor)
    
    return visited

# ==================== OPTIMIZED APPROACH ====================
def bfs_optimized(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Optimized BFS: Uses deque for O(1) queue operations
    
    Time Complexity: O(V + E) - visits each vertex and edge once
    Space Complexity: O(V) - queue and visited set
    
    Advantages:
    - O(1) queue operations with deque
    - Set lookup for visited (O(1))
    - Memory efficient
    """
    if start not in graph:
        # Check if start exists as a neighbor
        all_nodes = set(graph.keys())
        for neighbors in graph.values():
            all_nodes.update(neighbors)
        if start not in all_nodes:
            return [start]
    
    visited = set()
    result = []
    queue = deque([start])
    visited.add(start)
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result

def bfs_shortest_path(graph: Dict[int, List[int]], start: int, target: int) -> Optional[List[int]]:
    """
    Find shortest path using BFS
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    if start == target:
        return [start]
    
    visited = set([start])
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor in graph.get(node, []):
            if neighbor == target:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None

def bfs_with_distances(graph: Dict[int, List[int]], start: int) -> Dict[int, int]:
    """
    BFS with distance tracking from start node
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    distances = {start: 0}
    queue = deque([start])
    
    while queue:
        node = queue.popleft()
        current_dist = distances[node]
        
        for neighbor in graph.get(node, []):
            if neighbor not in distances:
                distances[neighbor] = current_dist + 1
                queue.append(neighbor)
    
    return distances

def bfs_level_order_traversal(graph: Dict[int, List[int]], start: int) -> List[List[int]]:
    """
    BFS level-order traversal - returns nodes grouped by level
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    if start not in graph:
        return [[start]]
    
    result = []
    queue = deque([start])
    visited = set([start])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        result.append(current_level)
    
    return result

def bfs_connected_components(graph: Dict[int, List[int]]) -> List[List[int]]:
    """
    Find all connected components using BFS
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    visited = set()
    components = []
    
    # Get all vertices
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)
    
    for vertex in all_vertices:
        if vertex not in visited:
            component = []
            queue = deque([vertex])
            visited.add(vertex)
            
            while queue:
                node = queue.popleft()
                component.append(node)
                
                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component)
    
    return components

def is_bipartite(graph: Dict[int, List[int]]) -> bool:
    """
    Check if graph is bipartite using BFS two-coloring
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    color = {}
    
    # Get all vertices
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)
    
    for start in all_vertices:
        if start not in color:
            queue = deque([start])
            color[start] = 0
            
            while queue:
                node = queue.popleft()
                
                for neighbor in graph.get(node, []):
                    if neighbor not in color:
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:
                        return False
    
    return True

# ==================== GRID BFS ====================
def bfs_grid_shortest_path(grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> int:
    """
    BFS on 2D grid to find shortest path
    0 = walkable, 1 = obstacle
    
    Time Complexity: O(rows * cols)
    Space Complexity: O(rows * cols)
    """
    if not grid or not grid[0]:
        return -1
    
    rows, cols = len(grid), len(grid[0])
    if grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1:
        return -1
    
    if start == end:
        return 0
    
    queue = deque([(start[0], start[1], 0)])
    visited = set([start])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    while queue:
        row, col, dist = queue.popleft()
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            if (0 <= new_row < rows and 0 <= new_col < cols and 
                grid[new_row][new_col] == 0 and (new_row, new_col) not in visited):
                
                if (new_row, new_col) == end:
                    return dist + 1
                
                visited.add((new_row, new_col))
                queue.append((new_row, new_col, dist + 1))
    
    return -1

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    # Create sample graph
    graph = {
        1: [2, 3],
        2: [4, 5],
        3: [6],
        4: [],
        5: [7],
        6: [],
        7: []
    }
    
    print("Graph:", graph)
    print("\n--- BFS Traversals ---")
    print("Brute Force BFS:", bfs_brute_force(graph, 1))
    print("Optimized BFS:", bfs_optimized(graph, 1))
    
    print("\n--- Shortest Path ---")
    path = bfs_shortest_path(graph, 1, 7)
    print(f"Shortest path from 1 to 7: {path}")
    
    print("\n--- Distances ---")
    distances = bfs_with_distances(graph, 1)
    print(f"Distances from node 1: {distances}")
    
    print("\n--- Level Order ---")
    levels = bfs_level_order_traversal(graph, 1)
    print(f"Level order traversal: {levels}")
    
    print("\n--- Bipartite Check ---")
    bipartite_graph = {1: [2, 4], 2: [1, 3], 3: [2, 4], 4: [1, 3]}
    print(f"Is bipartite: {is_bipartite(bipartite_graph)}")
    
    print("\n--- Grid BFS ---")
    grid = [
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]
    ]
    shortest = bfs_grid_shortest_path(grid, (0, 0), (3, 3))
    print(f"Shortest path in grid: {shortest}")

"""
COMMON BFS PATTERNS:

1. Shortest Path (Unweighted):
   - Level-by-level exploration guarantees shortest path
   - Use when all edges have equal weight

2. Level Order Traversal:
   - Process nodes level by level
   - Tree level order, graph layers

3. Connected Components:
   - Find all reachable nodes from a starting point
   - Count number of islands/components

4. Bipartite Check:
   - Two-coloring using BFS
   - Detect odd cycles

WHEN TO USE BFS:
- Finding shortest path in unweighted graphs
- Level-order traversal needed
- Finding minimum number of steps
- Exploring neighbors before going deeper

WHEN NOT TO USE BFS:
- Weighted graphs (use Dijkstra instead)
- Memory is very limited (DFS uses less memory)
- Need to explore all paths (use DFS)
- Very wide graphs (BFS queue can become very large)

BFS vs DFS:
- BFS: Breadth-first, uses queue, finds shortest path
- DFS: Depth-first, uses stack/recursion, uses less memory
"""