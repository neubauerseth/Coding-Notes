"""
Depth-First Search (DFS) Algorithm
==================================

Used for: Tree traversal, connected components, cycle detection, pathfinding

Time Complexity: O(V + E) where V = vertices, E = edges
Space Complexity: O(V) for recursion stack or explicit stack
"""

from collections import defaultdict, deque
from typing import List, Set, Dict, Optional

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
def dfs_brute_force(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Brute Force DFS: Uses simple recursion without optimization
    
    Time Complexity: O(V + E) - visits each vertex and edge once
    Space Complexity: O(V) - recursion stack depth can be V in worst case
    
    Problems: 
    - Can cause stack overflow for deep graphs
    - No cycle detection
    - Inefficient for repeated searches
    """
    visited = []
    
    def dfs_recursive(node):
        visited.append(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs_recursive(neighbor)
    
    dfs_recursive(start)
    return visited

# ==================== OPTIMIZED APPROACH ====================
def dfs_iterative_optimized(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Optimized DFS: Uses explicit stack to avoid recursion limits
    
    Time Complexity: O(V + E) - visits each vertex and edge once
    Space Complexity: O(V) - explicit stack and visited set
    
    Advantages:
    - No stack overflow issues
    - Can handle very large graphs
    - Memory efficient with set lookup
    """
    if start not in graph:
        return [start] if start is not None else []
    
    visited = set()
    result = []
    stack = [start]
    
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            
            # Add neighbors in reverse order to maintain left-to-right traversal
            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return result

def dfs_with_path_tracking(graph: Dict[int, List[int]], start: int, target: int) -> Optional[List[int]]:
    """
    DFS with path tracking - finds path from start to target
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    visited = set()
    path = []
    
    def dfs_path(node):
        if node in visited:
            return False
        
        visited.add(node)
        path.append(node)
        
        if node == target:
            return True
        
        for neighbor in graph.get(node, []):
            if dfs_path(neighbor):
                return True
        
        path.pop()  # Backtrack
        return False
    
    if dfs_path(start):
        return path
    return None

def detect_cycle_directed(graph: Dict[int, List[int]]) -> bool:
    """
    Detect cycle in directed graph using DFS
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = defaultdict(int)
    
    def has_cycle(node):
        if color[node] == GRAY:  # Back edge found
            return True
        if color[node] == BLACK:  # Already processed
            return False
        
        color[node] = GRAY
        for neighbor in graph.get(node, []):
            if has_cycle(neighbor):
                return True
        
        color[node] = BLACK
        return False
    
    for vertex in graph:
        if color[vertex] == WHITE:
            if has_cycle(vertex):
                return True
    return False

def connected_components(graph: Dict[int, List[int]]) -> List[List[int]]:
    """
    Find all connected components using DFS
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    visited = set()
    components = []
    
    def dfs_component(node, component):
        visited.add(node)
        component.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs_component(neighbor, component)
    
    # Get all vertices
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)
    
    for vertex in all_vertices:
        if vertex not in visited:
            component = []
            dfs_component(vertex, component)
            components.append(component)
    
    return components

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
    print("\n--- DFS Traversals ---")
    print("Brute Force DFS:", dfs_brute_force(graph, 1))
    print("Optimized DFS:", dfs_iterative_optimized(graph, 1))
    
    print("\n--- Path Finding ---")
    path = dfs_with_path_tracking(graph, 1, 7)
    print(f"Path from 1 to 7: {path}")
    
    print("\n--- Cycle Detection ---")
    # Cycle graph
    cycle_graph = {1: [2], 2: [3], 3: [1]}
    print(f"Has cycle: {detect_cycle_directed(cycle_graph)}")
    
    print("\n--- Connected Components ---")
    # Disconnected graph
    disconnected = {1: [2], 2: [1], 3: [4], 4: [3], 5: []}
    components = connected_components(disconnected)
    print(f"Connected components: {components}")

"""
COMMON DFS PATTERNS:

1. Tree Traversal:
   - Preorder: Process node, then children
   - Inorder: Left child, node, right child (binary trees)
   - Postorder: Children first, then node

2. Graph Problems:
   - Connected components
   - Cycle detection
   - Topological sorting (with modifications)
   - Path finding

3. Backtracking:
   - N-Queens
   - Sudoku
   - Maze solving
   - Permutations/Combinations

WHEN TO USE DFS:
- Tree/graph traversal needed
- Path finding with backtracking
- Detecting cycles
- Connected components
- When you need to explore as far as possible before backtracking

WHEN NOT TO USE DFS:
- Finding shortest path (use BFS instead)
- Level-order traversal needed
- Very deep graphs (stack overflow risk with recursive)
"""