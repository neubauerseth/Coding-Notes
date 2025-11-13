"""
Topological Sort Algorithm
==========================

Used for: DAG ordering, dependency resolution, scheduling tasks

Time Complexity: O(V + E) where V = vertices, E = edges
Space Complexity: O(V) for recursion stack or queue
"""

from collections import defaultdict, deque
from typing import List, Set, Dict, Optional

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.vertices = set()
    
    def add_edge(self, u, v):
        """Add directed edge to graph"""
        self.graph[u].append(v)
        self.vertices.add(u)
        self.vertices.add(v)

# ==================== BRUTE FORCE APPROACH ====================
def topological_sort_brute_force(graph: Dict[int, List[int]]) -> Optional[List[int]]:
    """
    Brute Force Topological Sort using DFS
    
    Time Complexity: O(V + E) - but with potential inefficiencies
    Space Complexity: O(V) - recursion stack
    
    Problems:
    - Deep recursion can cause stack overflow
    - No cycle detection optimization
    - Less readable implementation
    """
    # Get all vertices
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)
    
    visited = set()
    rec_stack = set()
    result = []
    
    def dfs(node):
        if node in rec_stack:  # Cycle detected
            return False
        if node in visited:
            return True
        
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph.get(node, []):
            if not dfs(neighbor):
                return False
        
        rec_stack.remove(node)
        result.append(node)  # Add to result after processing all dependencies
        return True
    
    # Process all vertices
    for vertex in all_vertices:
        if vertex not in visited:
            if not dfs(vertex):
                return None  # Cycle detected
    
    return result[::-1]  # Reverse to get correct topological order

# ==================== OPTIMIZED APPROACH ====================
def topological_sort_kahn(graph: Dict[int, List[int]]) -> Optional[List[int]]:
    """
    Kahn's Algorithm for Topological Sort (BFS-based)
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    
    Advantages:
    - No recursion (no stack overflow)
    - Easy cycle detection
    - Intuitive algorithm
    - Can process nodes as soon as ready
    """
    # Get all vertices
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)
    
    # Calculate in-degrees
    in_degree = {v: 0 for v in all_vertices}
    for vertex in graph:
        for neighbor in graph[vertex]:
            in_degree[neighbor] += 1
    
    # Find vertices with no incoming edges
    queue = deque([v for v in all_vertices if in_degree[v] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        # Remove edges from this node
        for neighbor in graph.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check for cycles
    if len(result) != len(all_vertices):
        return None  # Cycle detected
    
    return result

def topological_sort_dfs_optimized(graph: Dict[int, List[int]]) -> Optional[List[int]]:
    """
    Optimized DFS-based Topological Sort with better cycle detection
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    # Get all vertices
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)
    
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in all_vertices}
    result = []
    
    def dfs(node):
        if color[node] == GRAY:  # Back edge - cycle detected
            return False
        if color[node] == BLACK:  # Already processed
            return True
        
        color[node] = GRAY
        
        for neighbor in graph.get(node, []):
            if not dfs(neighbor):
                return False
        
        color[node] = BLACK
        result.append(node)
        return True
    
    # Process all vertices
    for vertex in all_vertices:
        if color[vertex] == WHITE:
            if not dfs(vertex):
                return None  # Cycle detected
    
    return result[::-1]

def find_all_topological_orders(graph: Dict[int, List[int]]) -> List[List[int]]:
    """
    Find all possible topological orderings
    
    Time Complexity: O(V! * (V + E)) - factorial in worst case
    Space Complexity: O(V)
    
    Use only for small graphs due to exponential complexity
    """
    # Get all vertices
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)
    
    # Calculate in-degrees
    in_degree = {v: 0 for v in all_vertices}
    for vertex in graph:
        for neighbor in graph[vertex]:
            in_degree[neighbor] += 1
    
    def backtrack(current_order, remaining_in_degree):
        if len(current_order) == len(all_vertices):
            return [current_order[:]]
        
        results = []
        # Find all vertices with in-degree 0
        candidates = [v for v in all_vertices if v not in current_order and remaining_in_degree[v] == 0]
        
        for candidate in candidates:
            # Choose candidate
            current_order.append(candidate)
            
            # Update in-degrees
            temp_in_degree = remaining_in_degree.copy()
            for neighbor in graph.get(candidate, []):
                temp_in_degree[neighbor] -= 1
            
            # Recurse
            results.extend(backtrack(current_order, temp_in_degree))
            
            # Backtrack
            current_order.pop()
        
        return results
    
    return backtrack([], in_degree.copy())

def longest_path_dag(graph: Dict[int, List[int]], weights: Dict[tuple, int] = None) -> Dict[int, int]:
    """
    Find longest path in DAG using topological sort
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    if weights is None:
        weights = {}  # Default weight of 1 for all edges
    
    topo_order = topological_sort_kahn(graph)
    if topo_order is None:
        return {}  # Graph has cycle
    
    # Initialize distances
    dist = {v: float('-inf') for v in topo_order}
    
    # Set distance to source nodes (nodes with no incoming edges)
    all_vertices = set(graph.keys())
    for neighbors in graph.values():
        all_vertices.update(neighbors)
    
    in_degree = {v: 0 for v in all_vertices}
    for vertex in graph:
        for neighbor in graph[vertex]:
            in_degree[neighbor] += 1
    
    for vertex in all_vertices:
        if in_degree[vertex] == 0:
            dist[vertex] = 0
    
    # Process vertices in topological order
    for vertex in topo_order:
        if dist[vertex] != float('-inf'):
            for neighbor in graph.get(vertex, []):
                edge_weight = weights.get((vertex, neighbor), 1)
                dist[neighbor] = max(dist[neighbor], dist[vertex] + edge_weight)
    
    return dist

def detect_cycle_in_dag(graph: Dict[int, List[int]]) -> bool:
    """
    Detect if the given graph has a cycle (making it not a DAG)
    
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    return topological_sort_kahn(graph) is None

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    # Example 1: Course prerequisites
    print("=== Course Prerequisites Example ===")
    # Courses: 0, 1, 2, 3, 4, 5
    # Prerequisites: (course, prerequisite)
    prerequisites = [(1, 0), (2, 0), (3, 1), (3, 2), (4, 1), (5, 3), (5, 4)]
    
    course_graph = defaultdict(list)
    for course, prereq in prerequisites:
        course_graph[prereq].append(course)
    
    print("Course dependencies:", dict(course_graph))
    
    # Try different algorithms
    print("\nBrute Force DFS:", topological_sort_brute_force(course_graph))
    print("Kahn's Algorithm:", topological_sort_kahn(course_graph))
    print("Optimized DFS:", topological_sort_dfs_optimized(course_graph))
    
    # Example 2: Task scheduling
    print("\n=== Task Scheduling Example ===")
    tasks = {
        'A': ['B', 'C'],  # A must be done before B and C
        'B': ['D'],       # B must be done before D
        'C': ['D'],       # C must be done before D
        'D': ['E'],       # D must be done before E
        'E': []           # E has no dependencies
    }
    
    print("Task dependencies:", tasks)
    order = topological_sort_kahn(tasks)
    print("Task execution order:", order)
    
    # Example 3: Cycle detection
    print("\n=== Cycle Detection Example ===")
    cyclic_graph = {
        1: [2],
        2: [3],
        3: [1]  # Creates a cycle
    }
    
    print("Cyclic graph:", cyclic_graph)
    print("Has cycle:", detect_cycle_in_dag(cyclic_graph))
    result = topological_sort_kahn(cyclic_graph)
    print("Topological sort result:", result)
    
    # Example 4: All topological orders (small graph)
    print("\n=== All Topological Orders ===")
    small_graph = {
        1: [2, 3],
        2: [4],
        3: [4],
        4: []
    }
    
    print("Small graph:", small_graph)
    all_orders = find_all_topological_orders(small_graph)
    print("All possible topological orders:")
    for i, order in enumerate(all_orders, 1):
        print(f"  {i}: {order}")
    
    # Example 5: Longest path in DAG
    print("\n=== Longest Path in DAG ===")
    dag_graph = {
        'A': ['B', 'C'],
        'B': ['D'],
        'C': ['D'],
        'D': ['E'],
        'E': []
    }
    
    # With custom weights
    edge_weights = {
        ('A', 'B'): 5,
        ('A', 'C'): 3,
        ('B', 'D'): 6,
        ('C', 'D'): 4,
        ('D', 'E'): 2
    }
    
    print("DAG with weights:", dag_graph)
    print("Edge weights:", edge_weights)
    longest_paths = longest_path_dag(dag_graph, edge_weights)
    print("Longest paths from sources:", longest_paths)

"""
COMMON TOPOLOGICAL SORT PATTERNS:

1. Course Prerequisites:
   - Course scheduling with prerequisites
   - Dependency resolution

2. Task Scheduling:
   - Build systems (compile order)
   - Project task dependencies

3. Package Dependencies:
   - Software package installation order
   - Library dependency resolution

4. Spreadsheet Calculations:
   - Cell formula dependencies
   - Calculation order

WHEN TO USE TOPOLOGICAL SORT:
- DAG (Directed Acyclic Graph) problems
- Dependency resolution needed
- Task scheduling with prerequisites
- Finding order of execution

ALGORITHM COMPARISON:
- Kahn's (BFS): Better for detecting cycles early, more intuitive
- DFS-based: Uses less extra space, natural recursion
- Both have O(V + E) time complexity

CYCLE DETECTION:
- If topological sort fails, graph has a cycle
- Kahn's: If result size < total vertices
- DFS: If back edge found (gray node revisited)
"""