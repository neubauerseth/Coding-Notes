"""
Disjoint Set Union (Union-Find) Algorithm
=========================================

Used for: Connected components, union-find operations, cycle detection

Time Complexity: O(α(n)) ≈ O(1) per operation with optimizations
Space Complexity: O(n) for parent and rank arrays
"""

from typing import List, Dict, Set, Optional, Tuple

# ==================== BRUTE FORCE APPROACH ====================
class DisjointSetBruteForce:
    """
    Brute Force Disjoint Set: Simple implementation without optimizations
    
    Time Complexity: O(n) per find operation in worst case
    Space Complexity: O(n)
    
    Problems:
    - Find operation can be O(n) in worst case (long chains)
    - Union operation inefficient
    - No balancing of tree structure
    """
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.n = n
    
    def find(self, x: int) -> int:
        """Find root of element x (no path compression)"""
        if self.parent[x] != x:
            return self.find(self.parent[x])  # No path compression
        return x
    
    def union(self, x: int, y: int) -> bool:
        """Union two sets (no union by rank)"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Simple union - just attach one to other
        self.parent[root_x] = root_y
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """Check if two elements are in same set"""
        return self.find(x) == self.find(y)
    
    def count_components(self) -> int:
        """Count number of connected components"""
        roots = set()
        for i in range(self.n):
            roots.add(self.find(i))
        return len(roots)

# ==================== OPTIMIZED APPROACH ====================
class DisjointSetOptimized:
    """
    Optimized Disjoint Set with Path Compression and Union by Rank
    
    Time Complexity: O(α(n)) ≈ O(1) per operation (amortized)
    Space Complexity: O(n)
    
    Optimizations:
    - Path compression: flattens tree during find
    - Union by rank: keeps tree balanced
    - Near constant time operations
    """
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n  # Height of subtree
        self.n = n
        self.components = n  # Track number of components
    
    def find(self, x: int) -> int:
        """Find root with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union by rank for balanced trees"""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False  # Already in same set
        
        # Union by rank - attach smaller tree under larger tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.components -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """Check if two elements are connected"""
        return self.find(x) == self.find(y)
    
    def count_components(self) -> int:
        """Get number of connected components"""
        return self.components
    
    def get_component_size(self, x: int) -> int:
        """Get size of component containing x"""
        root = self.find(x)
        count = 0
        for i in range(self.n):
            if self.find(i) == root:
                count += 1
        return count
    
    def get_all_components(self) -> List[List[int]]:
        """Get all connected components as lists"""
        components_map = {}
        for i in range(self.n):
            root = self.find(i)
            if root not in components_map:
                components_map[root] = []
            components_map[root].append(i)
        
        return list(components_map.values())

class WeightedDisjointSet:
    """
    Weighted Disjoint Set for problems requiring distance/weight information
    
    Useful for: Relative positioning, equation solving
    """
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.weight = [0] * n  # Weight relative to parent
        self.n = n
    
    def find(self, x: int) -> Tuple[int, int]:
        """Find root and accumulated weight"""
        if self.parent[x] != x:
            root, w = self.find(self.parent[x])
            self.weight[x] += w
            self.parent[x] = root
        return self.parent[x], self.weight[x]
    
    def union(self, x: int, y: int, w: int) -> bool:
        """Union with weight: weight[y] = weight[x] + w"""
        root_x, weight_x = self.find(x)
        root_y, weight_y = self.find(y)
        
        if root_x == root_y:
            return weight_y == weight_x + w  # Check consistency
        
        self.parent[root_y] = root_x
        self.weight[root_y] = weight_x + w - weight_y
        return True
    
    def get_difference(self, x: int, y: int) -> Optional[int]:
        """Get weight difference if connected"""
        root_x, weight_x = self.find(x)
        root_y, weight_y = self.find(y)
        
        if root_x != root_y:
            return None
        
        return weight_y - weight_x

# ==================== APPLICATIONS ====================
def find_connected_components_graph(edges: List[Tuple[int, int]], n: int) -> List[List[int]]:
    """
    Find connected components in undirected graph using Union-Find
    
    Time Complexity: O(E * α(V))
    Space Complexity: O(V)
    """
    uf = DisjointSetOptimized(n)
    
    for u, v in edges:
        uf.union(u, v)
    
    return uf.get_all_components()

def detect_cycle_undirected(edges: List[Tuple[int, int]], n: int) -> bool:
    """
    Detect cycle in undirected graph using Union-Find
    
    Time Complexity: O(E * α(V))
    Space Complexity: O(V)
    """
    uf = DisjointSetOptimized(n)
    
    for u, v in edges:
        if uf.connected(u, v):
            return True  # Cycle found
        uf.union(u, v)
    
    return False

def kruskal_minimum_spanning_tree(edges: List[Tuple[int, int, int]], n: int) -> List[Tuple[int, int, int]]:
    """
    Kruskal's MST algorithm using Union-Find
    
    Time Complexity: O(E log E)
    Space Complexity: O(V)
    """
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])
    
    uf = DisjointSetOptimized(n)
    mst = []
    total_weight = 0
    
    for u, v, weight in edges:
        if not uf.connected(u, v):
            uf.union(u, v)
            mst.append((u, v, weight))
            total_weight += weight
            
            if len(mst) == n - 1:  # MST complete
                break
    
    return mst

def count_islands_2d_grid(grid: List[List[int]]) -> int:
    """
    Count islands in 2D binary grid using Union-Find
    
    Time Complexity: O(m * n * α(m * n))
    Space Complexity: O(m * n)
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    uf = DisjointSetOptimized(rows * cols)
    
    def get_index(r: int, c: int) -> int:
        return r * cols + c
    
    # Process each cell
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                # Check 4 directions
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and 
                        grid[nr][nc] == 1):
                        uf.union(get_index(r, c), get_index(nr, nc))
    
    # Count unique components that are islands
    islands = set()
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                islands.add(uf.find(get_index(r, c)))
    
    return len(islands)

def account_merge(accounts: List[List[str]]) -> List[List[str]]:
    """
    Merge accounts with common emails using Union-Find
    
    Time Complexity: O(N * M * α(N * M)) where N = accounts, M = max emails
    Space Complexity: O(N * M)
    """
    email_to_id = {}
    email_to_name = {}
    
    # Map emails to indices
    email_id = 0
    for account in accounts:
        name = account[0]
        for email in account[1:]:
            if email not in email_to_id:
                email_to_id[email] = email_id
                email_to_name[email] = name
                email_id += 1
    
    uf = DisjointSetOptimized(email_id)
    
    # Union emails from same account
    for account in accounts:
        if len(account) > 1:
            first_email = account[1]
            for email in account[2:]:
                uf.union(email_to_id[first_email], email_to_id[email])
    
    # Group emails by component
    id_to_email = {v: k for k, v in email_to_id.items()}
    groups = {}
    
    for email_id in range(len(email_to_id)):
        root = uf.find(email_id)
        if root not in groups:
            groups[root] = []
        groups[root].append(id_to_email[email_id])
    
    # Format result
    result = []
    for emails in groups.values():
        emails.sort()
        name = email_to_name[emails[0]]
        result.append([name] + emails)
    
    return result

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("=== Basic Union-Find Operations ===")
    uf = DisjointSetOptimized(6)
    
    print("Initial components:", uf.count_components())
    
    # Connect some elements
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)
    
    print("After unions:")
    print("  Components:", uf.count_components())
    print("  0 and 2 connected:", uf.connected(0, 2))
    print("  2 and 3 connected:", uf.connected(2, 3))
    print("  All components:", uf.get_all_components())
    
    print("\n=== Connected Components in Graph ===")
    edges = [(0, 1), (1, 2), (3, 4), (5, 6), (6, 7)]
    components = find_connected_components_graph(edges, 8)
    print("Graph edges:", edges)
    print("Connected components:", components)
    
    print("\n=== Cycle Detection ===")
    cycle_edges = [(0, 1), (1, 2), (2, 0)]  # Triangle - has cycle
    no_cycle_edges = [(0, 1), (1, 2), (3, 4)]  # No cycle
    
    print("Cycle edges:", cycle_edges)
    print("Has cycle:", detect_cycle_undirected(cycle_edges, 4))
    print("No cycle edges:", no_cycle_edges)
    print("Has cycle:", detect_cycle_undirected(no_cycle_edges, 5))
    
    print("\n=== Minimum Spanning Tree ===")
    weighted_edges = [
        (0, 1, 10), (0, 2, 6), (0, 3, 5),
        (1, 3, 15), (2, 3, 4)
    ]
    
    mst = kruskal_minimum_spanning_tree(weighted_edges, 4)
    print("Weighted edges:", weighted_edges)
    print("Minimum Spanning Tree:", mst)
    print("Total MST weight:", sum(weight for _, _, weight in mst))
    
    print("\n=== Island Counting ===")
    island_grid = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1]
    ]
    
    print("Grid:")
    for row in island_grid:
        print(row)
    print("Number of islands:", count_islands_2d_grid(island_grid))
    
    print("\n=== Weighted Union-Find ===")
    wuf = WeightedDisjointSet(5)
    
    # x + 3 = y, so weight[y] = weight[x] + 3
    wuf.union(0, 1, 3)  # x[1] = x[0] + 3
    wuf.union(1, 2, 2)  # x[2] = x[1] + 2, so x[2] = x[0] + 5
    
    print("Weight difference between 0 and 2:", wuf.get_difference(0, 2))

"""
UNION-FIND PATTERNS:

1. Connected Components:
   - Social networks (friend groups)
   - Network connectivity
   - Island problems

2. Cycle Detection:
   - Undirected graph cycles
   - Redundant connections
   - Circular dependencies

3. Minimum Spanning Tree:
   - Network design
   - Clustering problems
   - Infrastructure planning

4. Dynamic Connectivity:
   - Online algorithms
   - Incremental connectivity
   - Percolation problems

WHEN TO USE UNION-FIND:
- Need to track connected components
- Dynamic connectivity queries
- Cycle detection in undirected graphs
- Kruskal's MST algorithm
- Equivalence relation problems

OPTIMIZATION TECHNIQUES:
- Path Compression: O(log n) → O(α(n))
- Union by Rank: Keeps trees balanced
- Union by Size: Alternative to rank
- Weighted Union-Find: For relative positioning

TIME COMPLEXITY:
- Without optimizations: O(n) per operation
- With path compression only: O(log n) amortized
- With union by rank only: O(log n) per operation
- With both optimizations: O(α(n)) ≈ O(1) amortized

α(n) is the inverse Ackermann function, extremely slow growing
For practical purposes, α(n) ≤ 4 for any reasonable input size
"""