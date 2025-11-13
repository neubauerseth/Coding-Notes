"""
Greedy Algorithms
================

Used for: Optimization problems where local optimal choices lead to global optimum

Time Complexity: Varies (typically O(n log n) due to sorting)
Space Complexity: O(1) to O(n) depending on problem
"""

from typing import List, Tuple, Optional, Dict
import heapq

# ==================== BRUTE FORCE APPROACHES ====================
def activity_selection_brute_force(start: List[int], finish: List[int]) -> List[int]:
    """
    Brute Force Activity Selection: Try all possible combinations
    
    Time Complexity: O(2^n) - exponential
    Space Complexity: O(n) - recursion depth
    
    Problems:
    - Exponential time complexity
    - Explores all possible combinations
    - Doesn't utilize greedy property
    """
    n = len(start)
    activities = list(range(n))
    
    def is_compatible(selected_activities):
        for i in range(len(selected_activities)):
            for j in range(i + 1, len(selected_activities)):
                act1, act2 = selected_activities[i], selected_activities[j]
                if not (finish[act1] <= start[act2] or finish[act2] <= start[act1]):
                    return False
        return True
    
    max_count = 0
    best_selection = []
    
    # Try all possible subsets
    for mask in range(1 << n):
        selected = []
        for i in range(n):
            if mask & (1 << i):
                selected.append(i)
        
        if is_compatible(selected) and len(selected) > max_count:
            max_count = len(selected)
            best_selection = selected
    
    return best_selection

def fractional_knapsack_brute_force(weights: List[int], values: List[int], capacity: int) -> float:
    """
    Brute Force Fractional Knapsack: Try all possible fractions
    
    Time Complexity: O(n!) - factorial
    Space Complexity: O(n)
    """
    n = len(weights)
    items = list(range(n))
    
    def get_value(permutation, remaining_capacity):
        total_value = 0
        for item in permutation:
            if weights[item] <= remaining_capacity:
                total_value += values[item]
                remaining_capacity -= weights[item]
            else:
                fraction = remaining_capacity / weights[item]
                total_value += fraction * values[item]
                break
        return total_value
    
    max_value = 0
    import itertools
    
    for perm in itertools.permutations(items):
        value = get_value(perm, capacity)
        max_value = max(max_value, value)
    
    return max_value

# ==================== OPTIMIZED GREEDY APPROACHES ====================

def activity_selection_greedy(start: List[int], finish: List[int]) -> List[int]:
    """
    Activity Selection using Greedy approach (earliest finish time)
    
    Time Complexity: O(n log n) - due to sorting
    Space Complexity: O(n) - for storing activities
    
    Greedy Choice: Always pick activity that finishes earliest
    """
    n = len(start)
    activities = [(finish[i], start[i], i) for i in range(n)]
    activities.sort()  # Sort by finish time
    
    selected = []
    last_finish_time = -1
    
    for finish_time, start_time, original_index in activities:
        if start_time >= last_finish_time:
            selected.append(original_index)
            last_finish_time = finish_time
    
    return selected

def fractional_knapsack_greedy(weights: List[int], values: List[int], capacity: int) -> float:
    """
    Fractional Knapsack using Greedy approach (value-to-weight ratio)
    
    Time Complexity: O(n log n) - due to sorting
    Space Complexity: O(n) - for storing items
    
    Greedy Choice: Always pick item with highest value-to-weight ratio
    """
    n = len(weights)
    items = [(values[i] / weights[i], weights[i], values[i], i) for i in range(n)]
    items.sort(reverse=True)  # Sort by value-to-weight ratio (descending)
    
    total_value = 0
    remaining_capacity = capacity
    
    for ratio, weight, value, original_index in items:
        if weight <= remaining_capacity:
            # Take entire item
            total_value += value
            remaining_capacity -= weight
        else:
            # Take fraction of item
            fraction = remaining_capacity / weight
            total_value += fraction * value
            break
    
    return total_value

def job_scheduling_greedy(jobs: List[Tuple[int, int, int]]) -> Tuple[int, int]:
    """
    Job Scheduling with Deadlines and Profits
    
    Input: jobs = [(job_id, deadline, profit), ...]
    Output: (number_of_jobs, total_profit)
    
    Time Complexity: O(n log n + n * max_deadline)
    Space Complexity: O(max_deadline)
    
    Greedy Choice: Schedule jobs in decreasing order of profit
    """
    # Sort jobs by profit in descending order
    jobs.sort(key=lambda x: x[2], reverse=True)
    
    if not jobs:
        return (0, 0)
    
    max_deadline = max(job[1] for job in jobs)
    schedule = [False] * (max_deadline + 1)
    
    total_jobs = 0
    total_profit = 0
    
    for job_id, deadline, profit in jobs:
        # Find latest available slot before deadline
        for slot in range(min(deadline, max_deadline), 0, -1):
            if not schedule[slot]:
                schedule[slot] = True
                total_jobs += 1
                total_profit += profit
                break
    
    return (total_jobs, total_profit)

def huffman_coding(frequencies: List[Tuple[str, int]]) -> Dict[str, str]:
    """
    Huffman Coding for optimal prefix-free codes
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Greedy Choice: Always merge two nodes with smallest frequencies
    """
    import heapq
    from collections import defaultdict
    
    if len(frequencies) == 1:
        return {frequencies[0][0]: '0'}
    
    # Min heap with (frequency, unique_id, node)
    heap = []
    for i, (char, freq) in enumerate(frequencies):
        heapq.heappush(heap, (freq, i, char))
    
    node_counter = len(frequencies)
    
    # Build Huffman tree
    while len(heap) > 1:
        freq1, id1, node1 = heapq.heappop(heap)
        freq2, id2, node2 = heapq.heappop(heap)
        
        merged_freq = freq1 + freq2
        merged_node = (node1, node2)
        
        heapq.heappush(heap, (merged_freq, node_counter, merged_node))
        node_counter += 1
    
    # Extract codes from tree
    _, _, root = heap[0]
    codes = {}
    
    def extract_codes(node, code=''):
        if isinstance(node, str):  # Leaf node
            codes[node] = code if code else '0'
        else:  # Internal node
            left, right = node
            extract_codes(left, code + '0')
            extract_codes(right, code + '1')
    
    extract_codes(root)
    return codes

def minimum_spanning_tree_prim(graph: Dict[int, List[Tuple[int, int]]]) -> List[Tuple[int, int, int]]:
    """
    Minimum Spanning Tree using Prim's algorithm
    
    Time Complexity: O((V + E) log V)
    Space Complexity: O(V)
    
    Greedy Choice: Always add minimum weight edge connecting tree to non-tree vertex
    """
    if not graph:
        return []
    
    start_vertex = next(iter(graph))
    mst = []
    visited = {start_vertex}
    
    # Priority queue: (weight, from_vertex, to_vertex)
    edges = []
    for neighbor, weight in graph[start_vertex]:
        heapq.heappush(edges, (weight, start_vertex, neighbor))
    
    while edges and len(visited) < len(graph):
        weight, from_vertex, to_vertex = heapq.heappop(edges)
        
        if to_vertex in visited:
            continue
        
        # Add edge to MST
        mst.append((from_vertex, to_vertex, weight))
        visited.add(to_vertex)
        
        # Add new edges from newly added vertex
        if to_vertex in graph:
            for neighbor, edge_weight in graph[to_vertex]:
                if neighbor not in visited:
                    heapq.heappush(edges, (edge_weight, to_vertex, neighbor))
    
    return mst

def gas_station_circuit(gas: List[int], cost: List[int]) -> int:
    """
    Gas Station Circuit - find starting station to complete circuit
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Greedy Choice: Start from station where we have enough gas to proceed
    """
    total_gas = sum(gas)
    total_cost = sum(cost)
    
    if total_gas < total_cost:
        return -1  # Impossible to complete circuit
    
    current_gas = 0
    start_station = 0
    
    for i in range(len(gas)):
        current_gas += gas[i] - cost[i]
        
        if current_gas < 0:
            # Can't reach next station from current start
            start_station = i + 1
            current_gas = 0
    
    return start_station

def jump_game_greedy(nums: List[int]) -> bool:
    """
    Jump Game - determine if you can reach the last index
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Greedy Choice: Keep track of farthest reachable position
    """
    farthest = 0
    
    for i in range(len(nums)):
        if i > farthest:
            return False  # Can't reach current position
        
        farthest = max(farthest, i + nums[i])
        
        if farthest >= len(nums) - 1:
            return True
    
    return farthest >= len(nums) - 1

def jump_game_min_jumps(nums: List[int]) -> int:
    """
    Jump Game II - minimum jumps to reach last index
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Greedy Choice: Jump as far as possible with each jump
    """
    if len(nums) <= 1:
        return 0
    
    jumps = 0
    current_reach = 0
    farthest_reach = 0
    
    for i in range(len(nums) - 1):
        farthest_reach = max(farthest_reach, i + nums[i])
        
        if i == current_reach:
            jumps += 1
            current_reach = farthest_reach
            
            if current_reach >= len(nums) - 1:
                break
    
    return jumps

def candy_distribution(ratings: List[int]) -> int:
    """
    Candy Distribution - minimum candies needed for rating-based distribution
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Greedy Choice: Two passes to handle left and right neighbor constraints
    """
    n = len(ratings)
    if n == 0:
        return 0
    
    candies = [1] * n
    
    # Left to right pass
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1
    
    # Right to left pass
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)
    
    return sum(candies)

def non_overlapping_intervals(intervals: List[List[int]]) -> int:
    """
    Minimum number of intervals to remove to make rest non-overlapping
    
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    
    Greedy Choice: Keep intervals that end earliest
    """
    if not intervals:
        return 0
    
    # Sort by end time
    intervals.sort(key=lambda x: x[1])
    
    count = 0
    last_end = intervals[0][1]
    
    for i in range(1, len(intervals)):
        if intervals[i][0] < last_end:
            # Overlapping interval, remove it
            count += 1
        else:
            # Non-overlapping, update last_end
            last_end = intervals[i][1]
    
    return count

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("=== Activity Selection ===")
    start = [1, 3, 0, 5, 8, 5]
    finish = [2, 4, 6, 7, 9, 9]
    
    print(f"Start times: {start}")
    print(f"Finish times: {finish}")
    print(f"Selected activities (greedy): {activity_selection_greedy(start, finish)}")
    
    print("\n=== Fractional Knapsack ===")
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    
    print(f"Weights: {weights}")
    print(f"Values: {values}")
    print(f"Capacity: {capacity}")
    print(f"Maximum value (greedy): {fractional_knapsack_greedy(weights, values, capacity)}")
    
    print("\n=== Job Scheduling ===")
    jobs = [(1, 4, 20), (2, 1, 10), (3, 1, 40), (4, 1, 30)]
    
    print(f"Jobs (id, deadline, profit): {jobs}")
    job_count, total_profit = job_scheduling_greedy(jobs)
    print(f"Scheduled jobs: {job_count}, Total profit: {total_profit}")
    
    print("\n=== Huffman Coding ===")
    frequencies = [('a', 5), ('b', 9), ('c', 12), ('d', 13), ('e', 16), ('f', 45)]
    
    print(f"Character frequencies: {frequencies}")
    codes = huffman_coding(frequencies)
    print(f"Huffman codes: {codes}")
    
    print("\n=== Gas Station Circuit ===")
    gas = [1, 2, 3, 4, 5]
    cost = [3, 4, 5, 1, 2]
    
    print(f"Gas: {gas}")
    print(f"Cost: {cost}")
    print(f"Starting station: {gas_station_circuit(gas, cost)}")
    
    print("\n=== Jump Games ===")
    nums1 = [2, 3, 1, 1, 4]
    nums2 = [3, 2, 1, 0, 4]
    
    print(f"Array 1: {nums1}")
    print(f"Can jump to end: {jump_game_greedy(nums1)}")
    print(f"Minimum jumps: {jump_game_min_jumps(nums1)}")
    
    print(f"Array 2: {nums2}")
    print(f"Can jump to end: {jump_game_greedy(nums2)}")
    
    print("\n=== Candy Distribution ===")
    ratings = [1, 0, 2]
    print(f"Ratings: {ratings}")
    print(f"Minimum candies: {candy_distribution(ratings)}")
    
    print("\n=== Non-overlapping Intervals ===")
    intervals = [[1, 2], [2, 3], [3, 4], [1, 3]]
    print(f"Intervals: {intervals}")
    print(f"Intervals to remove: {non_overlapping_intervals(intervals)}")

"""
GREEDY ALGORITHM PATTERNS:

1. Activity/Interval Selection:
   - Sort by end time, pick non-overlapping
   - Applications: Meeting rooms, task scheduling

2. Fractional Knapsack:
   - Sort by value-to-weight ratio
   - Take items greedily by ratio

3. Minimum Spanning Tree:
   - Prim's: Add minimum edge connecting to tree
   - Kruskal's: Add minimum edge that doesn't create cycle

4. Huffman Coding:
   - Always merge two smallest frequency nodes
   - Creates optimal prefix-free codes

5. Job Scheduling:
   - Sort by profit, assign to latest possible slot
   - Maximizes total profit

WHEN TO USE GREEDY:
- Problem has optimal substructure
- Greedy choice leads to optimal solution
- Local optimum leads to global optimum
- Can prove greedy choice is safe

GREEDY VS DYNAMIC PROGRAMMING:
- Greedy: Makes choice without considering future
- DP: Considers all possibilities before choosing
- Greedy: Faster (usually O(n log n))
- DP: More general but slower

PROVING GREEDY CORRECTNESS:
1. Greedy Choice Property: Local optimum leads to global optimum
2. Optimal Substructure: Optimal solution contains optimal subsolutions
3. Exchange Argument: Can transform any optimal solution to greedy solution

COMMON GREEDY PATTERNS:
- Sort by some criteria, then pick greedily
- Use heap/priority queue for dynamic selection
- Two-pass algorithms (left-to-right, right-to-left)
- Interval problems: sort by start/end time

WHEN GREEDY FAILS:
- 0/1 Knapsack (need DP)
- Longest Common Subsequence
- Edit Distance
- Problems where local optimum ≠ global optimum
"""