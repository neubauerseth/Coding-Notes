"""
Prefix Sums Algorithm
====================

Used for: Range sum queries, subarray sum problems

Time Complexity: O(n) preprocessing, O(1) query
Space Complexity: O(n) for prefix sum array
"""

from typing import List, Dict, Tuple
from collections import defaultdict

# ==================== BRUTE FORCE APPROACH ====================
def range_sum_query_brute_force(nums: List[int], queries: List[Tuple[int, int]]) -> List[int]:
    """
    Brute Force Range Sum: Calculate sum for each query separately
    
    Time Complexity: O(q * n) where q is number of queries, n is array length
    Space Complexity: O(1)
    
    Problems:
    - Recalculates sum for overlapping ranges
    - Inefficient for multiple queries
    - No preprocessing to optimize queries
    """
    results = []
    for left, right in queries:
        total = 0
        for i in range(left, right + 1):
            total += nums[i]
        results.append(total)
    return results

def subarray_sum_equals_k_brute_force(nums: List[int], k: int) -> int:
    """
    Brute Force: Check all possible subarrays
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    count = 0
    n = len(nums)
    
    for i in range(n):
        current_sum = 0
        for j in range(i, n):
            current_sum += nums[j]
            if current_sum == k:
                count += 1
    
    return count

# ==================== OPTIMIZED APPROACH ====================
class PrefixSumArray:
    """
    Prefix Sum Array for efficient range queries
    
    Time Complexity: O(n) preprocessing, O(1) per query
    Space Complexity: O(n)
    """
    
    def __init__(self, nums: List[int]):
        self.nums = nums
        self.prefix_sums = [0]
        
        # Build prefix sum array
        for num in nums:
            self.prefix_sums.append(self.prefix_sums[-1] + num)
    
    def range_sum(self, left: int, right: int) -> int:
        """Get sum of elements from index left to right (inclusive)"""
        return self.prefix_sums[right + 1] - self.prefix_sums[left]
    
    def update(self, index: int, val: int):
        """Update single element (requires rebuilding prefix sums)"""
        old_val = self.nums[index]
        self.nums[index] = val
        diff = val - old_val
        
        # Update all prefix sums from index onwards
        for i in range(index + 1, len(self.prefix_sums)):
            self.prefix_sums[i] += diff

def subarray_sum_equals_k_optimized(nums: List[int], k: int) -> int:
    """
    Count subarrays with sum equal to k using prefix sums
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    count = 0
    prefix_sum = 0
    prefix_sum_count = defaultdict(int)
    prefix_sum_count[0] = 1  # Empty prefix has sum 0
    
    for num in nums:
        prefix_sum += num
        
        # Check if there exists a prefix with sum = current_sum - k
        if prefix_sum - k in prefix_sum_count:
            count += prefix_sum_count[prefix_sum - k]
        
        prefix_sum_count[prefix_sum] += 1
    
    return count

def continuous_subarray_sum_divisible_by_k(nums: List[int], k: int) -> bool:
    """
    Check if there exists a continuous subarray with sum divisible by k
    
    Time Complexity: O(n)
    Space Complexity: O(k)
    """
    remainder_count = defaultdict(int)
    remainder_count[0] = 1  # Empty prefix has remainder 0
    
    prefix_sum = 0
    for i, num in enumerate(nums):
        prefix_sum += num
        remainder = prefix_sum % k
        
        if remainder in remainder_count:
            # If we've seen this remainder before, subarray sum is divisible by k
            if i - remainder_count[remainder] > 0:  # Ensure subarray length > 1
                return True
        else:
            remainder_count[remainder] = i + 1
    
    return False

def max_size_subarray_sum_equals_k(nums: List[int], k: int) -> int:
    """
    Find maximum length of subarray with sum equal to k
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    prefix_sum = 0
    prefix_sum_index = {0: -1}  # prefix_sum -> first index where this sum occurs
    max_length = 0
    
    for i, num in enumerate(nums):
        prefix_sum += num
        
        if prefix_sum - k in prefix_sum_index:
            length = i - prefix_sum_index[prefix_sum - k]
            max_length = max(max_length, length)
        
        # Only store first occurrence of each prefix sum
        if prefix_sum not in prefix_sum_index:
            prefix_sum_index[prefix_sum] = i
    
    return max_length

def product_except_self(nums: List[int]) -> List[int]:
    """
    Product of array except self using prefix/suffix products
    
    Time Complexity: O(n)
    Space Complexity: O(1) excluding output array
    """
    n = len(nums)
    result = [1] * n
    
    # Forward pass: result[i] contains product of all elements before i
    for i in range(1, n):
        result[i] = result[i - 1] * nums[i - 1]
    
    # Backward pass: multiply with product of all elements after i
    suffix_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= suffix_product
        suffix_product *= nums[i]
    
    return result

# ==================== 2D PREFIX SUMS ====================
class PrefixSum2D:
    """
    2D Prefix Sum for matrix range queries
    
    Time Complexity: O(m * n) preprocessing, O(1) per query
    Space Complexity: O(m * n)
    """
    
    def __init__(self, matrix: List[List[int]]):
        if not matrix or not matrix[0]:
            self.prefix = []
            return
        
        rows, cols = len(matrix), len(matrix[0])
        self.prefix = [[0] * (cols + 1) for _ in range(rows + 1)]
        
        # Build 2D prefix sum
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                self.prefix[i][j] = (matrix[i-1][j-1] + 
                                   self.prefix[i-1][j] + 
                                   self.prefix[i][j-1] - 
                                   self.prefix[i-1][j-1])
    
    def range_sum(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """Get sum of rectangle from (row1,col1) to (row2,col2) inclusive"""
        if not self.prefix:
            return 0
        
        # Convert to 1-indexed
        row1 += 1
        col1 += 1
        row2 += 1
        col2 += 1
        
        return (self.prefix[row2][col2] - 
                self.prefix[row1-1][col2] - 
                self.prefix[row2][col1-1] + 
                self.prefix[row1-1][col1-1])

def num_submatrices_sum_target(matrix: List[List[int]], target: int) -> int:
    """
    Number of submatrices with sum equal to target
    
    Time Complexity: O(m² * n)
    Space Complexity: O(n)
    """
    if not matrix or not matrix[0]:
        return 0
    
    rows, cols = len(matrix), len(matrix[0])
    count = 0
    
    # Try all pairs of rows
    for top in range(rows):
        col_sums = [0] * cols
        
        for bottom in range(top, rows):
            # Add current row to column sums
            for col in range(cols):
                col_sums[col] += matrix[bottom][col]
            
            # Find subarrays in col_sums with sum = target
            count += subarray_sum_equals_k_optimized(col_sums, target)
    
    return count

# ==================== ADVANCED PREFIX SUM PATTERNS ====================
def shortest_subarray_sum_at_least_k(nums: List[int], k: int) -> int:
    """
    Shortest subarray with sum at least k (handles negative numbers)
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    from collections import deque
    
    n = len(nums)
    prefix_sums = [0]
    
    # Build prefix sums
    for num in nums:
        prefix_sums.append(prefix_sums[-1] + num)
    
    deq = deque()  # Monotonic deque storing indices
    min_length = float('inf')
    
    for i in range(n + 1):
        # Check if current prefix sum can form valid subarray with previous ones
        while deq and prefix_sums[i] - prefix_sums[deq[0]] >= k:
            min_length = min(min_length, i - deq.popleft())
        
        # Maintain monotonic property (ascending prefix sums)
        while deq and prefix_sums[i] <= prefix_sums[deq[-1]]:
            deq.pop()
        
        deq.append(i)
    
    return min_length if min_length != float('inf') else -1

def running_sum_1d(nums: List[int]) -> List[int]:
    """
    Simple running sum calculation
    
    Time Complexity: O(n)
    Space Complexity: O(1) excluding output
    """
    result = []
    running_sum = 0
    
    for num in nums:
        running_sum += num
        result.append(running_sum)
    
    return result

def pivot_index(nums: List[int]) -> int:
    """
    Find pivot index where left sum equals right sum
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    total_sum = sum(nums)
    left_sum = 0
    
    for i, num in enumerate(nums):
        right_sum = total_sum - left_sum - num
        
        if left_sum == right_sum:
            return i
        
        left_sum += num
    
    return -1

def subarray_sums_divisible_by_k(nums: List[int], k: int) -> int:
    """
    Count subarrays with sum divisible by k
    
    Time Complexity: O(n)
    Space Complexity: O(k)
    """
    remainder_count = defaultdict(int)
    remainder_count[0] = 1  # Empty prefix
    
    prefix_sum = 0
    count = 0
    
    for num in nums:
        prefix_sum += num
        remainder = prefix_sum % k
        
        # In Python, remainder can be negative, so normalize it
        remainder = (remainder + k) % k
        
        count += remainder_count[remainder]
        remainder_count[remainder] += 1
    
    return count

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("=== Basic Prefix Sum ===")
    nums = [1, 2, 3, 4, 5]
    prefix_sum_array = PrefixSumArray(nums)
    
    print(f"Array: {nums}")
    print(f"Range sum [1, 3]: {prefix_sum_array.range_sum(1, 3)}")
    print(f"Range sum [0, 4]: {prefix_sum_array.range_sum(0, 4)}")
    
    queries = [(0, 2), (1, 3), (2, 4)]
    print(f"Queries {queries}:")
    print(f"  Brute force: {range_sum_query_brute_force(nums, queries)}")
    print(f"  Prefix sum: {[prefix_sum_array.range_sum(l, r) for l, r in queries]}")
    
    print("\n=== Subarray Sum Equals K ===")
    nums = [1, 1, 1]
    k = 2
    print(f"Array: {nums}, k: {k}")
    print(f"  Brute force: {subarray_sum_equals_k_brute_force(nums, k)}")
    print(f"  Optimized: {subarray_sum_equals_k_optimized(nums, k)}")
    
    print("\n=== Product Except Self ===")
    nums = [1, 2, 3, 4]
    print(f"Array: {nums}")
    print(f"Product except self: {product_except_self(nums)}")
    
    print("\n=== 2D Prefix Sum ===")
    matrix = [
        [3, 0, 1, 4, 2],
        [5, 6, 3, 2, 1],
        [1, 2, 0, 1, 5],
        [4, 1, 0, 1, 7],
        [1, 0, 3, 0, 5]
    ]
    
    prefix_2d = PrefixSum2D(matrix)
    print("Matrix:")
    for row in matrix:
        print(row)
    
    print(f"Sum of rectangle (2,1) to (4,3): {prefix_2d.range_sum(2, 1, 4, 3)}")
    
    print("\n=== Advanced Patterns ===")
    nums = [1, -1, 0]
    print(f"Array: {nums}")
    print(f"Running sum: {running_sum_1d(nums)}")
    print(f"Pivot index: {pivot_index(nums)}")
    
    nums = [4, 5, 0, -2, -3, 1]
    k = 5
    print(f"Array: {nums}, k: {k}")
    print(f"Subarrays divisible by {k}: {subarray_sums_divisible_by_k(nums, k)}")

"""
PREFIX SUMS PATTERNS:

1. Basic Range Sum Query:
   - Preprocess: prefix[i] = sum of elements [0, i-1]
   - Query: sum[l, r] = prefix[r+1] - prefix[l]

2. Subarray Sum Problems:
   - Use HashMap to store prefix sum frequencies
   - For sum = k: count prefix sums where current - k exists
   - For divisible by k: use remainder of prefix sums

3. 2D Prefix Sums:
   - prefix[i][j] = sum of rectangle from (0,0) to (i-1,j-1)
   - Query: use inclusion-exclusion principle

4. Product Arrays:
   - Forward pass: product of all elements before current
   - Backward pass: multiply with product of all elements after

WHEN TO USE PREFIX SUMS:
- Multiple range sum queries on static array
- Subarray sum problems (equal to k, divisible by k)
- Need to optimize from O(n) per query to O(1)
- Problems involving cumulative properties

KEY INSIGHTS:
- Prefix sum converts range queries to O(1)
- HashMap with prefix sums solves many subarray problems
- 2D prefix sums handle rectangle sum queries
- Can be extended to other operations (XOR, product)

COMMON PATTERNS:
1. Range Sum: prefix[j+1] - prefix[i]
2. Subarray Sum = k: count[prefix_sum - k]
3. Subarray Sum divisible by k: remainder frequency
4. Maximum subarray length: store first occurrence of each prefix sum

OPTIMIZATION TECHNIQUES:
- Space optimization: Sometimes can avoid storing entire prefix array
- Rolling hash: For string problems with prefix concept
- Coordinate compression: For sparse 2D problems
- Lazy propagation: For range update queries
"""