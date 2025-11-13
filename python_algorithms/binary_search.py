"""
Binary Search Algorithm
=======================

Used for: Finding elements in sorted arrays, optimization with monotonic conditions

Time Complexity: O(log n) - divides search space in half each iteration
Space Complexity: O(1) for iterative, O(log n) for recursive
"""

from typing import List, Optional, Callable
import bisect

# ==================== BRUTE FORCE APPROACH ====================
def linear_search(nums: List[int], target: int) -> int:
    """
    Brute Force Search: Linear scan through array
    
    Time Complexity: O(n) - check every element
    Space Complexity: O(1)
    
    Problems:
    - Linear time complexity
    - Doesn't utilize sorted property
    - Inefficient for large arrays
    """
    for i, num in enumerate(nums):
        if num == target:
            return i
    return -1

def find_first_occurrence_linear(nums: List[int], target: int) -> int:
    """
    Brute force find first occurrence
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    for i, num in enumerate(nums):
        if num == target:
            return i
    return -1

# ==================== OPTIMIZED APPROACH ====================
def binary_search_iterative(nums: List[int], target: int) -> int:
    """
    Classic Binary Search - Iterative Implementation
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    
    Advantages:
    - Logarithmic time complexity
    - No recursion overhead
    - Utilizes sorted property
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def binary_search_recursive(nums: List[int], target: int, left: int = 0, right: int = None) -> int:
    """
    Binary Search - Recursive Implementation
    
    Time Complexity: O(log n)
    Space Complexity: O(log n) - recursion stack
    """
    if right is None:
        right = len(nums) - 1
    
    if left > right:
        return -1
    
    mid = left + (right - left) // 2
    
    if nums[mid] == target:
        return mid
    elif nums[mid] < target:
        return binary_search_recursive(nums, target, mid + 1, right)
    else:
        return binary_search_recursive(nums, target, left, mid - 1)

def find_first_occurrence(nums: List[int], target: int) -> int:
    """
    Find first occurrence of target in sorted array with duplicates
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left for first occurrence
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

def find_last_occurrence(nums: List[int], target: int) -> int:
    """
    Find last occurrence of target in sorted array with duplicates
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    result = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right for last occurrence
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return result

def search_range(nums: List[int], target: int) -> List[int]:
    """
    Find first and last position of target
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    first = find_first_occurrence(nums, target)
    if first == -1:
        return [-1, -1]
    
    last = find_last_occurrence(nums, target)
    return [first, last]

def search_insert_position(nums: List[int], target: int) -> int:
    """
    Find position where target should be inserted to maintain sorted order
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return left  # Position to insert

def search_rotated_sorted_array(nums: List[int], target: int) -> int:
    """
    Search in rotated sorted array
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Determine which half is sorted
        if nums[left] <= nums[mid]:  # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1

def find_minimum_rotated_array(nums: List[int]) -> int:
    """
    Find minimum element in rotated sorted array
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[right]:
            # Minimum is in right half
            left = mid + 1
        else:
            # Minimum is in left half (including mid)
            right = mid
    
    return nums[left]

def find_peak_element(nums: List[int]) -> int:
    """
    Find any peak element (element greater than neighbors)
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[mid + 1]:
            # Peak is in left half (including mid)
            right = mid
        else:
            # Peak is in right half
            left = mid + 1
    
    return left

def sqrt_binary_search(x: int) -> int:
    """
    Integer square root using binary search
    
    Time Complexity: O(log x)
    Space Complexity: O(1)
    """
    if x < 2:
        return x
    
    left, right = 1, x // 2
    
    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # Largest integer whose square <= x

def search_2d_matrix(matrix: List[List[int]], target: int) -> bool:
    """
    Search target in 2D matrix (sorted row-wise and column-wise)
    
    Time Complexity: O(log(m * n))
    Space Complexity: O(1)
    """
    if not matrix or not matrix[0]:
        return False
    
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        mid_value = matrix[mid // cols][mid % cols]
        
        if mid_value == target:
            return True
        elif mid_value < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False

# ==================== BINARY SEARCH ON ANSWER ====================
def binary_search_on_condition(left: int, right: int, condition: Callable[[int], bool]) -> int:
    """
    Generic binary search on answer template
    Find the smallest/largest value that satisfies a condition
    
    Time Complexity: O(log(right - left) * condition_time)
    Space Complexity: O(1)
    """
    while left < right:
        mid = left + (right - left) // 2
        
        if condition(mid):
            right = mid  # For finding minimum satisfying condition
        else:
            left = mid + 1
    
    return left

def capacity_to_ship_packages(weights: List[int], days: int) -> int:
    """
    Find minimum ship capacity to ship all packages within given days
    
    Time Complexity: O(n * log(sum(weights)))
    Space Complexity: O(1)
    """
    def can_ship_in_days(capacity: int) -> bool:
        current_weight = 0
        days_needed = 1
        
        for weight in weights:
            if current_weight + weight > capacity:
                days_needed += 1
                current_weight = weight
            else:
                current_weight += weight
        
        return days_needed <= days
    
    left = max(weights)  # Minimum possible capacity
    right = sum(weights)  # Maximum possible capacity
    
    return binary_search_on_condition(left, right, can_ship_in_days)

def koko_eating_bananas(piles: List[int], hours: int) -> int:
    """
    Find minimum eating speed to finish all bananas within hours
    
    Time Complexity: O(n * log(max(piles)))
    Space Complexity: O(1)
    """
    def can_finish_in_time(speed: int) -> bool:
        time_needed = 0
        for pile in piles:
            time_needed += (pile + speed - 1) // speed  # Ceiling division
        return time_needed <= hours
    
    left, right = 1, max(piles)
    
    return binary_search_on_condition(left, right, can_finish_in_time)

def find_kth_smallest_in_matrix(matrix: List[List[int]], k: int) -> int:
    """
    Find kth smallest element in sorted matrix
    
    Time Complexity: O(n * log(max - min))
    Space Complexity: O(1)
    """
    def count_less_equal(target: int) -> int:
        count = 0
        n = len(matrix)
        row, col = 0, n - 1
        
        while row < n and col >= 0:
            if matrix[row][col] <= target:
                count += col + 1
                row += 1
            else:
                col -= 1
        
        return count
    
    left, right = matrix[0][0], matrix[-1][-1]
    
    while left < right:
        mid = left + (right - left) // 2
        
        if count_less_equal(mid) < k:
            left = mid + 1
        else:
            right = mid
    
    return left

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("=== Basic Binary Search ===")
    sorted_array = [1, 3, 5, 7, 9, 11, 13, 15]
    target = 7
    print(f"Array: {sorted_array}")
    print(f"Search for {target}:")
    print(f"  Linear search: {linear_search(sorted_array, target)}")
    print(f"  Binary search: {binary_search_iterative(sorted_array, target)}")
    
    print("\n=== Search with Duplicates ===")
    duplicate_array = [5, 7, 7, 8, 8, 10]
    target = 8
    print(f"Array: {duplicate_array}")
    print(f"Search for {target}:")
    print(f"  First occurrence: {find_first_occurrence(duplicate_array, target)}")
    print(f"  Last occurrence: {find_last_occurrence(duplicate_array, target)}")
    print(f"  Range: {search_range(duplicate_array, target)}")
    
    print("\n=== Insert Position ===")
    array = [1, 3, 5, 6]
    targets = [5, 2, 7, 0]
    print(f"Array: {array}")
    for t in targets:
        pos = search_insert_position(array, t)
        print(f"  Insert {t} at position: {pos}")
    
    print("\n=== Rotated Sorted Array ===")
    rotated = [4, 5, 6, 7, 0, 1, 2]
    search_targets = [0, 3]
    print(f"Rotated array: {rotated}")
    for t in search_targets:
        idx = search_rotated_sorted_array(rotated, t)
        print(f"  Search {t}: {'Found at ' + str(idx) if idx != -1 else 'Not found'}")
    
    print(f"  Minimum element: {find_minimum_rotated_array(rotated)}")
    
    print("\n=== Peak Element ===")
    peak_array = [1, 2, 3, 1]
    peak_idx = find_peak_element(peak_array)
    print(f"Array: {peak_array}")
    print(f"Peak element at index {peak_idx}: {peak_array[peak_idx]}")
    
    print("\n=== Square Root ===")
    numbers = [4, 8, 16, 25]
    for num in numbers:
        sqrt_val = sqrt_binary_search(num)
        print(f"  sqrt({num}) = {sqrt_val}")
    
    print("\n=== Binary Search on Answer ===")
    # Ship capacity problem
    weights = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    days = 5
    capacity = capacity_to_ship_packages(weights, days)
    print(f"Packages: {weights}")
    print(f"Ship within {days} days, minimum capacity: {capacity}")
    
    # Koko eating bananas
    piles = [3, 6, 7, 11]
    hours = 8
    speed = koko_eating_bananas(piles, hours)
    print(f"Banana piles: {piles}")
    print(f"Finish in {hours} hours, minimum speed: {speed}")

"""
BINARY SEARCH PATTERNS:

1. Classic Binary Search:
   - Find exact target in sorted array
   - Most basic form

2. Find Boundary:
   - First/last occurrence of target
   - Search insert position
   - Lower/upper bound

3. Search Space Reduction:
   - Rotated sorted array
   - Peak element
   - Mountain array

4. Binary Search on Answer:
   - Minimize/maximize some value
   - Capacity problems
   - Rate problems
   - Kth smallest/largest

WHEN TO USE BINARY SEARCH:
- Array is sorted (or can be conceptually sorted)
- Looking for specific value or position
- Optimization problems with monotonic properties
- Search space can be divided in half

KEY INSIGHTS:
- Always use mid = left + (right - left) // 2 to avoid overflow
- Be careful with boundary conditions (< vs <=)
- For finding boundaries, adjust search direction after finding target
- Binary search on answer: define condition function clearly

COMMON MISTAKES:
- Off-by-one errors in boundary conditions
- Infinite loops due to incorrect boundary updates
- Not handling edge cases (empty arrays)
- Wrong termination condition (< vs <=)

TEMPLATE FOR BINARY SEARCH ON ANSWER:
```python
def binary_search_condition(left, right, condition):
    while left < right:
        mid = left + (right - left) // 2
        if condition(mid):
            right = mid  # or left = mid + 1
        else:
            left = mid + 1  # or right = mid
    return left
```
"""