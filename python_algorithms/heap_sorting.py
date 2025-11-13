"""
Heap/Sorting Algorithms
======================

Used for: Priority queue, kth largest/smallest elements, sorting

Time Complexity: O(n log n) for sorting, O(log n) for heap operations
Space Complexity: O(1) to O(n) depending on algorithm
"""

import heapq
from typing import List, Tuple, Optional
import random

# ==================== BRUTE FORCE APPROACHES ====================
def find_kth_largest_brute_force(nums: List[int], k: int) -> int:
    """
    Brute Force: Sort entire array to find kth largest
    
    Time Complexity: O(n log n) - full sorting
    Space Complexity: O(1) or O(n) depending on sort implementation
    
    Problems:
    - Sorts entire array when only kth element needed
    - Doesn't utilize heap properties
    - Not efficient for small k
    """
    nums_sorted = sorted(nums, reverse=True)
    return nums_sorted[k - 1]

def merge_k_sorted_lists_brute_force(lists: List[List[int]]) -> List[int]:
    """
    Brute Force: Merge all lists then sort
    
    Time Complexity: O(n log n) where n is total elements
    Space Complexity: O(n)
    """
    all_elements = []
    for lst in lists:
        all_elements.extend(lst)
    
    return sorted(all_elements)

# ==================== OPTIMIZED HEAP APPROACHES ====================
def find_kth_largest_heap(nums: List[int], k: int) -> int:
    """
    Find kth largest using min heap of size k
    
    Time Complexity: O(n log k)
    Space Complexity: O(k)
    
    Advantages:
    - Only maintains k elements in heap
    - More efficient than full sorting for small k
    - Stable memory usage
    """
    heap = []
    
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, num)
        elif num > heap[0]:
            heapq.heappop(heap)
            heapq.heappush(heap, num)
    
    return heap[0]

def find_kth_smallest_heap(nums: List[int], k: int) -> int:
    """
    Find kth smallest using max heap of size k
    
    Time Complexity: O(n log k)
    Space Complexity: O(k)
    """
    heap = []  # Max heap (negate values for min heap behavior)
    
    for num in nums:
        if len(heap) < k:
            heapq.heappush(heap, -num)
        elif num < -heap[0]:
            heapq.heappop(heap)
            heapq.heappush(heap, -num)
    
    return -heap[0]

def top_k_frequent_elements(nums: List[int], k: int) -> List[int]:
    """
    Find k most frequent elements using heap
    
    Time Complexity: O(n log k)
    Space Complexity: O(n)
    """
    from collections import Counter
    
    count = Counter(nums)
    heap = []
    
    for num, freq in count.items():
        if len(heap) < k:
            heapq.heappush(heap, (freq, num))
        elif freq > heap[0][0]:
            heapq.heappop(heap)
            heapq.heappush(heap, (freq, num))
    
    return [num for freq, num in heap]

def merge_k_sorted_lists_heap(lists: List[List[int]]) -> List[int]:
    """
    Merge k sorted lists using min heap
    
    Time Complexity: O(n log k) where n is total elements, k is number of lists
    Space Complexity: O(k)
    """
    heap = []
    result = []
    
    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))  # (value, list_index, element_index)
    
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        # Add next element from same list
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    
    return result

def sliding_window_median(nums: List[int], k: int) -> List[float]:
    """
    Sliding window median using two heaps
    
    Time Complexity: O(n log k)
    Space Complexity: O(k)
    """
    from collections import defaultdict
    
    def get_median():
        if len(max_heap) == len(min_heap):
            return (-max_heap[0] + min_heap[0]) / 2.0
        else:
            return float(-max_heap[0])
    
    def add_num(num):
        if not max_heap or num <= -max_heap[0]:
            heapq.heappush(max_heap, -num)
        else:
            heapq.heappush(min_heap, num)
        balance_heaps()
    
    def remove_num(num):
        if num <= -max_heap[0]:
            hash_map[num] += 1
            if num == -max_heap[0]:
                while max_heap and hash_map[-max_heap[0]] > 0:
                    hash_map[-max_heap[0]] -= 1
                    heapq.heappop(max_heap)
        else:
            hash_map[num] += 1
            if num == min_heap[0]:
                while min_heap and hash_map[min_heap[0]] > 0:
                    hash_map[min_heap[0]] -= 1
                    heapq.heappop(min_heap)
        balance_heaps()
    
    def balance_heaps():
        if len(max_heap) > len(min_heap) + 1:
            heapq.heappush(min_heap, -heapq.heappop(max_heap))
        elif len(min_heap) > len(max_heap):
            heapq.heappush(max_heap, -heapq.heappop(min_heap))
    
    max_heap = []  # For smaller half (negated for max heap)
    min_heap = []  # For larger half
    hash_map = defaultdict(int)  # For lazy deletion
    result = []
    
    for i, num in enumerate(nums):
        add_num(num)
        
        if i >= k - 1:
            result.append(get_median())
            remove_num(nums[i - k + 1])
    
    return result

# ==================== SORTING ALGORITHMS ====================
def bubble_sort(nums: List[int]) -> List[int]:
    """
    Bubble Sort - Simple but inefficient
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    nums = nums.copy()
    n = len(nums)
    
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                swapped = True
        if not swapped:
            break
    
    return nums

def selection_sort(nums: List[int]) -> List[int]:
    """
    Selection Sort
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    nums = nums.copy()
    n = len(nums)
    
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if nums[j] < nums[min_idx]:
                min_idx = j
        nums[i], nums[min_idx] = nums[min_idx], nums[i]
    
    return nums

def insertion_sort(nums: List[int]) -> List[int]:
    """
    Insertion Sort - Efficient for small arrays
    
    Time Complexity: O(n²) worst case, O(n) best case
    Space Complexity: O(1)
    """
    nums = nums.copy()
    
    for i in range(1, len(nums)):
        key = nums[i]
        j = i - 1
        
        while j >= 0 and nums[j] > key:
            nums[j + 1] = nums[j]
            j -= 1
        
        nums[j + 1] = key
    
    return nums

def merge_sort(nums: List[int]) -> List[int]:
    """
    Merge Sort - Stable, guaranteed O(n log n)
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    if len(nums) <= 1:
        return nums
    
    mid = len(nums) // 2
    left = merge_sort(nums[:mid])
    right = merge_sort(nums[mid:])
    
    return merge(left, right)

def merge(left: List[int], right: List[int]) -> List[int]:
    """Helper function for merge sort"""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(nums: List[int]) -> List[int]:
    """
    Quick Sort - Average O(n log n), worst O(n²)
    
    Time Complexity: O(n log n) average, O(n²) worst
    Space Complexity: O(log n) average
    """
    nums = nums.copy()
    quick_sort_helper(nums, 0, len(nums) - 1)
    return nums

def quick_sort_helper(nums: List[int], low: int, high: int):
    """Helper function for quick sort"""
    if low < high:
        pi = partition(nums, low, high)
        quick_sort_helper(nums, low, pi - 1)
        quick_sort_helper(nums, pi + 1, high)

def partition(nums: List[int], low: int, high: int) -> int:
    """Partition function for quick sort"""
    pivot = nums[high]
    i = low - 1
    
    for j in range(low, high):
        if nums[j] <= pivot:
            i += 1
            nums[i], nums[j] = nums[j], nums[i]
    
    nums[i + 1], nums[high] = nums[high], nums[i + 1]
    return i + 1

def heap_sort(nums: List[int]) -> List[int]:
    """
    Heap Sort
    
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    """
    nums = nums.copy()
    n = len(nums)
    
    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(nums, n, i)
    
    # Extract elements from heap
    for i in range(n - 1, 0, -1):
        nums[0], nums[i] = nums[i], nums[0]
        heapify(nums, i, 0)
    
    return nums

def heapify(nums: List[int], n: int, i: int):
    """Helper function for heap sort"""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and nums[left] > nums[largest]:
        largest = left
    
    if right < n and nums[right] > nums[largest]:
        largest = right
    
    if largest != i:
        nums[i], nums[largest] = nums[largest], nums[i]
        heapify(nums, n, largest)

def counting_sort(nums: List[int]) -> List[int]:
    """
    Counting Sort - For integers in small range
    
    Time Complexity: O(n + k) where k is range of input
    Space Complexity: O(k)
    """
    if not nums:
        return nums
    
    min_val, max_val = min(nums), max(nums)
    range_val = max_val - min_val + 1
    
    count = [0] * range_val
    output = [0] * len(nums)
    
    # Count occurrences
    for num in nums:
        count[num - min_val] += 1
    
    # Calculate cumulative count
    for i in range(1, range_val):
        count[i] += count[i - 1]
    
    # Build output array
    for i in range(len(nums) - 1, -1, -1):
        output[count[nums[i] - min_val] - 1] = nums[i]
        count[nums[i] - min_val] -= 1
    
    return output

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("=== Kth Largest/Smallest ===")
    nums = [3, 2, 1, 5, 6, 4]
    k = 2
    print(f"Array: {nums}, k: {k}")
    print(f"Kth largest (brute force): {find_kth_largest_brute_force(nums, k)}")
    print(f"Kth largest (heap): {find_kth_largest_heap(nums, k)}")
    print(f"Kth smallest (heap): {find_kth_smallest_heap(nums, k)}")
    
    print("\n=== Top K Frequent ===")
    nums = [1, 1, 1, 2, 2, 3]
    k = 2
    print(f"Array: {nums}, k: {k}")
    print(f"Top {k} frequent: {top_k_frequent_elements(nums, k)}")
    
    print("\n=== Merge K Sorted Lists ===")
    lists = [[1, 4, 5], [1, 3, 4], [2, 6]]
    print(f"Lists: {lists}")
    print(f"Merged (brute force): {merge_k_sorted_lists_brute_force(lists)}")
    print(f"Merged (heap): {merge_k_sorted_lists_heap(lists)}")
    
    print("\n=== Sliding Window Median ===")
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    print(f"Array: {nums}, k: {k}")
    medians = sliding_window_median(nums, k)
    print(f"Sliding medians: {medians}")
    
    print("\n=== Sorting Algorithms Comparison ===")
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original: {test_array}")
    print(f"Bubble sort: {bubble_sort(test_array)}")
    print(f"Selection sort: {selection_sort(test_array)}")
    print(f"Insertion sort: {insertion_sort(test_array)}")
    print(f"Merge sort: {merge_sort(test_array)}")
    print(f"Quick sort: {quick_sort(test_array)}")
    print(f"Heap sort: {heap_sort(test_array)}")
    print(f"Counting sort: {counting_sort(test_array)}")
    
    print("\n=== Performance Characteristics ===")
    print("Bubble Sort:    O(n²) time, O(1) space - Simple but inefficient")
    print("Selection Sort: O(n²) time, O(1) space - Minimizes swaps")
    print("Insertion Sort: O(n²) time, O(1) space - Efficient for small/nearly sorted")
    print("Merge Sort:     O(n log n) time, O(n) space - Stable, guaranteed performance")
    print("Quick Sort:     O(n log n) avg, O(n²) worst, O(log n) space - Fast in practice")
    print("Heap Sort:      O(n log n) time, O(1) space - Guaranteed performance")
    print("Counting Sort:  O(n + k) time, O(k) space - For integers in small range")

"""
HEAP/SORTING PATTERNS:

1. Heap Applications:
   - Priority Queue: Always get min/max element
   - Top K Problems: Maintain heap of size k
   - Median Finding: Two heaps (max heap + min heap)
   - Merge K Sorted: Use heap to track smallest elements

2. When to Use Each Sort:
   - Bubble Sort: Educational purposes only
   - Selection Sort: When memory writes are expensive
   - Insertion Sort: Small arrays, nearly sorted data
   - Merge Sort: Need stable sort, guaranteed O(n log n)
   - Quick Sort: General purpose, average case performance
   - Heap Sort: Guaranteed O(n log n), in-place
   - Counting Sort: Integer data with small range

HEAP OPERATIONS:
- heappush(heap, item): Add item - O(log n)
- heappop(heap): Remove and return min - O(log n)
- heappushpop(heap, item): Push then pop - O(log n)
- heapreplace(heap, item): Pop then push - O(log n)
- heapify(iterable): Create heap from list - O(n)

SORTING STABILITY:
- Stable: Merge Sort, Insertion Sort, Bubble Sort, Counting Sort
- Unstable: Quick Sort, Heap Sort, Selection Sort

SORTING RECOMMENDATIONS:
- General purpose: Python's built-in sorted() (Timsort)
- Small arrays: Insertion Sort
- Memory constrained: Heap Sort or Quick Sort
- Need stability: Merge Sort
- Integer data: Counting Sort or Radix Sort
- Nearly sorted: Insertion Sort

HEAP TRICKS:
- For max heap, negate values in min heap
- For sliding window median, use two heaps
- For top K, use heap of size K
- For merge K sorted, use heap with (value, source) tuples
"""