"""
Sliding Window Algorithm
=======================

Used for: Subarray/substring problems, fixed or variable window sizes

Time Complexity: O(n) - single pass through array
Space Complexity: O(1) to O(k) depending on window requirements
"""

from typing import List, Dict, Optional
from collections import defaultdict, deque

# ==================== BRUTE FORCE APPROACH ====================
def max_sum_subarray_k_brute_force(nums: List[int], k: int) -> int:
    """
    Brute Force: Calculate sum for every k-length subarray
    
    Time Complexity: O(n * k) - for each position, calculate k-length sum
    Space Complexity: O(1)
    
    Problems:
    - Recalculates overlapping sums
    - Inefficient for large arrays or large k
    - Doesn't utilize the sliding nature
    """
    n = len(nums)
    max_sum = float('-inf')
    
    for i in range(n - k + 1):
        current_sum = 0
        for j in range(i, i + k):
            current_sum += nums[j]
        max_sum = max(max_sum, current_sum)
    
    return max_sum

def longest_substring_without_repeating_brute(s: str) -> int:
    """
    Brute Force: Check all substrings for uniqueness
    
    Time Complexity: O(n³) - generate all substrings O(n²), check uniqueness O(n)
    Space Complexity: O(min(m, n)) - for character set
    """
    n = len(s)
    max_length = 0
    
    for i in range(n):
        for j in range(i, n):
            substring = s[i:j+1]
            if len(set(substring)) == len(substring):  # All characters unique
                max_length = max(max_length, len(substring))
    
    return max_length

# ==================== OPTIMIZED APPROACH ====================

# 1. FIXED SIZE SLIDING WINDOW
def max_sum_subarray_k_optimized(nums: List[int], k: int) -> int:
    """
    Fixed Size Sliding Window for maximum sum subarray
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1)
    
    Advantages:
    - Only calculates each element's contribution once
    - Slides window efficiently
    - Linear time complexity
    """
    if len(nums) < k:
        return 0
    
    # Calculate sum of first window
    current_sum = sum(nums[:k])
    max_sum = current_sum
    
    # Slide the window
    for i in range(k, len(nums)):
        current_sum = current_sum - nums[i - k] + nums[i]
        max_sum = max(max_sum, current_sum)
    
    return max_sum

def average_subarrays_k(nums: List[int], k: int) -> List[float]:
    """
    Find averages of all subarrays of size k
    
    Time Complexity: O(n)
    Space Complexity: O(n) - for result array
    """
    if len(nums) < k:
        return []
    
    result = []
    window_sum = sum(nums[:k])
    result.append(window_sum / k)
    
    for i in range(k, len(nums)):
        window_sum = window_sum - nums[i - k] + nums[i]
        result.append(window_sum / k)
    
    return result

def max_sum_subarray_size_k_all_positions(nums: List[int], k: int) -> List[int]:
    """
    Return maximum sum for each k-size subarray
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if len(nums) < k:
        return []
    
    result = []
    current_sum = sum(nums[:k])
    result.append(current_sum)
    
    for i in range(k, len(nums)):
        current_sum = current_sum - nums[i - k] + nums[i]
        result.append(current_sum)
    
    return result

# 2. VARIABLE SIZE SLIDING WINDOW
def longest_substring_without_repeating_optimized(s: str) -> int:
    """
    Variable Size Sliding Window for longest substring without repeating chars
    
    Time Complexity: O(n) - each character visited at most twice
    Space Complexity: O(min(m, n)) - for character set
    """
    char_index = {}
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        if s[right] in char_index and char_index[s[right]] >= left:
            left = char_index[s[right]] + 1
        
        char_index[s[right]] = right
        max_length = max(max_length, right - left + 1)
    
    return max_length

def min_window_substring(s: str, t: str) -> str:
    """
    Minimum Window Substring containing all characters of t
    
    Time Complexity: O(n + m) - where n is len(s), m is len(t)
    Space Complexity: O(m) - for character frequency map
    """
    if not s or not t:
        return ""
    
    # Count characters in t
    t_count = defaultdict(int)
    for char in t:
        t_count[char] += 1
    
    required = len(t_count)  # Number of unique characters in t
    formed = 0  # Number of unique characters in current window with desired frequency
    
    window_counts = defaultdict(int)
    left = 0
    min_len = float('inf')
    min_start = 0
    
    for right in range(len(s)):
        # Add character from right to window
        char = s[right]
        window_counts[char] += 1
        
        # Check if current character's frequency matches desired frequency in t
        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1
        
        # Try to contract window from left
        while left <= right and formed == required:
            # Update minimum window if current is smaller
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_start = left
            
            # Remove character from left
            left_char = s[left]
            window_counts[left_char] -= 1
            if left_char in t_count and window_counts[left_char] < t_count[left_char]:
                formed -= 1
            
            left += 1
    
    return "" if min_len == float('inf') else s[min_start:min_start + min_len]

def longest_substring_k_distinct(s: str, k: int) -> int:
    """
    Longest substring with at most k distinct characters
    
    Time Complexity: O(n)
    Space Complexity: O(k)
    """
    if not s or k == 0:
        return 0
    
    char_count = defaultdict(int)
    left = 0
    max_length = 0
    
    for right in range(len(s)):
        # Add character to window
        char_count[s[right]] += 1
        
        # Contract window if more than k distinct characters
        while len(char_count) > k:
            left_char = s[left]
            char_count[left_char] -= 1
            if char_count[left_char] == 0:
                del char_count[left_char]
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length

def max_consecutive_ones_with_k_flips(nums: List[int], k: int) -> int:
    """
    Maximum consecutive 1s after flipping at most k zeros
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left = 0
    zeros_count = 0
    max_length = 0
    
    for right in range(len(nums)):
        if nums[right] == 0:
            zeros_count += 1
        
        # Contract window if more than k zeros
        while zeros_count > k:
            if nums[left] == 0:
                zeros_count -= 1
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length

def subarray_sum_equals_k(nums: List[int], k: int) -> int:
    """
    Count subarrays with sum equal to k
    
    Time Complexity: O(n)
    Space Complexity: O(n) - for prefix sum map
    
    Note: This uses prefix sum concept, not traditional sliding window
    """
    count = 0
    prefix_sum = 0
    prefix_sums = defaultdict(int)
    prefix_sums[0] = 1  # Empty prefix
    
    for num in nums:
        prefix_sum += num
        
        # Check if there's a prefix sum such that current - prefix = k
        if prefix_sum - k in prefix_sums:
            count += prefix_sums[prefix_sum - k]
        
        prefix_sums[prefix_sum] += 1
    
    return count

def min_subarray_sum_geq_target(nums: List[int], target: int) -> int:
    """
    Minimum length subarray with sum >= target
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left = 0
    current_sum = 0
    min_length = float('inf')
    
    for right in range(len(nums)):
        current_sum += nums[right]
        
        # Contract window while sum >= target
        while current_sum >= target:
            min_length = min(min_length, right - left + 1)
            current_sum -= nums[left]
            left += 1
    
    return min_length if min_length != float('inf') else 0

def sliding_window_maximum(nums: List[int], k: int) -> List[int]:
    """
    Sliding Window Maximum using deque
    
    Time Complexity: O(n) - each element added/removed at most once
    Space Complexity: O(k) - deque size
    """
    if not nums:
        return []
    
    dq = deque()  # Store indices
    result = []
    
    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # Remove indices with smaller values (they can't be maximum)
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()
        
        dq.append(i)
        
        # Add maximum to result if window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

def permutation_in_string(s1: str, s2: str) -> bool:
    """
    Check if any permutation of s1 is substring of s2
    
    Time Complexity: O(n) - where n is len(s2)
    Space Complexity: O(1) - fixed size character count
    """
    if len(s1) > len(s2):
        return False
    
    # Count characters in s1
    s1_count = [0] * 26
    for char in s1:
        s1_count[ord(char) - ord('a')] += 1
    
    window_size = len(s1)
    window_count = [0] * 26
    
    # Initialize first window
    for i in range(window_size):
        window_count[ord(s2[i]) - ord('a')] += 1
    
    if window_count == s1_count:
        return True
    
    # Slide window
    for i in range(window_size, len(s2)):
        # Add new character
        window_count[ord(s2[i]) - ord('a')] += 1
        # Remove old character
        window_count[ord(s2[i - window_size]) - ord('a')] -= 1
        
        if window_count == s1_count:
            return True
    
    return False

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("=== Fixed Size Sliding Window ===")
    nums = [2, 1, 5, 1, 3, 2]
    k = 3
    print(f"Array: {nums}, k: {k}")
    print(f"Max sum (brute force): {max_sum_subarray_k_brute_force(nums, k)}")
    print(f"Max sum (optimized): {max_sum_subarray_k_optimized(nums, k)}")
    print(f"All sums: {max_sum_subarray_size_k_all_positions(nums, k)}")
    
    print("\n=== Variable Size Sliding Window ===")
    s = "abcabcbb"
    print(f"String: '{s}'")
    print(f"Longest substring without repeating (brute): {longest_substring_without_repeating_brute(s)}")
    print(f"Longest substring without repeating (optimized): {longest_substring_without_repeating_optimized(s)}")
    
    print("\n=== Minimum Window Substring ===")
    s, t = "ADOBECODEBANC", "ABC"
    print(f"String: '{s}', Target: '{t}'")
    print(f"Minimum window: '{min_window_substring(s, t)}'")
    
    print("\n=== Longest Substring with K Distinct ===")
    s, k = "eceba", 2
    print(f"String: '{s}', k: {k}")
    print(f"Longest substring: {longest_substring_k_distinct(s, k)}")
    
    print("\n=== Max Consecutive Ones with K Flips ===")
    nums, k = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2
    print(f"Array: {nums}, k: {k}")
    print(f"Max consecutive ones: {max_consecutive_ones_with_k_flips(nums, k)}")
    
    print("\n=== Subarray Sum Equals K ===")
    nums, k = [1, 1, 1], 2
    print(f"Array: {nums}, k: {k}")
    print(f"Count of subarrays: {subarray_sum_equals_k(nums, k)}")
    
    print("\n=== Minimum Subarray Sum >= Target ===")
    nums, target = [2, 3, 1, 2, 4, 3], 7
    print(f"Array: {nums}, target: {target}")
    print(f"Minimum length: {min_subarray_sum_geq_target(nums, target)}")
    
    print("\n=== Sliding Window Maximum ===")
    nums, k = [1, 3, -1, -3, 5, 3, 6, 7], 3
    print(f"Array: {nums}, k: {k}")
    print(f"Sliding maximums: {sliding_window_maximum(nums, k)}")
    
    print("\n=== Permutation in String ===")
    s1, s2 = "ab", "eidbaooo"
    print(f"s1: '{s1}', s2: '{s2}'")
    print(f"Contains permutation: {permutation_in_string(s1, s2)}")

"""
SLIDING WINDOW PATTERNS:

1. Fixed Size Window:
   - Maximum/minimum sum of k elements
   - Average of subarrays
   - All problems with fixed window size
   
   Template:
   ```
   window_sum = sum(arr[:k])  # Initialize
   for i in range(k, len(arr)):
       window_sum = window_sum - arr[i-k] + arr[i]  # Slide
   ```

2. Variable Size Window - Expand until condition violated:
   - Longest substring with constraint
   - Maximum window satisfying condition
   
   Template:
   ```
   left = 0
   for right in range(len(arr)):
       # Add arr[right] to window
       while condition_violated:
           # Remove arr[left] from window
           left += 1
   ```

3. Variable Size Window - Shrink until condition satisfied:
   - Minimum window containing target
   - Shortest subarray with sum >= target
   
   Template:
   ```
   left = 0
   for right in range(len(arr)):
       # Add arr[right] to window
       while condition_satisfied:
           # Update result
           # Remove arr[left] from window
           left += 1
   ```

WHEN TO USE SLIDING WINDOW:
- Contiguous subarray/substring problems
- Need to find optimal window (min/max size)
- Constraint can be maintained incrementally
- Can avoid recalculating from scratch

KEY INSIGHTS:
- Two pointers: left (start), right (end) of window
- Maintain window state (sum, count, frequency map)
- Expand window by moving right pointer
- Shrink window by moving left pointer
- Update result at appropriate times

COMMON MISTAKES:
- Not updating window state correctly when sliding
- Wrong condition for expanding/shrinking window
- Forgetting to update result at right time
- Off-by-one errors in window boundaries
- Not handling edge cases (empty arrays, k > n)

OPTIMIZATION TECHNIQUES:
- Use deque for sliding window maximum/minimum
- Use hash map for character/element frequencies
- Use prefix sums for range sum queries
- Two pointers technique for specific patterns
"""