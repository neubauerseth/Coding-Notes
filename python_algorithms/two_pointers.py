"""
Two Pointers Algorithm
======================

Used for: Array/string problems, finding pairs, palindromes, sorted data

Time Complexity: O(n) - single pass through data
Space Complexity: O(1) - constant extra space
"""

from typing import List, Optional, Tuple

# ==================== BRUTE FORCE APPROACH ====================
def two_sum_brute_force(nums: List[int], target: int) -> Optional[List[int]]:
    """
    Brute Force Two Sum: Check all pairs
    
    Time Complexity: O(n²) - nested loops
    Space Complexity: O(1) - no extra space
    
    Problems:
    - Quadratic time complexity
    - Inefficient for large arrays
    - Multiple unnecessary comparisons
    """
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return None

def three_sum_brute_force(nums: List[int]) -> List[List[int]]:
    """
    Brute Force Three Sum: Check all triplets
    
    Time Complexity: O(n³) - three nested loops
    Space Complexity: O(1) - excluding result space
    """
    n = len(nums)
    result = []
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if nums[i] + nums[j] + nums[k] == 0:
                    triplet = sorted([nums[i], nums[j], nums[k]])
                    if triplet not in result:
                        result.append(triplet)
    
    return result

# ==================== OPTIMIZED APPROACH ====================
def two_sum_sorted(nums: List[int], target: int) -> Optional[List[int]]:
    """
    Two Pointers Two Sum for sorted array
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - constant space
    
    Advantages:
    - Linear time complexity
    - No extra space needed
    - Works on sorted arrays
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        current_sum = nums[left] + nums[right]
        
        if current_sum == target:
            return [left, right]
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    
    return None

def three_sum_optimized(nums: List[int]) -> List[List[int]]:
    """
    Optimized Three Sum using sorting + two pointers
    
    Time Complexity: O(n²) - sorting O(n log n) + O(n²) for pairs
    Space Complexity: O(1) - excluding result space
    """
    nums.sort()
    n = len(nums)
    result = []
    
    for i in range(n - 2):
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        
        left, right = i + 1, n - 1
        
        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]
            
            if current_sum == 0:
                result.append([nums[i], nums[left], nums[right]])
                
                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                
                left += 1
                right -= 1
            elif current_sum < 0:
                left += 1
            else:
                right -= 1
    
    return result

def container_with_most_water(heights: List[int]) -> int:
    """
    Container with most water using two pointers
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left, right = 0, len(heights) - 1
    max_area = 0
    
    while left < right:
        # Calculate area with current pointers
        width = right - left
        height = min(heights[left], heights[right])
        area = width * height
        max_area = max(max_area, area)
        
        # Move pointer with smaller height
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1
    
    return max_area

def is_palindrome_two_pointers(s: str) -> bool:
    """
    Check if string is palindrome using two pointers
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left, right = 0, len(s) - 1
    
    while left < right:
        # Skip non-alphanumeric characters
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1
        
        # Compare characters (case insensitive)
        if s[left].lower() != s[right].lower():
            return False
        
        left += 1
        right -= 1
    
    return True

def remove_duplicates_sorted_array(nums: List[int]) -> int:
    """
    Remove duplicates from sorted array in-place
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not nums:
        return 0
    
    slow = 0  # Pointer for next unique position
    
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    
    return slow + 1

def move_zeros_to_end(nums: List[int]) -> None:
    """
    Move all zeros to end while maintaining relative order
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    slow = 0  # Position for next non-zero element
    
    # Move all non-zero elements to front
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1
    
    # Fill remaining positions with zeros
    while slow < len(nums):
        nums[slow] = 0
        slow += 1

def partition_array(nums: List[int], pivot: int) -> int:
    """
    Partition array around pivot value (Dutch National Flag style)
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left = 0
    right = len(nums) - 1
    i = 0
    
    while i <= right:
        if nums[i] < pivot:
            nums[i], nums[left] = nums[left], nums[i]
            left += 1
            i += 1
        elif nums[i] > pivot:
            nums[i], nums[right] = nums[right], nums[i]
            right -= 1
            # Don't increment i here, need to check swapped element
        else:
            i += 1
    
    return left  # Return position where pivot region starts

def trapping_rain_water(heights: List[int]) -> int:
    """
    Calculate trapped rainwater using two pointers
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not heights:
        return 0
    
    left, right = 0, len(heights) - 1
    left_max = right_max = 0
    water_trapped = 0
    
    while left < right:
        if heights[left] < heights[right]:
            if heights[left] >= left_max:
                left_max = heights[left]
            else:
                water_trapped += left_max - heights[left]
            left += 1
        else:
            if heights[right] >= right_max:
                right_max = heights[right]
            else:
                water_trapped += right_max - heights[right]
            right -= 1
    
    return water_trapped

def longest_palindromic_substring(s: str) -> str:
    """
    Find longest palindromic substring using expand around centers
    
    Time Complexity: O(n²)
    Space Complexity: O(1)
    """
    if not s:
        return ""
    
    start = 0
    max_len = 1
    
    def expand_around_center(left: int, right: int) -> int:
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1
    
    for i in range(len(s)):
        # Check for odd length palindromes (center at i)
        len1 = expand_around_center(i, i)
        # Check for even length palindromes (center between i and i+1)
        len2 = expand_around_center(i, i + 1)
        
        current_max = max(len1, len2)
        if current_max > max_len:
            max_len = current_max
            start = i - (current_max - 1) // 2
    
    return s[start:start + max_len]

def sort_colors_dutch_flag(nums: List[int]) -> None:
    """
    Sort array of 0s, 1s, 2s using Dutch National Flag algorithm
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    red = 0    # Boundary for 0s
    white = 0  # Current position
    blue = len(nums) - 1  # Boundary for 2s
    
    while white <= blue:
        if nums[white] == 0:
            nums[red], nums[white] = nums[white], nums[red]
            red += 1
            white += 1
        elif nums[white] == 1:
            white += 1
        else:  # nums[white] == 2
            nums[white], nums[blue] = nums[blue], nums[white]
            blue -= 1
            # Don't increment white, need to check swapped element

def find_closest_pair_sum(nums: List[int], target: int) -> Tuple[int, int]:
    """
    Find pair with sum closest to target in sorted array
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    left, right = 0, len(nums) - 1
    closest_sum = float('inf')
    result_pair = (0, 0)
    
    while left < right:
        current_sum = nums[left] + nums[right]
        
        if abs(current_sum - target) < abs(closest_sum - target):
            closest_sum = current_sum
            result_pair = (nums[left], nums[right])
        
        if current_sum < target:
            left += 1
        else:
            right -= 1
    
    return result_pair

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("=== Two Sum ===")
    sorted_nums = [1, 2, 3, 4, 6, 8, 9]
    print(f"Array: {sorted_nums}")
    print(f"Two sum (target=10): {two_sum_sorted(sorted_nums, 10)}")
    
    print("\n=== Three Sum ===")
    nums = [-1, 0, 1, 2, -1, -4]
    print(f"Array: {nums}")
    print(f"Three sum (target=0): {three_sum_optimized(nums)}")
    
    print("\n=== Container With Most Water ===")
    heights = [1, 8, 6, 2, 5, 4, 8, 3, 7]
    print(f"Heights: {heights}")
    print(f"Max water: {container_with_most_water(heights)}")
    
    print("\n=== Palindrome Check ===")
    test_string = "A man, a plan, a canal: Panama"
    print(f"String: '{test_string}'")
    print(f"Is palindrome: {is_palindrome_two_pointers(test_string)}")
    
    print("\n=== Remove Duplicates ===")
    duplicate_array = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
    print(f"Original: {duplicate_array}")
    length = remove_duplicates_sorted_array(duplicate_array)
    print(f"After removal: {duplicate_array[:length]}")
    
    print("\n=== Trapping Rain Water ===")
    heights = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    print(f"Heights: {heights}")
    print(f"Trapped water: {trapping_rain_water(heights)}")
    
    print("\n=== Longest Palindromic Substring ===")
    text = "babad"
    print(f"String: '{text}'")
    print(f"Longest palindrome: '{longest_palindromic_substring(text)}'")
    
    print("\n=== Sort Colors (Dutch Flag) ===")
    colors = [2, 0, 2, 1, 1, 0]
    print(f"Before: {colors}")
    sort_colors_dutch_flag(colors)
    print(f"After: {colors}")

"""
TWO POINTERS PATTERNS:

1. Opposite Direction (Most Common):
   - Start from both ends, move towards center
   - Two sum in sorted array
   - Palindrome checking
   - Container with most water

2. Same Direction (Fast/Slow):
   - Both pointers start from beginning
   - Remove duplicates
   - Move zeros
   - Cycle detection in linked lists

3. Sliding Window Variation:
   - Maintain window with two pointers
   - Subarray problems
   - String matching

WHEN TO USE TWO POINTERS:
- Array is sorted or can be sorted
- Looking for pairs or triplets
- Need to reduce O(n²) to O(n)
- Palindrome-related problems
- Partitioning problems

ADVANTAGES:
- Reduces time complexity (often from O(n²) to O(n))
- Uses constant extra space O(1)
- Simple and intuitive approach
- No need for extra data structures

COMMON MISTAKES:
- Forgetting to handle duplicates
- Not considering edge cases (empty arrays)
- Moving wrong pointer in optimization problems
- Off-by-one errors in boundary conditions

RELATED TECHNIQUES:
- Sliding Window: Two pointers with variable distance
- Fast/Slow Pointers: Different speed movement
- Dutch National Flag: Three-way partitioning
"""