"""
Dynamic Programming Algorithm
============================

Used for: Overlapping subproblems, optimization problems, counting problems

Time Complexity: Varies (typically O(n²) to O(n³))
Space Complexity: O(n) to O(n²) depending on approach and optimization
"""

from typing import List, Dict, Tuple, Optional
from functools import lru_cache

# ==================== BRUTE FORCE APPROACH ====================
def fibonacci_recursive_brute(n: int) -> int:
    """
    Brute Force Fibonacci: Pure recursion without memoization
    
    Time Complexity: O(2^n) - exponential due to repeated calculations
    Space Complexity: O(n) - recursion stack depth
    
    Problems:
    - Exponential time complexity
    - Recalculates same subproblems multiple times
    - Inefficient for large inputs
    """
    if n <= 1:
        return n
    return fibonacci_recursive_brute(n - 1) + fibonacci_recursive_brute(n - 2)

def coin_change_brute_force(coins: List[int], amount: int) -> int:
    """
    Brute Force Coin Change: Try all combinations recursively
    
    Time Complexity: O(amount^coins) - exponential
    Space Complexity: O(amount) - recursion depth
    """
    def helper(remaining: int) -> int:
        if remaining == 0:
            return 0
        if remaining < 0:
            return -1
        
        min_coins = float('inf')
        for coin in coins:
            result = helper(remaining - coin)
            if result != -1:
                min_coins = min(min_coins, result + 1)
        
        return min_coins if min_coins != float('inf') else -1
    
    return helper(amount)

# ==================== OPTIMIZED APPROACH ====================

# 1. TOP-DOWN APPROACH (Memoization)
def fibonacci_memoized(n: int) -> int:
    """
    Fibonacci with Memoization (Top-Down DP)
    
    Time Complexity: O(n) - each subproblem calculated once
    Space Complexity: O(n) - memoization table + recursion stack
    """
    memo = {}
    
    def helper(num: int) -> int:
        if num in memo:
            return memo[num]
        
        if num <= 1:
            return num
        
        memo[num] = helper(num - 1) + helper(num - 2)
        return memo[num]
    
    return helper(n)

@lru_cache(maxsize=None)
def fibonacci_lru_cache(n: int) -> int:
    """
    Fibonacci using Python's built-in LRU cache
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if n <= 1:
        return n
    return fibonacci_lru_cache(n - 1) + fibonacci_lru_cache(n - 2)

# 2. BOTTOM-UP APPROACH (Tabulation)
def fibonacci_tabulation(n: int) -> int:
    """
    Fibonacci with Tabulation (Bottom-Up DP)
    
    Time Complexity: O(n)
    Space Complexity: O(n) - DP table
    """
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[1] = 1
    
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

def fibonacci_space_optimized(n: int) -> int:
    """
    Space-optimized Fibonacci
    
    Time Complexity: O(n)
    Space Complexity: O(1) - only store last two values
    """
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1

# 3. CLASSIC DP PROBLEMS

def coin_change_optimized(coins: List[int], amount: int) -> int:
    """
    Coin Change with DP (Bottom-Up)
    
    Time Complexity: O(amount * coins)
    Space Complexity: O(amount)
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    
    return dp[amount] if dp[amount] != float('inf') else -1

def longest_increasing_subsequence(nums: List[int]) -> int:
    """
    Longest Increasing Subsequence (LIS)
    
    Time Complexity: O(n²) - DP approach
    Space Complexity: O(n)
    """
    if not nums:
        return 0
    
    n = len(nums)
    dp = [1] * n  # dp[i] = length of LIS ending at index i
    
    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    return max(dp)

def longest_increasing_subsequence_optimized(nums: List[int]) -> int:
    """
    LIS with Binary Search optimization
    
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    """
    import bisect
    
    if not nums:
        return 0
    
    tails = []  # tails[i] = smallest ending element of LIS of length i+1
    
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    
    return len(tails)

def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Longest Common Subsequence (LCS)
    
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

def edit_distance(word1: str, word2: str) -> int:
    """
    Edit Distance (Levenshtein Distance)
    
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )
    
    return dp[m][n]

def knapsack_01(weights: List[int], values: List[int], capacity: int) -> int:
    """
    0/1 Knapsack Problem
    
    Time Complexity: O(n * capacity)
    Space Complexity: O(n * capacity)
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't take item i-1
            dp[i][w] = dp[i - 1][w]
            
            # Take item i-1 if possible
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
    
    return dp[n][capacity]

def knapsack_01_space_optimized(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Space-optimized 0/1 Knapsack
    
    Time Complexity: O(n * capacity)
    Space Complexity: O(capacity)
    """
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        # Traverse backwards to avoid using updated values
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]

def max_subarray_sum(nums: List[int]) -> int:
    """
    Maximum Subarray Sum (Kadane's Algorithm)
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    max_sum = current_sum = nums[0]
    
    for i in range(1, len(nums)):
        current_sum = max(nums[i], current_sum + nums[i])
        max_sum = max(max_sum, current_sum)
    
    return max_sum

def house_robber(nums: List[int]) -> int:
    """
    House Robber Problem
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2, prev1 = prev1, current
    
    return prev1

def climb_stairs(n: int) -> int:
    """
    Climbing Stairs Problem
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1

def unique_paths(m: int, n: int) -> int:
    """
    Unique Paths in Grid
    
    Time Complexity: O(m * n)
    Space Complexity: O(n) - space optimized
    """
    dp = [1] * n
    
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    
    return dp[n - 1]

def word_break(s: str, wordDict: List[str]) -> bool:
    """
    Word Break Problem
    
    Time Complexity: O(n² * m) where m is average word length
    Space Complexity: O(n)
    """
    word_set = set(wordDict)
    dp = [False] * (len(s) + 1)
    dp[0] = True
    
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    
    return dp[len(s)]

def palindrome_partitioning_min_cuts(s: str) -> int:
    """
    Minimum cuts needed for palindrome partitioning
    
    Time Complexity: O(n²)
    Space Complexity: O(n²)
    """
    n = len(s)
    
    # Precompute palindrome table
    is_palindrome = [[False] * n for _ in range(n)]
    
    # Single characters are palindromes
    for i in range(n):
        is_palindrome[i][i] = True
    
    # Check for palindromes of length 2
    for i in range(n - 1):
        is_palindrome[i][i + 1] = (s[i] == s[i + 1])
    
    # Check for palindromes of length 3 and more
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            is_palindrome[i][j] = (s[i] == s[j] and is_palindrome[i + 1][j - 1])
    
    # DP for minimum cuts
    cuts = [0] * n
    
    for i in range(n):
        if is_palindrome[0][i]:
            cuts[i] = 0
        else:
            cuts[i] = float('inf')
            for j in range(i):
                if is_palindrome[j + 1][i]:
                    cuts[i] = min(cuts[i], cuts[j] + 1)
    
    return cuts[n - 1]

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("=== Fibonacci Comparison ===")
    n = 10
    print(f"Fibonacci({n}):")
    print(f"  Brute force: {fibonacci_recursive_brute(n)}")
    print(f"  Memoized: {fibonacci_memoized(n)}")
    print(f"  Tabulation: {fibonacci_tabulation(n)}")
    print(f"  Space optimized: {fibonacci_space_optimized(n)}")
    
    print("\n=== Coin Change ===")
    coins = [1, 3, 4]
    amount = 6
    print(f"Coins: {coins}, Amount: {amount}")
    print(f"  Minimum coins needed: {coin_change_optimized(coins, amount)}")
    
    print("\n=== Longest Increasing Subsequence ===")
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    print(f"Array: {nums}")
    print(f"  LIS length (O(n²)): {longest_increasing_subsequence(nums)}")
    print(f"  LIS length (O(n log n)): {longest_increasing_subsequence_optimized(nums)}")
    
    print("\n=== Longest Common Subsequence ===")
    text1, text2 = "abcde", "ace"
    print(f"Text1: '{text1}', Text2: '{text2}'")
    print(f"  LCS length: {longest_common_subsequence(text1, text2)}")
    
    print("\n=== Edit Distance ===")
    word1, word2 = "horse", "ros"
    print(f"Word1: '{word1}', Word2: '{word2}'")
    print(f"  Edit distance: {edit_distance(word1, word2)}")
    
    print("\n=== 0/1 Knapsack ===")
    weights = [1, 3, 4, 5]
    values = [1, 4, 5, 7]
    capacity = 7
    print(f"Weights: {weights}, Values: {values}, Capacity: {capacity}")
    print(f"  Maximum value: {knapsack_01(weights, values, capacity)}")
    print(f"  Space optimized: {knapsack_01_space_optimized(weights, values, capacity)}")
    
    print("\n=== Maximum Subarray Sum ===")
    nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    print(f"Array: {nums}")
    print(f"  Maximum sum: {max_subarray_sum(nums)}")
    
    print("\n=== House Robber ===")
    houses = [2, 7, 9, 3, 1]
    print(f"House values: {houses}")
    print(f"  Maximum robbery: {house_robber(houses)}")
    
    print("\n=== Climbing Stairs ===")
    stairs = 5
    print(f"Number of stairs: {stairs}")
    print(f"  Ways to climb: {climb_stairs(stairs)}")
    
    print("\n=== Unique Paths ===")
    m, n = 3, 7
    print(f"Grid size: {m} x {n}")
    print(f"  Unique paths: {unique_paths(m, n)}")
    
    print("\n=== Word Break ===")
    s = "leetcode"
    wordDict = ["leet", "code"]
    print(f"String: '{s}', Dictionary: {wordDict}")
    print(f"  Can break: {word_break(s, wordDict)}")

"""
DYNAMIC PROGRAMMING PATTERNS:

1. Linear DP:
   - Fibonacci, climbing stairs
   - House robber, maximum subarray
   - O(n) time, O(1) space possible

2. 2D DP:
   - Longest common subsequence
   - Edit distance, unique paths
   - Knapsack problems
   - O(m*n) time and space

3. Interval DP:
   - Palindrome partitioning
   - Matrix chain multiplication
   - Burst balloons

4. Tree DP:
   - Binary tree problems
   - Maximum path sum
   - Diameter of tree

5. State Machine DP:
   - Stock problems
   - String matching with patterns

WHEN TO USE DP:
- Overlapping subproblems exist
- Optimal substructure property
- Can break problem into smaller subproblems
- Need to find optimal solution (min/max)
- Counting problems (number of ways)

TOP-DOWN vs BOTTOM-UP:
- Top-down (Memoization): 
  * Natural recursion with caching
  * Only computes needed subproblems
  * Higher space due to recursion stack

- Bottom-up (Tabulation):
  * Iterative approach
  * Computes all subproblems
  * Better space efficiency
  * Avoids recursion overhead

OPTIMIZATION TECHNIQUES:
- Space optimization: Use O(1) or O(n) instead of O(n²)
- Rolling array: Only keep necessary previous states
- State compression: Represent states more efficiently

COMMON MISTAKES:
- Not identifying overlapping subproblems
- Incorrect base cases
- Wrong transition formula
- Not considering all possible states
- Off-by-one errors in indexing
"""