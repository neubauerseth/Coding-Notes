"""
Backtracking Algorithm
=====================

Used for: Exhaustive search, constraint satisfaction, generating all solutions

Time Complexity: Exponential (varies by problem)
Space Complexity: O(depth) for recursion stack
"""

from typing import List, Set, Tuple, Optional, Dict
import copy

# ==================== BRUTE FORCE APPROACHES ====================
def generate_all_permutations_brute_force(nums: List[int]) -> List[List[int]]:
    """
    Brute Force Permutations: Generate all using library
    
    Time Complexity: O(n! * n) - n! permutations, O(n) to copy each
    Space Complexity: O(n! * n) - store all permutations
    
    Problems:
    - Uses library function (not algorithmic solution)
    - No control over generation process
    - Cannot easily add constraints
    """
    import itertools
    return list(itertools.permutations(nums))

def solve_n_queens_brute_force(n: int) -> List[List[str]]:
    """
    Brute Force N-Queens: Try all possible placements
    
    Time Complexity: O(n^(n²)) - exponential in board size
    Space Complexity: O(n²) - board storage
    
    Problems:
    - Tries invalid placements
    - No early pruning
    - Extremely inefficient
    """
    def is_safe(board, row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check diagonals
        for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
            if board[i][j] == 'Q':
                return False
        
        for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
            if board[i][j] == 'Q':
                return False
        
        return True
    
    def solve(board, row):
        if row == n:
            return [[''.join(row) for row in board]]
        
        solutions = []
        for col in range(n):
            if is_safe(board, row, col):
                board[row][col] = 'Q'
                solutions.extend(solve(board, row + 1))
                board[row][col] = '.'
        
        return solutions
    
    board = [['.' for _ in range(n)] for _ in range(n)]
    return solve(board, 0)

# ==================== OPTIMIZED BACKTRACKING APPROACHES ====================

def generate_permutations_backtrack(nums: List[int]) -> List[List[int]]:
    """
    Generate all permutations using backtracking
    
    Time Complexity: O(n! * n)
    Space Complexity: O(n) - recursion depth
    
    Advantages:
    - Explicit backtracking logic
    - Can easily add constraints
    - Memory efficient during generation
    """
    result = []
    
    def backtrack(current_permutation, remaining):
        if not remaining:
            result.append(current_permutation[:])
            return
        
        for i in range(len(remaining)):
            # Choose
            current_permutation.append(remaining[i])
            new_remaining = remaining[:i] + remaining[i+1:]
            
            # Explore
            backtrack(current_permutation, new_remaining)
            
            # Unchoose (backtrack)
            current_permutation.pop()
    
    backtrack([], nums)
    return result

def generate_permutations_optimized(nums: List[int]) -> List[List[int]]:
    """
    Optimized permutations using swapping
    
    Time Complexity: O(n! * n)
    Space Complexity: O(n) - recursion depth
    """
    result = []
    
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])
            return
        
        for i in range(start, len(nums)):
            # Swap
            nums[start], nums[i] = nums[i], nums[start]
            
            # Recurse
            backtrack(start + 1)
            
            # Backtrack
            nums[start], nums[i] = nums[i], nums[start]
    
    backtrack(0)
    return result

def generate_combinations(nums: List[int], k: int) -> List[List[int]]:
    """
    Generate all combinations of k elements
    
    Time Complexity: O(C(n,k) * k)
    Space Complexity: O(k) - recursion depth
    """
    result = []
    
    def backtrack(start, current_combination):
        if len(current_combination) == k:
            result.append(current_combination[:])
            return
        
        for i in range(start, len(nums)):
            # Choose
            current_combination.append(nums[i])
            
            # Explore
            backtrack(i + 1, current_combination)
            
            # Unchoose
            current_combination.pop()
    
    backtrack(0, [])
    return result

def generate_subsets(nums: List[int]) -> List[List[int]]:
    """
    Generate all subsets (power set)
    
    Time Complexity: O(2^n * n)
    Space Complexity: O(n) - recursion depth
    """
    result = []
    
    def backtrack(start, current_subset):
        # Add current subset to result
        result.append(current_subset[:])
        
        for i in range(start, len(nums)):
            # Include current element
            current_subset.append(nums[i])
            
            # Recurse
            backtrack(i + 1, current_subset)
            
            # Backtrack
            current_subset.pop()
    
    backtrack(0, [])
    return result

def solve_n_queens_optimized(n: int) -> List[List[str]]:
    """
    N-Queens with optimized conflict detection
    
    Time Complexity: O(n!) - much better than brute force
    Space Complexity: O(n) - recursion depth
    """
    result = []
    
    def is_safe(positions, row, col):
        for prev_row in range(row):
            prev_col = positions[prev_row]
            # Check column and diagonals
            if (prev_col == col or 
                prev_col - prev_row == col - row or 
                prev_col + prev_row == col + row):
                return False
        return True
    
    def backtrack(row, positions):
        if row == n:
            # Convert positions to board representation
            board = []
            for r in range(n):
                row_str = '.' * positions[r] + 'Q' + '.' * (n - positions[r] - 1)
                board.append(row_str)
            result.append(board)
            return
        
        for col in range(n):
            if is_safe(positions, row, col):
                positions[row] = col
                backtrack(row + 1, positions)
    
    positions = [-1] * n
    backtrack(0, positions)
    return result

def solve_sudoku(board: List[List[str]]) -> bool:
    """
    Solve Sudoku puzzle using backtracking
    
    Time Complexity: O(9^(n²)) where n² is number of empty cells
    Space Complexity: O(n²) - recursion depth
    """
    def is_valid(board, row, col, num):
        # Check row
        for j in range(9):
            if board[row][j] == num:
                return False
        
        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False
        
        # Check 3x3 box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] == num:
                    return False
        
        return True
    
    def solve():
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            
                            if solve():
                                return True
                            
                            board[i][j] = '.'  # Backtrack
                    
                    return False
        return True
    
    return solve()

def word_search(board: List[List[str]], word: str) -> bool:
    """
    Word Search in 2D grid using backtracking
    
    Time Complexity: O(m * n * 4^L) where L is word length
    Space Complexity: O(L) - recursion depth
    """
    if not board or not board[0]:
        return False
    
    rows, cols = len(board), len(board[0])
    
    def backtrack(row, col, index):
        if index == len(word):
            return True
        
        if (row < 0 or row >= rows or col < 0 or col >= cols or 
            board[row][col] != word[index]):
            return False
        
        # Mark as visited
        temp = board[row][col]
        board[row][col] = '#'
        
        # Explore all 4 directions
        found = (backtrack(row + 1, col, index + 1) or
                backtrack(row - 1, col, index + 1) or
                backtrack(row, col + 1, index + 1) or
                backtrack(row, col - 1, index + 1))
        
        # Backtrack
        board[row][col] = temp
        
        return found
    
    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0):
                return True
    
    return False

def palindrome_partitioning(s: str) -> List[List[str]]:
    """
    Palindrome partitioning using backtracking
    
    Time Complexity: O(2^n * n) - 2^n partitions, O(n) to check palindrome
    Space Complexity: O(n) - recursion depth
    """
    result = []
    
    def is_palindrome(string):
        return string == string[::-1]
    
    def backtrack(start, current_partition):
        if start == len(s):
            result.append(current_partition[:])
            return
        
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                current_partition.append(substring)
                backtrack(end, current_partition)
                current_partition.pop()
    
    backtrack(0, [])
    return result

def generate_parentheses(n: int) -> List[str]:
    """
    Generate all valid parentheses combinations
    
    Time Complexity: O(4^n / √n) - Catalan number
    Space Complexity: O(n) - recursion depth
    """
    result = []
    
    def backtrack(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return
        
        # Add opening parenthesis
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)
        
        # Add closing parenthesis
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)
    
    backtrack('', 0, 0)
    return result

def letter_combinations_phone(digits: str) -> List[str]:
    """
    Letter combinations of phone number
    
    Time Complexity: O(4^n) - worst case when all digits map to 4 letters
    Space Complexity: O(n) - recursion depth
    """
    if not digits:
        return []
    
    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }
    
    result = []
    
    def backtrack(index, current_combination):
        if index == len(digits):
            result.append(current_combination)
            return
        
        digit = digits[index]
        for letter in phone_map[digit]:
            backtrack(index + 1, current_combination + letter)
    
    backtrack(0, '')
    return result

def restore_ip_addresses(s: str) -> List[str]:
    """
    Restore valid IP addresses from string
    
    Time Complexity: O(1) - constant number of combinations
    Space Complexity: O(1) - constant depth
    """
    result = []
    
    def is_valid_part(part):
        if not part or len(part) > 3:
            return False
        if len(part) > 1 and part[0] == '0':
            return False
        return 0 <= int(part) <= 255
    
    def backtrack(start, parts):
        if len(parts) == 4:
            if start == len(s):
                result.append('.'.join(parts))
            return
        
        for end in range(start + 1, min(start + 4, len(s) + 1)):
            part = s[start:end]
            if is_valid_part(part):
                parts.append(part)
                backtrack(end, parts)
                parts.pop()
    
    if 4 <= len(s) <= 12:  # Valid IP address length range
        backtrack(0, [])
    
    return result

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("=== Permutations ===")
    nums = [1, 2, 3]
    print(f"Numbers: {nums}")
    print(f"Permutations: {generate_permutations_backtrack(nums)}")
    
    print("\n=== Combinations ===")
    combinations = generate_combinations([1, 2, 3, 4], 2)
    print(f"Combinations C(4,2): {combinations}")
    
    print("\n=== Subsets ===")
    subsets = generate_subsets([1, 2, 3])
    print(f"All subsets of [1,2,3]: {subsets}")
    
    print("\n=== N-Queens ===")
    solutions = solve_n_queens_optimized(4)
    print(f"4-Queens solutions: {len(solutions)}")
    if solutions:
        print("First solution:")
        for row in solutions[0]:
            print(f"  {row}")
    
    print("\n=== Sudoku Solver ===")
    sudoku_board = [
        ["5","3",".",".","7",".",".",".","."],
        ["6",".",".","1","9","5",".",".","."],
        [".","9","8",".",".",".",".","6","."],
        ["8",".",".",".","6",".",".",".","3"],
        ["4",".",".","8",".","3",".",".","1"],
        ["7",".",".",".","2",".",".",".","6"],
        [".","6",".",".",".",".","2","8","."],
        [".",".",".","4","1","9",".",".","5"],
        [".",".",".",".","8",".",".","7","9"]
    ]
    
    print("Sudoku puzzle:")
    for row in sudoku_board:
        print(f"  {' '.join(row)}")
    
    if solve_sudoku(sudoku_board):
        print("Solved:")
        for row in sudoku_board:
            print(f"  {' '.join(row)}")
    
    print("\n=== Generate Parentheses ===")
    parentheses = generate_parentheses(3)
    print(f"Valid parentheses for n=3: {parentheses}")
    
    print("\n=== Letter Combinations ===")
    combinations = letter_combinations_phone("23")
    print(f"Letter combinations for '23': {combinations}")
    
    print("\n=== Palindrome Partitioning ===")
    partitions = palindrome_partitioning("aab")
    print(f"Palindrome partitions of 'aab': {partitions}")

"""
BACKTRACKING PATTERNS:

1. Template Structure:
   ```python
   def backtrack(state):
       if is_complete(state):
           add_to_result(state)
           return
       
       for choice in get_choices(state):
           make_choice(choice)
           backtrack(new_state)
           undo_choice(choice)  # Backtrack
   ```

2. Common Applications:
   - Permutations and Combinations
   - N-Queens, Sudoku solving
   - Word search, path finding
   - Subset generation
   - Constraint satisfaction problems

3. Key Components:
   - Choose: Make a decision
   - Explore: Recurse with new state
   - Unchoose: Undo the decision (backtrack)

WHEN TO USE BACKTRACKING:
- Need to find all solutions
- Constraint satisfaction problems
- Exhaustive search with pruning
- Problems with "try all possibilities" nature

OPTIMIZATION TECHNIQUES:
- Early pruning: Stop exploring invalid paths
- Constraint propagation: Eliminate invalid choices early  
- Heuristics: Choose promising paths first
- Memoization: Cache results of subproblems (if applicable)

TIME COMPLEXITY PATTERNS:
- Permutations: O(n!)
- Combinations: O(C(n,k))
- Subsets: O(2^n)
- N-Queens: O(n!) with pruning
- Sudoku: O(9^(empty_cells))

BACKTRACKING VS OTHER APPROACHES:
- vs Brute Force: Backtracking prunes invalid paths
- vs Dynamic Programming: DP for overlapping subproblems
- vs Greedy: Backtracking explores all possibilities

COMMON MISTAKES:
- Forgetting to backtrack (undo choices)
- Not checking constraints early enough
- Modifying shared state without restoration
- Incorrect base case conditions
- Not copying result when needed

OPTIMIZATION TIPS:
- Use constraint checking to prune early
- Order choices by likelihood of success
- Use bit manipulation for set operations
- Avoid unnecessary copying of state
- Consider iterative approaches for deep recursion
"""