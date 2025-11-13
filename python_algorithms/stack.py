"""
Stack Algorithm
==============

Used for: Parsing, matching brackets, monotonic stack problems, expression evaluation

Time Complexity: O(n) for most operations
Space Complexity: O(n) for stack storage
"""

from typing import List, Optional, Dict, Tuple
from collections import deque

# ==================== BRUTE FORCE APPROACHES ====================
def valid_parentheses_brute_force(s: str) -> bool:
    """
    Brute Force Parentheses: Check all combinations by counting
    
    Time Complexity: O(n²) - nested loops for different bracket types
    Space Complexity: O(1)
    
    Problems:
    - Doesn't handle nested structures properly
    - Can't distinguish between different bracket types correctly
    - Inefficient for complex expressions
    """
    # Count each type of bracket
    round_count = square_count = curly_count = 0
    
    for char in s:
        if char == '(':
            round_count += 1
        elif char == ')':
            round_count -= 1
        elif char == '[':
            square_count += 1
        elif char == ']':
            square_count -= 1
        elif char == '{':
            curly_count += 1
        elif char == '}':
            curly_count -= 1
        
        # This approach fails for cases like "([)]"
        if round_count < 0 or square_count < 0 or curly_count < 0:
            return False
    
    return round_count == 0 and square_count == 0 and curly_count == 0

def evaluate_expression_brute_force(expression: str) -> int:
    """
    Brute Force Expression Evaluation: Use built-in eval
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Problems:
    - Security risk with eval()
    - No control over evaluation process
    - Doesn't demonstrate algorithmic approach
    """
    try:
        # This is unsafe and not algorithmic
        return eval(expression)
    except:
        return 0

# ==================== OPTIMIZED STACK APPROACHES ====================
def valid_parentheses_stack(s: str) -> bool:
    """
    Valid Parentheses using Stack
    
    Time Complexity: O(n)
    Space Complexity: O(n) - worst case all opening brackets
    
    Advantages:
    - Correctly handles nested structures
    - Distinguishes between bracket types
    - Linear time complexity
    """
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in s:
        if char in mapping:  # Closing bracket
            if not stack or stack.pop() != mapping[char]:
                return False
        else:  # Opening bracket
            stack.append(char)
    
    return len(stack) == 0

def evaluate_postfix_expression(tokens: List[str]) -> int:
    """
    Evaluate Reverse Polish Notation (Postfix) using Stack
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack = []
    
    for token in tokens:
        if token in ['+', '-', '*', '/']:
            # Pop two operands
            b = stack.pop()
            a = stack.pop()
            
            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            elif token == '/':
                # Handle division (truncate towards zero)
                result = int(a / b)
            
            stack.append(result)
        else:
            # Push operand
            stack.append(int(token))
    
    return stack[0]

def infix_to_postfix(expression: str) -> str:
    """
    Convert infix expression to postfix using stack
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}
    right_associative = {'^'}
    
    stack = []
    output = []
    
    for char in expression:
        if char.isalnum():  # Operand
            output.append(char)
        elif char == '(':
            stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(':
                output.append(stack.pop())
            stack.pop()  # Remove '('
        elif char in precedence:  # Operator
            while (stack and stack[-1] != '(' and
                   stack[-1] in precedence and
                   (precedence[stack[-1]] > precedence[char] or
                    (precedence[stack[-1]] == precedence[char] and
                     char not in right_associative))):
                output.append(stack.pop())
            stack.append(char)
    
    while stack:
        output.append(stack.pop())
    
    return ''.join(output)

def basic_calculator(s: str) -> int:
    """
    Basic Calculator with +, -, (, ) using Stack
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack = []
    result = 0
    number = 0
    sign = 1
    
    for char in s:
        if char.isdigit():
            number = number * 10 + int(char)
        elif char == '+':
            result += sign * number
            number = 0
            sign = 1
        elif char == '-':
            result += sign * number
            number = 0
            sign = -1
        elif char == '(':
            # Push current result and sign onto stack
            stack.append(result)
            stack.append(sign)
            # Reset for expression inside parentheses
            result = 0
            sign = 1
        elif char == ')':
            result += sign * number
            number = 0
            # Pop sign and previous result
            result *= stack.pop()  # sign
            result += stack.pop()  # previous result
    
    return result + sign * number

def daily_temperatures(temperatures: List[int]) -> List[int]:
    """
    Daily Temperatures using Monotonic Stack
    
    Time Complexity: O(n) - each element pushed/popped once
    Space Complexity: O(n) - stack size
    
    Find next warmer temperature for each day
    """
    result = [0] * len(temperatures)
    stack = []  # Monotonic decreasing stack (indices)
    
    for i, temp in enumerate(temperatures):
        # Pop all temperatures that are cooler than current
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        
        stack.append(i)
    
    return result

def next_greater_element(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Next Greater Element using Stack
    
    Time Complexity: O(n + m)
    Space Complexity: O(n)
    """
    # Build next greater element map for nums2
    next_greater = {}
    stack = []
    
    for num in nums2:
        while stack and stack[-1] < num:
            next_greater[stack.pop()] = num
        stack.append(num)
    
    # For remaining elements in stack, no greater element exists
    while stack:
        next_greater[stack.pop()] = -1
    
    # Build result for nums1
    return [next_greater[num] for num in nums1]

def largest_rectangle_histogram(heights: List[int]) -> int:
    """
    Largest Rectangle in Histogram using Stack
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack = []  # Stack of indices
    max_area = 0
    heights.append(0)  # Add sentinel to process remaining bars
    
    for i, height in enumerate(heights):
        while stack and heights[stack[-1]] > height:
            h = heights[stack.pop()]
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)
    
    heights.pop()  # Remove sentinel
    return max_area

def trapping_rain_water_stack(heights: List[int]) -> int:
    """
    Trapping Rain Water using Stack
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack = []
    water_trapped = 0
    
    for i, height in enumerate(heights):
        while stack and heights[stack[-1]] < height:
            bottom = stack.pop()
            
            if not stack:
                break
            
            distance = i - stack[-1] - 1
            bounded_height = min(height, heights[stack[-1]]) - heights[bottom]
            water_trapped += distance * bounded_height
        
        stack.append(i)
    
    return water_trapped

def remove_duplicate_letters(s: str) -> str:
    """
    Remove Duplicate Letters using Monotonic Stack
    
    Time Complexity: O(n)
    Space Complexity: O(1) - limited by alphabet size
    """
    # Count frequency of each character
    count = {}
    for char in s:
        count[char] = count.get(char, 0) + 1
    
    stack = []
    in_stack = set()
    
    for char in s:
        count[char] -= 1
        
        if char in in_stack:
            continue
        
        # Remove characters that are lexicographically larger
        # and appear later in the string
        while (stack and stack[-1] > char and 
               count[stack[-1]] > 0):
            removed = stack.pop()
            in_stack.remove(removed)
        
        stack.append(char)
        in_stack.add(char)
    
    return ''.join(stack)

def decode_string(s: str) -> str:
    """
    Decode String using Stack
    
    Time Complexity: O(n * max_k) where max_k is maximum multiplier
    Space Complexity: O(n)
    """
    stack = []
    current_num = 0
    current_string = ""
    
    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            # Push current state to stack
            stack.append((current_string, current_num))
            current_string = ""
            current_num = 0
        elif char == ']':
            # Pop from stack and decode
            prev_string, num = stack.pop()
            current_string = prev_string + current_string * num
        else:
            current_string += char
    
    return current_string

def min_stack_design():
    """
    Design Min Stack that supports getMin() in O(1)
    """
    class MinStack:
        def __init__(self):
            self.stack = []
            self.min_stack = []
        
        def push(self, val: int) -> None:
            self.stack.append(val)
            # Push to min_stack if it's empty or val is <= current min
            if not self.min_stack or val <= self.min_stack[-1]:
                self.min_stack.append(val)
        
        def pop(self) -> None:
            if self.stack:
                val = self.stack.pop()
                # Pop from min_stack if it's the minimum
                if self.min_stack and val == self.min_stack[-1]:
                    self.min_stack.pop()
        
        def top(self) -> int:
            return self.stack[-1] if self.stack else None
        
        def getMin(self) -> int:
            return self.min_stack[-1] if self.min_stack else None
    
    return MinStack()

def simplify_path(path: str) -> str:
    """
    Simplify Unix-style path using Stack
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack = []
    components = path.split('/')
    
    for component in components:
        if component == '' or component == '.':
            continue
        elif component == '..':
            if stack:
                stack.pop()
        else:
            stack.append(component)
    
    return '/' + '/'.join(stack)

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("=== Valid Parentheses ===")
    test_cases = ["()", "()[]{}", "(]", "([)]", "{[]}"]
    
    for test in test_cases:
        brute_result = valid_parentheses_brute_force(test)
        stack_result = valid_parentheses_stack(test)
        print(f"'{test}': brute={brute_result}, stack={stack_result}")
    
    print("\n=== Postfix Expression Evaluation ===")
    postfix = ["2", "1", "+", "3", "*"]
    result = evaluate_postfix_expression(postfix)
    print(f"Postfix {postfix} = {result}")
    
    print("\n=== Infix to Postfix Conversion ===")
    infix = "a+b*c"
    postfix_result = infix_to_postfix(infix)
    print(f"Infix '{infix}' -> Postfix '{postfix_result}'")
    
    print("\n=== Basic Calculator ===")
    expressions = ["1 + 1", "2-1 + 2", "(1+(4+5+2)-3)+(6+8)"]
    for expr in expressions:
        result = basic_calculator(expr)
        print(f"'{expr}' = {result}")
    
    print("\n=== Daily Temperatures ===")
    temperatures = [73, 74, 75, 71, 69, 72, 76, 73]
    result = daily_temperatures(temperatures)
    print(f"Temperatures: {temperatures}")
    print(f"Days to wait: {result}")
    
    print("\n=== Next Greater Element ===")
    nums1, nums2 = [4, 1, 2], [1, 3, 4, 2]
    result = next_greater_element(nums1, nums2)
    print(f"nums1: {nums1}, nums2: {nums2}")
    print(f"Next greater: {result}")
    
    print("\n=== Largest Rectangle in Histogram ===")
    heights = [2, 1, 5, 6, 2, 3]
    max_area = largest_rectangle_histogram(heights)
    print(f"Heights: {heights}")
    print(f"Largest rectangle area: {max_area}")
    
    print("\n=== Trapping Rain Water ===")
    heights = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    water = trapping_rain_water_stack(heights)
    print(f"Heights: {heights}")
    print(f"Water trapped: {water}")
    
    print("\n=== Remove Duplicate Letters ===")
    s = "bcabc"
    result = remove_duplicate_letters(s)
    print(f"String: '{s}' -> '{result}'")
    
    print("\n=== Decode String ===")
    s = "3[a2[c]]"
    result = decode_string(s)
    print(f"Encoded: '{s}' -> Decoded: '{result}'")
    
    print("\n=== Min Stack Demo ===")
    min_stack = min_stack_design()
    operations = [(-2, 'push'), (0, 'push'), (-3, 'push'), 
                  (None, 'getMin'), (None, 'pop'), (None, 'top'), (None, 'getMin')]
    
    for val, op in operations:
        if op == 'push':
            min_stack.push(val)
            print(f"Push {val}")
        elif op == 'pop':
            min_stack.pop()
            print("Pop")
        elif op == 'top':
            print(f"Top: {min_stack.top()}")
        elif op == 'getMin':
            print(f"Min: {min_stack.getMin()}")
    
    print("\n=== Simplify Path ===")
    paths = ["/home/", "/../", "/home//foo/", "/a/./b/../../c/"]
    for path in paths:
        simplified = simplify_path(path)
        print(f"'{path}' -> '{simplified}'")

"""
STACK PATTERNS:

1. Matching/Parsing Problems:
   - Parentheses validation
   - Expression evaluation
   - HTML/XML parsing
   - Template pattern matching

2. Monotonic Stack:
   - Next/Previous greater/smaller element
   - Daily temperatures, stock spans
   - Largest rectangle problems
   - Remove duplicates maintaining order

3. Expression Evaluation:
   - Infix to postfix conversion
   - Calculator implementations
   - Operator precedence handling

4. Backtracking Support:
   - DFS traversal
   - Undo operations
   - State management

WHEN TO USE STACK:
- Need LIFO (Last In, First Out) behavior
- Matching/balancing problems
- Expression parsing and evaluation
- Maintaining monotonic properties
- Recursive algorithms (implicit stack)

STACK vs OTHER DATA STRUCTURES:
- vs Queue: LIFO vs FIFO behavior
- vs Array: Dynamic size, O(1) top access
- vs Linked List: Restricted access pattern

MONOTONIC STACK TECHNIQUE:
- Maintain elements in increasing/decreasing order
- Pop elements that violate monotonic property
- Useful for "next greater/smaller" problems
- Each element pushed/popped at most once

COMMON STACK OPERATIONS:
- push(x): Add element to top - O(1)
- pop(): Remove top element - O(1)
- top()/peek(): View top element - O(1)
- isEmpty(): Check if empty - O(1)
- size(): Get number of elements - O(1)

IMPLEMENTATION CONSIDERATIONS:
- Array-based: Fixed size, cache-friendly
- Linked List-based: Dynamic size, extra memory overhead
- Language built-ins: Usually optimized implementations

ADVANCED PATTERNS:
- Two stacks for queue implementation
- Stack for recursive to iterative conversion
- Min/Max stack maintaining auxiliary information
- Stack-based tree traversal algorithms
"""