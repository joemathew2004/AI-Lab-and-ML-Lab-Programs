import numpy as np
n = int(input("Enter the value of n: "))
array = np.zeros((n, n), dtype=int)
print("Enter the values for the array row by row:")
for i in range(n):
    row_values = input(f"Row {i+1}: ").split()
    array[i] = [int(val) for val in row_values]
print("Generated Array:")
print(array)
def find_patterns(sequence):
    patterns = {}
    length = len(sequence)
    for pattern_length in range(2, length + 1):
        for i in range(length - pattern_length + 1):
            pattern = tuple(sequence[i:i + pattern_length])
            if pattern in patterns:
                patterns[pattern] += 1
            else:
                patterns[pattern] = 1
    return patterns
def find_repeating_patterns(arr):
    repeating_patterns = {}
    # Check row-wise patterns (left to right and right to left)
    for i in range(n):
        row = arr[i]
        row_patterns = find_patterns(row)
        reversed_row_patterns = find_patterns(row[::-1])
        for pattern, count in row_patterns.items():
            if pattern in repeating_patterns:
                repeating_patterns[pattern] += count
            else:
                repeating_patterns[pattern] = count
        for pattern, count in reversed_row_patterns.items():
            if pattern in repeating_patterns:
                repeating_patterns[pattern] += count
            else:
                repeating_patterns[pattern] = count
    # Check column-wise patterns (up to down and down to up)
    for j in range(n):
        col = arr[:, j]
        col_patterns = find_patterns(col)
        reversed_col_patterns = find_patterns(col[::-1])
        for pattern, count in col_patterns.items():
            if pattern in repeating_patterns:
                repeating_patterns[pattern] += count
            else:
                repeating_patterns[pattern] = count
        for pattern, count in reversed_col_patterns.items():
            if pattern in repeating_patterns:
                repeating_patterns[pattern] += count
            else:
                repeating_patterns[pattern] = count

    # Ensure we only count patterns that occur more than once in the array
    repeating_patterns = {k: v for k, v in repeating_patterns.items() if v > 1}
    return repeating_patterns
repeating_patterns = find_repeating_patterns(array)
if repeating_patterns:
    print("Repeating numerical patterns found:")
    for pattern, count in repeating_patterns.items():
        print(f"Pattern {pattern} is repeated {count} times")
else:
    print("No repeating numerical patterns found.")