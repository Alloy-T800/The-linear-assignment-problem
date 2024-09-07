import numpy as np
from scipy.optimize import linear_sum_assignment

# Define the matrix
matrix = np.array([
    [7, 9, 2, 6, 5],
    [8, 8, 5, 6, 7],
    [2, 8, 9, 6, 8],
    [7, 2, 6, 3, 9],
    [2, 9, 1, 7, 5]
])

# Use linear_sum_assignment to solve the assignment problem
row_ind, col_ind = linear_sum_assignment(-matrix)  # Negate the matrix to maximize the sum

# Calculate the maximum sum
max_sum = matrix[row_ind, col_ind].sum()

# Output the result
print(max_sum)
