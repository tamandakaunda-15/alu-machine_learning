#!/usr/bin/env python3
def determinant(matrix):
    """Calculates the determinant of a square matrix."""
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    
    if len(matrix) == 1:
        return matrix[0][0]
    
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for col in range(len(matrix)):
        sub_matrix = [row[:col] + row[col+1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(sub_matrix)
    
    return det

def minor(matrix):
    """Calculates the minor matrix of a square matrix."""
    # Validate that the matrix is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    # Validate that the matrix is square and non-empty
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    
    # If the matrix is 1x1, the minor is [[1]]
    if len(matrix) == 1:
        return [[1]]
    
    # Initialize the minor matrix
    minor_matrix = []
    
    # Loop through each element in the matrix
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix)):
            # Create the submatrix by excluding the i-th row and j-th column
            sub_matrix = [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]
            # Append the determinant of the submatrix to the minor_row
            minor_row.append(determinant(sub_matrix))
        # Append the minor_row to the minor_matrix
        minor_matrix.append(minor_row)
    
    return minor_matrix
