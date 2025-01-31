#!/usr/bin/env python3
def determinant(matrix):
    """Calculates the determinant of a matrix."""
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for i in range(len(matrix)):
        sub_matrix = [row[:i] + row[i+1:] for row in matrix[1:]]
        det += ((-1) ** i) * matrix[0][i] * determinant(sub_matrix)
    return det

def minor(matrix):
    """Calculates the minor matrix of a matrix."""
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")
    
    minors = []
    for i in range(len(matrix)):
        minors.append([])
        for j in range(len(matrix)):
            sub_matrix = [row[:j] + row[j+1:] for row in matrix[:i] + matrix[i+1:]]
            minors[i].append(determinant(sub_matrix))
    return minors

def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix."""
    minors = minor(matrix)
    for i in range(len(minors)):
        for j in range(len(minors)):
            minors[i][j] *= (-1) ** (i + j)
    return minors
