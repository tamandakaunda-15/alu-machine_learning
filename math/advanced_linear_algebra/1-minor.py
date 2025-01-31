#!/usr/bin/env python3

"""
This module contains functions for matrix operations, including calculating the determinant 
and minor matrix of a square matrix.

The `determinant` function calculates the determinant of a square matrix using recursion.
The `minor` function calculates the minor matrix of a given square matrix.

Functions:
    - determinant(matrix): Returns the determinant of the matrix.
    - minor(matrix): Returns the minor matrix of the matrix.
"""

def determinant(matrix):
    """Calculates the determinant of a square matrix."""
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return (matrix[0][0] * matrix[1][1] 
                - matrix[0][1] * matrix[1][0])

    det = 0
    for col in range(len(matrix)):
        sub_matrix = [row[:col] + row[col+1:] for row in matrix[1:]]
        det += ((-1) ** col) * matrix[0][col] * determinant(sub_matrix)

    return det
