#!/usr/bin/env python3
"""
This module contains a function to calculate the shape of a matrix.
"""

def matrix_shape(matrix):
    """Returns the shape of a matrix as a list of integers."""
def matrix_shape(matrix):
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
