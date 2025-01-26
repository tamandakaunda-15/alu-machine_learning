#!/usr/bin/env python3
"""
This module provides a function to compute the transpose of a matrix.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a given 2D matrix.

    Args:
        matrix (list of list): The input 2D matrix.

    Returns:
        list of list: The transposed matrix.
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
