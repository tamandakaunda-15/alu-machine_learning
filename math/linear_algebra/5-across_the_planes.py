#!/usr/bin/env python3
"""
This module provides a function to add two 2D matrices element-wise.
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise.

    Args:
        mat1 (list of lists of int/float): The first matrix.
        mat2 (list of lists of int/float): The second matrix.

    Returns:
        list: A new matrix containing the element-wise sum of mat1 and mat2.
        None: If the matrices are not the same shape.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    return [
        [
            mat1[i][j] + mat2[i][j]
            for j in range(len(mat1[0]))
        ]
        for i in range(len(mat1))
    ]
