#!/usr/bin/env python3

"""
This module provides a function to perform element-wise multiplication of two
matrices.
"""

def multiply_matrices(mat1, mat2):
    """
    Multiplies two matrices element-wise.

    Args:
        mat1 (list of lists of int/float): The first matrix.
        mat2 (list of lists of int/float): The second matrix.

    Returns:
        list: A new matrix containing the element-wise product of mat1 and mat2.
        None: If the matrices have different shapes.
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    return [[mat1[i][j] * mat2[i][j] for j in range(len(mat1[0]))] for i in range(len(mat1))]
