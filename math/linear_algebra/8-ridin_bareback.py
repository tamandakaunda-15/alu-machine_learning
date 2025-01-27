#!/usr/bin/env python3

"""
This module contains a function mat_mul that multiplies two 2D matrices.
If the matrices cannot be multiplied, the function returns None.
"""


def mat_mul(mat1, mat2):
    """
    Multiplies two matrices mat1 and mat2 and returns the resulting matrix.
    If the matrices cannot be multiplied, returns None.

    Args:
    mat1 (list of lists): The first matrix.
    mat2 (list of lists): The second matrix.

    Returns:
    list of lists: The result of multiplying mat1 by mat2.
    None: If the matrices cannot be multiplied.
    """
    if len(mat1[0]) != len(mat2):
        return None

    result = [
        [sum(a * b for a, b in zip(row, col)) for col in zip(*mat2)]
        for row in mat1
    ]
    return result
