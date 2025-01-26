#!/usr/bin/env python3

"""
This module provides a function to concatenate two 2D matrices 
along the given axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenates two 2D matrices along a specified axis.

    Args:
        mat1 (list of lists of int/float): The first matrix.
        mat2 (list of lists of int/float): The second matrix.
        axis (int, optional): The axis to concatenate on (0 for rows, 1 for columns).

    Returns:
        list: A new matrix with the concatenated result, or None if shapes are incompatible.
    """
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2

    if axis == 1 and len(mat1) == len(mat2):
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]

    return None
