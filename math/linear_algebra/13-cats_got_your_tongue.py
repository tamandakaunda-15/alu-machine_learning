#!/usr/bin/env python3
"""
This module provides functions to perform matrix operations using NumPy.
"""

import numpy as np


def np_matmul(mat1, mat2):
    """
    Perform matrix multiplication using NumPy's matmul function.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.

    Returns:
        numpy.ndarray: The result of the matrix multiplication.
    """
    return np.matmul(mat1, mat2)


def np_determinant(mat):
    """
    Calculate the determinant of a square matrix using NumPy's linalg.det function.

    Args:
        mat (numpy.ndarray): The square matrix.

    Returns:
        float: The determinant of the matrix.
    """
    return np.linalg.det(mat)


def np_cat(mat1, mat2, axis=0):
    """
    Concatenate two matrices along a specified axis using NumPy's concatenate function.

    Args:
        mat1 (numpy.ndarray): The first matrix.
        mat2 (numpy.ndarray): The second matrix.
        axis (int): The axis along which to concatenate. Default is 0.

    Returns:
        numpy.ndarray: The concatenated matrix.
    """
    return np.concatenate((mat1, mat2), axis=axis)
