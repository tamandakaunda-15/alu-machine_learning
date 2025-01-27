#!/usr/bin/env python3
"""
This module provides a function to perform matrix multiplication using NumPy.
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
