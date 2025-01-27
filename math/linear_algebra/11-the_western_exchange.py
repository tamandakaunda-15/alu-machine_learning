!/usr/bin/env python3
"""

This module provides a function to transpose a numpy-compatible matrix.
"""

def np_transpose(matrix):
    """
    Transposes a given numpy-compatible matrix.

    Args:
        matrix: A numpy.ndarray to be transposed.

    Returns:
        numpy.ndarray: A new array representing the transposed matrix.
    """
    return matrix.T
