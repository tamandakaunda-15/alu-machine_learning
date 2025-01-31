"""
This module contains a function to determine the definiteness of a matrix.

A matrix can be classified as:
- Positive definite
- Positive semi-definite
- Negative definite
- Negative semi-definite
- Indefinite

It uses eigenvalues of the matrix to make this determination.
"""

import numpy as np


def definiteness(matrix):
    """Determines the definiteness of a matrix."""
    # Validate input
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if matrix.size == 0 or matrix.shape[0] != matrix.shape[1]:
        return None  # Invalid input: return None
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    
    # Check definiteness conditions
    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
