#!/usr/bin/env python3
import numpy as np

"""
This module contains a function to calculate the correlation matrix from a covariance matrix.
"""

def correlation(C):
    """
    Calculates the correlation matrix from the covariance matrix.
    
    Args:
    C (numpy.ndarray): A 2D numpy array of shape (d, d) representing the covariance matrix.

    Returns:
    numpy.ndarray: A 2D numpy array of shape (d, d) representing the correlation matrix.
    
    Raises:
    TypeError: If C is not a numpy ndarray.
    ValueError: If C is not a 2D square matrix.
    """
    
    # Check if C is a numpy ndarray
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    
    # Check if C is a 2D square matrix
    d, n = C.shape
    if d != n:
        raise ValueError("C must be a 2D square matrix")
    
    # Calculate the correlation matrix
    # Compute the standard deviations of the variables (square roots of the diagonal elements of C)
    std_devs = np.sqrt(np.diagonal(C))
    
    # Calculate the correlation matrix by normalizing the covariance matrix
    corr_matrix = C / (std_devs[:, None] * std_devs)
    
    return corr_matrix
