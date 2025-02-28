#!/usr/bin/env python3
import numpy as np

"""
This module contains a function to calculate the correlation matrix from a covariance matrix.
"""

def correlation(C):
    """
    Calculates the correlation matrix from a covariance matrix.
    
    Args:
    C (numpy.ndarray): A 2D numpy array of shape (d, d) containing a covariance matrix.
    
    Returns:
    numpy.ndarray: A 2D numpy array of shape (d, d) containing the correlation matrix.
    
    Raises:
    TypeError: If C is not a numpy.ndarray.
    ValueError: If C does not have shape (d, d) (i.e., it's not a square matrix).
    """
    
    # Check if C is a numpy ndarray
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    
    # Check if C is a square matrix
    if C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    
    # Calculate the correlation matrix
    # First, compute the standard deviations (square roots of the variances)
    stddev = np.sqrt(np.diagonal(C))
    
    # Use broadcasting to calculate the correlation matrix
    corr = C / (stddev[:, None] * stddev)
    
    return corr
