#!/usr/bin/env python3
import numpy as np

"""
This module contains a function to calculate the mean and covariance of a dataset.
"""

def mean_cov(X):
    """
    Calculates the mean and covariance of a dataset.
    
    Args:
    X (numpy.ndarray): A 2D numpy array of shape (n, d) where n is the number
                        of data points, and d is the number of dimensions.

    Returns:
    tuple: A tuple containing:
        - mean (numpy.ndarray): A 1D numpy array of shape (d,) containing the mean of the dataset.
        - cov (numpy.ndarray): A 2D numpy array of shape (d, d) containing the covariance matrix of the dataset.
    
    Raises:
    TypeError: If X is not a 2D numpy array.
    ValueError: If X has fewer than 2 data points.
    """
    
    # Check if X is a 2D numpy array
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    
    # Get the number of data points (n) and dimensions (d)
    n, d = X.shape
    
    # Check if the dataset has at least two data points
    if n < 2:
        raise ValueError("X must contain multiple data points")
    
    # Calculate the mean of the dataset
    mean = np.mean(X, axis=0).reshape(1, -1)
    
    # Calculate the covariance matrix manually
    # Subtract the mean from each data point
    X_centered = X - mean
    
    # Calculate the covariance matrix: (1/(n-1)) * (X_centered.T @ X_centered)
    cov = np.dot(X_centered.T, X_centered) / (n - 1)
    
    return mean, cov
