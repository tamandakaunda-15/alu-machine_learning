#!/usr/bin/env python3
import numpy as np


class MultiNormal:
    """
    A class representing a Multivariate Normal distribution.

    Attributes:
        mean (numpy.ndarray): The mean vector of the data with shape (d, 1).
        cov (numpy.ndarray): The covariance matrix of the data with shape (d, d).

    Methods:
        __init__(self, data): Initializes the MultiNormal instance using
 the provided data.
    """
    def __init__(self, data):
        """
        Initializes the MultiNormal instance.

        Args:
            data (numpy.ndarray): A 2D array of shape (d, n), where d is the
 number of dimensions 
                                   and n is the number of data points.

        Raises:
            TypeError: If the data is not a 2D numpy.ndarray.
            ValueError: If the number of data points is less than 2.
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        
        d, n = data.shape
        
        if n < 2:
            raise ValueError("data must contain multiple data points")
        
        # Compute the mean (d, 1)
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        
        # Compute the covariance matrix (d, d) without using numpy.cov
        data_centered = data - self.mean  # Subtract the mean from the
 data points
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)
