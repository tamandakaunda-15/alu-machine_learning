#!/usr/bin/env python3
import numpy as np

class MultiNormal:
    """
    A class representing a Multivariate Normal distribution.

    Attributes:
        mean (numpy.ndarray): The mean vector of the data with shape (d, 1).
        cov (numpy.ndarray): The covariance matrix of the data with shape (d, d).

    Methods:
        __init__(self, data): Initializes the MultiNormal instance using the provided data.
        pdf(self, x): Calculates the PDF at the given data point x.
    """

    def __init__(self, data):
        """
        Initializes the MultiNormal instance.

        Args:
            data (numpy.ndarray): A 2D array of shape (d, n), where d is the number of dimensions 
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
        
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        data_centered = data - self.mean
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF of the multivariate normal distribution at a data point x.

        Args:
            x (numpy.ndarray): A 2D array of shape (d, 1) representing the data point.

        Raises:
            TypeError: If x is not a numpy.ndarray.
            ValueError: If x does not have the shape (d, 1).

        Returns:
            float: The value of the PDF at x.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        
        d, _ = self.mean.shape
        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # Calculate the PDF
        diff = x - self.mean
        cov_inv = np.linalg.inv(self.cov)
        exponent = -0.5 * np.dot(np.dot(diff.T, cov_inv), diff)
        norm_const = 1 / np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov))
        
        return norm_const * np.exp(exponent)
