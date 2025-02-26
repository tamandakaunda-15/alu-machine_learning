#!/usr/bin/env python3
"""
This module defines the Binomial class for working with binomial distributions.
The class allows you to estimate parameters from data or use predefined parameters.
"""

class Binomial:
    """
    Binomial distribution class.

    This class represents a binomial distribution. You can either provide data
    to estimate the distribution parameters or specify the number of trials (n)
    and the probability of success (p).

    Attributes:
        n (int): The number of trials (must be positive).
        p (float): The probability of success (must be between 0 and 1).
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution object.

        If data is provided, calculates n and p from it. Otherwise, uses the
        provided n and p.

        Args:
            data (list, optional): Data to estimate n and p. Defaults to None.
            n (int, optional): Number of trials. Defaults to 1.
            p (float, optional): Probability of success. Defaults to 0.5.

        Raises:
            ValueError: If n is not positive or if p is not between 0 and 1.
            TypeError: If data is not a list or if it contains fewer than two elements.
        """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            
            # Calculate p as the average of data, assuming data represents successes
            p = sum(data) / len(data) / n  # Average success rate (mean of data divided by n)
            n = round(sum(data) / len(data))  # Use the rounded mean as n
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

        self.n = n
        self.p = p

    def __repr__(self):
        """
        Returns a string representation of the Binomial distribution object.

        Returns:
            str: A string representation of the distribution parameters.
        """
        return f"Binomial(n={self.n}, p={self.p})"
