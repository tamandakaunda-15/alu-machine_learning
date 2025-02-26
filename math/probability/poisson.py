#!/usr/bin/env python3

import math

"""Module for Poisson distribution."""


class Poisson:
    """Represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.0):
        """Initializes the Poisson distribution.

        Args:
            data (list, optional): List of data points to estimate lambtha.
            lambtha (float, optional): Expected number of occurrences. 
                Defaults to 1.

        Raises:
            ValueError: If lambtha is not a positive value.
            TypeError: If data is not a list.
            ValueError: If data does not contain multiple values.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”."""
        # If k is not an integer, convert it to an integer
        if not isinstance(k, int):
            k = int(k)
        
        # If k is out of range (negative), return 0
        if k < 0:
            return 0
        
        # Calculate the CDF using the formula for Poisson distribution
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += (self.lambtha ** i) * math.exp(-self.lambtha) / math.factorial(i)
        
        return cdf_value
