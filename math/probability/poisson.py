#!/usr/bin/env python3

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

    def factorial(self, n):
        """Calculates the factorial of a number."""
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def exp(self, x):
        """Calculates the exponential of a number (e^x)."""
        result = 1
        term = 1
        n = 1
        # Use series expansion to approximate e^x
        while True:
            term *= x / n
            result += term
            n += 1
            if abs(term) < 1e-10:  # Convergence threshold
                break
        return result

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
            cdf_value += (self.lambtha ** i) * self.exp(-self.lambtha) / self.factorial(i)
        
        return cdf_value
