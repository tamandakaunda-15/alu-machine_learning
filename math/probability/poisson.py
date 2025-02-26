#!/usr/bin/env python3

class Poisson:
    """Represents a Poisson distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initializes the Poisson distribution.

        Args:
            data (list, optional): List of data points to estimate lambtha.
            lambtha (float, optional): Expected number of occurrences.
                Defaults to 1.0.

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
        """Computes factorial of n manually."""
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def exp(self, x):
        """Computes e^(-x) using a Taylor series approximation."""
        terms = 20  # More terms increase accuracy
        result = 1.0
        power = 1.0
        factorial = 1.0
        for i in range(1, terms):
            power *= -x
            factorial *= i
            result += power / factorial
        return result

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”.

        Args:
            k (int or float): The number of successes.

        Returns:
            float: The PMF value for k.
        """
        if not isinstance(k, int):
            k = int(k)  # Convert to integer

        if k < 0:
            return 0  # Poisson distribution is only defined for k >= 0

        # Poisson PMF formula: (λ^k * e^(-λ)) / k!
        lambtha_k = 1
        for _ in range(k):
            lambtha_k *= self.lambtha  # Compute lambtha^k manually

        return (lambtha_k * self.exp(self.lambtha)) / self.factorial(k)
