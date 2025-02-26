#!/usr/bin/env python3

"""Module for Poisson distribution."""


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

   def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”.

        Args:
            k (int): The number of successes.

        Returns:
            float: The PMF value for k.
        """
        if not isinstance(k, int):
            k = int(k)  # Convert to integer if k is not an integer

        if k < 0:
            return 0  # Poisson distribution is only defined for k >= 0

        # Poisson PMF formula: (λ^k * e^(-λ)) / k!
        return (self.lambtha ** k * math.exp(-self.lambtha)) / math.factorial(k)
