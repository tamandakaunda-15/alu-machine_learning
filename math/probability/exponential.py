#!/usr/bin/env python3

"""Module for Exponential distribution."""


class Exponential:
    """Represents an exponential distribution."""

    def __init__(self, data=None, lambtha=1.):
        """Initializes the Exponential distribution.

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
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """Calculates the probability density function (PDF) for a given
 time period x.

        Args:
            x (float): The time period.

        Returns:
            float: The PDF value for x.
        """
        if x < 0:
            return 0
        return self.lambtha * (2.7182818285 ** (-self.lambtha * x))

    def cdf(self, x):
        """Calculates the cumulative distribution function (CDF) for a given
 time period x.

        Args:
            x (float): The time period.

        Returns:
            float: The CDF value for x.
        """
        if x < 0:
            return 0
        return 1 - (2.7182818285 ** (-self.lambtha * x))
