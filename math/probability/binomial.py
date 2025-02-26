#!/usr/bin/env python3

"""
Module that defines a Binomial distribution class.
"""


class Binomial:
    """
    Represents a binomial distribution.

    Attributes:
        n (int): The number of Bernoulli trials.
        p (float): The probability of a "success".
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution with either data or
 parameters n and p.

        If data is provided, n and p are estimated from the data. Otherwise,
        the values of n and p are set by the user.

        Args:
            data (list, optional): List of observed binomial data
 to estimate the distribution.
            n (int, optional): The number of Bernoulli trials.
            p (float, optional): The probability of success.

        Raises:
            ValueError: If n is not a positive value or p is
 not in the valid range.
            TypeError: If data is not a list.
            ValueError: If data contains fewer than two data points.
        """
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Estimate p first (average successes per trial)
            p_estimate = sum(data) / (len(data) * n) 
           # This is the correct way to calculate p

           # Use the estimated p to calculate n (rounded to nearest integer)
           n_estimate = round(sum(data) / p_estimate) 
           # This ensures n is a rounded integer based on the estimated p

            self.n = n_estimate
            self.p = p_estimate
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
