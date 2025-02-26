#!/usr/bin/env python3

"""
Binomial Distribution Class

This module defines a class that represents a binomial distribution. It can
be initialized with data or the number of trials (n) and probability (p).
"""


class Binomial:
    """
    A class that represents a binomial distribution.

    Attributes:
        n (int): The number of Bernoulli trials.
        p (float): The probability of a success in a Bernoulli trial.

    Methods:
        __init__(self, data=None, n=1, p=0.5):
            Initializes a Binomial distribution with given data, or using
            the provided number of trials (n) and probability (p).
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution.
        If `data` is provided, `n` and `p` will be calculated based on the
 data.
        Otherwise, the provided `n` and `p` values will be used.

        Args:
            data (list, optional): A list of data to estimate the distribution.
            n (int): The number of Bernoulli trials. Must be a positive value.
            p (float): The probability of a success. Must be between 0 and 1.

        Raises:
            ValueError: If `n` is not positive or `p` is not between 0 and 1.
            TypeError: If `data` is not a list.
        """
        if data is None:
            # Case where data is not provided
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)  # Ensure n is an integer
            self.p = float(p)  # Ensure p is a float
        else:
            # Case where data is provided
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Print the data for debugging purposes
            print(f"Data: {data}")

            # Calculate p from data
            successes = sum(data)
            self.n = len(data)  # Number of trials
            self.p = successes / self.n  # p is the proportion of successes

            # Print the calculated p for debugging purposes
            print(f"Calculated p: {self.p}")

            # Validate p to be within the valid range
            if not (0 < self.p < 1):
                raise ValueError("p must be greater than 0 and less than 1")

            print(f"n: {self.n}, p: {self.p}")  # Print n and p for debugging
