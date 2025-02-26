#!/usr/bin/env python3

"""
This module contains the Binomial class, which represents a 
binomial distribution. It can estimate parameters from data and 
compute the probability mass function (PMF).
"""

class Binomial:
    """
    Class that represents a binomial distribution.
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes a Binomial instance.

        Args:
            data (list, optional): List of data to estimate the 
                                   distribution. Defaults to None.
            n (int, optional): Number of Bernoulli trials. 
                               Defaults to 1.
            p (float, optional): Probability of success. 
                                 Defaults to 0.5.

        Raises:
            ValueError: If n is not positive or p is invalid.
            TypeError: If data is not a list.
            ValueError: If data has fewer than two elements.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and "
                                 "less than 1")
            self.n = n
            self.p = p
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            p_est = sum(data) / len(data) / max(data)
            n_est = round(len(data) / p_est)

            if not (0 < p_est < 1):
                raise ValueError("p must be greater than 0 and "
                                 "less than 1")

            self.p = p_est
            self.n = n_est

    def factorial(self, num):
        """
        Computes the factorial of a number.

        Args:
            num (int): Number to compute the factorial for.

        Returns:
            int: Factorial of num.
        """
        if num == 0 or num == 1:
            return 1
        result = 1
        for i in range(2, num + 1):
            result *= i
        return result

    def pmf(self, k):
        """
        Computes the probability mass function (PMF) for k 
        successes.

        Args:
            k (int): Number of successes.

        Returns:
            float: PMF value for k.
        """
        k = int(k)  # Convert k to an integer

        if k < 0 or k > self.n:
            return 0

        comb = (self.factorial(self.n) // 
                (self.factorial(k) * self.factorial(self.n - k)))

        pmf_value = comb * (self.p ** k) * ((1 - self.p) ** 
                                            (self.n - k))
        return pmf_value
