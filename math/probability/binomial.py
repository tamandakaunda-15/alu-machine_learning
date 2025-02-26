#!/usr/bin/env python3

class Binomial:
    """
    Class that represents a binomial distribution
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes a Binomial instance.

        Args:
            data (list, optional): List of data to estimate the
 distribution. Defaults to None.
            n (int, optional): Number of Bernoulli trials. Defaults to 1.
            p (float, optional): Probability of success. Defaults to 0.5.

        Raises:
            ValueError: If n is not positive or p is not a valid probability.
            TypeError: If data is not a list.
            ValueError: If data contains fewer than two data points.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = n
            self.p = p
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            
            # Calculate p from data
            p_est = sum(data) / len(data) / max(data)
            n_est = round(len(data) / p_est)
            
            if not (0 < p_est < 1):
                raise ValueError("p must be greater than 0 and less than 1")

            self.p = p_est
            self.n = n_est
