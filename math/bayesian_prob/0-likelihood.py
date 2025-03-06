#!/usr/bin/env python3
"""
Calculates the likelihood of obtaining data given various hypothetical
probabilities of developing severe side effects.
"""

import numpy as np
from scipy.special import comb


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining data given various hypothetical
    probabilities of developing severe side effects.

    Parameters:
    - x (int): Number of patients that develop severe side effects.
    - n (int): Total number of patients observed.
    - P (numpy.ndarray): 1D array of hypothetical probabilities.

    Returns:
    - numpy.ndarray: Likelihood of obtaining the data for each probability in P.

    Raises:
    - ValueError: If n is not a positive integer.
    - ValueError: If x is not an integer >= 0.
    - ValueError: If x > n.
    - TypeError: If P is not a 1D numpy.ndarray.
    - ValueError: If any value in P is not in the range [0, 1].
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    binomial_coeff = comb(n, x)
    likelihoods = binomial_coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
