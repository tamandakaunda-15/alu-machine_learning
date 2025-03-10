#!/usr/bin/env python3
"""
This module contains the function `intersection` that calculates the intersection 
of obtaining data with hypothetical probabilities, considering prior beliefs.

It raises various exceptions for invalid inputs and returns a 1D numpy array of 
the intersection results.
"""

import numpy as np
from scipy.special import comb


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining data with the various hypothetical
    probabilities, considering prior beliefs.

    Parameters:
    - x (int): Number of patients that develop severe side effects.
    - n (int): Total number of patients observed.
    - P (numpy.ndarray): 1D array of hypothetical probabilities.
    - Pr (numpy.ndarray): 1D array of prior probabilities.

    Returns:
    - numpy.ndarray: Intersection of obtaining the data for each probability 
      in P.

    Raises:
    - ValueError: If n is not a positive integer.
    - ValueError: If x is not an integer >= 0.
    - ValueError: If x > n.
    - TypeError: If P is not a 1D numpy.ndarray.
    - TypeError: If Pr is not a 1D numpy.ndarray with the same shape as P.
    - ValueError: If any value in P or Pr is not in the range [0, 1].
    - ValueError: If Pr does not sum to 1.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.ndim != 1:
        raise TypeError("Pr must be a 1D numpy.ndarray")
    if P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the binomial coefficient
    binomial_coeff = comb(n, x)

    # Calculate the likelihood for each probability in P
    likelihoods = binomial_coeff * (P ** x) * ((1 - P) ** (n - x))

    # Calculate the intersection (likelihood * prior belief)
    intersection_result = likelihoods * Pr

    return intersection_result
