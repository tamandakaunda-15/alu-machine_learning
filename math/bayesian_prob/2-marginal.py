#!/usr/bin/env python3
"""
Calculates the marginal probability of obtaining data given various 
hypothetical probabilities of developing severe side effects and prior 
beliefs about those probabilities.
"""

import numpy as np
from scipy.special import comb


def marginal(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining data given various 
    hypothetical probabilities of developing severe side effects and prior 
    beliefs.

    Parameters:
    - x (int): Number of patients that develop severe side effects.
    - n (int): Total number of patients observed.
    - P (numpy.ndarray): 1D array of hypothetical probabilities.
    - Pr (numpy.ndarray): 1D array of prior probabilities.

    Returns:
    - float: Marginal probability of obtaining the data for x and n.

    Raises:
    - ValueError: If n is not a positive integer.
    - ValueError: If x is not an integer >= 0.
    - ValueError: If x is greater than n.
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

    # Calculate the marginal probability by summing the product of likelihoods and prior probabilities
    marginal_probability = np.sum(likelihoods * Pr)

    return marginal_probability
