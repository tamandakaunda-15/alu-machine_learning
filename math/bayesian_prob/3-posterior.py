#!/usr/bin/env python3
import numpy as np
from scipy.special import comb

def posterior(x, n, P, Pr):
    """
    Calculates the posterior probability for the various hypothetical 
    probabilities of developing severe side effects, given the data.

    Parameters:
    - x (int): Number of patients that develop severe side effects.
    - n (int): Total number of patients observed.
    - P (numpy.ndarray): 1D array of hypothetical probabilities.
    - Pr (numpy.ndarray): 1D array of prior probabilities.

    Returns:
    - numpy.ndarray: Posterior probabilities for each probability in P.

    Raises:
    - ValueError: If n is not a positive integer.
    - ValueError: If x is not an integer >= 0.
    - ValueError: If x is greater than n.
    - TypeError: If P is not a 1D numpy.ndarray.
    - TypeError: If Pr is not a numpy.ndarray with the same shape as P.
    - ValueError: If any value in P or Pr is not in the range [0, 1].
    - ValueError: If Pr does not sum to 1.
    """
    # Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    
    # Check if x is an integer >= 0
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")
    
    # Check if x is not greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")
    
    # Check if P is a 1D numpy.ndarray
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    
    # Check if Pr is a numpy.ndarray with the same shape as P
    if not isinstance(Pr, np.ndarray) or Pr.ndim != 1:
        raise TypeError("Pr must be a 1D numpy.ndarray")
    if P.shape != Pr.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    
    # Check if all values in P and Pr are within [0, 1]
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    
    # Check if Pr sums to 1
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Compute the likelihood for each value in P
    binomial_coeff = comb(n, x)  # Binomial coefficient C(n, x)
    likelihoods = binomial_coeff * (P ** x) * ((1 - P) ** (n - x))
    
    # Compute the marginal probability
    marginal_prob = np.sum(likelihoods * Pr)
    
    # Compute the posterior using Bayes' Theorem
    posterior_prob = (likelihoods * Pr) / marginal_prob

    return posterior_prob
