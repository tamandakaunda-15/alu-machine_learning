#!/usr/bin/env python3
"""Neuron module that defines a single neuron for binary classification"""

import numpy as np


class Neuron:
    """Class that defines a single neuron for binary classification"""

    def __init__(self, nx):
        """
        Constructor method
        Args:
            nx (int): number of input features
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Weight vector initialized using a random normal distribution
        self.W = np.random.randn(1, nx)
        # Bias initialized to 0
        self.b = 0
        # Activated output initialized to 0
        self.A = 0

