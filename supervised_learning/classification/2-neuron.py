#!/usr/bin/env python3
"""Neuron module with forward propagation"""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Initialize a Neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for weights"""
        return self.__W

    @property
    def b(self):
        """Getter for bias"""
        return self.__b

    @property
    def A(self):
        """Getter for activated output"""
        return self.__A

    def forward_prop(self, X):
        """
        Performs forward propagation using sigmoid activation

        Args:
            X (np.ndarray): input data of shape (nx, m)

        Returns:
            The activated output __A
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
