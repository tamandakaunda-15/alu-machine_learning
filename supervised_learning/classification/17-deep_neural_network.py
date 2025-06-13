#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network for binary classification"""

    def __init__(self, nx, layers):
        """
        Constructor for DeepNeuralNetwork

        Args:
            nx (int): number of input features
            layers (list): number of nodes in each layer

        Raises:
            TypeError: if nx is not int
            ValueError: if nx < 1
            TypeError: if layers is not a list or empty or contains non-positive integers
        """
        # nx validation
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # layers validation
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # He initialization using only one loop
        for l in range(1, self.__L + 1):
            prev_layer = nx if l == 1 else layers[l - 2]
            self.__weights[f"W{l}"] = (np.random.randn(layers[l - 1], prev_layer)
                                       * np.sqrt(2 / prev_layer))
            self.__weights[f"b{l}"] = np.zeros((layers[l - 1], 1))

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights
