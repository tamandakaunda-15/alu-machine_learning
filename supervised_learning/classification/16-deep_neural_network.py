#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network class for binary classification"""

    def __init__(self, nx, layers):
        """
        Initialize a deep neural network

        Args:
            nx (int): number of input features
            layers (list): list of nodes in each layer

        Raises:
            TypeError: if nx is not int
            ValueError: if nx < 1
            TypeError: if layers is not list or empty
            TypeError: if any layer size is not a positive integer
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)  # number of layers
        self.cache = {}  # dictionary to hold intermediary values
        self.weights = {}  # dictionary to hold weights and biases

        # Initialize weights and biases using He initialization (only one loop)
        for l in range(1, self.L + 1):
            nodes_prev = nx if l == 1 else layers[l - 2]
            nodes_curr = layers[l - 1]

            self.weights["W{}".format(l)] = (np.random.randn(nodes_curr, nodes_prev) *
                                            np.sqrt(2 / nodes_prev))
            self.weights["b{}".format(l)] = np.zeros((nodes_curr, 1))
