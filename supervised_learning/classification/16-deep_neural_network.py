#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network class for binary classification"""

    def __init__(self, nx, layers):
        """Initialize the deep neural network

        Args:
            nx (int): number of input features
            layers (list): number of nodes in each layer
        """
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate layers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(l, int) and l > 0 for l in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            layer_input_size = nx if l == 1 else layers[l - 2]
            self.weights[f"W{l}"] = (np.random.randn(layers[l - 1], layer_input_size) *
                                     np.sqrt(2 / layer_input_size))
            self.weights[f"b{l}"] = np.zeros((layers[l - 1], 1))
