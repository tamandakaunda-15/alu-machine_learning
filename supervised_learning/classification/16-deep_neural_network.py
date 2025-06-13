#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network class for binary classification"""

    def __init__(self, nx, layers):
        """
        Initialize a deep neural network

        Parameters:
        nx (int): number of input features
        layers (list): list of number of nodes in each layer
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

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        # Initialize weights and biases using He et al. method
        for l in range(1, self.L + 1):
            layer_input = nx if l == 1 else layers[l - 2]
            self.weights[f"W{l}"] = (
                np.random.randn(layers[l - 1], layer_input) *
                np.sqrt(2 / layer_input)
            )
            self.weights[f"b{l}"] = np.zeros((layers[l - 1], 1))
