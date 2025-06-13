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
            TypeError: if layers is not a list or empty
            TypeError: if any element in layers is not a positive integer
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

        self.L = len(layers)          # Number of layers in the network
        self.cache = {}               # To store intermediary values during forward propagation
        self.weights = {}             # To store weights and biases

        # Initialize weights and biases using He et al. method
        for l in range(1, self.L + 1):
            layer_size = layers[l - 1]
            prev_layer_size = nx if l == 1 else layers[l - 2]

            self.weights[f"W{l}"] = (np.random.randn(layer_size, prev_layer_size) *
                                     np.sqrt(2 / prev_layer_size))
            self.weights[f"b{l}"] = np.zeros((layer_size, 1))
