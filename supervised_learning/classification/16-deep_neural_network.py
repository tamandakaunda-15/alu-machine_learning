#!/usr/bin/env python3
"""Defines a deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """Deep neural network class for binary classification"""

    def __init__(self, nx, layers):
        """Class constructor

        Args:
            nx (int): number of input features
            layers (list): number of nodes in each layer
        """
        # Check if nx is an integer
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        # Check if nx is positive
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        # Check if layers is a list of positive integers
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(i, int) and i > 0 for i in layers):
            raise TypeError("layers must be a list of positive integers")

        self.nx = nx
        self.layers = layers
        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            layer_size = layers[l - 1]
            prev_layer_size = nx if l == 1 else layers[l - 2]
            # He initialization for weights
            self.weights['W' + str(l)] = (np.random.randn(layer_size, prev_layer_size) *
                                          np.sqrt(2 / prev_layer_size))
            self.weights['b' + str(l)] = np.zeros((layer_size, 1))
