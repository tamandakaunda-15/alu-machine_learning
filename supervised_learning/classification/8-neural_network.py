#!/usr/bin/env python3
"""NeuralNetwork with one hidden layer performing binary classification"""
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden layer"""
    def __init__(self, nx, nodes):
        # Validate nx
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Validate nodes
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Initialize hidden layer parameters
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0

        # Initialize output neuron parameters
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
