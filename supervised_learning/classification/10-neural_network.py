#!/usr/bin/env python3
"""NeuralNetwork with one hidden layer performing binary
classification (with forward propagation)"""
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

        # Private hidden layer parameters
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # Private output layer parameters
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    # Getter methods
    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network

        Parameters:
        - X: numpy.ndarray with shape (nx, m) input data

        Updates __A1 and __A2 with activated outputs
        Returns: __A1, __A2
        """
        # Linear combination for hidden layer
        Z1 = np.matmul(self.__W1, X) + self.__b1
        # Sigmoid activation for hidden layer
        self.__A1 = 1 / (1 + np.exp(-Z1))

        # Linear combination for output neuron
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        # Sigmoid activation for output neuron
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2
