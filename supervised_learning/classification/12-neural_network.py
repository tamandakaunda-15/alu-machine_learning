#!/usr/bin/env python3
"""Defines a neural network with one hidden layer"""
import numpy as np


class NeuralNetwork:
    """Neural Network class for binary classification"""

    def __init__(self, nx, nodes):
        """
        Initialize the neural network
        nx: number of input features
        nodes: number of nodes in the hidden layer
        """
        if not isinstance(nx, int) or nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int) or nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = np.zeros((nodes, 1))

        # Output layer
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    # Getters
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
        Performs forward propagation
        X: input data of shape (nx, m)
        Returns: A1, A2
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Computes cost using logistic regression
        Y: true labels of shape (1, m)
        A: predicted probabilities of shape (1, m)
        Returns: cost
        """
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the predictions of the neural network
        X: input data
        Y: true labels
        Returns: predictions and cost
        """
        A1, A2 = self.forward_prop(X)
        predictions = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return predictions, cost
