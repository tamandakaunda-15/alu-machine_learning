#!/usr/bin/env python3
"""Deep Neural Network class with save and load functionality"""
import numpy as np
import pickle
import os


class DeepNeuralNetwork:
    """Deep Neural Network performing binary classification"""
    def __init__(self, nx, layers):
        if not isinstance(nx, int) or nx < 1:
            raise ValueError("nx must be a positive integer")
        if (not isinstance(layers, list)) or len(layers) == 0 or not all(
                isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(1, self.L + 1):
            layer_size = layers[l - 1]
            prev_size = nx if l == 1 else layers[l - 2]
            self.weights[f'W{l}'] = (np.random.randn(layer_size, prev_size)
                                     * np.sqrt(2 / prev_size))
            self.weights[f'b{l}'] = np.zeros((layer_size, 1))

    def forward_prop(self, X):
        self.cache['A0'] = X
        for l in range(1, self.L + 1):
            Z = (self.weights[f'W{l}'] @ self.cache[f'A{l - 1}'] +
                 self.weights[f'b{l}'])
            if l != self.L:
                self.cache[f'A{l}'] = 1 / (1 + np.exp(-Z))  # Sigmoid
            else:
                self.cache[f'A{l}'] = 1 / (1 + np.exp(-Z))  # Sigmoid final
        return self.cache[f'A{self.L}'], self.cache

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A + 1e-8) +
                       (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        predictions = np.where(A >= 0.5, 1, 0)
        return predictions, self.cost(Y, A)

    def gradient_descent(self, Y, alpha=0.05):
        m = Y.shape[1]
        L = self.L
        A_prev = self.cache[f'A{L - 1}']
        A_curr = self.cache[f'A{L}']
        dZ = A_curr - Y

        for l in reversed(range(1, L + 1)):
            A_prev = self.cache[f'A{l - 1}']
            dW = dZ @ A_prev.T / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            self.weights[f'W{l}'] -= alpha * dW
            self.weights[f'b{l}'] -= alpha * db
            if l > 1:
                W_curr = self.weights[f'W{l}']
                dZ = (W_curr.T @ dZ) * (A_prev * (1 - A_prev))

    def train(self, X, Y, iterations=5000, alpha=0.05, graph=True, step=100):
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float) or alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)
            if i % step == 0 or i == iterations:
                cost = self.cost(Y, A)
                if graph:
                    print(f"Cost after {i} iterations: {cost}")
            self.gradient_descent(Y, alpha)
        return self.evaluate(X, Y)

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        if not os.path.isfile(filename):
            return None
        with open(filename, 'rb') as f:
            return pickle.load(f)
