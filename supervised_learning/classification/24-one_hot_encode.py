#!/usr/bin/env python3
"""Function to convert numeric class labels into a one-hot matrix"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix

    Parameters:
    Y (np.ndarray): array with shape (m,) containing numeric class labels
    classes (int): maximum number of classes

    Returns:
    np.ndarray: one-hot encoding of Y with shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if classes <= np.max(Y):
        return None

    try:
        one_hot = np.zeros((classes, Y.shape[0]))
        one_hot[Y, np.arange(Y.shape[0])] = 1
        return one_hot
    except Exception:
        return None
