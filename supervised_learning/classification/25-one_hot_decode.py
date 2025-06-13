#!/usr/bin/env python3
"""Function to decode a one-hot matrix into a vector of class labels"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels

    Parameters:
    one_hot (np.ndarray): one-hot encoded matrix with shape (classes, m)

    Returns:
    np.ndarray: vector with shape (m,) containing the numeric class labels,
                or None on failure
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    if len(one_hot.shape) != 2:
        return None

    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
