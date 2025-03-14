#!/usr/bin/env python3
"""Performs convolution on grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding

    Args:
        images (numpy.ndarray): Grayscale images of shape (m, h, w)
        kernel (numpy.ndarray): Kernel for convolution of shape (kh, kw)
        padding (tuple): Padding (ph, pw)

    Returns:
        numpy.ndarray: The convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Compute output dimensions
    h_new = h + 2 * ph - kh + 1
    w_new = w + 2 * pw - kw + 1

    # Pad images
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    # Initialize output
    convolved = np.zeros((m, h_new, w_new))

    # Perform convolution
    for i in range(h_new):
        for j in range(w_new):
            convolved[:, i, j] = np.sum(
                images_padded[:, i:i + kh, j:j + kw] * kernel,
                axis=(1, 2)
            )

    return convolved
