#!/usr/bin/env python3
"""Performs a convolution on grayscale images with padding and stride"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images

    Args:
        images (numpy.ndarray): Grayscale images of shape (m, h, w)
        kernel (numpy.ndarray): Kernel for convolution of shape (kh, kw)
        padding (tuple, str): Padding (ph, pw), 'same', or 'valid'
        stride (tuple): Stride (sh, sw)

    Returns:
        numpy.ndarray: The convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    else:
        ph, pw = padding

    # Compute output dimensions
    h_new = (h + 2 * ph - kh) // sh + 1
    w_new = (w + 2 * pw - kw) // sw + 1

    # Apply padding only if necessary
    if ph > 0 or pw > 0:
        images_padded = np.pad(
            images, ((0, 0), (ph, ph), (pw, pw)), mode='constant'
        )
    else:
        images_padded = images  # No padding for 'valid'

    # Initialize output
    convolved = np.zeros((m, h_new, w_new))

    # Perform convolution
    for i in range(h_new):
        for j in range(w_new):
            convolved[:, i, j] = np.sum(
                images_padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw]
                * kernel,
                axis=(1, 2)
            )

    return convolved
