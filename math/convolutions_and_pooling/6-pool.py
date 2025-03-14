#!/usr/bin/env python3
import numpy as np

"""
Module that performs pooling on images with channels. Supports max pooling 
and average pooling. The pooling operation is applied using a sliding window 
defined by the kernel shape and stride.
"""

def pool(images, kernel_shape, stride, mode='max'):
    """
    Perform pooling on images with channels.

    images: numpy.ndarray of shape (m, h, w, c) containing multiple images
    kernel_shape: tuple (kh, kw) for the kernel shape
    stride: tuple (sh, sw) indicating the stride in height and width
    mode: 'max' for max pooling, 'avg' for average pooling

    Returns: numpy.ndarray containing the pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Output dimensions
    out_h = (h - kh) // sh + 1
    out_w = (w - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, out_h, out_w, c))

    # Perform pooling
    for i in range(out_h):
        for j in range(out_w):
            # Extract the current region of the image
            region = images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]

            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

    return output
