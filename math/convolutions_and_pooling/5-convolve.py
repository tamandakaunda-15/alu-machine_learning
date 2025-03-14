#!/usr/bin/env python3
import numpy as np

"""
This module performs convolution on multiple images using multiple kernels.
It supports both 'same' and 'valid' padding, as well as customizable strides.
The `convolve` function takes images, kernels, padding, and stride as inputs,
and returns the convolved images.
"""

def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Perform convolution on multiple images using multiple kernels.

    images: numpy.ndarray of shape (m, h, w, c) containing multiple images
    kernels: numpy.ndarray of shape (kh, kw, c, nc) containing the kernels
    padding: 'same', 'valid', or a tuple (ph, pw) for padding
    stride: tuple (sh, sw) indicating the stride for the height and width

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    # Handle padding
    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
        padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    elif padding == 'valid':
        ph = pw = 0
        padded_images = images
    else:
        ph, pw = padding
        padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')

    # Output dimensions
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, out_h, out_w, nc))

    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            # Extract the region of the image that the kernel will convolve with
            region = padded_images[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]

            for k in range(nc):
                # Perform the convolution for each kernel
                output[:, i, j, k] = np.sum(region * kernels[:, :, :, k], axis=(1, 2, 3))

    return output
