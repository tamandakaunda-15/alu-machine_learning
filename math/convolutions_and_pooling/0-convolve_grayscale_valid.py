#!/usr/bin/env python3


"""
This module performs the convolution operation on grayscale 
images, specifically using 'valid' padding and iterating over 
the image with a given stride.
"""

import numpy as np

def convolve_grayscale_valid(images, kernel):
    """
    Performs a convolution on a grayscale image using 'valid' padding.

    Args:
        images (np.ndarray): Grayscale images.
        kernel (np.ndarray): The kernel/filter used for convolution.

    Returns:
        np.ndarray: Result of the convolution.
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate output image dimensions
    new_h = h - kh + 1
    new_w = w - kw + 1

    # Initialize output array
    output = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            for k in range(m):
                output[k, i, j] = np.sum(
                    images[k, i:i+kh, j:j+kw] * kernel
                )
    
    return output


def pool(images, kernel_size, stride, mode='max'):
    """
    Applies max or average pooling to images.

    Args:
        images (np.ndarray): Grayscale images.
        kernel_size (tuple): Pooling kernel size (height, width).
        stride (int): The stride for pooling.
        mode (str): 'max' or 'average' pooling mode.

    Returns:
        np.ndarray: Pooled images.
    """

    m, h, w = images.shape
    kh, kw = kernel_size

    # Calculate output dimensions
    new_h = (h - kh) // stride + 1
    new_w = (w - kw) // stride + 1

    # Initialize output array
    output = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            for k in range(m):
                img_slice = images[k, i*stride:i*stride+kh, j*stride:j*stride+kw]
                if mode == 'max':
                    output[k, i, j] = np.max(img_slice)
                elif mode == 'average':
                    output[k, i, j] = np.mean(img_slice)

    return output
