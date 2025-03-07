#!/usr/bin/env python3


"""
This module performs the convolution operation on grayscale images, 
specifically using 'valid' padding and iterating over the image with a
given stride.It includes functions for performing the convolution,
 max pooling, and average pooling.
"""

import numpy as np
from math import ceil, floor


def convolve_grayscale_valid(images, kernel):
    """
    Performs a convolution on a grayscale image using 'valid' padding.

    Args:
        images (numpy.ndarray): A numpy array containing multiple
                                grayscale images.
        kernel (numpy.ndarray): The kernel/filter used for the convolution.

    Returns:
        numpy.ndarray: A numpy array containing the result of the convolution.
    """

    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate the dimensions of the output image
    new_h = h - kh + 1
    new_w = w - kw + 1

    # Initialize the output array
    output = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            output[:, i, j] = np.sum(
                images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )
    
    return output


def pool(images, kernel_size, stride, mode='max'):
    """
    Applies max or average pooling to an image or a set of images.

    Args:
        images (numpy.ndarray): A numpy array containing grayscale images.
        kernel_size (tuple): The size of the pooling kernel (height, width).
        stride (int): The stride for pooling.
        mode (str): The pooling mode - 'max' or 'average'.

    Returns:
        numpy.ndarray: A numpy array containing the pooled images.
    """

    m, h, w = images.shape
    kh, kw = kernel_size

    # Calculate output dimensions
    new_h = (h - kh) // stride + 1
    new_w = (w - kw) // stride + 1

    # Initialize the output array
    output = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            # Get the current slice of the image
            img_slice = images[:, i*stride:i*stride+kh, j*stride:j*stride+kw]
            if mode == 'max':
                output[:, i, j] = np.max(img_slice, axis=(1, 2))
            elif mode == 'average':
                output[:, i, j] = np.mean(img_slice, axis=(1, 2))

    return output
