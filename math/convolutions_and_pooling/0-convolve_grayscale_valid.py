#!/usr/bin/env python3
"""
Performs a valid convolution on grayscale images.
"""
import numpy as np

def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
        images (numpy.ndarray): Shape (m, h, w) containing multiple grayscale images.
            m: Number of images.
            h: Height in pixels of the images.
            w: Width in pixels of the images.
        kernel (numpy.ndarray): Shape (kh, kw) containing the kernel for the convolution.
            kh: Height of the kernel.
            kw: Width of the kernel.
    
    Returns:
        numpy.ndarray: Convolved images with shape (m, new_h, new_w).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    new_h = h - kh + 1
    new_w = w - kw + 1
    convolved = np.zeros((m, new_h, new_w))
    
    for i in range(new_h):
        for j in range(new_w):
            convolved[:, i, j] = np.sum(
                images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )
    
    return convolved
