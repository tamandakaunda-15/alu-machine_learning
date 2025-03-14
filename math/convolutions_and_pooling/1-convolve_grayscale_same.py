#!/usr/bin/env python3
"""
Performs a same convolution on grayscale images.
"""
import numpy as np

def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
        images (numpy.ndarray): Shape (m, h, w) containing multiple grayscale images.
            m: Number of images.
            h: Height in pixels of the images.
            w: Width in pixels of the images.
        kernel (numpy.ndarray): Shape (kh, kw) containing the kernel for the convolution.
            kh: Height of the kernel.
            kw: Width of the kernel.

    Returns:
        numpy.ndarray: Convolved images with the same shape as the input images (m, h, w).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2
    
    # Pad images with zeros to maintain the original size after convolution
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    convolved = np.zeros((m, h, w))
    
    for i in range(h):
        for j in range(w):
            convolved[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2)
            )
    
    return convolved
