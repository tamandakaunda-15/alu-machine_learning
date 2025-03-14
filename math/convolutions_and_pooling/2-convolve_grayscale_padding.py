"""
Module for applying convolution to grayscale images with padding.
"""
import numpy as np

def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with padding.

    Parameters:
    images (numpy.ndarray): Input grayscale images of shape (m, h, w)
    kernel (numpy.ndarray): Kernel/filter for convolution of shape (kh, kw)
    padding (tuple): (ph, pw) padding for height and width

    Returns:
    numpy.ndarray: The convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    
    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant'
    )
    new_h = h + 2 * ph - kh + 1
    new_w = w + 2 * pw - kw + 1
    output = np.zeros((m, new_h, new_w))
    
    for i in range(new_h):
        for j in range(new_w):
            region = padded_images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))
    
    return output
