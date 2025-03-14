"""
Module for performing convolutions on images using multiple kernels.
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Perform convolution on images using multiple kernels.

    Args:
        images (numpy.ndarray): Input images of shape (m, h, w, c).
        kernels (numpy.ndarray): Kernels for convolution of shape (kh, kw, c, nc).
        padding (str/tuple): Padding type ('same', 'valid', or tuple (ph, pw)).
        stride (tuple): Stride for height and width (sh, sw).

    Returns:
        numpy.ndarray: Convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    # Handle padding
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
        images_padded = np.pad(images, ((0,), (ph,), (pw,), (0,)), mode='constant')
    elif padding == 'valid':
        ph = pw = 0
        images_padded = images
    else:
        ph, pw = padding
        images_padded = np.pad(images, ((0,), (ph,), (pw,), (0,)), mode='constant')

    # Output dimensions
    h_out = (h + 2 * ph - kh) // sh + 1
    w_out = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, h_out, w_out, nc))

    for i in range(m):
        for j in range(nc):
            for y in range(0, h_out * sh, sh):
                for x in range(0, w_out * sw, sw):
                    output[i, y // sh, x // sw, j] = np.sum(
                        images_padded[i, y:y+kh, x:x+kw, :] * kernels[:, :, :, j]
                    )
    
    return output
