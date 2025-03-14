#!/usr/bin/env python3
import numpy as np

def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Perform a convolution on images with channels.

    images: numpy.ndarray of shape (m, h, w, c) containing multiple images
    kernel: numpy.ndarray of shape (kh, kw, c) containing the kernel for the convolution
    padding: 'same', 'valid', or tuple (ph, pw) indicating padding in height and width
    stride: tuple (sh, sw) indicating the stride in height and width

    Returns: numpy.ndarray of the convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    # Handle padding
    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:  # Padding as tuple
        ph, pw = padding

    # Pad images
    images_padded = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant', constant_values=0
    )

    # Output dimensions
    out_h = (h + 2 * ph - kh) // sh + 1
    out_w = (w + 2 * pw - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, out_h, out_w))

    # Perform convolution
    for i in range(out_h):
        for j in range(out_w):
            # Extract the current region of the image
            region = images_padded[:, i * sh:i * sh + kh, 
                                   j * sw:j * sw + kw, :]
            # Perform element-wise multiplication with the kernel and sum
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))

    return output
