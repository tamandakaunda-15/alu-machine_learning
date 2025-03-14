#!/usr/bin/env python3
import numpy as np
from math import ceil, floor

def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels.
    
    Parameters:
    images (numpy.ndarray): Shape (m, h, w, c), containing multiple images.
    kernels (numpy.ndarray): Shape (kh, kw, c, nc), containing the kernels.
    padding (str or tuple): 'same' or 'valid' or a tuple (ph, pw) for custom padding.
    stride (tuple): (sh, sw) for strides in the height and width directions.
    
    Returns:
    numpy.ndarray: Convolved images with shape (m, oh, ow, nc).
    """
    
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride
    
    if padding == 'same':
        ph = (sh * (h - 1) + kh - h) // 2
        pw = (sw * (w - 1) + kw - w) // 2
    elif padding == 'valid':
        ph = pw = 0
    else:  # padding is a tuple (ph, pw)
        ph, pw = padding
    
    # Calculate output dimensions
    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    
    # Pad the image if necessary
    images_padded = np.pad(images, ((0,), (ph,), (pw,), (0,)), mode='constant')
    
    # Prepare output array
    output = np.zeros((m, oh, ow, nc))
    
    # Perform convolution
    for i in range(m):  # Loop over the images
        for j in range(oh):  # Loop over the output height
            for k in range(ow):  # Loop over the output width
                # Apply kernels for each image
                for l in range(nc):  # Loop over the kernels
                    # Extract the region of interest from the image
                    region = images_padded[i, j*sh:j*sh+kh, k*sw:k*sw+kw, :]
                    # Perform the convolution (element-wise multiplication and summation)
                    output[i, j, k, l] = np.sum(region * kernels[:, :, :, l])
    
    return output
