#!/usr/bin/env python3
import numpy as np

def convolve_grayscale_padding(images, kernel, padding):
    """
    Function to perform convolution with padding on grayscale images.

    Parameters:
    images (numpy.ndarray): Array of shape (m, h, w) containing m grayscale images.
    kernel (numpy.ndarray): Kernel/filter for the convolution.
    padding (tuple): Tuple (ph, pw) for padding in height and width directions.

    Returns:
    numpy.ndarray: Array of shape (m, h + 2*ph - kh + 1, w + 2*pw - kw + 1) with convolved images.
    """
    m, h, w = images.shape  # Extract number of images, height, and width
    kh, kw = kernel.shape  # Extract height and width of the kernel
    ph, pw = padding  # Extract padding values for height and width
    
    # Pad the images with zeros
    padded_images = np.pad(images, ((0,), (ph,), (pw,)), mode='constant', constant_values=0)
    
    # Calculate the dimensions of the output convolved image
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1
    
    # Prepare an array to hold the result of the convolution
    convolved_images = np.zeros((m, output_h, output_w))
    
    # Perform the convolution
    for i in range(m):
        for y in range(output_h):
            for x in range(output_w):
                # Extract the region of the image that corresponds to the kernel
                region = padded_images[i, y:y+kh, x:x+kw]
                # Apply the kernel on this region
                convolved_images[i, y, x] = np.sum(region * kernel)
    
    return convolved_images
