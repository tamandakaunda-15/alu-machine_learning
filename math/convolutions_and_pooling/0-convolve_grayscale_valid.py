#!/usr/bin/env python3
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images using a kernel.
    
    Parameters:
    images (numpy.ndarray): A 3D numpy array of shape (m, h, w), where m is the number of images, h is the height, and w is the width.
    kernel (numpy.ndarray): A 2D numpy array representing the convolution kernel (filter).
    
    Returns:
    numpy.ndarray: A 3D numpy array of shape (m, h-kh+1, w-kw+1), where kh and kw are the height and width of the kernel.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_height = h - kh + 1
    output_width = w - kw + 1

    output = np.zeros((m, output_height, output_width))

    for i in range(m):
        for j in range(output_height):
            for k in range(output_width):
                # Extract region of the image matching the kernel size
                region = images[i, j:j+kh, k:k+kw]
                # Perform element-wise multiplication and sum the result
                output[i, j, k] = np.sum(region * kernel)

    return output


def pool_grayscale_max(images, pool_size, stride):
    """
    Performs max pooling on grayscale images.
    
    Parameters:
    images (numpy.ndarray): A 3D numpy array of shape (m, h, w), where m is the number of images, h is the height, and w is the width.
    pool_size (int): The size of the pooling window (usually 2 for 2x2 pooling).
    stride (int): The stride of the pooling operation.
    
    Returns:
    numpy.ndarray: A 3D numpy array containing the pooled images.
    """
    m, h, w = images.shape
    ph, pw = pool_size, pool_size  # Pooling window size
    output_height = (h - ph) // stride + 1
    output_width = (w - pw) // stride + 1

    output = np.zeros((m, output_height, output_width))

    for i in range(m):
        for j in range(output_height):
            for k in range(output_width):
                # Define the region of the image for pooling
                region = images[i, j*stride:j*stride+ph, k*stride:k*stride+pw]
                # Apply max pooling (take the maximum value in the region)
                output[i, j, k] = np.max(region)

    return output


def pool_grayscale_avg(images, pool_size, stride):
    """
    Performs average pooling on grayscale images.
    
    Parameters:
    images (numpy.ndarray): A 3D numpy array of shape (m, h, w), where m is the number of images, h is the height, and w is the width.
    pool_size (int): The size of the pooling window (usually 2 for 2x2 pooling).
    stride (int): The stride of the pooling operation.
    
    Returns:
    numpy.ndarray: A 3D numpy array containing the pooled images.
    """
    m, h, w = images.shape
    ph, pw = pool_size, pool_size  # Pooling window size
    output_height = (h - ph) // stride + 1
    output_width = (w - pw) // stride + 1

    output = np.zeros((m, output_height, output_width))

    for i in range(m):
        for j in range(output_height):
            for k in range(output_width):
                # Define the region of the image for pooling
                region = images[i, j*stride:j*stride+ph, k*stride:k*stride+pw]
                # Apply average pooling (compute the mean value of the region)
                output[i, j, k] = np.mean(region)

    return output
