#!/usr/bin/env python3
"""
This module performs the convolution operation on grayscale 
images, specifically using 'valid' padding and iterating over 
the image with a given stride.
"""

def convolve_grayscale_valid(images, kernel):
    """
    Performs a convolution on a grayscale image using 'valid' padding.

    Args:
        images (list of lists of lists): Grayscale images.
        kernel (list of lists): The kernel/filter used for convolution.

    Returns:
        list of lists of lists: Result of the convolution.
    """

    m, h, w = len(images), len(images[0]), len(images[0][0])
    kh, kw = len(kernel), len(kernel[0])

    # Calculate output image dimensions
    new_h = h - kh + 1
    new_w = w - kw + 1

    # Initialize output array
    output = [[[0 for _ in range(new_w)] for _ in range(new_h)] 
              for _ in range(m)]

    for i in range(new_h):
        for j in range(new_w):
            for k in range(m):
                output[k][i][j] = sum(
                    images[k][i + x][j + y] * kernel[x][y] 
                    for x in range(kh) for y in range(kw)
                )
    
    return output


def pool(images, kernel_size, stride, mode='max'):
    """
    Applies max or average pooling to images.

    Args:
        images (list of lists of lists): Grayscale images.
        kernel_size (tuple): Pooling kernel size (height, width).
        stride (int): The stride for pooling.
        mode (str): 'max' or 'average' pooling mode.

    Returns:
        list of lists of lists: Pooled images.
    """

    m, h, w = len(images), len(images[0]), len(images[0][0])
    kh, kw = kernel_size

    # Calculate output dimensions
    new_h = (h - kh) // stride + 1
    new_w = (w - kw) // stride + 1

    # Initialize output array
    output = [[[0 for _ in range(new_w)] for _ in range(new_h)] 
              for _ in range(m)]

    for i in range(new_h):
        for j in range(new_w):
            for k in range(m):
                img_slice = [row[j*stride:j*stride+kw] 
                             for row in images[k][i*stride:i*stride+kh]]
                if mode == 'max':
                    output[k][i][j] = max(max(row) for row in img_slice)
                elif mode == 'average':
                    output[k][i][j] = sum(sum(row) for row in img_slice) / (
                        kh * kw
                    )

    return output
