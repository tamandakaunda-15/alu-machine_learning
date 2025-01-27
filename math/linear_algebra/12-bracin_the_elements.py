#!/usr/bin/env python3
import numpy as np

def np_elementwise(mat1, mat2):
    """
    Performs element-wise addition, subtraction, multiplication, and division
    of two matrices (or a matrix and a scalar).
    
    Args:
        mat1 (numpy.ndarray): The first input matrix.
        mat2 (numpy.ndarray): The second input matrix or scalar.
    
    Returns:
        tuple: A tuple containing the element-wise sum, difference, product, and quotient.
    """
    add = np.add(mat1, mat2)
    sub = np.subtract(mat1, mat2)
    mul = np.multiply(mat1, mat2)
    div = np.divide(mat1, mat2)
    
    return add, sub, mul, div
