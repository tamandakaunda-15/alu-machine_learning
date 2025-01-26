#!/usr/bin/env python3
import numpy as np

"""
This module contains a function that performs matrix multiplication.
It takes two matrices, mat1 and mat2, and returns their product.
If the matrices cannot be multiplied due to incompatible dimensions, None is returned.
"""

def mat_mul(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        return None
    return np.dot(mat1, mat2)
