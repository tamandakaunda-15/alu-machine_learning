#!/usr/bin/env python3
import numpy as np

"""
This module performs matrix multiplication using numpy arrays.
It provides a function to multiply two matrices, mat1 and mat2, 
and returns the resulting matrix.
"""

def mat_mul(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        return None
    return np.dot(mat1, mat2)
