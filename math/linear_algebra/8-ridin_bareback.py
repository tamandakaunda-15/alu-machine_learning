#!/usr/bin/env python3
import numpy as np

"""
This module contains a function to perform matrix multiplication.
It takes two matrices as input and returns their element-wise multiplication 
if the matrices are compatible. Otherwise, it returns None.
"""

def mat_mul(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        return None
    return np.dot(mat1, mat2)
